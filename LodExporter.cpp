#include "LodExporter.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <execution>

namespace fs = std::filesystem;

LodExporter::LodExporter(const GaussianCloud& cloud, const std::vector<uint8_t>& lods, const Options& options)
    : cloud_(cloud), lods_(lods), options_(options) {
}

LodExporter::Aabb LodExporter::compute_bounds(const std::vector<uint32_t>& indices) const {
    Aabb bounds = {
        {1e30f, 1e30f, 1e30f},
        {-1e30f, -1e30f, -1e30f}
    };
    
    // Compute bounds in parallel for large sets
    if (indices.size() > 10000) {
        #pragma omp parallel
        {
            Eigen::Vector3f local_min(1e30f, 1e30f, 1e30f);
            Eigen::Vector3f local_max(-1e30f, -1e30f, -1e30f);
            
            #pragma omp for
            for (size_t i = 0; i < indices.size(); ++i) {
                Eigen::Vector3f p = cloud_.positions.row(indices[i]);
                local_min = local_min.cwiseMin(p);
                local_max = local_max.cwiseMax(p);
            }
            
            #pragma omp critical
            {
                bounds.min = bounds.min.cwiseMin(local_min);
                bounds.max = bounds.max.cwiseMax(local_max);
            }
        }
    } else {
        for (uint32_t idx : indices) {
             Eigen::Vector3f p = cloud_.positions.row(idx);
             bounds.min = bounds.min.cwiseMin(p);
             bounds.max = bounds.max.cwiseMax(p);
        }
    }
    return bounds;
}

std::unique_ptr<LodExporter::Node> LodExporter::build_tree(std::vector<uint32_t>& indices) {
    auto node = std::make_unique<Node>();
    node->count = indices.size();
    node->bounds = compute_bounds(indices);
    
    int target_count = options_.chunk_count * 1024;
    float target_extent = (float)options_.chunk_extent;
    
    // Check split criteria
    if (node->count <= target_count && node->bounds.largest_dim() <= target_extent) {
        node->indices = std::move(indices);
        return node;
    }
    
    // Split
    int axis = 0;
    Eigen::Vector3f extent = node->bounds.max - node->bounds.min;
    if (extent.y() > extent.x() && extent.y() > extent.z()) axis = 1;
    if (extent.z() > extent.x() && extent.z() > extent.y()) axis = 2;
    
    size_t mid = indices.size() / 2;
    
    // Use nth_element for efficient median splitting
    std::nth_element(indices.begin(), indices.begin() + mid, indices.end(), 
        [&](uint32_t a, uint32_t b) {
            return cloud_.positions(a, axis) < cloud_.positions(b, axis);
        }
    );
    
    std::vector<uint32_t> left_indices(indices.begin(), indices.begin() + mid);
    std::vector<uint32_t> right_indices(indices.begin() + mid, indices.end());
    
    node->left = build_tree(left_indices);
    node->right = build_tree(right_indices);
    
    return node;
}

struct LodExporter::ChunkingState {
     // Map LOD level -> List of File Chunks (each is a list of indices)
    struct FileBucket {
        std::vector<std::vector<uint32_t>> chunks; // List of chunks, each chunk is list of indices
        size_t current_size = 0;
    };
    std::map<int, std::vector<FileBucket>> lod_files; // lod -> list of files
    std::vector<std::string> filenames;
    int max_lod = 0;
};

nlohmann::json LodExporter::process_node(Node* node, ChunkingState& state) {
    nlohmann::json json;
    
    // Bounds
    std::vector<float> min = {node->bounds.min.x(), node->bounds.min.y(), node->bounds.min.z()};
    std::vector<float> max = {node->bounds.max.x(), node->bounds.max.y(), node->bounds.max.z()};
    json["bound"] = {
        {"min", min},
        {"max", max}
    };
    
    if (!node->is_leaf()) {
        json["children"] = nlohmann::json::array();
        json["children"].push_back(process_node(node->left.get(), state));
        json["children"].push_back(process_node(node->right.get(), state));
        return json;
    }
    
    // Leaf node: bin indices by LOD
    std::map<int, std::vector<uint32_t>> bins;
    for (uint32_t idx : node->indices) {
        int lod = 0;
        if (!lods_.empty()) {
            lod = lods_[idx];
        }
        bins[lod].push_back(idx);
    }
    
    nlohmann::json lods_json = nlohmann::json::object();
    int target_bin_size = options_.chunk_count * 1024;

    for (auto& [lod, indices] : bins) {
        if (state.lod_files.find(lod) == state.lod_files.end()) {
             state.lod_files[lod].push_back(ChunkingState::FileBucket());
        }
        
        std::vector<ChunkingState::FileBucket>& file_list = state.lod_files[lod];
        
        // Check if current file is full
        if (file_list.back().current_size + indices.size() > target_bin_size && file_list.back().current_size > 0) {
            file_list.push_back(ChunkingState::FileBucket());
        }
        
        ChunkingState::FileBucket& current_bucket = file_list.back();
        int file_index = file_list.size() - 1;
        
        // Construct filename relative to output dir
        std::string filename = std::to_string(lod) + "_" + std::to_string(file_index) + "/meta.json";
        
        // Add filename to global list if new
        auto it = std::find(state.filenames.begin(), state.filenames.end(), filename);
        int global_file_index;
        if (it == state.filenames.end()) {
            state.filenames.push_back(filename);
            global_file_index = state.filenames.size() - 1;
        } else {
            global_file_index = std::distance(state.filenames.begin(), it);
        }
        
        // Offset is the accumulated size in this file bucket so far
        size_t offset = current_bucket.current_size;
        
        // Add indices to bucket
        current_bucket.chunks.push_back(indices);
        current_bucket.current_size += indices.size();
        
        nlohmann::json lod_entry;
        lod_entry["file"] = global_file_index;
        lod_entry["offset"] = offset;
        lod_entry["count"] = indices.size();
        
        lods_json[std::to_string(lod)] = lod_entry;
        
        state.max_lod = std::max(state.max_lod, lod + 1);
    }
    
    json["lods"] = lods_json;
    return json;
}

void LodExporter::export_lods() {
    std::cout << "Building acceleration structure for " << cloud_.size() << " gaussians..." << std::endl;
    
    std::vector<uint32_t> indices(cloud_.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    auto root = build_tree(indices);
    
    std::cout << "Processing chunks..." << std::endl;
    ChunkingState state;
    nlohmann::json tree_json = process_node(root.get(), state);
    
    // Write Lods
    std::cout << "Writing " << state.filenames.size() << " chunk files..." << std::endl;
    
    fs::path base_should_be_dir = fs::path(options_.output_path).parent_path();
    fs::create_directories(base_should_be_dir);

    // For each LOD level
    for (auto& [lod, file_buckets] : state.lod_files) {
        // For each file bucket in that LOD
        for (size_t i = 0; i < file_buckets.size(); ++i) {
            const auto& bucket = file_buckets[i];
            if (bucket.current_size == 0) continue;
            
            // Flatten indices
            std::vector<size_t> flat_indices;
            flat_indices.reserve(bucket.current_size);
            for (const auto& chunk : bucket.chunks) {
                flat_indices.insert(flat_indices.end(), chunk.begin(), chunk.end());
            }
            
            // Create subset
            GaussianCloud subset_cloud = cloud_.subset(flat_indices);
            
            // Setup output path: output_dir / <lod>_<i> / meta.json
            fs::path sub_dir = base_should_be_dir / (std::to_string(lod) + "_" + std::to_string(i));
            fs::create_directories(sub_dir);
            
            SogEncoder::Options enc_opts;
            enc_opts.output_path = sub_dir.string();
            enc_opts.sh_iterations = options_.sh_iterations;
            enc_opts.bundle = false; // Sub-chunks are always unbundled directories
            
            // Encode
            std::cout << "  Writing " << sub_dir << " (" << flat_indices.size() << " points)" << std::endl;
            SogEncoder encoder(subset_cloud, enc_opts);
            encoder.encode();
        }
    }
    
    // Write lod-meta.json
    nlohmann::json meta;
    meta["lodLevels"] = state.max_lod;
    meta["filenames"] = state.filenames;
    meta["tree"] = tree_json;
    
    std::ofstream file(options_.output_path);
    file << meta.dump(0); // Minimized JSON
    
    std::cout << "Wrote " << options_.output_path << std::endl;
}
