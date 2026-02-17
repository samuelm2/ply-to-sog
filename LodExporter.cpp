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
    float max_val = std::numeric_limits<float>::max();
    float lowest_val = std::numeric_limits<float>::lowest();
    Aabb bounds = {
        {max_val, max_val, max_val},
        {lowest_val, lowest_val, lowest_val}
    };
    
    // Compute bounds in parallel for large sets
    if (indices.size() > 10000) {
        #pragma omp parallel
        {
            float max_val = std::numeric_limits<float>::max();
            float lowest_val = std::numeric_limits<float>::lowest();
            Eigen::Vector3f local_min(max_val, max_val, max_val);
            Eigen::Vector3f local_max(lowest_val, lowest_val, lowest_val);
            
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
    
    // Stop splitting if node is small enough (matches splat-transform behavior)
    if (indices.size() <= 256) {
        node->indices = std::move(indices);
        return node;
    }

    // Check split criteria (target chunk size/extent)
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
    
    // Recursive case: Internal Node
    if (!node->is_leaf()) {
        json["children"] = nlohmann::json::array();
        
        nlohmann::json left_json = process_node(node->left.get(), state);
        nlohmann::json right_json = process_node(node->right.get(), state);
        
        json["children"].push_back(left_json);
        json["children"].push_back(right_json);
        
        // Compute union of children bounds for this node
        std::vector<float> min = {
            std::min((float)left_json["bound"]["min"][0], (float)right_json["bound"]["min"][0]),
            std::min((float)left_json["bound"]["min"][1], (float)right_json["bound"]["min"][1]),
            std::min((float)left_json["bound"]["min"][2], (float)right_json["bound"]["min"][2])
        };
        std::vector<float> max = {
            std::max((float)left_json["bound"]["max"][0], (float)right_json["bound"]["max"][0]),
            std::max((float)left_json["bound"]["max"][1], (float)right_json["bound"]["max"][1]),
            std::max((float)left_json["bound"]["max"][2], (float)right_json["bound"]["max"][2])
        };
        
        json["bound"] = {
            {"min", min},
            {"max", max}
        };
        
        return json;
    }
    
    // Base case: Leaf Node
    // Bin indices AND compute full bounds
    std::map<int, std::vector<uint32_t>> bins;
    
    // Initialize bounds with inverted infinity
    float inf = std::numeric_limits<float>::max();
    Eigen::Vector3f min_b(inf, inf, inf);
    Eigen::Vector3f max_b(-inf, -inf, -inf);
    
    // Bounds can be computed in parallel if node is large, but usually nodes are small (<=256 or so)
    // For safety with std::map access, we keep this serial or use thread-local storage.
    // Given the small node size, serial is fine.
    
    for (uint32_t idx : node->indices) {
        // 1. Binning
        int lod = 0;
        if (!lods_.empty()) {
            lod = lods_[idx];
        }
        bins[lod].push_back(idx);
        
        // 2. Bounds Calculation (Full Extent)
        Eigen::Vector3f p = cloud_.positions.row(idx);
        Eigen::Vector3f s = cloud_.scales.row(idx);
        // Exponentiate scales (log-scale in PLY)
        s = Eigen::Vector3f(std::exp(s.x()), std::exp(s.y()), std::exp(s.z()));
        
        Eigen::Vector4f q_raw = cloud_.rotations.row(idx);
        Eigen::Quaternionf q(q_raw(0), q_raw(1), q_raw(2), q_raw(3));
        q.normalize();
        Eigen::Matrix3f R = q.toRotationMatrix();
        
        // Radius along axes = Sum(|R_ij| * s_j)
        Eigen::Vector3f radius;
        radius.x() = std::abs(R(0,0))*s.x() + std::abs(R(0,1))*s.y() + std::abs(R(0,2))*s.z();
        radius.y() = std::abs(R(1,0))*s.x() + std::abs(R(1,1))*s.y() + std::abs(R(1,2))*s.z();
        radius.z() = std::abs(R(2,0))*s.x() + std::abs(R(2,1))*s.y() + std::abs(R(2,2))*s.z();
        
        Eigen::Vector3f local_min = p - radius;
        Eigen::Vector3f local_max = p + radius;

        // Skip invalid bounds (NaN/Inf) to match splat-transform robustness
        if (!std::isfinite(local_min.x()) || !std::isfinite(local_min.y()) || !std::isfinite(local_min.z()) ||
            !std::isfinite(local_max.x()) || !std::isfinite(local_max.y()) || !std::isfinite(local_max.z())) {
            continue;
        }
        
        min_b = min_b.cwiseMin(local_min);
        max_b = max_b.cwiseMax(local_max);
    }
    
    // Handle case where all points were invalid
    std::vector<float> min_vec, max_vec;
    if (min_b.x() > max_b.x()) {
         min_vec = {0,0,0};
         max_vec = {0,0,0};
    } else {
         min_vec = {min_b.x(), min_b.y(), min_b.z()};
         max_vec = {max_b.x(), max_b.y(), max_b.z()};
    }
    json["bound"] = {
        {"min", min_vec},
        {"max", max_vec}
    };
    
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
            
            for (auto& chunk : bucket.chunks) {
                // Sort chunk by Morton Order locally (using chunk bounds)
                if (chunk.empty()) continue;
                
                // Compute bounds for this chunk
                Eigen::Vector3f min(1e30f, 1e30f, 1e30f);
                Eigen::Vector3f max(-1e30f, -1e30f, -1e30f);
                for (uint32_t idx : chunk) {
                    Eigen::Vector3f p = cloud_.positions.row(idx);
                    min = min.cwiseMin(p);
                    max = max.cwiseMax(p);
                }
                // Expand slightly
                min.array() -= 0.01f;
                max.array() += 0.01f;
                
                struct IndexedMorton {
                    uint32_t index;
                    uint32_t code;
                };
                std::vector<IndexedMorton> morton(chunk.size());
                
                for (size_t k = 0; k < chunk.size(); ++k) {
                    uint32_t idx = chunk[k];
                    morton[k].index = idx;
                    Eigen::Vector3f p = cloud_.positions.row(idx);
                    morton[k].code = SogEncoder::morton3D(p.x(), p.y(), p.z(), min, max);
                }
                
                std::sort(morton.begin(), morton.end(), [](const IndexedMorton& a, const IndexedMorton& b) {
                    return a.code < b.code;
                });
                
                // Append sorted indices to flat list
                for (const auto& m : morton) {
                    flat_indices.push_back(m.index);
                }
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
            enc_opts.sort = false;   // Disable global sort, use our local chunked sort order
            
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
