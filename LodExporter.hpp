#pragma once

#include "GaussianCloud.hpp"
#include "SogEncoder.hpp"
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

class LodExporter {
public:
    struct Options {
        std::string output_path;
        int chunk_count = 512;   // Target count per chunk (512k) is handled by using 512 * 1024
        int chunk_extent = 16;   // Target size in world units
        int sh_iterations = 10;
        bool bundle = false;
    };

    LodExporter(const GaussianCloud& cloud, const std::vector<uint8_t>& lods, const Options& options);
    void export_lods();

private:
    const GaussianCloud& cloud_;
    const std::vector<uint8_t>& lods_;
    Options options_;
    
    struct Aabb {
        Eigen::Vector3f min;
        Eigen::Vector3f max;
        
        float largest_dim() const {
             Eigen::Vector3f dims = max - min;
             return std::max({dims.x(), dims.y(), dims.z()});
        }
    };

    struct Node {
        Aabb bounds;
        int count = 0;
        std::vector<uint32_t> indices; // Only leaf nodes have indices
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        bool is_leaf() const { return !left && !right; }
    };
    
    struct MetaLod {
        int file_index;
        size_t offset;
        int count;
    };

    // Tree building
    std::unique_ptr<Node> build_tree(std::vector<uint32_t>& indices);
    
    // Chunking and Writing
    struct ChunkingState;
    nlohmann::json process_node(Node* node, ChunkingState& state);
    
    // Helper to calculate bounds of a set of indices
    Aabb compute_bounds(const std::vector<uint32_t>& indices) const;
};
