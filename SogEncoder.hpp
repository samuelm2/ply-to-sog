#pragma once

#include "GaussianCloud.hpp"
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <nlohmann/json.hpp>

// WebP headers
#include <webp/encode.h>

class SogEncoder {
public:
    struct Options {
        std::string output_path;
        bool bundle = false;
        int sh_iterations = 2; // K-means iterations (default low for speed)
        int width_hint = 0; // 0 = auto
    };

    // Standard constructor (encodes entire cloud)
    SogEncoder(const GaussianCloud& cloud, const Options& options);

    // Subset constructor (encodes only specified indices)
    SogEncoder(const GaussianCloud& cloud, const std::vector<size_t>& subset_indices, const Options& options);
    
    // Main entry point
    void encode();

private:
    const GaussianCloud& cloud_;
    Options options_;
    
    // Morton sorted indices
    std::vector<uint32_t> indices_;
    
    // Dimensions
    int width_;
    int height_;
    int padded_count_;
    
    // Helpers
    void compute_morton_indices();
    void write_webp(const std::string& filename, const uint8_t* data, int w, int h) const;
    void write_file(const std::string& filename, const std::string& content) const;
    void write_file(const std::string& filename, const std::vector<uint8_t>& content) const;

    // Encoding steps
    nlohmann::json encode_positions(); // returns min/max
    void encode_quaternions();
    nlohmann::json encode_scales();    // returns codebook
    nlohmann::json encode_colors();    // returns codebook
    nlohmann::json encode_sh();        // returns shN object
};
