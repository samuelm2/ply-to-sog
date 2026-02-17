#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <miniz.h>
#include "SogEncoder.hpp"
#include "LodExporter.hpp"

namespace fs = std::filesystem;

void print_usage() {
    std::cout << "Usage: ply-to-sog [input.ply -l <lod> ...] <output_path> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --bundle       Create a bundled .sog file (zip) (Single input only)\n";
    std::cout << "  --sh-iter      <N>  K-Means iterations for SH clustering (default: 10)\n";
    std::cout << "  --chunk-count  <N>  Target count per chunk in K (default: 512)\n";
    std::cout << "  --chunk-extent <N>  Target size in world units (default: 16)\n";
    std::cout << "  -H, --filter-harmonics <N> Set max SH degree (0, 1, 2, 3). Default: keep all\n";
}

struct InputFile {
    std::string path;
    int lod = 0;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::vector<InputFile> inputs;
    std::string output_path;
    bool bundle = false;
    int sh_iter = 10;
    int chunk_count = 512;
    int chunk_extent = 16;
    int max_sh_degree = -1; // -1 = keep all
    
    // Parse arguments manually to handle mixed positional and optional args
    // Heuristic: Last positional arg is output.
    // Preceding positional args are inputs.
    // optional args start with -
    
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);

    // Parse arguments
    inputs.clear();
    output_path = "";
    
    for (size_t i = 0; i < args.size(); ++i) {
         if (args[i] == "--bundle") {
             bundle = true;
         } else if (args[i] == "--sh-iter" || args[i] == "--k-means-iter") {
             if (i + 1 < args.size()) sh_iter = std::stoi(args[++i]);
         } else if (args[i] == "--chunk-count" || args[i] == "-C") {
             if (i + 1 < args.size()) chunk_count = std::stoi(args[++i]);
         } else if (args[i] == "--chunk-extent" || args[i] == "-X") {
             if (i + 1 < args.size()) chunk_extent = std::stoi(args[++i]);
         } else if (args[i] == "--filter-harmonics" || args[i] == "-H") {
             if (i + 1 < args.size()) max_sh_degree = std::stoi(args[++i]);
         } else if (args[i] == "-l" || args[i] == "--lod") {
             if (i + 1 < args.size()) {
                 int lod = std::stoi(args[++i]);
                 if (!inputs.empty()) {
                     inputs.back().lod = lod;
                 } else {
                     std::cerr << "Error: -l flag must follow an input file." << std::endl;
                     return 1;
                 }
             }
         } else if (args[i][0] != '-') {
             // Positional: could be input or output.
             // We won't know until the end. So we assume it's input, and the very last one is output.
             inputs.push_back({args[i], 0});
         }
    }
    
    if (inputs.empty()) {
        print_usage();
        return 1;
    }
    
    // The last "input" is actually the output path
    output_path = inputs.back().path;
    inputs.pop_back();
    
    if (inputs.empty()) {
        std::cerr << "Error: No input files specified." << std::endl;
        return 1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        GaussianCloud merged_cloud;
        std::vector<uint8_t> all_lods;
        
        for (const auto& input : inputs) {
            std::cout << "reading '" << input.path << "' (LOD " << input.lod << ")..." << std::endl;
            GaussianCloud cloud = GaussianCloud::load_ply(input.path);
            
            // Append LODs
            size_t current_size = all_lods.size();
            all_lods.resize(current_size + cloud.size(), (uint8_t)input.lod);
            
            if (merged_cloud.size() == 0) {
                merged_cloud = std::move(cloud);
            } else {
                merged_cloud.append(cloud);
            }
        }
        
        std::cout << "Total loaded: " << merged_cloud.size() << " gaussians" << std::endl;
        
        if (max_sh_degree >= 0) {
            std::cout << "Filtering SH to degree " << max_sh_degree << "..." << std::endl;
            merged_cloud.filter_harmonics(max_sh_degree);
        }
        
        // Mode 1: Multi-LOD Chunked Export
        if (inputs.size() > 1 || fs::path(output_path).extension() == ".json") {
            LodExporter::Options opts;
            opts.output_path = output_path;
            opts.chunk_count = chunk_count;
            opts.chunk_extent = chunk_extent;
            opts.sh_iterations = sh_iter;
            
            LodExporter exporter(merged_cloud, all_lods, opts);
            exporter.export_lods();
            
        } else {
            // Mode 2: Single File Standard Export (Legacy Mode)
            
            fs::path out_dir; 
            if (bundle) {
                // Check if output zip already exists
                if (fs::exists(output_path)) {
                    throw std::runtime_error("Output file already exists: " + output_path);
                }

                // Create a unique temporary directory for files
                auto now = std::chrono::system_clock::now().time_since_epoch().count();
                out_dir = fs::path(output_path).parent_path() / ("tmp_" + fs::path(output_path).stem().string() + "_" + std::to_string(now));
                fs::create_directories(out_dir);
                
                std::cout << "writing '" << output_path << "'..." << std::endl;

            } else {
                out_dir = output_path;
                // Check if output directory already exists and is not empty
                if (fs::exists(out_dir) && !fs::is_empty(out_dir)) {
                    throw std::runtime_error("Output directory already exists and is not empty: " + output_path);
                }
                fs::create_directories(out_dir);
                std::cout << "writing '" << output_path << "'..." << std::endl;
            }

            // Standard bundle/unbundle
            SogEncoder::Options options;
            options.output_path = out_dir.string();
            options.bundle = bundle;
            options.sh_iterations = sh_iter;
            
            SogEncoder encoder(merged_cloud, options);
            encoder.encode();
            
            if (bundle) {
                std::cout << "Bundling into " << output_path << "..." << std::endl;
                
                mz_zip_archive zip_archive;
                memset(&zip_archive, 0, sizeof(zip_archive));

                if (!mz_zip_writer_init_file(&zip_archive, output_path.c_str(), 0)) {
                    throw std::runtime_error("Failed to create zip file");
                }

                for (const auto& entry : fs::directory_iterator(out_dir)) {
                    if (entry.is_regular_file()) {
                        std::string filename = entry.path().filename().string();
                        std::string ext = entry.path().extension().string();
                        
                        int level = MZ_BEST_COMPRESSION;
                        if (ext == ".webp") {
                            level = MZ_NO_COMPRESSION;
                        }

                        if (!mz_zip_writer_add_file(&zip_archive, filename.c_str(), entry.path().string().c_str(), nullptr, 0, level)) {
                             mz_zip_writer_end(&zip_archive);
                             throw std::runtime_error("Failed to add file to zip: " + filename);
                        }
                    }
                }

                if (!mz_zip_writer_finalize_archive(&zip_archive)) {
                    mz_zip_writer_end(&zip_archive);
                    throw std::runtime_error("Failed to finalize zip archive");
                }

                mz_zip_writer_end(&zip_archive);
                
                // Cleanup
                fs::remove_all(out_dir);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Done in " << elapsed.count() << "s." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
