#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "GaussianCloud.hpp"
#include "SogEncoder.hpp"

namespace fs = std::filesystem;

void print_usage() {
    std::cout << "Usage: splat-converter <input.ply> <output_path> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --bundle       Create a bundled .sog file (zip)\n";
    std::cout << "  --sh-iter <N>  K-Means iterations for SH (default: 2)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    bool bundle = false;
    int sh_iter = 2;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bundle") {
            bundle = true;
        } else if (arg == "--sh-iter" && i + 1 < argc) {
            sh_iter = std::stoi(argv[++i]);
        }
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Loading PLY: " << input_path << std::endl;
        GaussianCloud cloud = GaussianCloud::load_ply(input_path);
        
        std::cout << "Loaded " << cloud.size() << " gaussians." << std::endl;
        
        fs::path out_dir; 
        if (bundle) {
            // Create a temporary directory for files
            out_dir = fs::path(output_path).parent_path() / ("tmp_" + fs::path(output_path).stem().string());
        } else {
            out_dir = output_path;
        }

        if (fs::exists(out_dir)) {
            fs::remove_all(out_dir);
        }
        fs::create_directories(out_dir);

        // Standard bundle/unbundle
        SogEncoder::Options options;
        options.output_path = out_dir.string();
        options.bundle = bundle;
        options.sh_iterations = sh_iter;
        
        std::cout << "Encoding to " << (bundle ? "bundled SOG" : "SOG directory") << "..." << std::endl;
        SogEncoder encoder(cloud, options);
        encoder.encode();
        
        if (bundle) {
            std::cout << "Bundling into " << output_path << "..." << std::endl;
            // Use system zip command
            // -j: junk paths (store just filenames), -r: recursive (not strictly needed for flat)
            // -q: quiet
            std::string cmd = "zip -j -q " + output_path + " " + out_dir.string() + "/*";
            int ret = std::system(cmd.c_str());
            if (ret != 0) {
                std::cerr << "Warning: zip command failed with code " << ret << std::endl;
            } else {
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
