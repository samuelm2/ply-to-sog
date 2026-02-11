#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <algorithm>
#include <Eigen/Dense>
#include <omp.h>

// SH Constants
constexpr float SH_C0 = 0.28209479177387814f;
constexpr float SH_C1 = 0.4886025119029199f;

struct GaussianCloud {
    // Core data (N gaussians)
    Eigen::MatrixXf positions;     // N x 3
    Eigen::MatrixXf scales;        // N x 3
    Eigen::MatrixXf rotations;     // N x 4 (quaternions: w, x, y, z)
    Eigen::VectorXf opacities;     // N
    Eigen::MatrixXf sh_dc;         // N x 3
    Eigen::MatrixXf sh_rest;       // N x (n_coeffs * 3)
    
    // Cached covariances (computed lazily)
    mutable std::vector<Eigen::Matrix3f> covariances;
    bool covariances_valid = false;
    bool lazy_covariance = false;  // If true, compute on-demand instead of storing
    
    size_t size() const { return positions.rows(); }

    // Compute covariance for a single gaussian
    Eigen::Matrix3f get_covariance(size_t i) const {
        if (!lazy_covariance && covariances_valid) {
            return covariances[i];
        }
        
        // Compute on-demand
        Eigen::Vector4f q = rotations.row(i).normalized();
        float w = q(0), x = q(1), y = q(2), z = q(3);
        
        Eigen::Matrix3f R;
        R << 1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y,
             2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x,
             2*x*z - 2*w*y,         2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y;
        
        Eigen::Vector3f s = scales.row(i);
        Eigen::Matrix3f S_sq = Eigen::Vector3f(s(0)*s(0), s(1)*s(1), s(2)*s(2)).asDiagonal();
        
        return R * S_sq * R.transpose();
    }
    
    // Set covariance (only stores if not in lazy mode)
    void set_covariance(size_t i, const Eigen::Matrix3f& cov) {
        if (!lazy_covariance) {
            if (covariances.size() <= i) {
                covariances.resize(i + 1);
            }
            covariances[i] = cov;
        }
    }
    
    void compute_covariances() {
        if (covariances_valid || lazy_covariance) return;
        
        size_t n = size();
        covariances.resize(n);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            covariances[i] = get_covariance(i);
        }
        covariances_valid = true;
    }
    
    // Subset extraction
    GaussianCloud subset(const std::vector<size_t>& indices) const {
        GaussianCloud result;
        size_t n = indices.size();
        
        result.positions.resize(n, 3);
        result.scales.resize(n, 3);
        result.rotations.resize(n, 4);
        result.opacities.resize(n);
        result.sh_dc.resize(n, 3);
        if (sh_rest.cols() > 0) {
            result.sh_rest.resize(n, sh_rest.cols());
        } else {
             result.sh_rest.resize(n, 0);
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t idx = indices[i];
            result.positions.row(i) = positions.row(idx);
            result.scales.row(i) = scales.row(idx);
            result.rotations.row(i) = rotations.row(idx);
            result.opacities(i) = opacities(idx);
            result.sh_dc.row(i) = sh_dc.row(idx);
            if (sh_rest.cols() > 0) {
                result.sh_rest.row(i) = sh_rest.row(idx);
            }
        }
        
        return result;
    }

    // Manual Append
    void append(const GaussianCloud& other) {
        size_t current_n = size();
        size_t other_n = other.size();
        size_t new_n = current_n + other_n;

        positions.conservativeResize(new_n, Eigen::NoChange);
        scales.conservativeResize(new_n, Eigen::NoChange);
        rotations.conservativeResize(new_n, Eigen::NoChange);
        opacities.conservativeResize(new_n);
        sh_dc.conservativeResize(new_n, Eigen::NoChange);
        if (sh_rest.cols() > 0 || other.sh_rest.cols() > 0) {
             if (sh_rest.cols() == 0 && other.sh_rest.cols() > 0) sh_rest.resize(current_n, other.sh_rest.cols()); // init if empty
             sh_rest.conservativeResize(new_n, Eigen::NoChange);
        }

        positions.bottomRows(other_n) = other.positions;
        scales.bottomRows(other_n) = other.scales;
        rotations.bottomRows(other_n) = other.rotations;
        opacities.tail(other_n) = other.opacities;
        sh_dc.bottomRows(other_n) = other.sh_dc;
        if (sh_rest.cols() > 0) {
             sh_rest.bottomRows(other_n) = other.sh_rest;
        }
    }
    
    // Load from PLY
    static GaussianCloud load_ply(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        // Parse Header
        std::string line;
        int vertex_count = 0;
        std::map<std::string, std::string> property_types;
        std::map<std::string, int> property_offsets; // byte offset valid for mixed types if we did that, but assume all float for splats usually?
        // Actually, splats are often all floats, but rotation/scale are often just float.
        // Standard Gaussian Splat PLY: x,y,z, nx,ny,nz, f_dc_0,1,2, f_rest_..., opacity, scale_0,1,2, rot_0,1,2,3
        
        std::vector<std::string> properties_in_order;
        
        while (std::getline(file, line)) {
            if (line == "end_header") break;
            std::stringstream ss(line);
            std::string token;
            ss >> token;
            if (token == "element") {
                ss >> token;
                if (token == "vertex") {
                    ss >> vertex_count;
                }
            } else if (token == "property") {
                std::string type, name;
                ss >> type >> name;
                properties_in_order.push_back(name);
            }
        }

        if (vertex_count == 0) throw std::runtime_error("No vertices found in PLY");

        GaussianCloud cloud;
        cloud.positions.resize(vertex_count, 3);
        cloud.scales.resize(vertex_count, 3);
        cloud.rotations.resize(vertex_count, 4);
        cloud.opacities.resize(vertex_count);
        cloud.sh_dc.resize(vertex_count, 3);
        
        // Count f_rest to size sh_rest
        int f_rest_count = 0;
        for (const auto& p : properties_in_order) {
            if (p.find("f_rest_") == 0) f_rest_count++;
        }
        cloud.sh_rest.resize(vertex_count, f_rest_count);

        // Read Data
        // Assumes all data is float32 (standard for these PLYs)
        // If not, would need more complex parsing. 
        // We will read vertex by vertex.
        
        // Mapping from property index to destination
        struct PropTarget {
            enum Type { POS, SCALE, ROT, OPACITY, SH_DC, SH_REST, IGNORE };
            Type type;
            int index;
        };
        
        std::vector<PropTarget> targets;
        for (const auto& name : properties_in_order) {
            PropTarget t = {PropTarget::IGNORE, 0};
            if (name == "x") { t.type = PropTarget::POS; t.index = 0; }
            else if (name == "y") { t.type = PropTarget::POS; t.index = 1; }
            else if (name == "z") { t.type = PropTarget::POS; t.index = 2; }
            else if (name == "scale_0") { t.type = PropTarget::SCALE; t.index = 0; }
            else if (name == "scale_1") { t.type = PropTarget::SCALE; t.index = 1; }
            else if (name == "scale_2") { t.type = PropTarget::SCALE; t.index = 2; }
            else if (name == "rot_0") { t.type = PropTarget::ROT; t.index = 0; }
            else if (name == "rot_1") { t.type = PropTarget::ROT; t.index = 1; }
            else if (name == "rot_2") { t.type = PropTarget::ROT; t.index = 2; }
            else if (name == "rot_3") { t.type = PropTarget::ROT; t.index = 3; }
            else if (name == "opacity") { t.type = PropTarget::OPACITY; t.index = 0; }
            else if (name == "f_dc_0") { t.type = PropTarget::SH_DC; t.index = 0; }
            else if (name == "f_dc_1") { t.type = PropTarget::SH_DC; t.index = 1; }
            else if (name == "f_dc_2") { t.type = PropTarget::SH_DC; t.index = 2; }
            else if (name.find("f_rest_") == 0) {
                t.type = PropTarget::SH_REST;
                t.index = std::stoi(name.substr(7));
            }
            targets.push_back(t);
        }

        // Read all data into a buffer first for speed? Or simpler loop.
        // vertices * props * 4 bytes. 
        size_t props_per_vertex = properties_in_order.size();
        std::vector<float> vertex_buffer(props_per_vertex); // buffer for one vertex

        for (int i = 0; i < vertex_count; ++i) {
             file.read(reinterpret_cast<char*>(vertex_buffer.data()), props_per_vertex * sizeof(float));
             if (file.gcount() != static_cast<std::streamsize>(props_per_vertex * sizeof(float))) {
                 break;
             }

             for (size_t p = 0; p < props_per_vertex; ++p) {
                 float val = vertex_buffer[p];
                 const auto& target = targets[p];
                 
                 switch (target.type) {
                     case PropTarget::POS: cloud.positions(i, target.index) = val; break;
                     case PropTarget::SCALE: cloud.scales(i, target.index) = val; break;
                     case PropTarget::ROT: cloud.rotations(i, target.index) = val; break;
                     case PropTarget::OPACITY: cloud.opacities(i) = val; break;
                     case PropTarget::SH_DC: cloud.sh_dc(i, target.index) = val; break;
                     case PropTarget::SH_REST: cloud.sh_rest(i, target.index) = val; break;
                     default: break; 
                 }
             }
        }
        
        // Normalize quaternions if needed? Usually input is normalized. 
        // But let's ensure exp(scale) or sigmoid(opacity) is handled in Encoder, not here.
        // PLY usually stores raw values. 
        // Vanilla 3DGS stores opacity as logit, scale as log. 
        // SOG expects positions, rotations active, subset, etc. 
        
        return cloud;
    }
};
