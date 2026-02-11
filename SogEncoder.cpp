#include "SogEncoder.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <limits>
#include <omp.h>

// Helper math
inline float log_transform(float value) {
    return std::copysign(std::log(std::abs(value) + 1.0f), value);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Morton Code Implementation (interleave bits)
// 10 bits per axis -> 30 bits total
inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline uint32_t morton3D(float x, float y, float z, const Eigen::Vector3f& min, const Eigen::Vector3f& max) {
    float x_norm = std::clamp((x - min.x()) / (max.x() - min.x()), 0.0f, 1.0f);
    float y_norm = std::clamp((y - min.y()) / (max.y() - min.y()), 0.0f, 1.0f);
    float z_norm = std::clamp((z - min.z()) / (max.z() - min.z()), 0.0f, 1.0f);
    
    uint32_t xx = expand_bits((uint32_t)(x_norm * 1023.0f));
    uint32_t yy = expand_bits((uint32_t)(y_norm * 1023.0f));
    uint32_t zz = expand_bits((uint32_t)(z_norm * 1023.0f));
    
    return xx * 4 + yy * 2 + zz;
}

// K-Means 1D
struct KMeansResult1D {
    std::vector<float> centroids;
    std::vector<uint8_t> labels; // 256 max
};

KMeansResult1D kmeans1d(const std::vector<float>& data, int k, int iterations = 10) {
    if (data.empty()) return {};
    k = std::min(k, (int)data.size()); // clamp k
    
    // Initialize centroids (k-means++ ish or random)
    std::vector<float> centroids(k);
    
    // Simple init: quantiles
    std::vector<float> sorted_sample = data;
    std::sort(sorted_sample.begin(), sorted_sample.end());
    for (int i=0; i<k; ++i) {
        centroids[i] = sorted_sample[i * sorted_sample.size() / k];
    }
    
    std::vector<uint8_t> labels(data.size());
    std::vector<double> sums(k, 0.0);
    std::vector<int> counts(k, 0);

    for (int iter = 0; iter < iterations; ++iter) {
        // Assignment
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) {
            float val = data[i];
            float min_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int j = 0; j < k; ++j) {
                float dist = std::abs(val - centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_c = j;
                }
            }
            labels[i] = best_c;
        }
        
        // Update
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        
        for (size_t i = 0; i < data.size(); ++i) {
            sums[labels[i]] += data[i];
            counts[labels[i]]++;
        }
        
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                centroids[j] = sums[j] / counts[j];
            }
        }
    }
    
    // Remap centroids to sorted order for better looking codebook
    std::vector<std::pair<float, int>> sorted_centroids(k);
    for (int i=0; i<k; ++i) sorted_centroids[i] = {centroids[i], i};
    std::sort(sorted_centroids.begin(), sorted_centroids.end());
    
    std::vector<float> final_centroids(k);
    std::vector<int> mapping(k);
    for (int i=0; i<k; ++i) {
        final_centroids[i] = sorted_centroids[i].first;
        mapping[sorted_centroids[i].second] = i;
    }
    
    for (size_t i=0; i<labels.size(); ++i) {
        labels[i] = mapping[labels[i]];
    }
    
    return {final_centroids, labels};
}

#include <nanoflann.hpp>

// Adapter for nanoflann to access std::vector<float> as a point cloud
// We are building an index of CENTROIDS, which are stored flat in std::vector<float>
struct CentroidPointCloud {
    const std::vector<float>& data;
    int dim;
    int num_points;

    CentroidPointCloud(const std::vector<float>& d, int dim, int n) : data(d), dim(dim), num_points(n) {}

    inline size_t kdtree_get_point_count() const { return num_points; }

    inline float kdtree_get_pt(const size_t idx, const size_t dim_idx) const {
        return data[idx * dim + dim_idx];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

using MyKdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, CentroidPointCloud>,
    CentroidPointCloud,
    -1 // runtime dim
>;

// Vector K-Means
struct KMeansResultVec {
    std::vector<float> centroids; // flattened k * dim
    std::vector<uint16_t> labels; // 65536 max
};

KMeansResultVec kmeans_vec(const Eigen::MatrixXf& data, int k, int dim, int iterations = 10) {
    if (data.rows() == 0) return {};
    k = std::min(k, (int)data.rows());
    
    // Init centroids (random sample)
    std::vector<float> centroids(k * dim);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> uni(0, data.rows() - 1);
    
    for (int i=0; i<k; ++i) {
        int idx = uni(rng);
        for (int d=0; d<dim; ++d) {
            centroids[i*dim + d] = data(idx, d);
        }
    }
    
    std::vector<uint16_t> labels(data.rows());
    std::vector<double> sums(k * dim, 0.0);
    std::vector<int> counts(k, 0);
    
    for (int iter = 0; iter < iterations; ++iter) {
        // Build KD-Tree on centroids
        CentroidPointCloud point_cloud(centroids, dim, k);
        MyKdTree index(dim, point_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        // Assignment step using KD-tree
        #pragma omp parallel for
        for (int i = 0; i < data.rows(); ++i) {
            std::vector<float> query_pt(dim);
            for(int d=0; d<dim; ++d) query_pt[d] = data(i,d);

            size_t ret_index;
            float out_dist_sqr;
            nanoflann::KNNResultSet<float> resultSet(1);
            resultSet.init(&ret_index, &out_dist_sqr);
            index.findNeighbors(resultSet, query_pt.data(), nanoflann::SearchParameters(10));
            
            labels[i] = (uint16_t)ret_index;
        }
        
        // Update
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        
        #pragma omp parallel
        {
             std::vector<double> local_sums(k * dim, 0.0);
             std::vector<int> local_counts(k, 0);
             
             #pragma omp for
             for (int i = 0; i < data.rows(); ++i) {
                 int label = labels[i];
                 local_counts[label]++;
                 for (int d = 0; d < dim; ++d) {
                     local_sums[label*dim + d] += data(i, d);
                 }
             }
             
             #pragma omp critical
             {
                 for (int j=0; j<k; ++j) {
                     counts[j] += local_counts[j];
                     for (int d=0; d<dim; ++d) {
                         sums[j*dim + d] += local_sums[j*dim + d];
                     }
                 }
             }
        }

        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                for (int d=0; d<dim; ++d) {
                    centroids[j*dim + d] = sums[j*dim + d] / counts[j];
                }
            }
        }
    }
    
    return {centroids, labels};
}


SogEncoder::SogEncoder(const GaussianCloud& cloud, const Options& options)
    : cloud_(cloud), options_(options) {
    if (cloud.size() > 0) {
        indices_.resize(cloud_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
    }
    
    // Auto-calculate dimensions based on count
    // Use float to avoid overflow before sqrt
    float count = (float)indices_.size();
    if (options.width_hint > 0) {
        width_ = options.width_hint;
    } else {
        width_ = std::ceil(std::sqrt(count) / 4.0f) * 4.0f;
        if (width_ < 4) width_ = 4;
    }
    height_ = std::ceil(count / width_ / 4.0f) * 4.0f;
    if (height_ < 4) height_ = 4;
    padded_count_ = width_ * height_;
}

SogEncoder::SogEncoder(const GaussianCloud& cloud, const std::vector<size_t>& subset_indices, const Options& options)
    : cloud_(cloud), options_(options) {
    indices_.reserve(subset_indices.size());
    for (size_t idx : subset_indices) {
        indices_.push_back((uint32_t)idx);
    }

    // Auto-calculate dimensions based on subset count
    float count = (float)indices_.size();
    if (options.width_hint > 0) {
        width_ = options.width_hint; 
    } else {
        width_ = std::ceil(std::sqrt(count) / 4.0f) * 4.0f;
        if (width_ < 4) width_ = 4;
    }
    height_ = std::ceil(count / width_ / 4.0f) * 4.0f;
    if (height_ < 4) height_ = 4;
    padded_count_ = width_ * height_;
}

void SogEncoder::compute_morton_indices() {
    if (indices_.empty()) return;

    // Compute bbox of the active indices
    Eigen::Vector3f min(1e30f, 1e30f, 1e30f);
    Eigen::Vector3f max(-1e30f, -1e30f, -1e30f);
    
    // Only check bounds of the indices we care about
    #pragma omp parallel
    {
        Eigen::Vector3f local_min(1e30f, 1e30f, 1e30f);
        Eigen::Vector3f local_max(-1e30f, -1e30f, -1e30f);
        
        #pragma omp for
        for (size_t i = 0; i < indices_.size(); ++i) {
             Eigen::Vector3f p = cloud_.positions.row(indices_[i]);
             local_min = local_min.cwiseMin(p);
             local_max = local_max.cwiseMax(p);
        }
        
        #pragma omp critical
        {
            min = min.cwiseMin(local_min);
            max = max.cwiseMax(local_max);
        }
    }
    
    // Expand bounds slightly to avoid edge cases
    min.array() -= 0.01f;
    max.array() += 0.01f;

    struct IndexedMorton {
        uint32_t index;
        uint32_t code;
    };
    std::vector<IndexedMorton> morton(cloud_.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        morton[i].index = i;
        morton[i].code = morton3D(cloud_.positions(i, 0), cloud_.positions(i, 1), cloud_.positions(i, 2), min, max);
    }
    
    std::sort(morton.begin(), morton.end(), [](const IndexedMorton& a, const IndexedMorton& b) {
        return a.code < b.code;
    });
    
    indices_.resize(cloud_.size());
    for (size_t i = 0; i < cloud_.size(); ++i) {
        indices_[i] = morton[i].index;
    }
}

void SogEncoder::write_webp(const std::string& filename, const uint8_t* data, int w, int h) const {
    uint8_t* output;
    size_t size = WebPEncodeLosslessRGBA(data, w, h, w * 4, &output);
    if (size == 0) {
        throw std::runtime_error("Failed to encode WebP");
    }
    
    write_file(filename, std::vector<uint8_t>(output, output + size));
    WebPFree(output);
}

void SogEncoder::write_file(const std::string& filename, const std::string& content) const {
    std::filesystem::path path = std::filesystem::path(options_.output_path) / filename;
    std::ofstream file(path, std::ios::binary);
    file.write(content.data(), content.size());
}

void SogEncoder::write_file(const std::string& filename, const std::vector<uint8_t>& content) const {
    std::filesystem::path path = std::filesystem::path(options_.output_path) / filename;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(content.data()), content.size());
}

// Encoding implementations...

nlohmann::json SogEncoder::encode_positions() {
    std::vector<uint8_t> means_l(width_ * height_ * 4, 0); // Initialize with 0 for padding
    std::vector<uint8_t> means_u(width_ * height_ * 4, 0);
    
    // Compute log transformed min/max
    Eigen::Vector3f min_log(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Eigen::Vector3f max_log(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    
    std::vector<Eigen::Vector3f> log_pos(cloud_.size());

    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        log_pos[i](0) = log_transform(cloud_.positions(i, 0));
        log_pos[i](1) = log_transform(cloud_.positions(i, 1));
        log_pos[i](2) = log_transform(cloud_.positions(i, 2));
    }

    // Reduction for min/max
    for (const auto& p : log_pos) {
        min_log = min_log.cwiseMin(p);
        max_log = max_log.cwiseMax(p);
    }
    
    Eigen::Vector3f range = max_log - min_log;
    // Avoid division by zero
    if (range.x() == 0) range.x() = 1.0f;
    if (range.y() == 0) range.y() = 1.0f;
    if (range.z() == 0) range.z() = 1.0f;
    
    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        size_t idx = indices_[i]; // Encoded in sorted order
        
        // 16-bit quantization
        uint16_t x = (uint16_t)(65535.0f * (log_pos[idx](0) - min_log(0)) / range(0));
        uint16_t y = (uint16_t)(65535.0f * (log_pos[idx](1) - min_log(1)) / range(1));
        uint16_t z = (uint16_t)(65535.0f * (log_pos[idx](2) - min_log(2)) / range(2));
        
        // Layout: Identity (no block swizzling for now as per reference code comments saying "no packing")
        size_t ti = i; 
        
        means_l[ti * 4 + 0] = x & 0xff;
        means_l[ti * 4 + 1] = y & 0xff;
        means_l[ti * 4 + 2] = z & 0xff;
        means_l[ti * 4 + 3] = 0xff; // Alpha
        
        means_u[ti * 4 + 0] = (x >> 8) & 0xff;
        means_u[ti * 4 + 1] = (y >> 8) & 0xff;
        means_u[ti * 4 + 2] = (z >> 8) & 0xff;
        means_u[ti * 4 + 3] = 0xff; // Alpha
    }
    
    // Fill padding with 0 (already done by init) except alpha
    for (size_t i = cloud_.size(); i < padded_count_; ++i) {
        means_l[i * 4 + 3] = 0xff;
        means_u[i * 4 + 3] = 0xff;
    }

    write_webp("means_l.webp", means_l.data(), width_, height_);
    write_webp("means_u.webp", means_u.data(), width_, height_);
    
    return {
        {"mins", {min_log(0), min_log(1), min_log(2)}},
        {"maxs", {max_log(0), max_log(1), max_log(2)}},
        {"files", {"means_l.webp", "means_u.webp"}}
    };
}

void SogEncoder::encode_quaternions() {
    std::vector<uint8_t> quats_data(width_ * height_ * 4, 0);

    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        size_t idx = indices_[i];
        Eigen::Vector4f q = cloud_.rotations.row(idx).normalized();
        
        // Find max component
        int max_comp = 0;
        float max_val = std::abs(q(0));
        for (int k = 1; k < 4; ++k) {
            float val = std::abs(q(k));
            if (val > max_val) {
                max_val = val;
                max_comp = k;
            }
        }
        
        // Invert if max component is negative (double cover)
        if (q(max_comp) < 0) q = -q;
        
        // Drop max component and scale by sqrt(2)
        // Indices map based on max_comp (same as TS code)
        int idx0, idx1, idx2;
        if (max_comp == 0) { idx0=1; idx1=2; idx2=3; }
        else if (max_comp == 1) { idx0=0; idx1=2; idx2=3; }
        else if (max_comp == 2) { idx0=0; idx1=1; idx2=3; }
        else { idx0=0; idx1=1; idx2=2; }
        
        float sqrt2 = std::sqrt(2.0f);
        uint8_t v0 = (uint8_t)(255.0f * (q(idx0) * sqrt2 * 0.5f + 0.5f));
        uint8_t v1 = (uint8_t)(255.0f * (q(idx1) * sqrt2 * 0.5f + 0.5f));
        uint8_t v2 = (uint8_t)(255.0f * (q(idx2) * sqrt2 * 0.5f + 0.5f));
        
        size_t ti = i;
        quats_data[ti * 4 + 0] = v0;
        quats_data[ti * 4 + 1] = v1;
        quats_data[ti * 4 + 2] = v2;
        quats_data[ti * 4 + 3] = 252 + max_comp;
    }
    
    // Padding
    for (size_t i = cloud_.size(); i < padded_count_; ++i) {
        quats_data[i * 4 + 3] = 0; // Default tag? Or 255. TS code seems to use 252+comp, default logic handles invalid tags as identity.
    }

    write_webp("quats.webp", quats_data.data(), width_, height_);
}

nlohmann::json SogEncoder::encode_scales() {
    // Collect data for each axis
    std::vector<float> s0(cloud_.size()), s1(cloud_.size()), s2(cloud_.size());
    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        s0[i] = cloud_.scales(i, 0);
        s1[i] = cloud_.scales(i, 1);
        s2[i] = cloud_.scales(i, 2);
    }
    
    // K-Means for each axis (256 clusters)
    auto km0 = kmeans1d(s0, 256, options_.sh_iterations);
    auto km1 = kmeans1d(s1, 256, options_.sh_iterations);
    auto km2 = kmeans1d(s2, 256, options_.sh_iterations);
    
    // Flatten codebooks into one array
    std::vector<float> codebook;
    codebook.insert(codebook.end(), km0.centroids.begin(), km0.centroids.end());
    codebook.insert(codebook.end(), km1.centroids.begin(), km1.centroids.end());
    codebook.insert(codebook.end(), km2.centroids.begin(), km2.centroids.end()); // Wait, TS quantizes ALL scales together? 
    // Checking TS code:
    // const scaleData = quantize1d(new DataTable(['scale_0', 'scale_1', 'scale_2'].map(...)));
    // quantize1d takes a DataTable. If it has multiple columns, it treats them as... a vector?
    // "quantize1d" implies 1D...
    // Let's re-read `read-sog.ts`.
    // `meta.scales.codebook` is a single array.
    // `columns[3].data[i] = sCode[sl[o]]`
    // `columns[4].data[i] = sCode[sl[o + 1]]`
    // It seems each component (sx, sy, sz) points to the SAME codebook.
    // So we should pool all scale values together, cluster them, and use the same codebook for x, y, z.
    
    std::vector<float> all_scales;
    all_scales.reserve(cloud_.size() * 3);
    for (size_t i=0; i<cloud_.size(); ++i) {
        all_scales.push_back(cloud_.scales(i, 0));
        all_scales.push_back(cloud_.scales(i, 1));
        all_scales.push_back(cloud_.scales(i, 2));
    }
    
    auto km_scales = kmeans1d(all_scales, 256, options_.sh_iterations);
    
    // Now map original values to labels
    std::vector<uint8_t> scales_data(width_ * height_ * 4, 0);

    // Need a fast lookup map? Or just re-find nearest.
    // Since we just ran k-means, we can just re-assign. Or better, `kmeans1d` returns labels for the input data.
    // Since `all_scales` was structured as [s0_x, s0_y, s0_z, s1_x, ...], labels correspond linearly.
    
    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        size_t idx = indices_[i];
        
        // Find labels for s[idx]
        // Since we did pool, labels are in `km_scales.labels`.
        // The indices in `all_scales` were linear: 0,1,2 for gaussian 0, etc.
        // Wait, `all_scales` order was just simple push_back.
        // But we need to use `indices_` for the output texture order.
        
        // We need the label for `cloud_.scales(idx, 0)`.
        // The label is in `km_scales.labels[idx * 3 + 0]`.
        
        uint8_t l0 = km_scales.labels[idx * 3 + 0];
        uint8_t l1 = km_scales.labels[idx * 3 + 1];
        uint8_t l2 = km_scales.labels[idx * 3 + 2];
        
        size_t ti = i;
        scales_data[ti * 4 + 0] = l0;
        scales_data[ti * 4 + 1] = l1;
        scales_data[ti * 4 + 2] = l2;
        scales_data[ti * 4 + 3] = 255;
    }
    
    // Padding
    for (size_t i = cloud_.size(); i < padded_count_; ++i) {
        scales_data[i * 4 + 3] = 255;
    }

    write_webp("scales.webp", scales_data.data(), width_, height_);
    
    return {
        {"files", {"scales.webp"}},
        {"codebook", km_scales.centroids}
    };
}

nlohmann::json SogEncoder::encode_colors() {
    // Similar to scales, but for DC colors. Pooled or separate?
    // TS code: `quantize1d(new DataTable(['f_dc_0', 'f_dc_1', 'f_dc_2']))`
    // Yes, pooled.
    
    std::vector<float> all_colors;
    all_colors.reserve(cloud_.size() * 3);
    for (size_t i=0; i<cloud_.size(); ++i) {
        all_colors.push_back(cloud_.sh_dc(i, 0));
        all_colors.push_back(cloud_.sh_dc(i, 1));
        all_colors.push_back(cloud_.sh_dc(i, 2));
    }
    
    auto km_colors = kmeans1d(all_colors, 256, options_.sh_iterations);
    
    std::vector<uint8_t> colors_data(width_ * height_ * 4, 0);

    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        size_t idx = indices_[i];
        
        uint8_t l0 = km_colors.labels[idx * 3 + 0];
        uint8_t l1 = km_colors.labels[idx * 3 + 1];
        uint8_t l2 = km_colors.labels[idx * 3 + 2];
        
        // Opacity
        float op = sigmoid(cloud_.opacities(idx));
        uint8_t op_byte = (uint8_t)(std::clamp(op * 255.0f, 0.0f, 255.0f));
        
        size_t ti = i;
        colors_data[ti * 4 + 0] = l0;
        colors_data[ti * 4 + 1] = l1;
        colors_data[ti * 4 + 2] = l2;
        colors_data[ti * 4 + 3] = op_byte;
    }
    
     // Padding
    for (size_t i = cloud_.size(); i < padded_count_; ++i) {
        colors_data[i * 4 + 3] = 0; // Transparent
    }

    write_webp("sh0.webp", colors_data.data(), width_, height_);
    
     return {
        {"files", {"sh0.webp"}},
        {"codebook", km_colors.centroids}
    };
}

nlohmann::json SogEncoder::encode_sh() {
    if (cloud_.sh_rest.cols() == 0) return nullptr;
    
    // Adaptive palette size
    int palette_size = std::min(64, (int)std::pow(2, std::floor(std::log2(cloud_.size() / 1024.0)))) * 1024;
    palette_size = std::max(1024, palette_size); // Ensure at least 1024? TS code says min(64, ...)*1024, if small clouds, pow can be small.
    // TS: Math.min(64, 2 ** Math.floor(Math.log2(indices.length / 1024))) * 1024;
    // If length < 1024, log2 is negative, floor is negative, 2**neg is small.
    // e.g. 500 points -> floor(log2(0.48)) = -2 -> 0.25 -> 256. 
    if (cloud_.size() < 1024) palette_size = 256; 
    
    std::cout << "Clustering SH with " << palette_size << " clusters..." << std::endl;
    
    int sh_coeffs = cloud_.sh_rest.cols() / 3;
    if (sh_coeffs == 0) return nullptr;
    
    // Determined by options or input size.
    // If input has 45 columns (Band 3), but we want fewer?
    // Let's stick to input size unless forced?
    // JS tool seems to default to Band 2 (8 coeffs)?
    // Let's just process what we have, but ensure we don't crash if 45.
    
    std::cout << "Compressing SH: " << cloud_.sh_rest.rows() << " points, " << cloud_.sh_rest.cols() << " dims." << std::endl;

    // Run Vector K-Means
    auto km_sh = kmeans_vec(cloud_.sh_rest, palette_size, cloud_.sh_rest.cols(), options_.sh_iterations);
    
    // Write Centroids Texture
    // 64 * shCoeffs width, ceil(numRows/64) height.
    // Texture stores centroids.
    // Layout: Rows of 64 centroids. Each centroid spans `shCoeffs` pixels horizontally.
    // Pixel (x, y) contains (c_r, c_g, c_b, 0xff).
    
    int input_coeffs = cloud_.sh_rest.cols() / 3;
    int target_coeffs = input_coeffs;

    int tex_width = 64 * input_coeffs;
    int tex_height = (km_sh.centroids.size() / (cloud_.sh_rest.cols()) + 63) / 64; // num_centroids / 64
    
    std::vector<uint8_t> centroid_data(tex_width * tex_height * 4, 0);
    
    // Quantize centroids (quantize1d in TS code).
    // TS: const codebook = quantize1d(centroids);
    // This implies 1D quantization of the centroid values themselves to 8-bit?
    // "Quantizing spherical harmonics" step in TS.
    // Yes, `quantize1d` on the centroids table. 
    // So the centroids texture stores INDICES into a codebook?
    // TS: `centroidsBuf[i ...] = x` where x comes from `centroidsRow`.
    // `centroidsRow` comes from `codebook.labels`.
    // So yes, the SH centroids themselves are quantized to 8-bit via a codebook.
    
    // Step 1: Flatten centroids and run 1D k-means (256 clusters).
    auto km_codebook = kmeans1d(km_sh.centroids, 256, options_.sh_iterations);
    
    // Step 2: Write centroids texture using labels from km_codebook.
    // km_sh.centroids is flattened: [C0_d0, C0_d1, ... C0_dN, C1_d0, ...]
    // km_codebook.labels maps each float to a byte.
    
    // km_codebook.labels maps EACH FLOAT in the flattened centroids array to a codebook index.
    // km_sh.centroids has size `palette_size * cloud_.sh_rest.cols()`.
    
    // We need to write these labels into the texture.
    int num_centroids = km_sh.centroids.size() / cloud_.sh_rest.cols();
    
    for (int i = 0; i < num_centroids; ++i) {
        for (int j = 0; j < target_coeffs; ++j) {
            // Each centroid has `target_coeffs` * 3 values.
            // Stored in texture as `target_coeffs` pixels.
            // Pixel j corresponds to (R, G, B) = (coeff_j, coeff_j+N, coeff_j+2N)
            // But from flattened centroid array which has length `input_coeffs * 3`.
            
            // If we are truncating, we only read the first `target_coeffs` from each block?
            // Wait, km_sh was run on `cloud_.sh_rest`. If that is 45 dims, centroids are 45 dims.
            // If we want 24 dims output, do we just drop the rest?
            // Used `input_coeffs` for striding.
            
            int offset = i * cloud_.sh_rest.cols();
            uint8_t r = km_codebook.labels[offset + j];
            uint8_t g = km_codebook.labels[offset + j + input_coeffs];
            uint8_t b = km_codebook.labels[offset + j + 2 * input_coeffs];
            
            // Texture layout calculation
            // i is centroid index.
            // i % 64 is X grid (times target_coeffs).
            // i / 64 is Y grid.
            
            int cx = (i % 64) * target_coeffs + j;
            int cy = i / 64;
            
            int ti = cy * tex_width + cx;
            
            centroid_data[ti * 4 + 0] = r;
            centroid_data[ti * 4 + 1] = g;
            centroid_data[ti * 4 + 2] = b;
            centroid_data[ti * 4 + 3] = 255;
        }
    }
    
    write_webp("shN_centroids.webp", centroid_data.data(), tex_width, tex_height);
    
    // Write Labels Texture for points
    // Stores 16-bit indices into the palette.
    // split into R, G.
    
    std::vector<uint8_t> label_data(width_ * height_ * 4, 0);
    
    #pragma omp parallel for
    for (size_t i = 0; i < cloud_.size(); ++i) {
        size_t idx = indices_[i];
        
        uint16_t label = km_sh.labels[idx];
        
        size_t ti = i;
        label_data[ti * 4 + 0] = label & 0xff;
        label_data[ti * 4 + 1] = (label >> 8) & 0xff;
        label_data[ti * 4 + 2] = 0;
        label_data[ti * 4 + 3] = 255;
    }

    write_webp("shN_labels.webp", label_data.data(), width_, height_);

    return {
        {"count", num_centroids},
        {"bands", (target_coeffs == 3 ? 1 : (target_coeffs == 8 ? 2 : (target_coeffs == 15 ? 3 : 0)))},
        {"codebook", km_codebook.centroids},
        {"files", {"shN_centroids.webp", "shN_labels.webp"}}
    };
}

void SogEncoder::encode() {
    compute_morton_indices();
    
    nlohmann::json meta_pos = encode_positions();
    encode_quaternions();
    nlohmann::json meta_scales = encode_scales();
    nlohmann::json meta_colors = encode_colors();
    nlohmann::json meta_sh = encode_sh();
    
    nlohmann::json meta = {
        {"version", 2},
        {"asset", {{"generator", "splat-converter-cpp"}}},
        {"count", cloud_.size()},
        {"means", meta_pos},
        {"scales", meta_scales},
        {"quats", {{"files", {"quats.webp"}}}},
        {"sh0", meta_colors}
    };
    
    if (meta_sh != nullptr) {
        meta["shN"] = meta_sh;
    }
    
    write_file("meta.json", meta.dump(4));
}
