/**
 * Python bindings for CUDA SIFT
 * 
 * This file provides pybind11 bindings for the CUDA SIFT implementation,
 * exposing the core C++ functionality to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cuda_runtime.h>

// Include existing CUDA SIFT headers
#include "cudaSift.h"
#include "cudaImage.h"
#include "siftConfigTxt.h"
#include "cudautils.h"

// Forward declaration for ImproveHomography function
extern int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

namespace py = pybind11;

// Forward declarations for functions we'll implement
class PythonSiftExtractor;
class PythonSiftMatcher;

// Forward declarations for functions we'll implement
class PythonSiftExtractor;
class PythonSiftMatcher;

/**
 * Python wrapper for SiftConfigTxt
 */
class PythonSiftConfig {
public:
    SiftConfigTxt config;
    
    PythonSiftConfig() {
        // Initialize with default values - constructor already handles this
    }
    
    PythonSiftConfig(const std::string& config_file) {
        config.loadFromFile(config_file);
    }
    
    // Parameter getters/setters
    float get_dog_threshold() const { return config.params.dog_threshold; }
    void set_dog_threshold(float value) { config.params.dog_threshold = value; }
    
    int get_num_octaves() const { return config.params.num_octaves; }
    void set_num_octaves(int value) { config.params.num_octaves = value; }
    
    float get_initial_blur() const { return config.params.initial_blur; }
    void set_initial_blur(float value) { config.params.initial_blur = value; }
    
    float get_lowest_scale() const { return config.params.lowest_scale; }
    void set_lowest_scale(float value) { config.params.lowest_scale = value; }
    
    bool get_scale_up() const { return config.params.scale_up; }
    void set_scale_up(bool value) { config.params.scale_up = value; }
    
    int get_max_features() const { return config.params.max_features; }
    void set_max_features(int value) { config.params.max_features = value; }
    
    // Validation and utility functions
    bool validate() const {
        return config.validateParams();
    }
};

/**
 * Python wrapper for SIFT feature extraction
 */
class PythonSiftExtractor {
private:
    PythonSiftConfig config_;
    float* temp_memory_;
    bool initialized_;
    bool cuda_initialized_;
    bool external_context_;
    cudaStream_t cuda_stream_;
    bool owns_stream_;
    
public:
    PythonSiftExtractor(const PythonSiftConfig& config, bool external_context = false) 
        : config_(config), temp_memory_(nullptr), initialized_(false), 
          cuda_initialized_(false), external_context_(external_context),
          cuda_stream_(nullptr), owns_stream_(false) {
        
        if (!external_context_) {
            // Initialize CUDA context only if not using external context
            InitCuda(config_.config.params.cuda_device);
            cuda_initialized_ = true;
        }
        
        // Create default stream if not using external context
        if (!external_context_) {
            cudaStreamCreate(&cuda_stream_);
            owns_stream_ = true;
        }
    }
    
    ~PythonSiftExtractor() {
        if (temp_memory_) {
            FreeSiftTempMemory(temp_memory_);
        }
        
        // Clean up stream if we own it
        if (owns_stream_ && cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
        }
    }
    
    // Set external CUDA stream (for PyCUDA integration)
    void set_cuda_stream(uintptr_t stream_handle) {
        // Destroy existing stream if we own it
        if (owns_stream_ && cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
            owns_stream_ = false;
        }
        
        // Set external stream
        cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_handle);
        owns_stream_ = false;
    }
    
    // Get current CUDA stream handle (for PyCUDA integration)
    uintptr_t get_cuda_stream() const {
        return reinterpret_cast<uintptr_t>(cuda_stream_);
    }
    
    // Synchronize CUDA stream
    void synchronize() {
        if (cuda_stream_) {
            cudaStreamSynchronize(cuda_stream_);
        }
    }
    
    // Get configuration parameters
    py::dict get_params() const {
        py::dict params;
        params["dog_threshold"] = config_.get_dog_threshold();
        params["num_octaves"] = config_.get_num_octaves();
        params["initial_blur"] = config_.get_initial_blur();
        params["lowest_scale"] = config_.get_lowest_scale();
        params["scale_up"] = config_.get_scale_up();
        params["max_features"] = config_.get_max_features();
        params["external_context"] = external_context_;
        return params;
    }
    
    // Update configuration parameters
    void set_params(py::dict params) {
        if (params.contains("dog_threshold")) {
            config_.set_dog_threshold(params["dog_threshold"].cast<float>());
        }
        if (params.contains("num_octaves")) {
            config_.set_num_octaves(params["num_octaves"].cast<int>());
        }
        if (params.contains("initial_blur")) {
            config_.set_initial_blur(params["initial_blur"].cast<float>());
        }
        if (params.contains("lowest_scale")) {
            config_.set_lowest_scale(params["lowest_scale"].cast<float>());
        }
        if (params.contains("scale_up")) {
            config_.set_scale_up(params["scale_up"].cast<bool>());
        }
        if (params.contains("max_features")) {
            config_.set_max_features(params["max_features"].cast<int>());
        }
    }
    
    // Extract SIFT features from a numpy array (grayscale image)
    py::dict extract(py::array_t<float> image) {
        // Check input array
        if (image.ndim() != 2) {
            throw std::runtime_error("Input image must be 2D (height, width)");
        }
        
        auto buf = image.request();
        int height = buf.shape[0];
        int width = buf.shape[1];
        
        // Create CudaImage from numpy array
        CudaImage cuda_img;
        cuda_img.Allocate(width, height, iAlignUp(width, 128), false, NULL, (float*)buf.ptr);
        cuda_img.Download();  // Upload to GPU
        
        // Allocate temporary memory if needed
        if (!temp_memory_) {
            temp_memory_ = AllocSiftTempMemory(width, height, config_.get_num_octaves(), config_.get_scale_up());
        }
        
        // Initialize SIFT data structure
        SiftData sift_data;
        InitSiftData(sift_data, config_.get_max_features(), true, true);
        
        // Extract SIFT features
        ExtractSift(sift_data, cuda_img, 
                   config_.get_num_octaves(),
                   config_.get_initial_blur(),
                   config_.get_dog_threshold(),
                   config_.get_lowest_scale(),
                   config_.get_scale_up(),
                   temp_memory_);
        
        // Convert results to Python format
        py::dict result;
        result["num_features"] = sift_data.numPts;
        
        // Create numpy arrays for feature data
        auto positions = py::array_t<float>({sift_data.numPts, 2});
        auto scales = py::array_t<float>(sift_data.numPts);
        auto orientations = py::array_t<float>(sift_data.numPts);
        auto descriptors = py::array_t<float>({sift_data.numPts, 128});
        
        auto pos_buf = positions.request();
        auto scale_buf = scales.request();
        auto orient_buf = orientations.request();
        auto desc_buf = descriptors.request();
        
        float* pos_ptr = (float*)pos_buf.ptr;
        float* scale_ptr = (float*)scale_buf.ptr;
        float* orient_ptr = (float*)orient_buf.ptr;
        float* desc_ptr = (float*)desc_buf.ptr;
        
        // Copy feature data
        for (int i = 0; i < sift_data.numPts; i++) {
            SiftPoint& pt = sift_data.h_data[i];
            pos_ptr[i*2] = pt.xpos;
            pos_ptr[i*2+1] = pt.ypos;
            scale_ptr[i] = pt.scale;
            orient_ptr[i] = pt.orientation;
            
            // Copy descriptor
            for (int j = 0; j < 128; j++) {
                desc_ptr[i*128 + j] = pt.data[j];
            }
        }
        
        result["positions"] = positions;
        result["scales"] = scales;
        result["orientations"] = orientations;
        result["descriptors"] = descriptors;
        
        // Clean up
        FreeSiftData(sift_data);
        
        return result;
    }
};

/**
 * Python wrapper for SIFT feature matching
 */
class PythonSiftMatcher {
private:
    float min_score_;
    float max_ambiguity_;
    bool external_context_;
    cudaStream_t cuda_stream_;
    bool owns_stream_;
    
public:
    PythonSiftMatcher(float min_score = 0.85f, float max_ambiguity = 0.95f, bool external_context = false) 
        : min_score_(min_score), max_ambiguity_(max_ambiguity), external_context_(external_context),
          cuda_stream_(nullptr), owns_stream_(false) {
        
        // Create default stream if not using external context
        if (!external_context_) {
            cudaStreamCreate(&cuda_stream_);
            owns_stream_ = true;
        }
    }
    
    ~PythonSiftMatcher() {
        // Clean up stream if we own it
        if (owns_stream_ && cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
        }
    }
    
    // Set external CUDA stream (for PyCUDA integration)
    void set_cuda_stream(uintptr_t stream_handle) {
        // Destroy existing stream if we own it
        if (owns_stream_ && cuda_stream_) {
            cudaStreamDestroy(cuda_stream_);
            owns_stream_ = false;
        }
        
        // Set external stream
        cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_handle);
        owns_stream_ = false;
    }
    
    // Get current CUDA stream handle (for PyCUDA integration)
    uintptr_t get_cuda_stream() const {
        return reinterpret_cast<uintptr_t>(cuda_stream_);
    }
    
    // Synchronize CUDA stream
    void synchronize() {
        if (cuda_stream_) {
            cudaStreamSynchronize(cuda_stream_);
        }
    }
    
    // Get matching parameters
    py::dict get_params() const {
        py::dict params;
        params["min_score"] = min_score_;
        params["max_ambiguity"] = max_ambiguity_;
        params["external_context"] = external_context_;
        return params;
    }
    
    // Update matching parameters
    void set_params(py::dict params) {
        if (params.contains("min_score")) {
            min_score_ = params["min_score"].cast<float>();
        }
        if (params.contains("max_ambiguity")) {
            max_ambiguity_ = params["max_ambiguity"].cast<float>();
        }
    }
    
    // Match two sets of SIFT features
    py::dict match(py::dict features1, py::dict features2) {
        // Extract feature counts
        int num_pts1 = features1["num_features"].cast<int>();
        int num_pts2 = features2["num_features"].cast<int>();
        
        if (num_pts1 == 0 || num_pts2 == 0) {
            py::dict result;
            result["num_matches"] = 0;
            auto empty_matches = py::array_t<int>({0, 2}, nullptr);
            result["matches"] = empty_matches;
            result["match_score"] = 0.0;
            return result;
        }
        
        // Create SiftData structures
        SiftData sift_data1, sift_data2;
        InitSiftData(sift_data1, num_pts1, true, true);
        InitSiftData(sift_data2, num_pts2, true, true);
        
        sift_data1.numPts = num_pts1;
        sift_data2.numPts = num_pts2;
        
        // Copy feature data from Python arrays
        auto positions1 = features1["positions"].cast<py::array_t<float>>();
        auto positions2 = features2["positions"].cast<py::array_t<float>>();
        auto scales1 = features1["scales"].cast<py::array_t<float>>();
        auto scales2 = features2["scales"].cast<py::array_t<float>>();
        auto orientations1 = features1["orientations"].cast<py::array_t<float>>();
        auto orientations2 = features2["orientations"].cast<py::array_t<float>>();
        auto descriptors1 = features1["descriptors"].cast<py::array_t<float>>();
        auto descriptors2 = features2["descriptors"].cast<py::array_t<float>>();
        
        // Fill SiftData structures
        fillSiftData(sift_data1, positions1, scales1, orientations1, descriptors1);
        fillSiftData(sift_data2, positions2, scales2, orientations2, descriptors2);
        
        // Upload data to GPU for matching
        cudaMemcpy(sift_data1.d_data, sift_data1.h_data, sizeof(SiftPoint)*sift_data1.numPts, cudaMemcpyHostToDevice);
        cudaMemcpy(sift_data2.d_data, sift_data2.h_data, sizeof(SiftPoint)*sift_data2.numPts, cudaMemcpyHostToDevice);
        
        // Perform matching
        double match_score = MatchSiftData(sift_data1, sift_data2);
        
        // Download match results from GPU
        cudaMemcpy(sift_data1.h_data, sift_data1.d_data, sizeof(SiftPoint)*sift_data1.numPts, cudaMemcpyDeviceToHost);
        
        // Extract matches
        std::vector<std::pair<int, int>> matches;
        for (int i = 0; i < sift_data1.numPts; i++) {
            if (sift_data1.h_data[i].match >= 0) {
                matches.push_back({i, sift_data1.h_data[i].match});
            }
        }
        
        // Create results
        py::dict result;
        result["num_matches"] = matches.size();
        result["match_score"] = match_score;
        
        if (matches.size() > 0) {
            auto match_array = py::array_t<int>({(int)matches.size(), 2});
            auto buf = match_array.request();
            int* ptr = (int*)buf.ptr;
            
            for (size_t i = 0; i < matches.size(); i++) {
                ptr[i*2] = matches[i].first;
                ptr[i*2+1] = matches[i].second;
            }
            result["matches"] = match_array;
        } else {
            auto empty_matches = py::array_t<int>({0, 2}, nullptr);
            result["matches"] = empty_matches;
        }
        
        // Clean up
        FreeSiftData(sift_data1);
        FreeSiftData(sift_data2);
        
        return result;
    }
    
    // High-efficiency combined matching and homography computation
    py::dict match_and_compute_homography(py::dict features1, py::dict features2, 
                                         int num_loops = 1000, float thresh = 5.0f, 
                                         bool use_improve = true, int improve_loops = 5) {
        
        // Extract feature counts
        int num_pts1 = features1["num_features"].cast<int>();
        int num_pts2 = features2["num_features"].cast<int>();
        
        if (num_pts1 == 0 || num_pts2 == 0) {
            py::dict result;
            auto identity_h = py::array_t<float>({3, 3});
            auto buf = identity_h.request();
            float* ptr = (float*)buf.ptr;
            // Initialize as identity matrix
            for (int i = 0; i < 9; i++) ptr[i] = 0.0f;
            ptr[0] = ptr[4] = ptr[8] = 1.0f;
            
            result["num_matches"] = 0;
            result["num_inliers"] = 0;
            result["num_refined"] = 0;
            result["homography"] = identity_h;
            result["match_score"] = 0.0;
            result["homography_score"] = 0.0;
            auto empty_matches = py::array_t<int>({0, 2}, nullptr);
            result["matches"] = empty_matches;
            return result;
        }
        
        // Create SiftData structures
        SiftData sift_data1, sift_data2;
        InitSiftData(sift_data1, num_pts1, true, true);
        InitSiftData(sift_data2, num_pts2, true, true);
        
        sift_data1.numPts = num_pts1;
        sift_data2.numPts = num_pts2;
        
        // Copy feature data from Python arrays
        auto positions1 = features1["positions"].cast<py::array_t<float>>();
        auto positions2 = features2["positions"].cast<py::array_t<float>>();
        auto scales1 = features1["scales"].cast<py::array_t<float>>();
        auto scales2 = features2["scales"].cast<py::array_t<float>>();
        auto orientations1 = features1["orientations"].cast<py::array_t<float>>();
        auto orientations2 = features2["orientations"].cast<py::array_t<float>>();
        auto descriptors1 = features1["descriptors"].cast<py::array_t<float>>();
        auto descriptors2 = features2["descriptors"].cast<py::array_t<float>>();
        
        // Fill SiftData structures
        fillSiftData(sift_data1, positions1, scales1, orientations1, descriptors1);
        fillSiftData(sift_data2, positions2, scales2, orientations2, descriptors2);
        
        // Upload data to GPU for matching (single transfer)
        safeCall(cudaMemcpy(sift_data1.d_data, sift_data1.h_data, 
                          sizeof(SiftPoint)*num_pts1, cudaMemcpyHostToDevice));
        safeCall(cudaMemcpy(sift_data2.d_data, sift_data2.h_data, 
                          sizeof(SiftPoint)*num_pts2, cudaMemcpyHostToDevice));
        
        // Step 1: Perform matching
        double match_score = MatchSiftData(sift_data1, sift_data2);
        
        // Download match results to get matched pairs for return value
        safeCall(cudaMemcpy(sift_data1.h_data, sift_data1.d_data, 
                          sizeof(SiftPoint)*num_pts1, cudaMemcpyDeviceToHost));
        
        // Extract match list for return value
        std::vector<std::pair<int, int>> matches;
        for (int i = 0; i < sift_data1.numPts; i++) {
            if (sift_data1.h_data[i].match >= 0) {
                matches.push_back({i, sift_data1.h_data[i].match});
            }
        }
        
        // Step 2: Compute homography using RANSAC (FindHomography)
        float homography[9];
        int num_inliers = 0;
        double homography_score = 0.0;
        int num_refined = 0;
        
        if (matches.size() >= 4) {
            // Upload the matched data back to GPU (already has match info)
            safeCall(cudaMemcpy(sift_data1.d_data, sift_data1.h_data, 
                              sizeof(SiftPoint)*num_pts1, cudaMemcpyHostToDevice));
            
            // RANSAC-based homography estimation
            homography_score = FindHomography(sift_data1, homography, &num_inliers, 
                                            num_loops, min_score_, max_ambiguity_, thresh);
            
            // Step 3: Refine homography using least squares (ImproveHomography)
            if (use_improve && num_inliers >= 8) {
                // Download data for CPU-based refinement
                safeCall(cudaMemcpy(sift_data1.h_data, sift_data1.d_data, 
                                  sizeof(SiftPoint)*num_pts1, cudaMemcpyDeviceToHost));
                
                // Refine using weighted least squares
                num_refined = ImproveHomography(sift_data1, homography, improve_loops, 
                                              min_score_, max_ambiguity_, thresh);
            } else {
                num_refined = num_inliers;  // No improvement applied
            }
        }
        
        // Create comprehensive result
        py::dict result;
        
        // Matching results
        result["num_matches"] = matches.size();
        result["match_score"] = match_score;
        
        if (matches.size() > 0) {
            auto match_array = py::array_t<int>({(int)matches.size(), 2});
            auto buf = match_array.request();
            int* ptr = (int*)buf.ptr;
            
            for (size_t i = 0; i < matches.size(); i++) {
                ptr[i*2] = matches[i].first;
                ptr[i*2+1] = matches[i].second;
            }
            result["matches"] = match_array;
        } else {
            auto empty_matches = py::array_t<int>({0, 2}, nullptr);
            result["matches"] = empty_matches;
        }
        
        // Homography results
        auto h_array = py::array_t<float>({3, 3});
        auto h_buf = h_array.request();
        float* h_ptr = (float*)h_buf.ptr;
        
        for (int i = 0; i < 9; i++) {
            h_ptr[i] = homography[i];
        }
        
        result["homography"] = h_array;
        result["num_inliers"] = num_inliers;        // RANSAC result
        result["num_refined"] = num_refined;        // Improved result
        result["homography_score"] = homography_score;
        
        // Clean up
        FreeSiftData(sift_data1);
        FreeSiftData(sift_data2);
        
        return result;
    }
    
    // Compute homography from matched features (original API for compatibility)
    py::dict compute_homography(py::dict matches_result, py::dict features1, py::dict features2, 
                               int num_loops = 1000, float thresh = 5.0f) {
        
        // Check if we have matches
        int num_matches = matches_result["num_matches"].cast<int>();
        if (num_matches == 0) {
            py::dict result;
            auto identity_h = py::array_t<float>({3, 3});
            auto buf = identity_h.request();
            float* ptr = (float*)buf.ptr;
            // Initialize as identity matrix
            for (int i = 0; i < 9; i++) ptr[i] = 0.0f;
            ptr[0] = ptr[4] = ptr[8] = 1.0f;
            result["homography"] = identity_h;
            result["num_inliers"] = 0;
            result["score"] = 0.0;
            return result;
        }
        
        // Get the match indices
        auto matches = matches_result["matches"].cast<py::array_t<int>>();
        auto match_buf = matches.request();
        int* match_ptr = (int*)match_buf.ptr;
        
        // Get feature data
        int num_pts1 = features1["num_features"].cast<int>();
        auto positions1 = features1["positions"].cast<py::array_t<float>>();
        auto positions2 = features2["positions"].cast<py::array_t<float>>();
        auto scales1 = features1["scales"].cast<py::array_t<float>>();
        auto scales2 = features2["scales"].cast<py::array_t<float>>();
        auto orientations1 = features1["orientations"].cast<py::array_t<float>>();
        auto orientations2 = features2["orientations"].cast<py::array_t<float>>();
        auto descriptors1 = features1["descriptors"].cast<py::array_t<float>>();
        auto descriptors2 = features2["descriptors"].cast<py::array_t<float>>();
        
        // Create SiftData with match information already filled
        SiftData sift_data1;
        InitSiftData(sift_data1, num_pts1, true, true);
        sift_data1.numPts = num_pts1;
        
        // Fill first dataset with features and match information
        fillSiftData(sift_data1, positions1, scales1, orientations1, descriptors1);
        
        // Fill match information from the match results
        auto pos1_buf = positions1.request();
        auto pos2_buf = positions2.request();
        float* pos1_ptr = (float*)pos1_buf.ptr;
        float* pos2_ptr = (float*)pos2_buf.ptr;
        
        for (int i = 0; i < sift_data1.numPts; i++) {
            sift_data1.h_data[i].match = -1;  // Initialize as no match
        }
        
        // Fill match information based on the matches array
        for (int i = 0; i < num_matches; i++) {
            int idx1 = match_ptr[i*2];
            int idx2 = match_ptr[i*2+1];
            if (idx1 < num_pts1) {
                sift_data1.h_data[idx1].match = idx2;
                sift_data1.h_data[idx1].match_xpos = pos2_ptr[idx2*2];
                sift_data1.h_data[idx1].match_ypos = pos2_ptr[idx2*2+1];
                // Set reasonable score and ambiguity for matched points
                sift_data1.h_data[idx1].score = 1.0f;      // High score for matched points
                sift_data1.h_data[idx1].ambiguity = 0.5f;  // Low ambiguity for matched points
            }
        }
        
        // Upload to GPU
        safeCall(cudaMemcpy(sift_data1.d_data, sift_data1.h_data, 
                          sizeof(SiftPoint)*num_pts1, cudaMemcpyHostToDevice));
        
        // Now compute homography using the matched data
        float homography[9];
        int num_inliers = 0;
        double result_score = FindHomography(sift_data1, homography, &num_inliers, 
                                           num_loops, min_score_, max_ambiguity_, thresh);
        
        // Create result
        py::dict result;
        auto h_array = py::array_t<float>({3, 3});
        auto buf = h_array.request();
        float* ptr = (float*)buf.ptr;
        
        for (int i = 0; i < 9; i++) {
            ptr[i] = homography[i];
        }
        
        result["homography"] = h_array;
        result["num_inliers"] = num_inliers;
        result["score"] = result_score;
        
        // Clean up
        FreeSiftData(sift_data1);
        
        return result;
    }

private:
    void fillSiftData(SiftData& sift_data, 
                     py::array_t<float>& positions,
                     py::array_t<float>& scales, 
                     py::array_t<float>& orientations,
                     py::array_t<float>& descriptors) {
        
        auto pos_buf = positions.request();
        auto scale_buf = scales.request();
        auto orient_buf = orientations.request();
        auto desc_buf = descriptors.request();
        
        float* pos_ptr = (float*)pos_buf.ptr;
        float* scale_ptr = (float*)scale_buf.ptr;
        float* orient_ptr = (float*)orient_buf.ptr;
        float* desc_ptr = (float*)desc_buf.ptr;
        
        for (int i = 0; i < sift_data.numPts; i++) {
            SiftPoint& pt = sift_data.h_data[i];
            pt.xpos = pos_ptr[i*2];
            pt.ypos = pos_ptr[i*2+1];
            pt.scale = scale_ptr[i];
            pt.orientation = orient_ptr[i];
            
            // Copy descriptor
            for (int j = 0; j < 128; j++) {
                pt.data[j] = desc_ptr[i*128 + j];
            }
            
            // Initialize other fields
            pt.match = -1;
            pt.score = 0.0f;
            pt.ambiguity = 1.0f;
        }
    }
};

/**
 * pybind11 module definition
 */
PYBIND11_MODULE(cuda_sift, m) {
    m.doc() = "CUDA SIFT Python bindings";
    
    // Bind configuration class
    py::class_<PythonSiftConfig>(m, "SiftConfig")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def_property("dog_threshold", 
                     &PythonSiftConfig::get_dog_threshold,
                     &PythonSiftConfig::set_dog_threshold,
                     "DoG response threshold")
        .def_property("num_octaves",
                     &PythonSiftConfig::get_num_octaves, 
                     &PythonSiftConfig::set_num_octaves,
                     "Number of octaves in pyramid")
        .def_property("initial_blur",
                     &PythonSiftConfig::get_initial_blur,
                     &PythonSiftConfig::set_initial_blur, 
                     "Initial blur value")
        .def_property("lowest_scale",
                     &PythonSiftConfig::get_lowest_scale,
                     &PythonSiftConfig::set_lowest_scale,
                     "Lowest scale value")
        .def_property("scale_up",
                     &PythonSiftConfig::get_scale_up,
                     &PythonSiftConfig::set_scale_up,
                     "Whether to scale up the image")
        .def_property("max_features",
                     &PythonSiftConfig::get_max_features,
                     &PythonSiftConfig::set_max_features,
                     "Maximum number of features to extract")
        .def("validate", &PythonSiftConfig::validate,
             "Validate configuration parameters");
    
    // Bind feature extractor
    py::class_<PythonSiftExtractor>(m, "SiftExtractor")
        .def(py::init<const PythonSiftConfig&>())
        .def(py::init<const PythonSiftConfig&, bool>(),
             py::arg("config"), py::arg("external_context") = false,
             "Initialize extractor with optional external context support")
        .def("extract", &PythonSiftExtractor::extract,
             "Extract SIFT features from image",
             "Extract SIFT features from a 2D numpy array (grayscale image). "
             "Returns a dictionary with 'num_features', 'positions', 'scales', "
             "'orientations', and 'descriptors'.")
        .def("set_cuda_stream", &PythonSiftExtractor::set_cuda_stream,
             "Set external CUDA stream handle (for PyCUDA integration)",
             py::arg("stream_handle"))
        .def("get_cuda_stream", &PythonSiftExtractor::get_cuda_stream,
             "Get current CUDA stream handle")
        .def("synchronize", &PythonSiftExtractor::synchronize,
             "Synchronize CUDA stream")
        .def("get_params", &PythonSiftExtractor::get_params,
             "Get current configuration parameters")
        .def("set_params", &PythonSiftExtractor::set_params,
             "Update configuration parameters",
             py::arg("params"));
    
    // Bind feature matcher
    py::class_<PythonSiftMatcher>(m, "SiftMatcher")
        .def(py::init<>())
        .def(py::init<float, float>())
        .def(py::init<float, float, bool>(),
             py::arg("min_score") = 0.85f, py::arg("max_ambiguity") = 0.95f, 
             py::arg("external_context") = false,
             "Initialize matcher with optional external context support")
        .def("match", &PythonSiftMatcher::match,
             "Match SIFT features between two sets",
             "Match features from two dictionaries returned by SiftExtractor.extract(). "
             "Returns a dictionary with 'num_matches', 'matches', and 'match_score'.")
        .def("match_and_compute_homography", &PythonSiftMatcher::match_and_compute_homography,
             "Efficiently match features and compute homography in one call",
             py::arg("features1"), py::arg("features2"), 
             py::arg("num_loops") = 1000, py::arg("thresh") = 5.0f,
             py::arg("use_improve") = true, py::arg("improve_loops") = 5,
             "Match features and compute homography transformation with optional refinement. "
             "Set use_improve=False for fastest speed, use_improve=True for best accuracy. "
             "Returns a comprehensive dictionary with matching and homography results.")
        .def("compute_homography", &PythonSiftMatcher::compute_homography,
             "Compute homography from existing match results",
             py::arg("matches_result"), py::arg("features1"), py::arg("features2"), 
             py::arg("num_loops") = 1000, py::arg("thresh") = 5.0f,
             "Compute homography transformation from pre-computed match results. "
             "Returns a dictionary with 'homography', 'num_inliers', and 'score'.")
        .def("set_cuda_stream", &PythonSiftMatcher::set_cuda_stream,
             "Set external CUDA stream handle (for PyCUDA integration)",
             py::arg("stream_handle"))
        .def("get_cuda_stream", &PythonSiftMatcher::get_cuda_stream,
             "Get current CUDA stream handle")
        .def("synchronize", &PythonSiftMatcher::synchronize,
             "Synchronize CUDA stream")
        .def("get_params", &PythonSiftMatcher::get_params,
             "Get current matching parameters")
        .def("set_params", &PythonSiftMatcher::set_params,
             "Update matching parameters",
             py::arg("params"));
             
    // Add some utility functions
    m.def("init_cuda", &InitCuda, py::arg("device") = 0,
          "Initialize CUDA context on specified device");
}
