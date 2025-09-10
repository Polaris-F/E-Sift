#include "siftConfigTxt.h"
#include <algorithm>
#include <cctype>

SiftConfigTxt::SiftConfigTxt(const std::string& config_file) {
    if (!config_file.empty()) {
        loadFromFile(config_file);
    }
}

bool SiftConfigTxt::loadFromFile(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open config file: " << config_file << std::endl;
        std::cout << "Using default parameters." << std::endl;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    parseTxtContent(content);
    return true;
}

void SiftConfigTxt::parseTxtContent(const std::string& content) {
    std::istringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = trim(line);
        
        // 跳过空行和注释行
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::string key, value;
        if (!parseKeyValue(line, key, value)) {
            continue;
        }
        
        // 设置参数值
        setParameter(key, value);
    }
}

bool SiftConfigTxt::parseKeyValue(const std::string& line, std::string& key, std::string& value) {
    size_t equal_pos = line.find('=');
    if (equal_pos == std::string::npos) {
        return false;
    }
    
    key = trim(line.substr(0, equal_pos));
    value = trim(line.substr(equal_pos + 1));
    
    // 移除值中的注释部分
    size_t comment_pos = value.find('#');
    if (comment_pos != std::string::npos) {
        value = trim(value.substr(0, comment_pos));
    }
    
    return !key.empty() && !value.empty();
}

bool SiftConfigTxt::setParameter(const std::string& key, const std::string& value) {
    // 核心算法参数
    if (key == "initial_blur") {
        params.initial_blur = stringToFloat(value, 1.0f);
    } else if (key == "dog_threshold") {
        params.dog_threshold = stringToFloat(value, 3.0f);
    } else if (key == "num_octaves") {
        params.num_octaves = stringToInt(value, 5);
    } else if (key == "lowest_scale") {
        params.lowest_scale = stringToFloat(value, 0.0f);
    } else if (key == "edge_limit") {
        params.edge_limit = stringToFloat(value, 10.0f);
    } else if (key == "scale_up") {
        params.scale_up = stringToBool(value, false);
    }
    // 特征匹配参数
    else if (key == "min_score") {
        params.min_score = stringToFloat(value, 0.85f);
    } else if (key == "max_ambiguity") {
        params.max_ambiguity = stringToFloat(value, 0.95f);
    }
    // 单应性估计参数
    else if (key == "ransac_iterations") {
        params.ransac_iterations = stringToInt(value, 10000);
    } else if (key == "inlier_threshold") {
        params.inlier_threshold = stringToFloat(value, 5.0f);
    } else if (key == "optimization_iterations") {
        params.optimization_iterations = stringToInt(value, 5);
    } else if (key == "optimization_threshold") {
        params.optimization_threshold = stringToFloat(value, 3.0f);
    }
    // 内存和性能参数
    else if (key == "max_features") {
        params.max_features = stringToInt(value, 32768);
    } else if (key == "cuda_device") {
        params.cuda_device = stringToInt(value, 0);
    }
    // 输入输出配置
    else if (key == "image_set") {
        params.image_set = stringToInt(value, 0);
    } else if (key == "image1_path") {
        params.image1_path = removeQuotes(value);
    } else if (key == "image2_path") {
        params.image2_path = removeQuotes(value);
    } else if (key == "alt_image1_path") {
        params.alt_image1_path = removeQuotes(value);
    } else if (key == "alt_image2_path") {
        params.alt_image2_path = removeQuotes(value);
    }
    // 调试和输出参数
    else if (key == "verbose") {
        params.verbose = stringToBool(value, false);
    } else if (key == "show_matches") {
        params.show_matches = stringToBool(value, true);
    } else if (key == "save_intermediate") {
        params.save_intermediate = stringToBool(value, false);
    } else if (key == "output_path") {
        params.output_path = removeQuotes(value);
    }
    // 可视化调试参数
    else if (key == "enable_visualization") {
        params.enable_visualization = stringToBool(value, true);
    } else if (key == "save_visualization") {
        params.save_visualization = stringToBool(value, true);
    } else if (key == "show_visualization_window") {
        params.show_visualization_window = stringToBool(value, false);
    } else if (key == "feature_circle_radius") {
        params.feature_circle_radius = stringToInt(value, 3);
    } else if (key == "match_line_thickness") {
        params.match_line_thickness = stringToInt(value, 1);
    } else if (key == "show_only_good_matches") {
        params.show_only_good_matches = stringToBool(value, true);
    } else if (key == "overlay_alpha") {
        params.overlay_alpha = stringToFloat(value, 0.5f);
    } else if (key == "visualization_output_path") {
        params.visualization_output_path = removeQuotes(value);
    }
    // 高级参数
    else if (key == "num_scales") {
        params.num_scales = stringToInt(value, 5);
    } else if (key == "laplace_radius") {
        params.laplace_radius = stringToInt(value, 4);
    } else if (key == "lowpass_radius") {
        params.lowpass_radius = stringToInt(value, 4);
    }
    // 线程块配置
    else if (key == "scaledown_width") {
        params.scaledown_width = stringToInt(value, 64);
    } else if (key == "scaledown_height") {
        params.scaledown_height = stringToInt(value, 16);
    } else if (key == "scaleup_width") {
        params.scaleup_width = stringToInt(value, 64);
    } else if (key == "scaleup_height") {
        params.scaleup_height = stringToInt(value, 8);
    } else if (key == "minmax_width") {
        params.minmax_width = stringToInt(value, 30);
    } else if (key == "minmax_height") {
        params.minmax_height = stringToInt(value, 8);
    } else if (key == "laplace_width") {
        params.laplace_width = stringToInt(value, 128);
    } else if (key == "laplace_height") {
        params.laplace_height = stringToInt(value, 4);
    } else if (key == "lowpass_width") {
        params.lowpass_width = stringToInt(value, 24);
    } else if (key == "lowpass_height") {
        params.lowpass_height = stringToInt(value, 32);
    }
    else {
        std::cerr << "Warning: Unknown parameter: " << key << std::endl;
        return false;
    }
    
    return true;
}

std::string SiftConfigTxt::getParameter(const std::string& key) const {
    if (key == "initial_blur") return std::to_string(params.initial_blur);
    if (key == "dog_threshold") return std::to_string(params.dog_threshold);
    if (key == "num_octaves") return std::to_string(params.num_octaves);
    if (key == "lowest_scale") return std::to_string(params.lowest_scale);
    if (key == "edge_limit") return std::to_string(params.edge_limit);
    if (key == "scale_up") return params.scale_up ? "true" : "false";
    if (key == "min_score") return std::to_string(params.min_score);
    if (key == "max_ambiguity") return std::to_string(params.max_ambiguity);
    if (key == "ransac_iterations") return std::to_string(params.ransac_iterations);
    if (key == "inlier_threshold") return std::to_string(params.inlier_threshold);
    if (key == "optimization_iterations") return std::to_string(params.optimization_iterations);
    if (key == "optimization_threshold") return std::to_string(params.optimization_threshold);
    if (key == "max_features") return std::to_string(params.max_features);
    if (key == "cuda_device") return std::to_string(params.cuda_device);
    if (key == "image_set") return std::to_string(params.image_set);
    if (key == "image1_path") return params.image1_path;
    if (key == "image2_path") return params.image2_path;
    if (key == "alt_image1_path") return params.alt_image1_path;
    if (key == "alt_image2_path") return params.alt_image2_path;
    if (key == "verbose") return params.verbose ? "true" : "false";
    if (key == "show_matches") return params.show_matches ? "true" : "false";
    if (key == "save_intermediate") return params.save_intermediate ? "true" : "false";
    if (key == "output_path") return params.output_path;
    if (key == "enable_visualization") return params.enable_visualization ? "true" : "false";
    if (key == "save_visualization") return params.save_visualization ? "true" : "false";
    if (key == "show_visualization_window") return params.show_visualization_window ? "true" : "false";
    if (key == "feature_circle_radius") return std::to_string(params.feature_circle_radius);
    if (key == "match_line_thickness") return std::to_string(params.match_line_thickness);
    if (key == "show_only_good_matches") return params.show_only_good_matches ? "true" : "false";
    if (key == "overlay_alpha") return std::to_string(params.overlay_alpha);
    if (key == "visualization_output_path") return params.visualization_output_path;
    
    return "";
}

std::vector<std::string> SiftConfigTxt::getParameterNames() const {
    return {
        "initial_blur", "dog_threshold", "num_octaves", "lowest_scale", "edge_limit", "scale_up",
        "min_score", "max_ambiguity",
        "ransac_iterations", "inlier_threshold", "optimization_iterations", "optimization_threshold",
        "max_features", "cuda_device",
        "image_set", "image1_path", "image2_path", "alt_image1_path", "alt_image2_path",
        "verbose", "show_matches", "save_intermediate", "output_path",
        "enable_visualization", "save_visualization", "show_visualization_window", 
        "feature_circle_radius", "match_line_thickness", "show_only_good_matches", "overlay_alpha", "visualization_output_path",
        "num_scales", "laplace_radius", "lowpass_radius",
        "scaledown_width", "scaledown_height", "scaleup_width", "scaleup_height",
        "minmax_width", "minmax_height", "laplace_width", "laplace_height",
        "lowpass_width", "lowpass_height"
    };
}

std::string SiftConfigTxt::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::string SiftConfigTxt::removeQuotes(const std::string& str) {
    std::string result = str;
    if (result.length() >= 2 && 
        ((result.front() == '"' && result.back() == '"') ||
         (result.front() == '\'' && result.back() == '\''))) {
        result = result.substr(1, result.length() - 2);
    }
    return result;
}

float SiftConfigTxt::stringToFloat(const std::string& str, float default_val) {
    try {
        return std::stof(str);
    } catch (const std::exception&) {
        return default_val;
    }
}

int SiftConfigTxt::stringToInt(const std::string& str, int default_val) {
    try {
        return std::stoi(str);
    } catch (const std::exception&) {
        return default_val;
    }
}

bool SiftConfigTxt::stringToBool(const std::string& str, bool default_val) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    if (lower_str == "true" || lower_str == "1" || lower_str == "yes" || lower_str == "on") {
        return true;
    } else if (lower_str == "false" || lower_str == "0" || lower_str == "no" || lower_str == "off") {
        return false;
    }
    
    return default_val;
}

bool SiftConfigTxt::validateParams() const {
    bool valid = true;
    
    // 验证SIFT提取参数
    if (params.initial_blur < 0.5f || params.initial_blur > 2.0f) {
        std::cerr << "Warning: initial_blur should be between 0.5 and 2.0 (current: " << params.initial_blur << ")" << std::endl;
        valid = false;
    }
    
    if (params.dog_threshold < 1.0f || params.dog_threshold > 10.0f) {
        std::cerr << "Warning: dog_threshold should be between 1.0 and 10.0 (current: " << params.dog_threshold << ")" << std::endl;
        valid = false;
    }
    
    if (params.num_octaves < 3 || params.num_octaves > 8) {
        std::cerr << "Warning: num_octaves should be between 3 and 8 (current: " << params.num_octaves << ")" << std::endl;
        valid = false;
    }
    
    if (params.edge_limit < 5.0f || params.edge_limit > 20.0f) {
        std::cerr << "Warning: edge_limit should be between 5.0 and 20.0 (current: " << params.edge_limit << ")" << std::endl;
        valid = false;
    }
    
    // 验证匹配参数
    if (params.min_score < 0.0f || params.min_score > 1.0f) {
        std::cerr << "Warning: min_score should be between 0.0 and 1.0 (current: " << params.min_score << ")" << std::endl;
        valid = false;
    }
    
    if (params.max_ambiguity < 0.0f || params.max_ambiguity > 1.0f) {
        std::cerr << "Warning: max_ambiguity should be between 0.0 and 1.0 (current: " << params.max_ambiguity << ")" << std::endl;
        valid = false;
    }
    
    // 验证单应性参数
    if (params.ransac_iterations < 1000 || params.ransac_iterations > 50000) {
        std::cerr << "Warning: ransac_iterations should be between 1000 and 50000 (current: " << params.ransac_iterations << ")" << std::endl;
        valid = false;
    }
    
    if (params.inlier_threshold < 1.0f || params.inlier_threshold > 10.0f) {
        std::cerr << "Warning: inlier_threshold should be between 1.0 and 10.0 (current: " << params.inlier_threshold << ")" << std::endl;
        valid = false;
    }
    
    // 验证性能参数
    if (params.max_features < 4096 || params.max_features > 65536) {
        std::cerr << "Warning: max_features should be between 4096 and 65536 (current: " << params.max_features << ")" << std::endl;
        valid = false;
    }
    
    return valid;
}

void SiftConfigTxt::printConfig() const {
    std::cout << "=== SIFT Configuration (TXT Format) ===" << std::endl;
    std::cout << "Core Algorithm Parameters:" << std::endl;
    std::cout << "  Initial Blur: " << params.initial_blur << std::endl;
    std::cout << "  DoG Threshold: " << params.dog_threshold << std::endl;
    std::cout << "  Num Octaves: " << params.num_octaves << std::endl;
    std::cout << "  Lowest Scale: " << params.lowest_scale << std::endl;
    std::cout << "  Edge Limit: " << params.edge_limit << std::endl;
    std::cout << "  Scale Up: " << (params.scale_up ? "true" : "false") << std::endl;
    
    std::cout << "Matching Parameters:" << std::endl;
    std::cout << "  Min Score: " << params.min_score << std::endl;
    std::cout << "  Max Ambiguity: " << params.max_ambiguity << std::endl;
    
    std::cout << "Homography Parameters:" << std::endl;
    std::cout << "  RANSAC Iterations: " << params.ransac_iterations << std::endl;
    std::cout << "  Inlier Threshold: " << params.inlier_threshold << std::endl;
    std::cout << "  Optimization Iterations: " << params.optimization_iterations << std::endl;
    std::cout << "  Optimization Threshold: " << params.optimization_threshold << std::endl;
    
    std::cout << "Performance Parameters:" << std::endl;
    std::cout << "  Max Features: " << params.max_features << std::endl;
    std::cout << "  CUDA Device: " << params.cuda_device << std::endl;
    
    std::cout << "Input/Output Parameters:" << std::endl;
    std::cout << "  Image Set: " << params.image_set << std::endl;
    std::cout << "  Image1 Path: " << params.image1_path << std::endl;
    std::cout << "  Image2 Path: " << params.image2_path << std::endl;
    std::cout << "  Verbose: " << (params.verbose ? "true" : "false") << std::endl;
    std::cout << "  Show Matches: " << (params.show_matches ? "true" : "false") << std::endl;
}

void SiftConfigTxt::getImagePaths(std::string& img1_path, std::string& img2_path) const {
    if (params.image_set == 0) {
        img1_path = params.image1_path;
        img2_path = params.image2_path;
    } else {
        img1_path = params.alt_image1_path;
        img2_path = params.alt_image2_path;
    }
}

bool SiftConfigTxt::saveToFile(const std::string& config_file) {
    std::ofstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write config file: " << config_file << std::endl;
        return false;
    }
    
    file << "# Auto-generated SIFT configuration file (TXT format)" << std::endl;
    file << "# Format: parameter_name = value" << std::endl;
    file << std::endl;
    
    file << "# Core Algorithm Parameters" << std::endl;
    file << "initial_blur = " << params.initial_blur << std::endl;
    file << "dog_threshold = " << params.dog_threshold << std::endl;
    file << "num_octaves = " << params.num_octaves << std::endl;
    file << "lowest_scale = " << params.lowest_scale << std::endl;
    file << "edge_limit = " << params.edge_limit << std::endl;
    file << "scale_up = " << (params.scale_up ? "true" : "false") << std::endl;
    file << std::endl;
    
    file << "# Matching Parameters" << std::endl;
    file << "min_score = " << params.min_score << std::endl;
    file << "max_ambiguity = " << params.max_ambiguity << std::endl;
    file << std::endl;
    
    file << "# Homography Parameters" << std::endl;
    file << "ransac_iterations = " << params.ransac_iterations << std::endl;
    file << "inlier_threshold = " << params.inlier_threshold << std::endl;
    file << "optimization_iterations = " << params.optimization_iterations << std::endl;
    file << "optimization_threshold = " << params.optimization_threshold << std::endl;
    file << std::endl;
    
    file << "# Performance Parameters" << std::endl;
    file << "max_features = " << params.max_features << std::endl;
    file << "cuda_device = " << params.cuda_device << std::endl;
    file << std::endl;
    
    file << "# Input/Output Parameters" << std::endl;
    file << "image_set = " << params.image_set << std::endl;
    file << "image1_path = " << params.image1_path << std::endl;
    file << "image2_path = " << params.image2_path << std::endl;
    file << "alt_image1_path = " << params.alt_image1_path << std::endl;
    file << "alt_image2_path = " << params.alt_image2_path << std::endl;
    file << std::endl;
    
    file << "# Debug Parameters" << std::endl;
    file << "verbose = " << (params.verbose ? "true" : "false") << std::endl;
    file << "show_matches = " << (params.show_matches ? "true" : "false") << std::endl;
    file << "save_intermediate = " << (params.save_intermediate ? "true" : "false") << std::endl;
    file << "output_path = " << params.output_path << std::endl;
    file << std::endl;
    
    file << "# Visualization Parameters" << std::endl;
    file << "enable_visualization = " << (params.enable_visualization ? "true" : "false") << std::endl;
    file << "save_visualization = " << (params.save_visualization ? "true" : "false") << std::endl;
    file << "show_visualization_window = " << (params.show_visualization_window ? "true" : "false") << std::endl;
    file << "feature_circle_radius = " << params.feature_circle_radius << std::endl;
    file << "match_line_thickness = " << params.match_line_thickness << std::endl;
    file << "show_only_good_matches = " << (params.show_only_good_matches ? "true" : "false") << std::endl;
    file << "visualization_output_path = " << params.visualization_output_path << std::endl;
    file << std::endl;
    
    file.close();
    return true;
}
