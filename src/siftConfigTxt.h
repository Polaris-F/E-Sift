#ifndef SIFT_CONFIG_TXT_H
#define SIFT_CONFIG_TXT_H

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

/**
 * @brief SIFT算法配置参数管理类 (TXT格式)
 * 
 * 该类负责从TXT格式配置文件中读取SIFT算法的各种参数，
 * 并提供访问这些参数的接口。
 * TXT格式更简单，使用 key = value 的形式。
 */
class SiftConfigTxt {
public:
    // 所有配置参数存储在一个结构体中
    struct Parameters {
        // 核心算法参数
        float initial_blur = 1.0f;
        float dog_threshold = 3.0f;
        int num_octaves = 5;
        float lowest_scale = 0.0f;
        float edge_limit = 10.0f;
        bool scale_up = false;
        
        // 特征匹配参数
        float min_score = 0.85f;
        float max_ambiguity = 0.95f;
        
        // 单应性估计参数
        int ransac_iterations = 10000;
        float inlier_threshold = 5.0f;
        int optimization_iterations = 5;
        float optimization_threshold = 3.0f;
        
        // 内存和性能参数
        int max_features = 32768;
        int cuda_device = 0;
        
        // 输入输出配置
        int image_set = 0;
        std::string image1_path = "data/img1.jpg";
        std::string image2_path = "data/img2.jpg";
        std::string alt_image1_path = "data/left.pgm";
        std::string alt_image2_path = "data/righ.pgm";
        
        // 调试和输出参数
        bool verbose = false;
        bool show_matches = true;
        bool save_intermediate = false;
        std::string output_path = "tmp/";
        
        // 可视化调试参数
        bool enable_visualization = true;
        bool save_visualization = true;
        bool show_visualization_window = false;
        int feature_circle_radius = 3;
        int match_line_thickness = 1;
        bool show_only_good_matches = true;
        float overlay_alpha = 0.5f;
        std::string visualization_output_path = "tmp/";
        
        // 高级参数（编译时参数）
        int num_scales = 5;
        int laplace_radius = 4;
        int lowpass_radius = 4;
        
        // 线程块配置
        int scaledown_width = 64;
        int scaledown_height = 16;
        int scaleup_width = 64;
        int scaleup_height = 8;
        int minmax_width = 30;
        int minmax_height = 8;
        int laplace_width = 128;
        int laplace_height = 4;
        int lowpass_width = 24;
        int lowpass_height = 32;
    } params;

public:
    /**
     * @brief 构造函数
     * @param config_file 配置文件路径
     */
    SiftConfigTxt(const std::string& config_file = "config/sift_config.txt");

    /**
     * @brief 从配置文件加载参数
     * @param config_file 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool loadFromFile(const std::string& config_file);

    /**
     * @brief 保存当前配置到文件
     * @param config_file 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool saveToFile(const std::string& config_file);

    /**
     * @brief 打印当前配置参数
     */
    void printConfig() const;

    /**
     * @brief 验证参数有效性
     * @return 参数有效返回true，无效返回false
     */
    bool validateParams() const;

    /**
     * @brief 获取图像路径（根据image_set自动选择）
     * @param img1_path 输出第一张图像路径
     * @param img2_path 输出第二张图像路径
     */
    void getImagePaths(std::string& img1_path, std::string& img2_path) const;

    /**
     * @brief 设置参数值
     * @param key 参数名
     * @param value 参数值（字符串格式）
     * @return 设置成功返回true
     */
    bool setParameter(const std::string& key, const std::string& value);

    /**
     * @brief 获取参数值
     * @param key 参数名
     * @return 参数值（字符串格式）
     */
    std::string getParameter(const std::string& key) const;

    /**
     * @brief 列出所有可用的参数名
     * @return 参数名列表
     */
    std::vector<std::string> getParameterNames() const;

private:
    /**
     * @brief 解析TXT格式的配置文件
     * @param content 文件内容
     */
    void parseTxtContent(const std::string& content);

    /**
     * @brief 去除字符串首尾空白字符
     * @param str 输入字符串
     * @return 处理后的字符串
     */
    std::string trim(const std::string& str);

    /**
     * @brief 解析键值对
     * @param line 配置行
     * @param key 输出键
     * @param value 输出值
     * @return 解析成功返回true
     */
    bool parseKeyValue(const std::string& line, std::string& key, std::string& value);

    /**
     * @brief 转换字符串为浮点数
     * @param str 字符串
     * @param default_val 默认值
     * @return 转换结果
     */
    float stringToFloat(const std::string& str, float default_val = 0.0f);

    /**
     * @brief 转换字符串为整数
     * @param str 字符串
     * @param default_val 默认值
     * @return 转换结果
     */
    int stringToInt(const std::string& str, int default_val = 0);

    /**
     * @brief 转换字符串为布尔值
     * @param str 字符串
     * @param default_val 默认值
     * @return 转换结果
     */
    bool stringToBool(const std::string& str, bool default_val = false);

    /**
     * @brief 移除字符串中的引号
     * @param str 输入字符串
     * @return 移除引号后的字符串
     */
    std::string removeQuotes(const std::string& str);
};

#endif // SIFT_CONFIG_TXT_H
