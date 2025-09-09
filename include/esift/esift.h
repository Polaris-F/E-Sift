#ifndef ESIFT_H
#define ESIFT_H

/**
 * @file esift.h
 * @brief Enhanced SIFT (E-Sift) - 优化的CUDA SIFT实现
 * 
 * 专为Jetson AGX Orin优化的SIFT特征提取和匹配库
 * 主要用于时序图像的前后帧匹配和图像对齐
 */

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace esift {

/**
 * @brief SIFT特征点结构
 */
struct Feature {
    float x, y;              // 特征点坐标
    float scale;             // 尺度
    float orientation;       // 方向
    float response;          // 响应强度
    std::vector<float> descriptor;  // 128维描述符
    
    Feature() : x(0), y(0), scale(0), orientation(0), response(0) {
        descriptor.resize(128, 0.0f);
    }
};

/**
 * @brief 特征匹配结果
 */
struct Match {
    int queryIdx;           // 查询图像特征索引
    int trainIdx;           // 训练图像特征索引
    float distance;         // 匹配距离
    
    Match(int q, int t, float d) : queryIdx(q), trainIdx(t), distance(d) {}
};

/**
 * @brief 单应性变换矩阵
 */
using Homography = cv::Mat;

/**
 * @brief SIFT检测器配置参数
 */
struct SiftConfig {
    int maxFeatures = 5000;         // 最大特征数量
    float threshold = 3.5f;         // DoG阈值
    float edgeThreshold = 10.0f;    // 边缘阈值
    float sigma = 1.6f;             // 高斯参数
    int octaves = 4;                // 金字塔层数
    bool useFP16 = true;            // 使用半精度优化
    bool useUnifiedMemory = true;   // 使用统一内存
    
    // Jetson优化参数
    int blockSizeX = 16;            // CUDA block X维度
    int blockSizeY = 16;            // CUDA block Y维度
    bool enableProfiling = false;   // 启用性能分析
};

/**
 * @brief 匹配器配置参数
 */
struct MatcherConfig {
    float ratioThreshold = 0.7f;    // Lowe's ratio test阈值
    int maxDistance = 256;          // 最大匹配距离
    bool crossCheck = true;         // 交叉验证
    bool useGPU = true;             // 使用GPU加速
};

/**
 * @brief 前向声明
 */
class SiftDetectorImpl;
class FeatureMatcherImpl;

/**
 * @brief SIFT特征检测器
 */
class SiftDetector {
public:
    explicit SiftDetector(const SiftConfig& config = SiftConfig());
    ~SiftDetector();

    /**
     * @brief 检测并计算SIFT特征
     * @param image 输入图像 (CV_8UC1 或 CV_32FC1)
     * @return 检测到的特征点列表
     */
    std::vector<Feature> detectAndCompute(const cv::Mat& image);

    /**
     * @brief 仅检测特征点
     * @param image 输入图像
     * @return 特征点列表(不含描述符)
     */
    std::vector<Feature> detect(const cv::Mat& image);

    /**
     * @brief 为已检测的特征点计算描述符
     * @param image 输入图像
     * @param features 输入输出特征点列表
     */
    void compute(const cv::Mat& image, std::vector<Feature>& features);

    /**
     * @brief 获取配置参数
     */
    const SiftConfig& getConfig() const;

    /**
     * @brief 获取上次检测的性能统计
     */
    struct Stats {
        float detectionTime;    // 检测耗时(ms)
        float computeTime;      // 描述符计算耗时(ms)
        float totalTime;        // 总耗时(ms)
        int numFeatures;        // 检测到的特征数量
        float gpuUtilization;   // GPU利用率
    };
    Stats getLastStats() const;

private:
    std::unique_ptr<SiftDetectorImpl> impl_;
};

/**
 * @brief 特征匹配器
 */
class FeatureMatcher {
public:
    explicit FeatureMatcher(const MatcherConfig& config = MatcherConfig());
    ~FeatureMatcher();

    /**
     * @brief 匹配两组特征
     * @param features1 第一组特征
     * @param features2 第二组特征
     * @return 匹配结果
     */
    std::vector<Match> match(const std::vector<Feature>& features1,
                            const std::vector<Feature>& features2);

    /**
     * @brief KNN匹配
     * @param features1 第一组特征
     * @param features2 第二组特征
     * @param k 每个特征的匹配数量
     * @return 匹配结果(每个查询特征对应k个匹配)
     */
    std::vector<std::vector<Match>> knnMatch(const std::vector<Feature>& features1,
                                            const std::vector<Feature>& features2,
                                            int k = 2);

private:
    std::unique_ptr<FeatureMatcherImpl> impl_;
};

/**
 * @brief 时序帧匹配器 - 专为连续帧优化
 */
class SequenceMatcher {
public:
    struct Config {
        SiftConfig siftConfig;
        MatcherConfig matcherConfig;
        int maxFrameHistory = 5;        // 保持的历史帧数
        float minMatchRatio = 0.3f;     // 最小匹配比例
        bool enablePrediction = true;   // 启用运动预测
    };

    explicit SequenceMatcher(const Config& config = Config());
    ~SequenceMatcher();

    /**
     * @brief 处理新的帧
     * @param frame 当前帧
     * @return 相对于上一帧的单应性变换
     */
    Homography processFrame(const cv::Mat& frame);

    /**
     * @brief 获取相对于参考帧的累积变换
     * @return 累积单应性变换
     */
    Homography getCumulativeTransform() const;

    /**
     * @brief 重置参考帧
     * @param referenceFrame 新的参考帧
     */
    void setReferenceFrame(const cv::Mat& referenceFrame);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 几何变换工具函数
 */
namespace geometry {

/**
 * @brief 从匹配点计算单应性变换
 * @param matches 匹配结果
 * @param features1 第一组特征
 * @param features2 第二组特征
 * @param method RANSAC方法
 * @param ransacThreshold RANSAC阈值
 * @return 单应性变换矩阵
 */
Homography findHomography(const std::vector<Match>& matches,
                         const std::vector<Feature>& features1,
                         const std::vector<Feature>& features2,
                         int method = cv::RANSAC,
                         double ransacThreshold = 3.0);

/**
 * @brief 应用单应性变换到图像
 * @param src 源图像
 * @param homography 变换矩阵
 * @param dsize 输出图像尺寸
 * @return 变换后的图像
 */
cv::Mat warpPerspective(const cv::Mat& src, 
                       const Homography& homography,
                       cv::Size dsize);

/**
 * @brief 计算变换质量评估
 * @param matches 匹配结果
 * @param features1 第一组特征
 * @param features2 第二组特征  
 * @param homography 变换矩阵
 * @return 质量分数 [0,1]
 */
float evaluateTransformQuality(const std::vector<Match>& matches,
                              const std::vector<Feature>& features1,
                              const std::vector<Feature>& features2,
                              const Homography& homography);

} // namespace geometry

/**
 * @brief 工具函数
 */
namespace utils {

/**
 * @brief 加载图像并转换为浮点格式
 * @param filename 图像文件路径
 * @return 浮点格式图像
 */
cv::Mat loadImage(const std::string& filename);

/**
 * @brief 可视化特征点
 * @param image 输入图像
 * @param features 特征点
 * @return 带有特征点标记的图像
 */
cv::Mat drawFeatures(const cv::Mat& image, const std::vector<Feature>& features);

/**
 * @brief 可视化匹配结果
 * @param img1 第一幅图像
 * @param img2 第二幅图像
 * @param features1 第一组特征
 * @param features2 第二组特征
 * @param matches 匹配结果
 * @return 匹配可视化图像
 */
cv::Mat drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                   const std::vector<Feature>& features1,
                   const std::vector<Feature>& features2,
                   const std::vector<Match>& matches);

/**
 * @brief 性能分析工具
 */
class Profiler {
public:
    static void enable(bool enabled = true);
    static void printStats();
    static void resetStats();
};

} // namespace utils

} // namespace esift

#endif // ESIFT_H
