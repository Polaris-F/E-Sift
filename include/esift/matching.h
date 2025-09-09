#ifndef ESIFT_MATCHING_H
#define ESIFT_MATCHING_H

#include "features.h"
#include <vector>
#include <cuda_runtime.h>

namespace esift {

/**
 * @brief GPU匹配结果
 */
struct GpuMatch {
    int queryIdx;
    int trainIdx;
    float distance;
};

/**
 * @brief GPU匹配器
 */
class GpuMatcher {
public:
    struct Config {
        float ratioThreshold = 0.7f;    // Lowe's ratio test
        int maxMatches = 10000;         // 最大匹配数
        bool crossCheck = true;         // 交叉验证
        bool useFP16 = true;           // 半精度优化
    };

    explicit GpuMatcher(const Config& config);
    ~GpuMatcher();

    /**
     * @brief 暴力匹配
     * @param features1 第一组特征
     * @param features2 第二组特征
     * @param matches 输出匹配结果
     * @return 匹配数量
     */
    int bruteForceMatch(const GpuFeatureSet& features1,
                       const GpuFeatureSet& features2,
                       std::vector<Match>& matches);

    /**
     * @brief KNN匹配
     * @param features1 第一组特征
     * @param features2 第二组特征
     * @param matches 输出匹配结果
     * @param k KNN参数
     * @return 匹配数量
     */
    int knnMatch(const GpuFeatureSet& features1,
                const GpuFeatureSet& features2,
                std::vector<std::vector<Match>>& matches,
                int k = 2);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 快速近似匹配器(使用共享内存优化)
 */
class FastMatcher {
public:
    struct Config {
        int tileSize = 32;             // 共享内存tile大小
        float ratioThreshold = 0.7f;   // ratio test阈值
        bool useTextureMemory = true;  // 使用纹理内存
    };

    explicit FastMatcher(const Config& config);
    ~FastMatcher();

    /**
     * @brief 快速匹配算法
     * @param features1 第一组特征
     * @param features2 第二组特征
     * @param matches 输出匹配结果
     * @return 匹配数量
     */
    int fastMatch(const GpuFeatureSet& features1,
                 const GpuFeatureSet& features2,
                 std::vector<Match>& matches);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 序列匹配器 - 专为视频序列优化
 */
class SequentialMatcher {
public:
    struct Config {
        int maxHistory = 3;            // 历史帧数
        float temporalWeight = 0.8f;   // 时序权重
        float spatialThreshold = 50.0f; // 空间阈值
        bool enablePrediction = true;   // 运动预测
    };

    explicit SequentialMatcher(const Config& config);
    ~SequentialMatcher();

    /**
     * @brief 匹配当前帧到历史帧
     * @param currentFeatures 当前帧特征
     * @param matches 输出匹配结果
     * @return 匹配数量
     */
    int matchToHistory(const GpuFeatureSet& currentFeatures,
                      std::vector<Match>& matches);

    /**
     * @brief 更新历史记录
     * @param features 新的特征集合
     */
    void updateHistory(const GpuFeatureSet& features);

    /**
     * @brief 重置历史记录
     */
    void resetHistory();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 匹配核函数包装器
 */
namespace matching_kernels {

/**
 * @brief 启动暴力匹配kernel
 */
void launchBruteForceKernel(const GpuFeatureSet& features1,
                           const GpuFeatureSet& features2,
                           GpuMatch* matches,
                           int* matchCount,
                           float ratioThreshold,
                           cudaStream_t stream = 0);

/**
 * @brief 启动快速匹配kernel (使用共享内存)
 */
void launchFastMatchKernel(const GpuFeatureSet& features1,
                          const GpuFeatureSet& features2,
                          GpuMatch* matches,
                          int* matchCount,
                          int tileSize,
                          float ratioThreshold,
                          cudaStream_t stream = 0);

/**
 * @brief 启动交叉验证kernel
 */
void launchCrossCheckKernel(GpuMatch* matches,
                           int matchCount,
                           bool* validMatches,
                           cudaStream_t stream = 0);

/**
 * @brief 启动匹配过滤kernel
 */
void launchMatchFilterKernel(GpuMatch* matches,
                            int* matchCount,
                            float maxDistance,
                            cudaStream_t stream = 0);

} // namespace matching_kernels

} // namespace esift

#endif // ESIFT_MATCHING_H
