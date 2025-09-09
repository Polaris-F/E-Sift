#ifndef ESIFT_FEATURES_H
#define ESIFT_FEATURES_H

#include "image.h"
#include <vector>
#include <cuda_runtime.h>

namespace esift {

/**
 * @brief GPU特征点数据结构
 */
struct GpuFeature {
    float x, y;              // 特征点坐标
    float scale;             // 尺度
    float orientation;       // 方向
    float response;          // 响应强度
    float data[128];        // 描述符数据
};

/**
 * @brief GPU特征集合
 */
class GpuFeatureSet {
public:
    GpuFeatureSet(int maxFeatures = 5000);
    ~GpuFeatureSet();

    /**
     * @brief 从CPU特征转换
     * @param features CPU特征列表
     */
    void uploadFeatures(const std::vector<Feature>& features);

    /**
     * @brief 下载到CPU特征
     * @return CPU特征列表
     */
    std::vector<Feature> downloadFeatures() const;

    /**
     * @brief 重置特征数量
     * @param count 新的特征数量
     */
    void setCount(int count) { numFeatures_ = count; }

    int getCount() const { return numFeatures_; }
    int getMaxCount() const { return maxFeatures_; }
    GpuFeature* getPtr() const { return features_; }

private:
    GpuFeature* features_;
    int numFeatures_;
    int maxFeatures_;
};

/**
 * @brief 特征检测核心类
 */
class FeatureDetector {
public:
    struct Config {
        float threshold = 0.04f;        // 响应阈值
        float edgeThreshold = 10.0f;    // 边缘阈值
        int maxFeatures = 5000;         // 最大特征数
        bool useFP16 = true;           // 半精度优化
    };

    explicit FeatureDetector(const Config& config);
    ~FeatureDetector();

    /**
     * @brief 在DoG金字塔中检测特征
     * @param pyramid 图像金字塔
     * @param features 输出特征集合
     * @return 检测到的特征数量
     */
    int detectKeypoints(const ImagePyramid& pyramid, GpuFeatureSet& features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief 描述符计算类
 */
class DescriptorComputer {
public:
    struct Config {
        int histogramBins = 8;         // 方向直方图bin数
        int descriptorSize = 128;      // 描述符维度
        float descriptorMagnification = 3.0f; // 描述符放大因子
        bool useFP16 = true;          // 半精度优化
    };

    explicit DescriptorComputer(const Config& config);
    ~DescriptorComputer();

    /**
     * @brief 计算特征描述符
     * @param pyramid 图像金字塔
     * @param features 输入输出特征集合
     */
    void computeDescriptors(const ImagePyramid& pyramid, GpuFeatureSet& features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief CUDA kernel调用包装器
 */
namespace kernels {

/**
 * @brief 启动DoG特征检测kernel
 */
void launchFeatureDetection(const ImagePyramid& pyramid,
                           GpuFeatureSet& features,
                           float threshold,
                           float edgeThreshold,
                           cudaStream_t stream = 0);

/**
 * @brief 启动描述符计算kernel
 */
void launchDescriptorComputation(const ImagePyramid& pyramid,
                                GpuFeatureSet& features,
                                const DescriptorComputer::Config& config,
                                cudaStream_t stream = 0);

/**
 * @brief 启动非极大值抑制kernel
 */
void launchNonMaxSuppression(GpuFeatureSet& features,
                            float suppressionRadius,
                            cudaStream_t stream = 0);

} // namespace kernels

} // namespace esift

#endif // ESIFT_FEATURES_H
