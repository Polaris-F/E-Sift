#ifndef ESIFT_IMAGE_H
#define ESIFT_IMAGE_H

#include <opencv2/opencv.hpp>
#include <memory>

namespace esift {

/**
 * @brief GPU图像容器 - 封装CUDA内存管理
 */
class GpuImage {
public:
    GpuImage();
    explicit GpuImage(const cv::Mat& hostImage);
    GpuImage(int width, int height, int type);
    ~GpuImage();

    // 禁用拷贝，启用移动
    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;
    GpuImage(GpuImage&& other) noexcept;
    GpuImage& operator=(GpuImage&& other) noexcept;

    /**
     * @brief 从主机内存上传图像到GPU
     * @param hostImage 主机图像
     */
    void upload(const cv::Mat& hostImage);

    /**
     * @brief 从GPU下载图像到主机内存
     * @return 主机图像
     */
    cv::Mat download() const;

    /**
     * @brief 在GPU上创建指定尺寸的图像
     * @param width 宽度
     * @param height 高度
     * @param type OpenCV类型
     */
    void create(int width, int height, int type);

    /**
     * @brief 释放GPU内存
     */
    void release();

    // 访问器
    int width() const { return width_; }
    int height() const { return height_; }
    int type() const { return type_; }
    size_t step() const { return step_; }
    bool empty() const { return data_ == nullptr; }

    // 获取GPU指针(内部使用)
    void* ptr() const { return data_; }
    float* ptrFloat() const { return static_cast<float*>(data_); }

private:
    void* data_;
    int width_, height_, type_;
    size_t step_;
};

/**
 * @brief 图像金字塔 - 多尺度图像表示
 */
class ImagePyramid {
public:
    ImagePyramid(int octaves = 4, int scales = 3);
    ~ImagePyramid();

    /**
     * @brief 从输入图像构建金字塔
     * @param input 输入图像
     * @param sigma 初始高斯参数
     */
    void build(const GpuImage& input, float sigma = 1.6f);

    /**
     * @brief 获取指定层的图像
     * @param octave 组索引
     * @param scale 尺度索引
     * @return GPU图像
     */
    const GpuImage& getImage(int octave, int scale) const;

    /**
     * @brief 获取DoG图像(差分高斯)
     * @param octave 组索引
     * @param scale 尺度索引
     * @return DoG图像
     */
    const GpuImage& getDoG(int octave, int scale) const;

    int getOctaves() const { return octaves_; }
    int getScales() const { return scales_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int octaves_, scales_;
};

} // namespace esift

#endif // ESIFT_IMAGE_H
