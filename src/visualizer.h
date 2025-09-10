#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "cudaSift.h"
#include "cudaImage.h"

class SiftVisualizer {
public:
    // 构造函数
    SiftVisualizer(bool save_to_file = true, bool show_window = false);
    
    // 可视化单张图像的SIFT特征点
    cv::Mat visualizeSiftFeatures(const cv::Mat& image, const SiftData& siftData, 
                                  const std::string& title = "SIFT Features");
    
    // 可视化两张图像的特征点匹配
    cv::Mat visualizeSiftMatches(const cv::Mat& img1, const cv::Mat& img2,
                                const SiftData& siftData1, const SiftData& siftData2,
                                const std::string& title = "SIFT Matches");
    
    // 可视化特征点的尺度和方向信息
    cv::Mat visualizeSiftDetails(const cv::Mat& image, const SiftData& siftData,
                                 const std::string& title = "SIFT Details");
    
    // 可视化匹配的质量分布
    void visualizeMatchQuality(const SiftData& siftData1, const SiftData& siftData2,
                               const std::string& title = "Match Quality");
    
    // 可视化图像叠加 - 将两张图半透明叠加显示配准效果
    cv::Mat visualizeImageOverlay(const cv::Mat& img1, const cv::Mat& img2,
                                  const float* homography = nullptr,
                                  float alpha = 0.5f,
                                  const std::string& title = "Image Overlay");
    
    // 可视化变换后的图像叠加 - 使用单应性矩阵变换后叠加
    cv::Mat visualizeTransformedOverlay(const cv::Mat& img1, const cv::Mat& img2,
                                       const float* homography,
                                       float alpha = 0.5f,
                                       const std::string& title = "Transformed Overlay");
    
    // 保存可视化结果
    void saveVisualization(const cv::Mat& vis_image, const std::string& filename);
    
    // 显示可视化结果（如果启用窗口显示）
    void showVisualization(const cv::Mat& vis_image, const std::string& title);
    
    // 设置可视化参数
    void setFeatureCircleRadius(int radius) { feature_circle_radius_ = radius; }
    void setMatchLineThickness(int thickness) { match_line_thickness_ = thickness; }
    void setShowOnlyGoodMatches(bool show_only_good) { show_only_good_matches_ = show_only_good; }
    void setMatchErrorThreshold(float threshold) { match_error_threshold_ = threshold; }
    void setOverlayAlpha(float alpha) { overlay_alpha_ = alpha; }
    
private:
    bool save_to_file_;
    bool show_window_;
    int feature_circle_radius_;
    int match_line_thickness_;
    bool show_only_good_matches_;
    float match_error_threshold_;
    float overlay_alpha_;
    
    // 辅助函数
    cv::Scalar getColorByScore(float score);
    cv::Scalar getColorByScale(float scale);
    void drawFeaturePoint(cv::Mat& image, const SiftPoint& point, const cv::Scalar& color);
    void drawMatchLine(cv::Mat& image, const cv::Point2f& pt1, const cv::Point2f& pt2, 
                      const cv::Scalar& color, int thickness);
    
    // 图像变换辅助函数
    cv::Mat applyHomography(const cv::Mat& image, const cv::Mat& H);
    cv::Mat convertHomographyMatrix(const float* homography_array);
};

#endif // VISUALIZER_H
