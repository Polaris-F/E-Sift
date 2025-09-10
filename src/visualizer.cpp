#include "visualizer.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

SiftVisualizer::SiftVisualizer(bool save_to_file, bool show_window)
    : save_to_file_(save_to_file), show_window_(show_window),
      feature_circle_radius_(3), match_line_thickness_(1),
      show_only_good_matches_(true), match_error_threshold_(5.0f),
      overlay_alpha_(0.5f) {
}

cv::Mat SiftVisualizer::visualizeSiftFeatures(const cv::Mat& image, const SiftData& siftData, 
                                              const std::string& title) {
    cv::Mat vis_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, vis_image, cv::COLOR_GRAY2BGR);
    } else {
        vis_image = image.clone();
    }
    
    std::cout << "=== SIFT特征点可视化 ===" << std::endl;
    std::cout << "图像: " << title << std::endl;
    std::cout << "检测到的特征点数量: " << siftData.numPts << std::endl;
    
#ifdef MANAGEDMEM
    SiftPoint *sift_points = siftData.m_data;
#else
    SiftPoint *sift_points = siftData.h_data;
#endif

    // 统计特征点信息
    float min_scale = FLT_MAX, max_scale = 0;
    float min_score = FLT_MAX, max_score = 0;
    
    for (int i = 0; i < siftData.numPts; i++) {
        min_scale = std::min(min_scale, sift_points[i].scale);
        max_scale = std::max(max_scale, sift_points[i].scale);
        min_score = std::min(min_score, sift_points[i].score);
        max_score = std::max(max_score, sift_points[i].score);
    }
    
    std::cout << "尺度范围: " << std::fixed << std::setprecision(2) 
              << min_scale << " - " << max_scale << std::endl;
    std::cout << "得分范围: " << min_score << " - " << max_score << std::endl;
    
    // 绘制特征点
    for (int i = 0; i < siftData.numPts; i++) {
        SiftPoint& point = sift_points[i];
        
        // 根据尺度确定颜色和大小
        cv::Scalar color = getColorByScale(point.scale);
        int radius = std::max(2, (int)(feature_circle_radius_ * point.scale / max_scale * 2));
        
        // 绘制特征点圆圈
        cv::Point2f center(point.xpos, point.ypos);
        cv::circle(vis_image, center, radius, color, 1);
        
        // 绘制方向线（如果尺度足够大）
        if (point.scale > 2.0f) {
            float angle = point.orientation * M_PI / 180.0f;
            cv::Point2f direction(
                center.x + cos(angle) * radius * 1.5f,
                center.y + sin(angle) * radius * 1.5f
            );
            cv::line(vis_image, center, direction, color, 1);
        }
    }
    
    // 添加图例
    int legend_y = 30;
    cv::putText(vis_image, "Features: " + std::to_string(siftData.numPts), 
                cv::Point(10, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(vis_image, "Scale: " + std::to_string(min_scale) + "-" + std::to_string(max_scale), 
                cv::Point(10, legend_y + 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    if (save_to_file_) {
        saveVisualization(vis_image, "tmp/sift_features_" + title + ".jpg");
    }
    
    if (show_window_) {
        showVisualization(vis_image, title);
    }
    
    return vis_image;
}

cv::Mat SiftVisualizer::visualizeSiftMatches(const cv::Mat& img1, const cv::Mat& img2,
                                            const SiftData& siftData1, const SiftData& siftData2,
                                            const std::string& title) {
    // 创建并排显示的图像
    cv::Mat vis_img1, vis_img2;
    if (img1.channels() == 1) {
        cv::cvtColor(img1, vis_img1, cv::COLOR_GRAY2BGR);
    } else {
        vis_img1 = img1.clone();
    }
    
    if (img2.channels() == 1) {
        cv::cvtColor(img2, vis_img2, cv::COLOR_GRAY2BGR);
    } else {
        vis_img2 = img2.clone();
    }
    
    // 创建组合图像
    cv::Mat combined_image;
    cv::hconcat(vis_img1, vis_img2, combined_image);
    
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
    SiftPoint *sift2 = siftData2.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
    SiftPoint *sift2 = siftData2.h_data;
#endif

    std::cout << "\n=== SIFT特征点匹配可视化 ===" << std::endl;
    std::cout << "图像1特征点: " << siftData1.numPts << std::endl;
    std::cout << "图像2特征点: " << siftData2.numPts << std::endl;
    
    int good_matches = 0;
    int total_matches = 0;
    float total_error = 0.0f;
    
    // 绘制所有特征点
    for (int i = 0; i < siftData1.numPts; i++) {
        cv::Point2f pt1(sift1[i].xpos, sift1[i].ypos);
        cv::circle(combined_image, pt1, 2, cv::Scalar(100, 100, 100), 1);
    }
    
    for (int i = 0; i < siftData2.numPts; i++) {
        cv::Point2f pt2(sift2[i].xpos + img1.cols, sift2[i].ypos);
        cv::circle(combined_image, pt2, 2, cv::Scalar(100, 100, 100), 1);
    }
    
    // 绘制匹配线
    for (int i = 0; i < siftData1.numPts; i++) {
        if (sift1[i].match >= 0 && sift1[i].match < siftData2.numPts) {
            total_matches++;
            
            bool is_good_match = sift1[i].match_error < match_error_threshold_;
            if (show_only_good_matches_ && !is_good_match) {
                continue;
            }
            
            if (is_good_match) {
                good_matches++;
                total_error += sift1[i].match_error;
            }
            
            cv::Point2f pt1(sift1[i].xpos, sift1[i].ypos);
            cv::Point2f pt2(sift2[sift1[i].match].xpos + img1.cols, sift2[sift1[i].match].ypos);
            
            // 根据匹配质量选择颜色
            cv::Scalar color;
            if (is_good_match) {
                color = cv::Scalar(0, 255, 0); // 绿色 - 好匹配
            } else {
                color = cv::Scalar(0, 0, 255); // 红色 - 差匹配
            }
            
            // 绘制匹配线
            cv::line(combined_image, pt1, pt2, color, match_line_thickness_);
            
            // 绘制特征点
            cv::circle(combined_image, pt1, 3, color, 2);
            cv::circle(combined_image, pt2, 3, color, 2);
        }
    }
    
    float avg_error = (good_matches > 0) ? total_error / good_matches : 0.0f;
    
    std::cout << "总匹配数: " << total_matches << std::endl;
    std::cout << "好匹配数: " << good_matches << " (误差 < " << match_error_threshold_ << ")" << std::endl;
    std::cout << "匹配率: " << std::fixed << std::setprecision(1) 
              << (100.0f * good_matches / std::max(1, siftData1.numPts)) << "%" << std::endl;
    std::cout << "平均匹配误差: " << std::fixed << std::setprecision(2) << avg_error << std::endl;
    
    // 添加信息文本
    int legend_y = 30;
    cv::putText(combined_image, "Good Matches: " + std::to_string(good_matches) + "/" + std::to_string(total_matches), 
                cv::Point(10, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined_image, "Avg Error: " + std::to_string(avg_error), 
                cv::Point(10, legend_y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined_image, "Green: Good, Red: Poor", 
                cv::Point(10, legend_y + 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    if (save_to_file_) {
        saveVisualization(combined_image, "tmp/sift_matches_" + title + ".jpg");
    }
    
    if (show_window_) {
        showVisualization(combined_image, title + " - Matches");
    }
    
    return combined_image;
}

cv::Mat SiftVisualizer::visualizeSiftDetails(const cv::Mat& image, const SiftData& siftData,
                                            const std::string& title) {
    cv::Mat vis_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, vis_image, cv::COLOR_GRAY2BGR);
    } else {
        vis_image = image.clone();
    }
    
#ifdef MANAGEDMEM
    SiftPoint *sift_points = siftData.m_data;
#else
    SiftPoint *sift_points = siftData.h_data;
#endif

    std::cout << "\n=== SIFT特征点详细信息 ===" << std::endl;
    
    // 按尺度排序显示前10个特征点的详细信息
    std::vector<std::pair<float, int>> scale_indices;
    for (int i = 0; i < siftData.numPts; i++) {
        scale_indices.push_back({sift_points[i].scale, i});
    }
    std::sort(scale_indices.rbegin(), scale_indices.rend()); // 降序排列
    
    std::cout << "前10个最大尺度的特征点:" << std::endl;
    std::cout << "ID\tX\tY\tScale\tOrient\tScore" << std::endl;
    
    for (int i = 0; i < std::min(10, (int)scale_indices.size()); i++) {
        int idx = scale_indices[i].second;
        SiftPoint& point = sift_points[idx];
        
        std::cout << idx << "\t" 
                  << std::fixed << std::setprecision(1) << point.xpos << "\t"
                  << point.ypos << "\t"
                  << std::setprecision(2) << point.scale << "\t"
                  << std::setprecision(0) << point.orientation << "\t"
                  << std::setprecision(3) << point.score << std::endl;
        
        // 绘制详细的特征点信息
        cv::Point2f center(point.xpos, point.ypos);
        int radius = (int)(point.scale * 2);
        
        // 根据尺度大小使用不同颜色
        cv::Scalar color = getColorByScale(point.scale);
        
        // 绘制多层圆圈表示尺度
        cv::circle(vis_image, center, radius, color, 2);
        cv::circle(vis_image, center, radius/2, color, 1);
        
        // 绘制方向箭头
        float angle = point.orientation * M_PI / 180.0f;
        cv::Point2f arrow_end(
            center.x + cos(angle) * radius,
            center.y + sin(angle) * radius
        );
        cv::arrowedLine(vis_image, center, arrow_end, color, 2);
        
        // 添加索引标签
        cv::putText(vis_image, std::to_string(idx), 
                    cv::Point(center.x + radius + 5, center.y), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
    }
    
    if (save_to_file_) {
        saveVisualization(vis_image, "tmp/sift_details_" + title + ".jpg");
    }
    
    if (show_window_) {
        showVisualization(vis_image, title + " - Details");
    }
    
    return vis_image;
}

void SiftVisualizer::visualizeMatchQuality(const SiftData& siftData1, const SiftData& siftData2,
                                          const std::string& title) {
    std::cout << "\n=== 匹配质量分析 ===" << std::endl;
    
#ifdef MANAGEDMEM
    SiftPoint *sift1 = siftData1.m_data;
#else
    SiftPoint *sift1 = siftData1.h_data;
#endif

    std::vector<float> match_errors;
    std::vector<float> match_scores;
    std::vector<float> ambiguities;
    
    int error_buckets[5] = {0}; // <1, 1-2, 2-5, 5-10, >10
    
    for (int i = 0; i < siftData1.numPts; i++) {
        if (sift1[i].match >= 0) {
            match_errors.push_back(sift1[i].match_error);
            match_scores.push_back(sift1[i].score);
            ambiguities.push_back(sift1[i].ambiguity);
            
            // 统计误差分布
            if (sift1[i].match_error < 1.0f) error_buckets[0]++;
            else if (sift1[i].match_error < 2.0f) error_buckets[1]++;
            else if (sift1[i].match_error < 5.0f) error_buckets[2]++;
            else if (sift1[i].match_error < 10.0f) error_buckets[3]++;
            else error_buckets[4]++;
        }
    }
    
    if (!match_errors.empty()) {
        std::sort(match_errors.begin(), match_errors.end());
        std::sort(match_scores.begin(), match_scores.end());
        std::sort(ambiguities.begin(), ambiguities.end());
        
        std::cout << "匹配误差统计:" << std::endl;
        std::cout << "  < 1.0: " << error_buckets[0] << " 个" << std::endl;
        std::cout << "  1-2: " << error_buckets[1] << " 个" << std::endl;
        std::cout << "  2-5: " << error_buckets[2] << " 个" << std::endl;
        std::cout << "  5-10: " << error_buckets[3] << " 个" << std::endl;
        std::cout << "  > 10: " << error_buckets[4] << " 个" << std::endl;
        
        std::cout << "\n匹配质量统计:" << std::endl;
        std::cout << "误差中位数: " << std::fixed << std::setprecision(2) 
                  << match_errors[match_errors.size()/2] << std::endl;
        std::cout << "得分中位数: " << match_scores[match_scores.size()/2] << std::endl;
        std::cout << "歧义度中位数: " << ambiguities[ambiguities.size()/2] << std::endl;
    }
}

void SiftVisualizer::saveVisualization(const cv::Mat& vis_image, const std::string& filename) {
    if (cv::imwrite(filename, vis_image)) {
        std::cout << "可视化结果已保存到: " << filename << std::endl;
    } else {
        std::cerr << "保存可视化结果失败: " << filename << std::endl;
    }
}

void SiftVisualizer::showVisualization(const cv::Mat& vis_image, const std::string& title) {
    cv::imshow(title, vis_image);
    std::cout << "按任意键继续..." << std::endl;
    cv::waitKey(0);
    cv::destroyWindow(title);
}

cv::Scalar SiftVisualizer::getColorByScore(float score) {
    // 根据得分返回颜色 (蓝色到红色渐变)
    float normalized = std::min(1.0f, std::max(0.0f, score / 100.0f));
    return cv::Scalar(255 * (1 - normalized), 0, 255 * normalized);
}

cv::Scalar SiftVisualizer::getColorByScale(float scale) {
    // 根据尺度返回颜色 (绿色到黄色到红色)
    float normalized = std::min(1.0f, std::max(0.0f, scale / 10.0f));
    if (normalized < 0.5f) {
        return cv::Scalar(0, 255, 255 * 2 * normalized); // 绿到黄
    } else {
        return cv::Scalar(0, 255 * 2 * (1 - normalized), 255); // 黄到红
    }
}

void SiftVisualizer::drawFeaturePoint(cv::Mat& image, const SiftPoint& point, const cv::Scalar& color) {
    cv::Point2f center(point.xpos, point.ypos);
    int radius = std::max(2, (int)(point.scale));
    cv::circle(image, center, radius, color, 1);
    
    // 绘制方向
    float angle = point.orientation * M_PI / 180.0f;
    cv::Point2f direction(
        center.x + cos(angle) * radius,
        center.y + sin(angle) * radius
    );
    cv::line(image, center, direction, color, 1);
}

void SiftVisualizer::drawMatchLine(cv::Mat& image, const cv::Point2f& pt1, const cv::Point2f& pt2, 
                                  const cv::Scalar& color, int thickness) {
    cv::line(image, pt1, pt2, color, thickness);
}

cv::Mat SiftVisualizer::visualizeImageOverlay(const cv::Mat& img1, const cv::Mat& img2,
                                              const float* homography,
                                              float alpha,
                                              const std::string& title) {
    std::cout << "\n=== 图像叠加可视化 ===" << std::endl;
    std::cout << "叠加透明度: " << alpha << std::endl;
    
    // 转换为彩色图像
    cv::Mat color_img1, color_img2;
    if (img1.channels() == 1) {
        cv::cvtColor(img1, color_img1, cv::COLOR_GRAY2BGR);
    } else {
        color_img1 = img1.clone();
    }
    
    if (img2.channels() == 1) {
        cv::cvtColor(img2, color_img2, cv::COLOR_GRAY2BGR);
    } else {
        color_img2 = img2.clone();
    }
    
    cv::Mat overlay_result;
    
    if (homography != nullptr) {
        // 使用单应性矩阵变换后叠加
        overlay_result = visualizeTransformedOverlay(color_img1, color_img2, homography, alpha, title);
    } else {
        // 直接叠加（假设图像已经对齐）
        if (color_img1.size() != color_img2.size()) {
            // 调整图像大小到相同尺寸
            cv::Size target_size(std::max(color_img1.cols, color_img2.cols), 
                               std::max(color_img1.rows, color_img2.rows));
            cv::resize(color_img1, color_img1, target_size);
            cv::resize(color_img2, color_img2, target_size);
        }
        
        // 确保两张图像都是3通道BGR格式
        cv::Mat bgr_img1, bgr_img2;
        if (color_img1.channels() == 3) {
            bgr_img1 = color_img1.clone();
        } else {
            cv::cvtColor(color_img1, bgr_img1, cv::COLOR_GRAY2BGR);
        }
        
        if (color_img2.channels() == 3) {
            bgr_img2 = color_img2.clone();
        } else {
            cv::cvtColor(color_img2, bgr_img2, cv::COLOR_GRAY2BGR);
        }
        
        // 给两张图像添加不同的颜色调色以便区分
        cv::Mat tinted_img1, tinted_img2;
        
        // 创建彩色版本 - 图像1偏红色，图像2偏绿色
        std::vector<cv::Mat> channels1, channels2;
        cv::split(bgr_img1, channels1);
        cv::split(bgr_img2, channels2);
        
        // 图像1: 保持红色通道，降低绿蓝通道
        cv::merge(std::vector<cv::Mat>{channels1[0] * 0.7, channels1[1] * 0.7, channels1[2]}, tinted_img1);
        
        // 图像2: 保持绿色通道，降低红蓝通道
        cv::merge(std::vector<cv::Mat>{channels2[0] * 0.7, channels2[1], channels2[2] * 0.7}, tinted_img2);
        
        // 半透明叠加
        cv::addWeighted(tinted_img1, 1.0 - alpha, tinted_img2, alpha, 0, overlay_result);
        
        std::cout << "直接叠加完成 - 红色区域主要是图像1，绿色区域主要是图像2，黄色区域是重叠区域" << std::endl;
    }
    
    // 添加标题信息
    cv::putText(overlay_result, "Image Overlay (Alpha=" + std::to_string(alpha) + ")", 
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(overlay_result, "Red: Img1, Green: Img2, Yellow: Overlap", 
                cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    if (save_to_file_) {
        saveVisualization(overlay_result, "tmp/image_overlay_" + title + ".jpg");
    }
    
    if (show_window_) {
        showVisualization(overlay_result, title + " - Overlay");
    }
    
    return overlay_result;
}

cv::Mat SiftVisualizer::visualizeTransformedOverlay(const cv::Mat& img1, const cv::Mat& img2,
                                                   const float* homography,
                                                   float alpha,
                                                   const std::string& title) {
    std::cout << "\n=== 变换叠加可视化 ===" << std::endl;
    std::cout << "使用单应性矩阵进行图像变换叠加" << std::endl;
    
    // 转换单应性矩阵
    cv::Mat H = convertHomographyMatrix(homography);
    
    std::cout << "单应性矩阵:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << std::fixed << std::setprecision(6) << H.at<double>(i, j) << "\t";
        }
        std::cout << std::endl;
    }
    
    // 转换为彩色图像
    cv::Mat color_img1, color_img2;
    if (img1.channels() == 1) {
        cv::cvtColor(img1, color_img1, cv::COLOR_GRAY2BGR);
    } else {
        color_img1 = img1.clone();
    }
    
    if (img2.channels() == 1) {
        cv::cvtColor(img2, color_img2, cv::COLOR_GRAY2BGR);
    } else {
        color_img2 = img2.clone();
    }
    
    // 计算变换后的图像边界
    std::vector<cv::Point2f> corners1 = {
        cv::Point2f(0, 0),
        cv::Point2f(img1.cols, 0),
        cv::Point2f(img1.cols, img1.rows),
        cv::Point2f(0, img1.rows)
    };
    
    std::vector<cv::Point2f> transformed_corners;
    cv::perspectiveTransform(corners1, transformed_corners, H);
    
    // 计算结果图像的大小
    float min_x = 0, max_x = img2.cols;
    float min_y = 0, max_y = img2.rows;
    
    for (const auto& corner : transformed_corners) {
        min_x = std::min(min_x, corner.x);
        max_x = std::max(max_x, corner.x);
        min_y = std::min(min_y, corner.y);
        max_y = std::max(max_y, corner.y);
    }
    
    // 创建平移矩阵以确保所有内容都在可见区域内
    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 2) = -min_x;
    translation.at<double>(1, 2) = -min_y;
    
    cv::Mat H_adjusted = translation * H;
    
    cv::Size result_size(std::ceil(max_x - min_x), std::ceil(max_y - min_y));
    
    // 变换第一张图像
    cv::Mat transformed_img1;
    cv::warpPerspective(color_img1, transformed_img1, H_adjusted, result_size);
    
    // 将第二张图像放置在相同大小的画布上
    cv::Mat canvas_img2 = cv::Mat::zeros(result_size, transformed_img1.type());
    int offset_x = std::max(0, (int)(-min_x));
    int offset_y = std::max(0, (int)(-min_y));
    
    if (offset_x + color_img2.cols <= canvas_img2.cols && offset_y + color_img2.rows <= canvas_img2.rows) {
        cv::Mat roi = canvas_img2(cv::Rect(offset_x, offset_y, color_img2.cols, color_img2.rows));
        color_img2.copyTo(roi);
    }
    
    // 给两张图像添加不同的颜色调色以便区分
    cv::Mat tinted_transformed, tinted_canvas;
    
    // 确保图像是3通道
    if (transformed_img1.channels() == 3) {
        tinted_transformed = transformed_img1.clone();
    } else {
        cv::cvtColor(transformed_img1, tinted_transformed, cv::COLOR_GRAY2BGR);
    }
    
    if (canvas_img2.channels() == 3) {
        tinted_canvas = canvas_img2.clone();
    } else {
        cv::cvtColor(canvas_img2, tinted_canvas, cv::COLOR_GRAY2BGR);
    }
    
    // 应用颜色调色
    std::vector<cv::Mat> channels_trans, channels_canvas;
    cv::split(tinted_transformed, channels_trans);
    cv::split(tinted_canvas, channels_canvas);
    
    // 变换后的图像偏红色
    cv::merge(std::vector<cv::Mat>{channels_trans[0] * 0.7, channels_trans[1] * 0.7, channels_trans[2]}, tinted_transformed);
    
    // 原始图像2偏绿色
    cv::merge(std::vector<cv::Mat>{channels_canvas[0] * 0.7, channels_canvas[1], channels_canvas[2] * 0.7}, tinted_canvas);
    
    // 创建叠加结果
    cv::Mat overlay_result;
    cv::addWeighted(tinted_transformed, 1.0 - alpha, tinted_canvas, alpha, 0, overlay_result);
    
    // 绘制变换后的图像边界
    std::vector<cv::Point> boundary_points;
    for (const auto& corner : transformed_corners) {
        boundary_points.push_back(cv::Point(corner.x - min_x, corner.y - min_y));
    }
    
    if (boundary_points.size() == 4) {
        cv::polylines(overlay_result, boundary_points, true, cv::Scalar(0, 255, 255), 2);
    }
    
    std::cout << "变换叠加完成 - 黄色边框显示图像1变换后的边界" << std::endl;
    std::cout << "结果图像大小: " << result_size.width << "x" << result_size.height << std::endl;
    
    // 添加标题信息
    cv::putText(overlay_result, "Transformed Overlay (Alpha=" + std::to_string(alpha) + ")", 
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(overlay_result, "Red: Transformed Img1, Green: Img2, Yellow: Boundary", 
                cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    if (save_to_file_) {
        saveVisualization(overlay_result, "tmp/transformed_overlay_" + title + ".jpg");
    }
    
    if (show_window_) {
        showVisualization(overlay_result, title + " - Transformed Overlay");
    }
    
    return overlay_result;
}

cv::Mat SiftVisualizer::applyHomography(const cv::Mat& image, const cv::Mat& H) {
    cv::Mat result;
    cv::warpPerspective(image, result, H, image.size());
    return result;
}

cv::Mat SiftVisualizer::convertHomographyMatrix(const float* homography_array) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            H.at<double>(i, j) = static_cast<double>(homography_array[i * 3 + j]);
        }
    }
    return H;
}
