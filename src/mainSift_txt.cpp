//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//  

#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cudaImage.h"
#include "cudaSift.h"
#include "siftConfigTxt.h"
#include "visualizer.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img);
void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography);

double ScaleUp(CudaImage &res, CudaImage &src);

///////////////////////////////////////////////////////////////////////////////
// Main program with TXT configuration support
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) 
{    
  // 加载TXT格式配置文件
  std::string config_file = "config/sift_config.txt";
  if (argc > 3) {
    config_file = argv[3];
  }
  
  std::cout << "Loading configuration from: " << config_file << std::endl;
  SiftConfigTxt config(config_file);
  
  if (!config.validateParams()) {
    std::cout << "Warning: Some parameters are out of recommended ranges." << std::endl;
  }
  
  if (config.params.verbose) {
    config.printConfig();
  }

  // 解析命令行参数（保持向后兼容）
  int devNum = config.params.cuda_device;
  int imgSet = config.params.image_set;
  if (argc > 1)
    devNum = std::atoi(argv[1]);
  if (argc > 2)
    imgSet = std::atoi(argv[2]);

  // 获取图像路径
  std::string img1_path, img2_path;
  if (imgSet) {
    img1_path = config.params.alt_image1_path;
    img2_path = config.params.alt_image2_path;
  } else {
    img1_path = config.params.image1_path;
    img2_path = config.params.image2_path;
  }

  // Read images using OpenCV
  cv::Mat limg, rimg;
  std::cout << "Loading images: " << img1_path << " and " << img2_path << std::endl;
  cv::imread(img1_path, 0).convertTo(limg, CV_32FC1);
  cv::imread(img2_path, 0).convertTo(rimg, CV_32FC1);
  
  if (limg.empty() || rimg.empty()) {
    std::cerr << "Error: Could not load images!" << std::endl;
    std::cerr << "Image 1: " << img1_path << (limg.empty() ? " (FAILED)" : " (OK)") << std::endl;
    std::cerr << "Image 2: " << img2_path << (rimg.empty() ? " (FAILED)" : " (OK)") << std::endl;
    return -1;
  }
  
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;
  
  // Initial Cuda images and download images to device
  std::cout << "Initializing CUDA device " << devNum << "..." << std::endl;
  InitCuda(devNum); 
  CudaImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  img1.Download();
  img2.Download(); 

  // Extract Sift features from images using config parameters
  SiftData siftData1, siftData2;
  InitSiftData(siftData1, config.params.max_features, true, true); 
  InitSiftData(siftData2, config.params.max_features, true, true);
  
  std::cout << "\n=== SIFT Configuration Parameters ===" << std::endl;
  std::cout << "Initial Blur: " << config.params.initial_blur << std::endl;
  std::cout << "DoG Threshold: " << config.params.dog_threshold << std::endl;
  std::cout << "Number of Octaves: " << config.params.num_octaves << std::endl;
  std::cout << "Lowest Scale: " << config.params.lowest_scale << std::endl;
  std::cout << "Edge Limit: " << config.params.edge_limit << std::endl;
  std::cout << "Scale Up: " << (config.params.scale_up ? "Yes" : "No") << std::endl;
  std::cout << "Max Features: " << config.params.max_features << std::endl;
  std::cout << "Min Score: " << config.params.min_score << std::endl;
  std::cout << "Max Ambiguity: " << config.params.max_ambiguity << std::endl;
  std::cout << "RANSAC Iterations: " << config.params.ransac_iterations << std::endl;
  std::cout << "Inlier Threshold: " << config.params.inlier_threshold << std::endl;
  std::cout << "=================================" << std::endl;
  
  // Allocate temporary memory for SIFT processing
  float *memoryTmp = AllocSiftTempMemory(w, h, config.params.num_octaves, config.params.scale_up);
  
  // A bit of benchmarking with configurable parameters
  std::cout << "Running SIFT extraction..." << std::endl;
  for (int i = 0; i < 1000; i++) {
    ExtractSift(siftData1, img1, 
                config.params.num_octaves,
                config.params.initial_blur, 
                config.params.dog_threshold, 
                config.params.lowest_scale, 
                config.params.scale_up, 
                memoryTmp);
    ExtractSift(siftData2, img2, 
                config.params.num_octaves,
                config.params.initial_blur, 
                config.params.dog_threshold, 
                config.params.lowest_scale, 
                config.params.scale_up, 
                memoryTmp);
  }
  FreeSiftTempMemory(memoryTmp);
    
  // Match Sift features and find a homography using config parameters
  std::cout << "Matching SIFT features..." << std::endl;
  for (int i = 0; i < 1; i++)
    MatchSiftData(siftData1, siftData2);
  
  float homography[9];
  int numMatches;
  std::cout << "Computing homography..." << std::endl;
  FindHomography(siftData1, homography, &numMatches, 
                 config.params.ransac_iterations, 
                 config.params.min_score, 
                 config.params.max_ambiguity, 
                 config.params.inlier_threshold);
  
  int numFit = ImproveHomography(siftData1, homography, 
                                config.params.optimization_iterations, 
                                config.params.min_score, 
                                config.params.max_ambiguity, 
                                config.params.optimization_threshold);
    
  std::cout << "\n=== SIFT Processing Results ===" << std::endl;
  std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
  std::cout << "Number of matching features: " << numFit << " " << numMatches << std::endl;
  float match_ratio = 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts);
  std::cout << "Match ratio: " << match_ratio << "%" << std::endl;
  std::cout << "Parameters used - Blur: " << config.params.initial_blur << 
               ", Threshold: " << config.params.dog_threshold << std::endl;
  std::cout << "==============================" << std::endl;
  
  // === 可视化调试功能 ===
  if (config.params.enable_visualization) {
    std::cout << "\n=== 开始可视化调试 ===" << std::endl;
    
    // 创建可视化器
    SiftVisualizer visualizer(config.params.save_visualization, config.params.show_visualization_window);
    
    // 设置可视化参数
    visualizer.setFeatureCircleRadius(config.params.feature_circle_radius);
    visualizer.setMatchLineThickness(config.params.match_line_thickness);
    visualizer.setShowOnlyGoodMatches(config.params.show_only_good_matches);
    visualizer.setMatchErrorThreshold(config.params.inlier_threshold);
    visualizer.setOverlayAlpha(config.params.overlay_alpha);
    
    // 转换CudaImage到OpenCV Mat格式
    cv::Mat cv_img1(img1.height, img1.width, CV_32FC1, img1.h_data);
    cv::Mat cv_img2(img2.height, img2.width, CV_32FC1, img2.h_data);
    
    // 转换为8位图像用于可视化
    cv::Mat display_img1, display_img2;
    cv_img1.convertTo(display_img1, CV_8UC1);
    cv_img2.convertTo(display_img2, CV_8UC1);
    
    // 生成核心可视化结果（仅4张关键图片）
    
    // 1. 可视化图像1的特征点
    std::cout << "生成图像1特征点可视化..." << std::endl;
    cv::Mat features_vis1 = visualizer.visualizeSiftFeatures(display_img1, siftData1, "img1");
    
    // 2. 可视化图像2的特征点
    std::cout << "生成图像2特征点可视化..." << std::endl;
    cv::Mat features_vis2 = visualizer.visualizeSiftFeatures(display_img2, siftData2, "img2");
    
    // 3. 可视化特征点匹配
    std::cout << "生成特征点匹配可视化..." << std::endl;
    cv::Mat matches_vis = visualizer.visualizeSiftMatches(display_img1, display_img2, siftData1, siftData2, "match");
    
    // 4. 可视化变换叠加 - 使用单应性矩阵
    std::cout << "生成变换叠加可视化..." << std::endl;
    cv::Mat transformed_overlay_vis = visualizer.visualizeTransformedOverlay(display_img1, display_img2, homography, config.params.overlay_alpha, "transformed");
    
    std::cout << "可视化调试完成！结果已保存到 " << config.params.visualization_output_path << " 目录" << std::endl;
    std::cout << "================================" << std::endl;
  } else {
    std::cout << "可视化调试已禁用" << std::endl;
  }
  
  // Print out and store summary data
  if (config.params.show_matches) {
    std::cout << "Generating match visualization..." << std::endl;
    PrintMatchData(siftData1, siftData2, img1);
    cv::imwrite("data/limg_pts.pgm", limg);
  }

  // Optional: save results to specified output paths
  if (config.params.save_intermediate) {
    std::cout << "Saving intermediate results to " << config.params.output_path << std::endl;
    // Additional saving logic can be added here
    config.saveToFile("output_config.txt");
    std::cout << "Current configuration saved to output_config.txt" << std::endl;
  }

  // Optional: run all matching analysis
  // if (config.params.verbose) {
  //   MatchAll(siftData1, siftData2, homography);
  // }
  
  // Free Sift data from device
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
  
  std::cout << "SIFT processing completed successfully." << std::endl;
  return 0;
}

void MatchAll(SiftData &siftData1, SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;
#if 1
  homography[0] = homography[4] = -1.0f;
  homography[1] = homography[3] = homography[6] = homography[7] = 0.0f;
  homography[2] = 1279.0f;
  homography[5] = 959.0f;
#endif
  for (int i=0;i<numPts1;i++) {
    float *data1 = sift1[i].data;
    std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << " " << sift1[i].xpos << " " << sift1[i].ypos << std::endl;
    bool found = false;
    for (int j=0;j<numPts2;j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k=0;k<128;k++) 
	sum += data1[k]*data2[k];    
      float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
      float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
      float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
      float err = dx*dx + dy*dy;
      if (err<100.0f) // 100.0
	found = true;
      if (err<100.0f || j==sift1[i].match) { // 100.0
	if (j==sift1[i].match && err<100.0f)
	  std::cout << " *";
	else if (j==sift1[i].match) 
	  std::cout << " -";
	else if (err<100.0f)
	  std::cout << " +";
	else
	  std::cout << "  ";
	std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << " " << sift2[j].xpos << " " << sift2[j].ypos << " " << (int)dx << " " << (int)dy << std::endl;
      }
    }
    std::cout << std::endl;
    if (found)
      numFound++;
  }
  std::cout << "Number of finds: " << numFound << " / " << numPts1 << std::endl;
  std::cout << homography[0] << " " << homography[1] << " " << homography[2] << std::endl;//%%%
  std::cout << homography[3] << " " << homography[4] << " " << homography[5] << std::endl;//%%%
  std::cout << homography[6] << " " << homography[7] << " " << homography[8] << std::endl;//%%%
}

void PrintMatchData(SiftData &siftData1, SiftData &siftData2, CudaImage &img)
{
  int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = siftData1.m_data;
  SiftPoint *sift2 = siftData2.m_data;
#else
  SiftPoint *sift1 = siftData1.h_data;
  SiftPoint *sift2 = siftData2.h_data;
#endif
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  std::cout << std::setprecision(3);
  for (int j=0;j<numPts;j++) { 
    int k = sift1[j].match;
    if (sift1[j].match_error<5) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
      if (false && sift1[j].xpos>550 && sift1[j].xpos<600) {
	std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
	std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
	std::cout << "scale=" << sift1[j].scale << "  ";
	std::cout << "error=" << (int)sift1[j].match_error << "  ";
	std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
	std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
      }
#endif
#if 1
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
	int x = (int)(sift1[j].xpos + dx*l/len);
	int y = (int)(sift1[j].ypos + dy*l/len);
	h_img[y*w+x] = 255.0f;
      }
#endif
    }
    int x = (int)(sift1[j].xpos+0.5);
    int y = (int)(sift1[j].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++) 
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
  }
  std::cout << std::setprecision(6);
}
