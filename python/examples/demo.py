

import sys
import os
import cv2
import numpy as np
import time


sys.path.insert(0, "/home/jetson/lhf/workspace_2/E-Sift/build/python")

import cuda_sift

config = cuda_sift.SiftConfig("/home/jetson/lhf/workspace_2/E-Sift/config/test_config.txt")
sift_extractor = cuda_sift.SiftExtractor(config)
matcher = cuda_sift.SiftMatcher()


image1 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("/home/jetson/lhf/workspace_2/E-Sift/data/img2.jpg", cv2.IMREAD_GRAYSCALE)

for _ in range(100):
    start_time = time.time()
    features1 = sift_extractor.extract(image1)
    extract_time = (time.time() - start_time) * 1000

print(f"✓ image1 提取到 {features1['num_features']} 个特征点 ({extract_time:.2f}ms)")

for _ in range(100):
    start_time = time.time()
    features2 = sift_extractor.extract(image2)
    extract_time2 = (time.time() - start_time) * 1000

print(f"✓ image2 提取到 {features2['num_features']} 个特征点 ({extract_time2:.2f}ms)")

for _ in range(1):
    start_time = time.time()
    result = matcher.match_and_compute_homography(
        features1, features2,
        use_improve=False  # 速度优先
    )
    match_time = (time.time() - start_time) * 1000
np.set_printoptions(suppress=True, precision=3)
print(f"✓ match_and_compute_homography (use_improve=False) 得到 {result['num_inliers']} 个内点 ({match_time:.2f}ms) 单应性变换矩阵:\n{result['homography']}")