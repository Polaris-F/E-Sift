# SIFT配置系统使用说明

本项目现在支持通过配置文件设置SIFT算法的各种参数，提供了更好的灵活性和可配置性。

## 编译

```bash
mkdir build
cd build
cmake ..
make
```

编译完成后会生成三个可执行文件：
- `cudasift`: 原始版本，使用硬编码参数
- `cudasift_configurable`: 使用YAML格式配置文件
- `cudasift_txt`: 使用TXT格式配置文件（推荐）

## 配置文件格式

### TXT格式 (推荐)
位置: `config/sift_config.txt`

格式简单，使用 `参数名 = 数值` 的形式：
```
# 注释以#开头
dog_threshold = 3.0
num_octaves = 5
scale_up = false
```

### YAML格式
位置: `config/sift_config.yaml`

使用标准YAML格式，支持分组：
```yaml
sift_extraction:
  dog_threshold: 3.0
  num_octaves: 5
  scale_up: false
```

## 运行程序

### 使用TXT配置（推荐）
```bash
# 使用默认配置文件 config/sift_config.txt
./cudasift_txt

# 指定设备和图像集
./cudasift_txt 0 0

# 指定自定义配置文件
./cudasift_txt 0 0 my_config.txt
```

### 使用YAML配置
```bash
# 使用默认配置文件 config/sift_config.yaml
./cudasift_configurable

# 指定自定义配置文件
./cudasift_configurable 0 0 my_config.yaml
```

## 主要参数说明

### 核心算法参数
- `initial_blur`: 初始模糊值 (0.5-2.0)，控制初始平滑程度
- `dog_threshold`: DoG响应阈值 (1.0-10.0)，控制特征点数量和质量
- `num_octaves`: 金字塔层数 (3-8)，影响检测尺度范围
- `edge_limit`: 边缘响应限制 (5.0-20.0)，过滤边缘响应
- `scale_up`: 是否启用尺度放大 (true/false)

### 匹配参数
- `min_score`: 最小匹配得分 (0.0-1.0)，控制匹配质量
- `max_ambiguity`: 最大歧义度 (0.0-1.0)，控制匹配唯一性

### 单应性估计参数
- `ransac_iterations`: RANSAC迭代次数 (1000-50000)
- `inlier_threshold`: 内点阈值 (1.0-10.0)
- `optimization_iterations`: 优化迭代次数 (1-20)

### 性能参数
- `max_features`: 最大特征点数量 (4096-65536)
- `cuda_device`: CUDA设备编号

### 输入输出配置
- `image_set`: 图像集选择 (0: jpg图像, 1: pgm图像)
- `image1_path`, `image2_path`: 输入图像路径
- `verbose`: 是否显示详细信息
- `show_matches`: 是否显示匹配结果

## 参数调优建议

### 提高特征点数量
- 降低 `dog_threshold` (例如从3.0降到2.0)
- 增加 `num_octaves` (例如从5增到6)
- 启用 `scale_up = true`

### 提高匹配质量
- 提高 `min_score` (例如从0.85提到0.90)
- 降低 `max_ambiguity` (例如从0.95降到0.90)
- 增加 `ransac_iterations`

### 提高处理速度
- 提高 `dog_threshold`
- 减少 `num_octaves`
- 减少 `max_features`
- 减少 `ransac_iterations`

### 处理低质量图像
- 提高 `dog_threshold` 到 4.5
- 增加 `initial_blur` 到 1.5
- 启用 `scale_up = true`

## 示例配置

### 高质量匹配配置
```
dog_threshold = 2.5
num_octaves = 6
min_score = 0.90
max_ambiguity = 0.85
ransac_iterations = 15000
scale_up = true
```

### 快速处理配置
```
dog_threshold = 4.0
num_octaves = 4
max_features = 16384
ransac_iterations = 5000
scale_up = false
```

### 低质量图像配置
```
dog_threshold = 4.5
initial_blur = 1.5
edge_limit = 15.0
min_score = 0.80
scale_up = true
```

## 故障排除

1. **特征点过少**: 降低 `dog_threshold`，增加 `num_octaves`，启用 `scale_up`
2. **匹配效果差**: 调整 `min_score` 和 `max_ambiguity`，增加 `ransac_iterations`
3. **处理速度慢**: 提高 `dog_threshold`，减少 `num_octaves` 和 `max_features`
4. **配置文件错误**: 检查参数名拼写，确保数值在有效范围内

## 注意事项

1. 某些高级参数（如 `num_scales`）需要重新编译才能生效
2. 参数验证会在运行时检查，超出范围的参数会显示警告
3. 建议先使用默认配置，然后根据具体需求逐步调整
4. TXT格式配置文件更简单易用，推荐使用
