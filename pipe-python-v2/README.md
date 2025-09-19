# 钢管检测和定位系统 (Pipe Detection and Localization System)

基于红外图像和深度图像的钢管识别和定位系统，能够检测图像中的钢管截面（管口），定位其圆心位置并计算钢管方向。

## 系统概述

本系统分为5个步骤：

1. **数据加载** (`step1_data_loading.py`): 从snapshot文件夹加载红外图像(IR)和深度图像(Depth)
2. **圆形检测** (`step2_circle_detection.py`): 增强IR图像并检测圆形结构
3. **深度分析** (`step3_depth_analysis.py`): 分析圆形区域的深度信息
4. **管道检测** (`step4_pipe_detection.py`): 判断是否为管道结构并计算方向
5. **结果可视化** (`step5_visualization.py`): 生成最终的可视化结果

## 安装依赖

```bash
cd pipe-python-v2
pip install -r requirements.txt
```

## 使用方法

### 1. 运行完整管道

```bash
python main_pipeline.py
```

### 2. 单独测试各个步骤

```bash
# 测试数据加载
python step1_data_loading.py

# 测试圆形检测
python step2_circle_detection.py

# 测试深度分析
python step3_depth_analysis.py

# 测试管道检测
python step4_pipe_detection.py

# 测试可视化
python step5_visualization.py
```

## 输入数据格式

系统需要以下输入文件（在snapshot文件夹中）：

- `IR_xxxxxx.png`: 红外图像
- `Depth_xxxxxx.png`: 深度图像
- `params.json`: 参数文件（可选）

## 输出结果

系统会在 `results/[folder_name]/` 目录下生成以下文件：

- `step1_loaded_images.png`: 原始图像显示
- `step2_circle_detection.png`: 圆形检测过程和结果
- `step3_depth_analysis.png`: 深度分析结果
- `step5_final_visualization.png`: 完整的分析和结果可视化
- `final_result.png`: 最终检测结果图像

## 结果解释

### 颜色标记

- **红色圆圈**: 检测到圆形但未检测到管道结构
- **黄色圆圈**: 检测到圆形且确认为管道结构
- **蓝色箭头**: 管道轴线方向指示

### 输出信息

- 圆心位置坐标
- 检测置信度
- 管道3D方向信息（如果检测到管道）:
  - **方位角 (Azimuth)**: 管道在图像平面内的方向角度
  - **仰角 (Elevation)**: 管道相对于图像平面的角度
  - **深度方向**: 管道是指向相机内侧还是外侧
  - **图像平面角度**: 管道轴线与图像平面的夹角

## 参数调整

可以在各个模块中调整以下参数：

### 圆形检测参数 (`step2_circle_detection.py`)

```python
self.detection_params = {
    'dp': 1,                # 图像分辨率的倒数比例
    'min_dist': 50,         # 圆心之间的最小距离
    'param1': 50,           # Canny边缘检测的高阈值
    'param2': 30,           # 累加器阈值
    'min_radius': 20,       # 最小圆半径
    'max_radius': 200       # 最大圆半径
}
```

### 管道检测参数 (`step4_pipe_detection.py`)

```python
self.pipe_detection_params = {
    'min_depth_variation': 20,           # 最小深度变化
    'gradient_threshold': 5.0,           # 梯度阈值
    'depth_consistency_threshold': 0.3   # 深度一致性阈值
}
```

### 深度数据有效性

系统只使用100-5000范围内的深度值进行分析：
- 深度值 < 100: 视为无效（太近或噪声）
- 深度值 > 5000: 视为无效（太远或无效测量）
- 只有在有效范围内的深度值才参与管道检测和方向计算

### 3D方向计算原理

系统使用深度信息将2D图像坐标转换为3D点云，然后通过3D主成分分析计算管道的真实空间方向：

1. **3D点云构建**: 将图像坐标(x,y)与深度值(z)组合成3D点
2. **3D PCA分析**: 对3D点云进行主成分分析，第一主成分即为管道轴线方向
3. **球面坐标转换**: 将3D轴线向量转换为方位角和仰角
4. **方向解释**:
   - **仰角 > 0°**: 管道指向相机方向（into_image）
   - **仰角 < 0°**: 管道指向远离相机方向（out_of_image）
   - **图像平面角度**: 0°表示垂直于图像平面，90°表示平行于图像平面

**示例结果解释**:
```
Azimuth: -80.5° (图像平面内的方向)
Elevation: 86.2° (几乎垂直指向相机)
Depth direction: into_image (指向相机内侧)
Image plane angle: 3.8° (几乎垂直于图像平面)
3D axis: [0.011, -0.065, 0.998] (归一化的轴线向量)
```

## 技术特点

- **自适应图像增强**: 针对暗淡的红外图像进行CLAHE增强和伽马校正
- **多层次分析**: 结合圆形检测、深度分析和梯度分析
- **深度数据过滤**: 仅使用100-5000范围内的有效深度值进行分析
- **3D方向计算**: 使用深度信息和3D主成分分析计算管道在三维空间中的轴线方向
- **置信度评估**: 基于多个特征的综合置信度评分
- **完整可视化**: 详细的调试输出和最终结果展示

## 系统要求

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+
- Scikit-learn 1.3+
- SciPy 1.11+

## 故障排除

1. **未检测到圆形**: 调整圆形检测参数，特别是`param2`和半径范围
2. **管道检测不准确**: 调整深度变化阈值和梯度阈值
3. **方向计算错误**: 检查深度数据质量和梯度分析结果

## 开发说明

每个步骤都是独立的模块，便于调试和优化。所有中间结果都会保存为图像文件，方便分析和改进算法。 