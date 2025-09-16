# 深度图像恢复工具

这个工具用于从深度相机拍摄的16位深度PNG图像中恢复真实的深度信息（毫米单位）。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 命令行使用

```bash
# 基础用法
python depth_recovery.py path/to/depth_image.png

# 指定参数文件
python depth_recovery.py path/to/depth_image.png --params path/to/params.json

# 指定输出目录
python depth_recovery.py path/to/depth_image.png --output ./output/

# 不显示可视化
python depth_recovery.py path/to/depth_image.png --no-viz
```

### 2. 作为Python模块使用

```python
from depth_recovery import DepthImageProcessor

# 创建处理器
processor = DepthImageProcessor("params.json")  # 可选参数文件

# 处理深度图像
depth_data = processor.process_depth_image("depth_image.png")

# 获取统计信息
stats = processor.get_depth_statistics(depth_data)
print(stats)
```

### 3. 批量处理

```python
import os
from depth_recovery import DepthImageProcessor

processor = DepthImageProcessor()

# 处理目录中的所有深度图像
depth_dir = "snapshot/"
for root, dirs, files in os.walk(depth_dir):
    for file in files:
        if file.startswith("Depth_") and file.endswith(".png"):
            depth_path = os.path.join(root, file)
            params_path = os.path.join(root, "params.json")
            
            # 使用对应的参数文件（如果存在）
            if os.path.exists(params_path):
                processor.load_params(params_path)
            
            try:
                depth_data = processor.process_depth_image(depth_path)
                print(f"处理完成: {depth_path}")
            except Exception as e:
                print(f"处理失败 {depth_path}: {e}")
```

## 输出文件

处理完成后，工具会在 `depth_analysis_output/` 目录下创建带时间戳的子目录，包含以下文件：

1. `*_depth.npy` - NumPy数组格式的深度数据
2. `*_depth.csv` - CSV格式的深度数据（x, y, depth_mm）
3. `*_depth_processed.png` - 处理后的16位PNG深度图像
4. `*_analysis.png` - 主要分析图表（6个子图）
5. `*_analysis_color_map.png` - 彩色深度图
6. `*_analysis_grayscale.png` - 灰度深度图  
7. `*_analysis_histogram.png` - 深度值分布直方图

### 交互式功能

在主分析图表中，将鼠标移动到彩色深度图上可以实时显示：
- 像素坐标 (x, y)
- 该像素的深度值（毫米）

## 特性

- **自动深度范围检测**：从参数文件中读取深度范围设置
- **多种输出格式**：支持NumPy、CSV、PNG格式输出
- **交互式可视化**：鼠标悬停显示像素深度值
- **多图像输出**：生成彩色映射、灰度图、直方图等
- **英文界面**：所有图表和输出使用英文标签
- **组织化输出**：自动创建时间戳目录存储结果
- **统计信息**：提供详细的深度数据统计
- **批量处理**：支持处理多个深度图像

## 参数文件格式

参数文件应为JSON格式，包含相机配置信息：

```json
{
    "Control": {
        "depthColorMapMin": 1,
        "depthColorMapMax": 3049
    }
}
```

## 直接运行

如果不提供命令行参数直接运行脚本，它会自动处理当前目录下的示例深度图像：

```bash
python depth_recovery.py
```

这会处理 `snapshot/` 目录中的所有深度图像文件。

### 快速测试

使用提供的测试脚本快速验证功能：

```bash
python test_depth_analysis.py
```

这会：
1. 自动找到示例深度图像
2. 应用相机参数
3. 生成所有分析图表
4. 展示交互式可视化功能 