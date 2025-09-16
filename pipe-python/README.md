# Circular Plane Detection Pipeline

This project implements a complete pipeline for detecting circular and elliptical planar surfaces in RGB-D images captured by depth cameras. The pipeline processes RGB and depth images to identify circular/elliptical shapes, analyzes their depth characteristics to determine if they represent planar surfaces, calculates their 3D centers and normal vectors, and creates comprehensive visualizations.

## Features

- **Step 1**: Circle and ellipse detection in RGB images using OpenCV
- **Step 2**: Depth analysis to determine if detected shapes represent planar surfaces
- **Step 3**: 3D plane center and normal vector calculation
- **Step 4**: Final visualization with planes and normal vectors marked on original images

## Project Structure

```
pipe-python/
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── main_pipeline.py             # Main pipeline orchestrator
├── step1_circle_detection.py    # Circle/ellipse detection
├── step2_depth_analysis.py      # Depth analysis and plane fitting
├── step3_plane_calculation.py   # Plane geometry calculation
└── step4_final_visualization.py # Final visualization generation
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "import cv2, numpy, matplotlib, scipy, skimage, sklearn; print('All dependencies installed successfully!')"
   ```

## Input Data Format

The pipeline expects RGB-D image pairs with the following specifications:

- **RGB Image**: JPEG format, any resolution (tested with 1600x1200)
- **Depth Image**: 16-bit PNG format, typically lower resolution (tested with 640x480)
- **File Structure**: RGB and depth images should be in the same directory

Example data structure:
```
snapshot/
└── DS87_2025_09_15_16_40_50_0268/
    ├── Color_00000000.jpg    # RGB image
    ├── Depth_00000000.png    # Depth image
    └── params.json           # Camera parameters (optional)
```

## Usage

### Basic Usage

```python
from main_pipeline import CircularPlaneDetectionPipeline

# Initialize pipeline
pipeline = CircularPlaneDetectionPipeline(output_base_dir="results")

# Process a single RGB-D image pair
rgb_path = "path/to/Color_00000000.jpg"
depth_path = "path/to/Depth_00000000.png"

results = pipeline.run_pipeline(rgb_path, depth_path, session_name="my_analysis")
```

### Command Line Usage

```bash
cd pipe-python
python main_pipeline.py
```

This will process the example images in the `../snapshot/` directory.

### Batch Processing

```python
# Process multiple image pairs
image_pairs = [
    {"rgb_path": "path/to/image1_rgb.jpg", "depth_path": "path/to/image1_depth.png"},
    {"rgb_path": "path/to/image2_rgb.jpg", "depth_path": "path/to/image2_depth.png"},
]

batch_results = pipeline.run_batch_processing(image_pairs)
```

## Pipeline Steps

### Step 1: Circle and Ellipse Detection

- **Input**: RGB image
- **Methods**: 
  - Hough Circle Transform for circle detection
  - Contour detection and ellipse fitting for ellipse detection
- **Output**: 
  - Detected circles (center coordinates and radius)
  - Detected ellipses (center, axes, angle)
  - Visualization with detected shapes marked

### Step 2: Depth Analysis

- **Input**: RGB detection results + depth image
- **Methods**:
  - Coordinate scaling between RGB and depth images
  - Depth value extraction within detected shapes
  - RANSAC-based plane fitting
  - Planarity assessment using residual analysis
- **Output**:
  - Depth statistics for each detected shape
  - Plane fitting parameters
  - Planarity confidence scores

### Step 3: Plane Geometry Calculation

- **Input**: Depth analysis results
- **Methods**:
  - 3D center calculation using fitted plane equations
  - Normal vector refinement using depth gradients
  - Geometric property calculation (area, orientation angles)
- **Output**:
  - 3D coordinates of plane centers
  - Refined normal vectors
  - Orientation angles and geometric properties

### Step 4: Final Visualization

- **Input**: All previous results + original images
- **Methods**:
  - 3D to 2D projection for visualization
  - Comprehensive multi-panel visualizations
  - Statistical summaries
- **Output**:
  - Final annotated RGB image with detected planes and normal vectors
  - Comprehensive analysis report with 3D visualizations
  - Statistical tables and summaries

## Output Files

The pipeline generates organized outputs in the specified results directory:

```
results/
└── [session_name]/
    ├── step1_circle_detection/
    │   └── step1_detections.jpg        # RGB with detected shapes
    ├── step2_depth_analysis/
    │   └── step2_depth_analysis.png    # Depth analysis visualization
    ├── step3_plane_calculation/
    │   └── step3_plane_geometry.png    # 3D geometry visualization
    ├── step4_final_visualization/
    │   ├── step4_comprehensive_results.png  # Complete analysis report
    │   └── step4_final_result.jpg          # Final annotated image
    └── pipeline_results.json          # Complete results in JSON format
```

## Algorithm Parameters

### Detection Parameters
- **Circle detection**: Adjustable min/max radius (default: 20-200 pixels)
- **Ellipse detection**: Area and axis ratio filtering
- **Preprocessing**: Gaussian blur with configurable kernel size

### Plane Fitting Parameters
- **RANSAC tolerance**: Maximum distance from plane (default: 15 depth units)
- **Minimum inlier ratio**: Required for planarity classification (default: 0.7)
- **Minimum points**: Required for plane fitting (default: 10)

### Visualization Parameters
- **Normal vector length**: Adjustable for visualization clarity
- **Color coding**: Configurable colors for different elements
- **Text annotations**: Customizable labels and information display

## Camera Parameters (Optional)

If camera intrinsic parameters are available, they can be provided for more accurate 3D calculations:

```python
camera_params = {
    'focal_length': (fx, fy),      # Focal lengths in pixels
    'principal_point': (cx, cy),   # Principal point coordinates
}

results = pipeline.run_pipeline(rgb_path, depth_path, camera_params=camera_params)
```

## Troubleshooting

### Common Issues

1. **No shapes detected**:
   - Adjust circle detection parameters (min/max radius)
   - Check image preprocessing settings
   - Verify image quality and lighting conditions

2. **No planar surfaces found**:
   - Adjust plane fitting tolerance
   - Check depth image quality
   - Verify coordinate scaling between RGB and depth

3. **Visualization issues**:
   - Check output directory permissions
   - Verify matplotlib backend configuration
   - Ensure sufficient disk space

### Performance Optimization

- **Large images**: Consider resizing for faster processing
- **Batch processing**: Use multiprocessing for large datasets
- **Memory usage**: Monitor for large depth images

## Example Results

The pipeline can successfully detect:
- Circular objects like pipes, cylinders, or round surfaces
- Elliptical projections of circular objects viewed at angles
- Planar circular surfaces with accurate normal vector estimation

## Dependencies

- OpenCV >= 4.8.0 (computer vision operations)
- NumPy >= 1.21.0 (numerical computations)
- Matplotlib >= 3.5.0 (visualizations)
- SciPy >= 1.7.0 (scientific computing)
- scikit-image >= 0.19.0 (image processing)
- scikit-learn >= 1.0.0 (machine learning algorithms)
- Pillow >= 9.0.0 (image I/O)

## License

This project is developed for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Citation

If you use this code in your research, please cite appropriately. 