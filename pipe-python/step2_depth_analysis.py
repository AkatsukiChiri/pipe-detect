"""
Step 2: Depth Analysis and Plane Detection
This module analyzes depth data within detected circular/elliptical regions to determine if they represent planar surfaces.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os
from scipy import stats
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures


class DepthPlaneAnalyzer:
    def __init__(self, depth_scale: float = 1.0, plane_tolerance: float = 10.0, 
                 min_valid_pixels: int = 10, min_depth: float = 100.0, max_depth: float = 5000.0,
                 sigma_threshold: float = 2.0):
        """
        Initialize the depth plane analyzer.
        
        Args:
            depth_scale: Scale factor for depth values (if needed)
            plane_tolerance: Maximum deviation from plane (in depth units) to consider as planar
            min_valid_pixels: Minimum number of valid pixels required for plane analysis
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            sigma_threshold: Threshold for 3-sigma outlier removal (default: 3.0)
        """
        self.depth_scale = depth_scale
        self.plane_tolerance = plane_tolerance
        self.min_valid_pixels = min_valid_pixels
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.sigma_threshold = sigma_threshold
        
    def load_depth_image(self, depth_path: str) -> np.ndarray:
        """
        Load 16-bit depth image.
        
        Args:
            depth_path: Path to depth image
            
        Returns:
            Depth array as float32
        """
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            raise ValueError(f"Could not load depth image: {depth_path}")
        
        # Convert to float and apply scale
        depth_float = depth_img.astype(np.float32) * self.depth_scale
        
        # Filter out invalid depth values - only keep values in valid range
        depth_float[(depth_float < self.min_depth) | (depth_float > self.max_depth)] = np.nan
        
        return depth_float
    
    def scale_coordinates(self, ir_coords: Tuple[int, int], ir_shape: Tuple[int, int, int], 
                         depth_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Scale coordinates from IR image space to depth image space.
        
        Args:
            ir_coords: (x, y) coordinates in IR image
            ir_shape: Shape of IR image (H, W, C)
            depth_shape: Shape of depth image (H, W)
            
        Returns:
            Scaled coordinates for depth image
        """
        x, y = ir_coords
        ir_h, ir_w = ir_shape[:2]
        depth_h, depth_w = depth_shape
        
        # Scale coordinates
        scaled_x = int(x * depth_w / ir_w)
        scaled_y = int(y * depth_h / ir_h)
        
        # Ensure coordinates are within bounds
        scaled_x = max(0, min(scaled_x, depth_w - 1))
        scaled_y = max(0, min(scaled_y, depth_h - 1))
        
        return scaled_x, scaled_y
    
    def extract_circle_depth_region(self, depth_image: np.ndarray, center: Tuple[int, int], 
                                  radius: int, ring_width: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract depth values from the circular ring/circumference (for pipe detection).
        Only extracts points on the circle's edge, not the entire filled circle.
        
        Args:
            depth_image: Depth image array
            center: Center coordinates (x, y)
            radius: Circle radius
            ring_width: Width of the ring to extract (in pixels)
            
        Returns:
            Tuple of (depth_values, x_coords, y_coords) for valid pixels on the ring
        """
        x_center, y_center = center
        h, w = depth_image.shape
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center
        distance_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
        
        # Create ring mask - only points on the circumference
        inner_radius = max(0, radius - ring_width // 2)
        outer_radius = radius + ring_width // 2
        mask = (distance_from_center >= inner_radius) & (distance_from_center <= outer_radius)
        
        # Extract valid depth values
        depth_values = depth_image[mask]
        y_coords, x_coords = np.where(mask)
        
        # Remove NaN values
        valid_mask = ~np.isnan(depth_values)
        depth_values = depth_values[valid_mask]
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        print(f"Extracted {len(depth_values)} valid depth points from circle")
        return depth_values, x_coords, y_coords
    
    def extract_ellipse_depth_region(self, depth_image: np.ndarray, ellipse_params: Dict, ring_width: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract depth values from the elliptical ring/circumference (for pipe detection).
        Only extracts points on the ellipse's edge, not the entire filled ellipse.
        
        Args:
            depth_image: Depth image array
            ellipse_params: Ellipse parameters from detection
            ring_width: Width of the ring to extract (in pixels)
            
        Returns:
            Tuple of (depth_values, x_coords, y_coords) for valid pixels on the ellipse ring
        """
        center = ellipse_params['center']
        axes = ellipse_params['axes']
        angle = ellipse_params['angle']
        
        h, w = depth_image.shape
        
        # Create ellipse masks - outer and inner
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create outer ellipse
        ellipse_cv_outer = (tuple(map(int, center)), tuple(map(int, axes)), angle)
        cv2.ellipse(outer_mask, ellipse_cv_outer, 255, thickness=-1)
        
        # Create inner ellipse (smaller by ring_width)
        inner_axes = (max(1, axes[0] - ring_width), max(1, axes[1] - ring_width))
        ellipse_cv_inner = (tuple(map(int, center)), tuple(map(int, inner_axes)), angle)
        cv2.ellipse(inner_mask, ellipse_cv_inner, 255, thickness=-1)
        
        # Create ring mask by subtracting inner from outer
        ring_mask = outer_mask.astype(bool) & ~inner_mask.astype(bool)
        mask = ring_mask
        
        # Extract valid depth values
        depth_values = depth_image[mask]
        y_coords, x_coords = np.where(mask)
        
        # Remove NaN values
        valid_mask = ~np.isnan(depth_values)
        depth_values = depth_values[valid_mask]
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        return depth_values, x_coords, y_coords
    
    def fit_plane_ransac(self, depth_values: np.ndarray, x_coords: np.ndarray, 
                        y_coords: np.ndarray) -> Optional[Dict]:
        """
        Fit a plane to 3D points using RANSAC.
        
        Args:
            depth_values: Z coordinates (depth values)
            x_coords: X coordinates
            y_coords: Y coordinates
            
        Returns:
            Dictionary with plane parameters and statistics, or None if fitting fails
        """
        # Filter out NaN values before plane fitting
        valid_mask = ~np.isnan(depth_values)
        depth_values_clean = depth_values[valid_mask]
        x_coords_clean = x_coords[valid_mask]
        y_coords_clean = y_coords[valid_mask]

        print(depth_values_clean)
        
        if len(depth_values_clean) < self.min_valid_pixels:  # Need minimum valid points for plane fitting
            return None
        
        # Apply 3-sigma rule to remove outliers for improved robustness
        depth_mean = np.mean(depth_values_clean)
        depth_std = np.std(depth_values_clean)
        
        # Only apply 3-sigma filtering if we have enough variation (std > 0)
        if depth_std > 0:
            # Define outliers as points beyond sigma_threshold standard deviations
            lower_bound = depth_mean - self.sigma_threshold * depth_std
            upper_bound = depth_mean + self.sigma_threshold * depth_std
            
            # Create mask for inlier points (within 3-sigma range)
            sigma_mask = (depth_values_clean >= lower_bound) & (depth_values_clean <= upper_bound)
            
            # Apply the mask
            depth_values_final = depth_values_clean[sigma_mask]
            x_coords_final = x_coords_clean[sigma_mask]
            y_coords_final = y_coords_clean[sigma_mask]
            
            outliers_removed = len(depth_values_clean) - len(depth_values_final)
            print(f"  {self.sigma_threshold}-sigma filtering: removed {outliers_removed} outliers from {len(depth_values_clean)} points")
            print(f"  Depth statistics: mean={depth_mean:.1f}, std={depth_std:.1f}, bounds=[{lower_bound:.1f}, {upper_bound:.1f}]")
        else:
            # If std is 0 (all values are the same), no outliers to remove
            depth_values_final = depth_values_clean
            x_coords_final = x_coords_clean
            y_coords_final = y_coords_clean
            print(f"  {self.sigma_threshold}-sigma filtering: skipped (std=0), all {len(depth_values_clean)} points retained")
        
        # Check if we still have enough points after outlier removal
        if len(depth_values_final) < self.min_valid_pixels:
            print(f"  Insufficient points after {self.sigma_threshold}-sigma filtering: {len(depth_values_final)} < {self.min_valid_pixels}")
            return None
        
        # Prepare data for plane fitting: Z = aX + bY + c
        X = np.column_stack([x_coords_final, y_coords_final])
        y = depth_values_final
        
        print(depth_values_final)

        try:
            # Fit plane using RANSAC
            ransac = RANSACRegressor(
                residual_threshold=self.plane_tolerance,
                min_samples=3,
                max_trials=1000,
                random_state=42
            )
            ransac.fit(X, y)
            
            # Get plane parameters
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            
            # Calculate residuals for final filtered points only
            z_predicted = a * x_coords_final + b * y_coords_final + c
            residuals = np.abs(depth_values_final - z_predicted)
            
            # Calculate statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            inlier_ratio = np.sum(ransac.inlier_mask_) / len(depth_values_final)
            
            # Plane normal vector (normalized)
            normal_vector = np.array([-a, -b, 1])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            
            plane_info = {
                'coefficients': (a, b, c),  # Z = aX + bY + c
                'normal_vector': normal_vector,
                'mean_residual': mean_residual,
                'std_residual': std_residual,
                'inlier_ratio': inlier_ratio,
                'inlier_mask': ransac.inlier_mask_,
                'is_planar': mean_residual < self.plane_tolerance and inlier_ratio > 0.7
            }
            
            return plane_info
            
        except Exception as e:
            print(f"Error fitting plane: {e}")
            return None
    
    def analyze_depth_statistics(self, depth_values: np.ndarray) -> Dict:
        """
        Calculate statistical properties of depth values.
        
        Args:
            depth_values: Array of depth values
            
        Returns:
            Dictionary with depth statistics
        """
        if len(depth_values) == 0:
            return {'valid': False}
        
        stats_dict = {
            'valid': True,
            'count': len(depth_values),
            'mean': np.mean(depth_values),
            'std': np.std(depth_values),
            'min': np.min(depth_values),
            'max': np.max(depth_values),
            'range': np.max(depth_values) - np.min(depth_values),
            'median': np.median(depth_values)
        }
        
        return stats_dict
    
    def visualize_depth_analysis(self, depth_image: np.ndarray, analysis_results: List[Dict], 
                               output_path: str, ir_detection_data: Dict = None) -> None:
        """
        Create visualization of depth analysis results.
        
        Args:
            depth_image: Original depth image
            analysis_results: List of analysis results for each detected shape
            output_path: Path to save visualization
            ir_detection_data: Original IR detection data for comparison (optional)
        """
        fig, ax_array = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original depth image with coordinate comparison
        ax_array[0, 0].imshow(depth_image, cmap='viridis')
        
        # If IR detection data is provided, show scaled vs original coordinates
        if ir_detection_data is not None:
            ir_shape = ir_detection_data['image_shape']
            depth_shape = depth_image.shape
            
            # Draw original IR circles scaled to depth image size for comparison
            for i, (x, y, r) in enumerate(ir_detection_data['circles']):
                # Manual scaling for comparison
                scaled_x = int(x * depth_shape[1] / ir_shape[1])
                scaled_y = int(y * depth_shape[0] / ir_shape[0])
                scaled_r = int(r * depth_shape[1] / ir_shape[1])
                
                # Draw circle in red for comparison
                circle = plt.Circle((scaled_x, scaled_y), scaled_r, fill=False, color='red', linewidth=1, alpha=0.7)
                ax_array[0, 0].add_patch(circle)
                ax_array[0, 0].text(scaled_x, scaled_y, f'IR{i}', color='red', fontsize=8, ha='center', va='center')
        
        ax_array[0, 0].set_title('Original Depth Image (with IR circles in red)')
        ax_array[0, 0].axis('off')
        
        # Depth image with detected regions
        depth_viz = depth_image.copy()
        valid_planes = []
        
        for i, result in enumerate(analysis_results):
            if result['plane_info'] and result['plane_info']['is_planar']:
                valid_planes.append(result)
                
                # Draw region boundary
                if result['shape_type'] == 'circle':
                    center = result['scaled_center']
                    radius = result['scaled_radius']
                    # Use a fixed color value instead of depth max
                    max_val = np.nanmax(depth_image)
                    color_val = float(max_val) if not np.isnan(max_val) else 255.0
                    cv2.circle(depth_viz, center, radius, color_val, 2)
                    
                    # Also draw on matplotlib plot for better visualization
                    circle_plot = plt.Circle(center, radius, fill=False, color='yellow', linewidth=2, alpha=0.9)
                    ax_array[0, 1].add_patch(circle_plot)
                    ax_array[0, 1].text(center[0], center[1], f'D{result["shape_index"]}', 
                                      color='yellow', fontsize=8, ha='center', va='center', weight='bold')
                elif result['shape_type'] == 'ellipse':
                    ellipse_params = result['ellipse_params']
                    # Fix ellipse parameters format and color
                    center, ellipse_axes, angle = ellipse_params
                    ellipse_cv = (tuple(map(int, center)), tuple(map(int, ellipse_axes)), angle)
                    max_val = np.nanmax(depth_image)
                    color_val = float(max_val) if not np.isnan(max_val) else 255.0
                    cv2.ellipse(depth_viz, ellipse_cv, color_val, 2)
        
        ax_array[0, 1].imshow(depth_viz, cmap='viridis')
        ax_array[0, 1].set_title(f'Detected Planar Regions ({len(valid_planes)} found) - Yellow: Depth Analysis')
        ax_array[0, 1].axis('off')
        
        # Depth statistics histogram
        if valid_planes:
            all_depths = []
            for result in valid_planes:
                all_depths.extend(result['depth_values'])
            
            ax_array[1, 0].hist(all_depths, bins=50, alpha=0.7, edgecolor='black')
            ax_array[1, 0].set_title('Depth Distribution in Planar Regions')
            ax_array[1, 0].set_xlabel('Depth Value')
            ax_array[1, 0].set_ylabel('Frequency')
        else:
            ax_array[1, 0].text(0.5, 0.5, 'No planar regions detected', 
                          ha='center', va='center', transform=ax_array[1, 0].transAxes)
            ax_array[1, 0].set_title('No Data Available')
        
        # Plane fitting quality metrics
        if valid_planes:
            residuals = [result['plane_info']['mean_residual'] for result in valid_planes]
            inlier_ratios = [result['plane_info']['inlier_ratio'] for result in valid_planes]
            
            x_pos = np.arange(len(valid_planes))
            ax_array[1, 1].bar(x_pos - 0.2, residuals, 0.4, label='Mean Residual', alpha=0.7)
            ax_array[1, 1].bar(x_pos + 0.2, [r * max(residuals) for r in inlier_ratios], 
                          0.4, label='Inlier Ratio (scaled)', alpha=0.7)
            ax_array[1, 1].set_title('Plane Fitting Quality')
            ax_array[1, 1].set_xlabel('Detected Plane Index')
            ax_array[1, 1].set_ylabel('Quality Metric')
            ax_array[1, 1].legend()
            ax_array[1, 1].set_xticks(x_pos)
        else:
            ax_array[1, 1].text(0.5, 0.5, 'No planar regions detected', 
                          ha='center', va='center', transform=ax_array[1, 1].transAxes)
            ax_array[1, 1].set_title('No Data Available')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def analyze_shapes_depth(ir_detection_data: Dict, depth_image_path: str, 
                        output_dir: str, filename_prefix: str = "", ring_width: int = 3) -> List[Dict]:
    """
    Analyze depth data for detected circular and elliptical shapes.
    For pipe detection, only analyzes the ring/circumference points, not the entire filled area.
    
    Args:
        ir_detection_data: Detection results from step 1 (IR image)
        depth_image_path: Path to corresponding depth image
        output_dir: Directory to save results
        filename_prefix: Prefix for output filenames
        ring_width: Width of the ring to extract for circular/elliptical analysis (in pixels)
        
    Returns:
        List of analysis results for each detected shape
    """
    # Initialize analyzer
    analyzer = DepthPlaneAnalyzer(depth_scale=1.0, plane_tolerance=15.0)
    
    # Load depth image
    depth_image = analyzer.load_depth_image(depth_image_path)
    
    # Get image shapes
    ir_shape = ir_detection_data['image_shape']
    depth_shape = depth_image.shape
    
    print(f"IR image shape: {ir_shape}")
    print(f"Depth image shape: {depth_shape}")
    
    analysis_results = []
    
    # Analyze circles
    for i, (x, y, r) in enumerate(ir_detection_data['circles']):
        # Scale coordinates to depth image space
        scaled_x, scaled_y = analyzer.scale_coordinates((x, y), ir_shape, depth_shape)
        scaled_r = int(r * depth_shape[1] / ir_shape[1])  # Scale radius
        
        print(f"Analyzing circle {i}: IR({x},{y},r={r}) -> Depth({scaled_x},{scaled_y},r={scaled_r})")
        print(f"  IR shape: {ir_shape}, Depth shape: {depth_shape}")
        print(f"  Scale factors: x={depth_shape[1]/ir_shape[1]:.3f}, y={depth_shape[0]/ir_shape[0]:.3f}")
        
        # Extract depth region (only ring/circumference for pipe detection)
        depth_values, x_coords, y_coords = analyzer.extract_circle_depth_region(
            depth_image, (scaled_x, scaled_y), scaled_r, ring_width)
        
        if len(depth_values) >= analyzer.min_valid_pixels:
            # Analyze depth statistics
            depth_stats = analyzer.analyze_depth_statistics(depth_values)
            
            # Fit plane
            plane_info = analyzer.fit_plane_ransac(depth_values, x_coords, y_coords)
            
            print(f"  -> {len(depth_values)} valid depth points found")
            print(f"  -> Coordinates: IR({x},{y},r={r}) -> Depth({scaled_x},{scaled_y},r={scaled_r})")
            
            result = {
                'shape_type': 'circle',
                'shape_index': i,
                'original_center': (x, y),
                'original_radius': r,
                'scaled_center': (scaled_x, scaled_y),
                'scaled_radius': scaled_r,
                'depth_values': depth_values,
                'depth_stats': depth_stats,
                'plane_info': plane_info
            }
            
            analysis_results.append(result)
            
            if plane_info and plane_info['is_planar']:
                print(f"  -> Planar surface detected! Residual: {plane_info['mean_residual']:.2f}, "
                      f"Inlier ratio: {plane_info['inlier_ratio']:.2f}")
            else:
                print(f"  -> Not a planar surface")
        else:
            valid_count = len(depth_values) if len(depth_values) > 0 else 0
            print(f"  -> Insufficient valid depth data: {valid_count} points (need {analyzer.min_valid_pixels})")
    
    # Analyze ellipses
    for i, ellipse_data in enumerate(ir_detection_data['ellipses']):
        center = ellipse_data['center']
        axes = ellipse_data['axes']
        angle = ellipse_data['angle']
        
        # Scale parameters to depth image space
        scaled_center = analyzer.scale_coordinates(center, ir_shape, depth_shape)
        scale_x = depth_shape[1] / ir_shape[1]
        scale_y = depth_shape[0] / ir_shape[0]
        scaled_axes = (axes[0] * scale_x, axes[1] * scale_y)
        
        scaled_ellipse_params = (scaled_center, scaled_axes, angle)
        
        print(f"Analyzing ellipse {i}: IR{center} -> Depth{scaled_center}")
        
        # Extract depth region
        scaled_ellipse_dict = {
            'center': scaled_center,
            'axes': scaled_axes,
            'angle': angle
        }
        
        depth_values, x_coords, y_coords = analyzer.extract_ellipse_depth_region(
            depth_image, scaled_ellipse_dict, ring_width)
        
        if len(depth_values) >= analyzer.min_valid_pixels:
            # Analyze depth statistics
            depth_stats = analyzer.analyze_depth_statistics(depth_values)
            
            # Fit plane
            plane_info = analyzer.fit_plane_ransac(depth_values, x_coords, y_coords)
            
            print(f"  -> {len(depth_values)} valid depth points found")
            
            result = {
                'shape_type': 'ellipse',
                'shape_index': i,
                'original_center': center,
                'original_axes': axes,
                'original_angle': angle,
                'scaled_center': scaled_center,
                'scaled_axes': scaled_axes,
                'ellipse_params': scaled_ellipse_params,
                'depth_values': depth_values,
                'depth_stats': depth_stats,
                'plane_info': plane_info
            }
            
            analysis_results.append(result)
            
            if plane_info and plane_info['is_planar']:
                print(f"  -> Planar surface detected! Residual: {plane_info['mean_residual']:.2f}, "
                      f"Inlier ratio: {plane_info['inlier_ratio']:.2f}")
            else:
                print(f"  -> Not a planar surface")
        else:
            valid_count = len(depth_values) if len(depth_values) > 0 else 0
            print(f"  -> Insufficient valid depth data: {valid_count} points (need {analyzer.min_valid_pixels})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize results
    viz_path = os.path.join(output_dir, f"{filename_prefix}step2_depth_analysis.png")
    analyzer.visualize_depth_analysis(depth_image, analysis_results, viz_path, ir_detection_data)
    
    return analysis_results


# Backward compatibility alias
def analyze_shapes_depth_rgb(rgb_detection_data: Dict, depth_image_path: str, 
                            output_dir: str, filename_prefix: str = "", ring_width: int = 3) -> List[Dict]:
    """
    Backward compatibility function - now processes IR detection data.
    """
    return analyze_shapes_depth(rgb_detection_data, depth_image_path, output_dir, filename_prefix, ring_width)


if __name__ == "__main__":
    # This would be called after step 1
    print("Step 2: Depth Analysis module")
    print("This module should be called after step 1 circle detection.") 