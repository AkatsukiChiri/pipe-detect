#!/usr/bin/env python3
"""
Step 3: Depth Analysis Module
Analyze depth information in the region around detected circles and ellipses
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


class DepthAnalyzer:
    def __init__(self):
        """Initialize the depth analyzer"""
        pass
    
    def analyze_shape_depth(self, depth_image, shape, radius_multiplier=2):
        """
        Analyze depth information in a square region around the detected shape
        For ellipses, uses major axis * radius_multiplier as the square side length
        
        Args:
            depth_image: Input depth image
            shape: Dictionary with shape information (circle or ellipse)
            radius_multiplier: Factor to determine analysis region size
            
        Returns:
            dict: Depth analysis results
        """
        center_x, center_y = shape['center']
        
        # Determine region size based on shape type
        if shape['type'] == 'ellipse':
            # For ellipses, use major axis * radius_multiplier as square side
            region_size = int((shape['major_axis'] / 2) * radius_multiplier)
        else:
            # For circles, use radius * radius_multiplier as before
            region_size = int(shape['radius'] * radius_multiplier)
        
        # Ensure region is within image bounds
        x_start = max(0, center_x - region_size)
        x_end = min(depth_image.shape[1], center_x + region_size)
        y_start = max(0, center_y - region_size)
        y_end = min(depth_image.shape[0], center_y + region_size)
        
        # Extract depth region
        depth_region = depth_image[y_start:y_end, x_start:x_end].copy()
        
        # Filter out invalid depth values (valid range: 100-5000)
        valid_mask = (depth_region >= 100) & (depth_region <= 5000)
        if np.sum(valid_mask) == 0:
            return {
                'shape': shape,
                'region_bounds': (x_start, y_start, x_end, y_end),
                'depth_region': depth_region,
                'valid_depths': np.array([]),
                'depth_stats': {},
                'gradient_analysis': {},
                'has_valid_data': False,
                'region_size': region_size
            }
        
        valid_depths = depth_region[valid_mask]
        
        # Calculate depth statistics
        depth_stats = {
            'mean': np.mean(valid_depths),
            'median': np.median(valid_depths),
            'std': np.std(valid_depths),
            'min': np.min(valid_depths),
            'max': np.max(valid_depths),
            'range': np.max(valid_depths) - np.min(valid_depths)
        }
        
        # Gradient analysis for pipe detection
        gradient_analysis = self._analyze_depth_gradients(depth_region, valid_mask, shape)
        
        result = {
            'shape': shape,
            'region_bounds': (x_start, y_start, x_end, y_end),
            'depth_region': depth_region,
            'valid_depths': valid_depths,
            'depth_stats': depth_stats,
            'gradient_analysis': gradient_analysis,
            'has_valid_data': True,
            'region_size': region_size
        }
        
        return result
    
    def _analyze_depth_gradients(self, depth_region, valid_mask, shape):
        """
        Analyze depth gradients to identify pipe-like structures
        Enhanced for elliptical shapes
        
        Args:
            depth_region: Depth data in the analysis region
            valid_mask: Mask of valid depth values
            shape: Original shape information (circle or ellipse)
            
        Returns:
            dict: Gradient analysis results
        """
        if np.sum(valid_mask) < 10:  # Need enough valid points
            return {'has_gradient_pattern': False}
        
        # Apply depth validity check (100-5000 range) to the valid_mask
        depth_validity_mask = (depth_region >= 100) & (depth_region <= 5000)
        valid_mask = valid_mask & depth_validity_mask
        
        # Create a cleaned depth region for gradient analysis
        cleaned_depth = depth_region.copy().astype(np.float32)
        cleaned_depth[~valid_mask] = np.nan
        
        # Fill NaN values using interpolation
        if np.sum(~np.isnan(cleaned_depth)) > 0:
            # Use median filter to fill invalid regions
            kernel = np.ones((5, 5), np.uint8)
            mask_dilated = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=2)
            
            # Simple inpainting for missing values
            for i in range(cleaned_depth.shape[0]):
                for j in range(cleaned_depth.shape[1]):
                    if np.isnan(cleaned_depth[i, j]) and mask_dilated[i, j]:
                        # Find nearest valid neighbors
                        valid_neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if (0 <= ni < cleaned_depth.shape[0] and 
                                    0 <= nj < cleaned_depth.shape[1] and
                                    not np.isnan(cleaned_depth[ni, nj])):
                                    valid_neighbors.append(cleaned_depth[ni, nj])
                        if valid_neighbors:
                            cleaned_depth[i, j] = np.median(valid_neighbors)
        
        # Calculate gradients
        grad_x = cv2.Sobel(cleaned_depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(cleaned_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Remove NaN values from gradient
        gradient_magnitude = np.nan_to_num(gradient_magnitude)
        
        # Analyze gradient patterns
        grad_stats = {
            'mean_gradient': np.mean(gradient_magnitude[valid_mask]),
            'max_gradient': np.max(gradient_magnitude[valid_mask]) if np.sum(valid_mask) > 0 else 0,
            'gradient_std': np.std(gradient_magnitude[valid_mask]) if np.sum(valid_mask) > 0 else 0
        }
        
        # Check for pipe patterns (adapted for ellipses)
        center = (cleaned_depth.shape[1] // 2, cleaned_depth.shape[0] // 2)
        has_gradient_pattern = self._detect_pipe_gradient_pattern(
            gradient_magnitude, center, valid_mask, shape
        )
        
        return {
            'gradient_magnitude': gradient_magnitude,
            'gradient_stats': grad_stats,
            'has_gradient_pattern': has_gradient_pattern,
            'cleaned_depth': cleaned_depth
        }
    
    def _detect_pipe_gradient_pattern(self, gradient_magnitude, center, valid_mask, shape):
        """
        Detect if the gradient pattern suggests a pipe structure
        Enhanced to handle elliptical shapes
        """
        if np.sum(valid_mask) < 20:
            return False
        
        # Check if gradients are higher near the edges (suggesting depth discontinuity)
        height, width = gradient_magnitude.shape
        cx, cy = center
        
        if shape['type'] == 'ellipse':
            # For ellipses, analyze along major and minor axes
            return self._analyze_elliptical_gradient_pattern(
                gradient_magnitude, center, valid_mask, shape
            )
        else:
            # For circles, use the original circular analysis
            return self._analyze_circular_gradient_pattern(
                gradient_magnitude, center, valid_mask
            )
    
    def _analyze_elliptical_gradient_pattern(self, gradient_magnitude, center, valid_mask, shape):
        """
        Analyze gradient patterns for elliptical shapes
        """
        height, width = gradient_magnitude.shape
        cx, cy = center
        
        # Get ellipse parameters in the analysis region coordinate system
        # Note: The shape angles and axes are in the original image coordinates
        # We need to adapt them to the cropped region
        
        # For simplicity, we'll analyze both along the major axis direction
        # and perpendicular to it
        
        # Create distance map from center
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Analyze concentric elliptical rings
        max_dist = min(width, height) // 2
        if max_dist < 5:
            return False
        
        ring_gradients = []
        for r in range(5, max_dist, 3):  # Smaller step for ellipses
            ring_mask = ((distances >= r-2) & (distances < r+2) & valid_mask)
            if np.sum(ring_mask) > 0:
                avg_gradient = np.mean(gradient_magnitude[ring_mask])
                ring_gradients.append(avg_gradient)
        
        if len(ring_gradients) < 2:
            return False
        
        # Check if outer rings have higher gradients (pipe edge effect)
        # For ellipses, this pattern might be less pronounced
        outer_gradient = np.mean(ring_gradients[-2:])
        inner_gradient = np.mean(ring_gradients[:2])
        
        # Use a lower threshold for ellipses as they might have less symmetric gradients
        return outer_gradient > inner_gradient * 1.1
    
    def _analyze_circular_gradient_pattern(self, gradient_magnitude, center, valid_mask):
        """
        Original circular gradient pattern analysis
        """
        height, width = gradient_magnitude.shape
        cx, cy = center
        
        # Create distance map from center
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Divide into rings and compare gradient levels
        max_dist = min(width, height) // 2
        if max_dist < 5:
            return False
        
        ring_gradients = []
        for r in range(5, max_dist, 5):
            ring_mask = ((distances >= r-2) & (distances < r+2) & valid_mask)
            if np.sum(ring_mask) > 0:
                avg_gradient = np.mean(gradient_magnitude[ring_mask])
                ring_gradients.append(avg_gradient)
        
        if len(ring_gradients) < 2:
            return False
        
        # Check if outer rings have higher gradients (pipe edge effect)
        outer_gradient = np.mean(ring_gradients[-2:])
        inner_gradient = np.mean(ring_gradients[:2])
        
        return outer_gradient > inner_gradient * 1.2
    
    def save_debug_visualization(self, depth_analysis_results, folder_name):
        """Save debug visualization of depth analysis"""
        if not depth_analysis_results:
            return
        
        n_shapes = len(depth_analysis_results)
        fig, axes = plt.subplots(n_shapes, 4, figsize=(16, 4 * n_shapes))
        
        if n_shapes == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(depth_analysis_results):
            if not result['has_valid_data']:
                continue
            
            shape = result['shape']
            depth_region = result['depth_region']
            gradient_analysis = result['gradient_analysis']
            
            # Shape identifier for title
            if shape['type'] == 'circle':
                shape_desc = f"Circle {i+1}"
            else:
                shape_desc = f"Ellipse {i+1}"
            
            # Original depth region
            axes[i, 0].imshow(depth_region, cmap='jet')
            axes[i, 0].set_title(f'{shape_desc}: Depth Region')
            axes[i, 0].axis('off')
            
            # Cleaned depth for analysis
            if 'cleaned_depth' in gradient_analysis:
                axes[i, 1].imshow(gradient_analysis['cleaned_depth'], cmap='jet')
                axes[i, 1].set_title(f'{shape_desc}: Cleaned Depth')
            else:
                axes[i, 1].imshow(depth_region, cmap='jet')
                axes[i, 1].set_title(f'{shape_desc}: Depth (No Cleaning)')
            axes[i, 1].axis('off')
            
            # Gradient magnitude
            if 'gradient_magnitude' in gradient_analysis:
                axes[i, 2].imshow(gradient_analysis['gradient_magnitude'], cmap='hot')
                axes[i, 2].set_title(f'{shape_desc}: Gradient Magnitude')
            else:
                axes[i, 2].text(0.5, 0.5, 'No Gradient Data', 
                               ha='center', va='center', transform=axes[i, 2].transAxes)
                axes[i, 2].set_title(f'{shape_desc}: No Gradient')
            axes[i, 2].axis('off')
            
            # Statistics and analysis
            stats_text = f"Shape: {shape['type']}\n"
            if shape['type'] == 'ellipse':
                stats_text += f"Major axis: {shape['major_axis']}\n"
                stats_text += f"Minor axis: {shape['minor_axis']}\n"
                stats_text += f"Angle: {shape['angle']:.1f}°\n"
                stats_text += f"Ratio: {shape.get('axis_ratio', 0):.2f}\n"
            else:
                stats_text += f"Radius: {shape['radius']}\n"
            
            stats_text += f"\nDepth Stats:\n"
            stats_text += f"Mean: {result['depth_stats']['mean']:.1f}\n"
            stats_text += f"Std: {result['depth_stats']['std']:.1f}\n"
            stats_text += f"Range: {result['depth_stats']['range']:.1f}\n"
            
            if 'gradient_stats' in gradient_analysis:
                stats_text += f"\nGradient Stats:\n"
                stats_text += f"Mean: {gradient_analysis['gradient_stats']['mean_gradient']:.2f}\n"
                stats_text += f"Max: {gradient_analysis['gradient_stats']['max_gradient']:.2f}\n"
                stats_text += f"Pattern: {gradient_analysis.get('has_gradient_pattern', False)}"
            
            axes[i, 3].text(0.05, 0.95, stats_text, transform=axes[i, 3].transAxes, 
                           verticalalignment='top', fontfamily='monospace', fontsize=8)
            axes[i, 3].set_title(f'{shape_desc}: Analysis')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        # Save result
        results_dir = Path("results") / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / "step3_depth_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug output saved: results/{folder_name}/step3_depth_analysis.png")

    # Keep backward compatibility
    def analyze_circle_depth(self, depth_image, circle, radius_multiplier=3):
        """Backward compatibility method"""
        return self.analyze_shape_depth(depth_image, circle, radius_multiplier)


def test_depth_analysis():
    """Test the depth analysis with sample data"""
    from step1_data_loading import DataLoader
    from step2_circle_detection import CircleEllipseDetector
    
    snapshot_base = Path("../snapshot")
    available_folders = [f.name for f in snapshot_base.iterdir() if f.is_dir()]
    
    if available_folders:
        test_folder = available_folders[0]
        print(f"Testing depth analysis with folder: {test_folder}")
        
        # Load data and detect shapes
        loader = DataLoader()
        ir_image, depth_image, params = loader.load_data(snapshot_base / test_folder)
        
        detector = CircleEllipseDetector()
        enhanced_ir, shapes = detector.detect_circles_and_ellipses(ir_image)
        
        if shapes:
            # Analyze depth for each shape
            analyzer = DepthAnalyzer()
            depth_analysis_results = []
            
            for shape in shapes:
                depth_info = analyzer.analyze_shape_depth(depth_image, shape, radius_multiplier=2)
                depth_analysis_results.append(depth_info)
                
                shape_desc = f"{shape['type'].capitalize()} at {shape['center']}"
                if shape['type'] == 'ellipse':
                    shape_desc += f" (major={shape['major_axis']}, minor={shape['minor_axis']}, angle={shape['angle']:.1f}°)"
                else:
                    shape_desc += f" (r={shape['radius']})"
                
                print(f"{shape_desc}:")
                if depth_info['has_valid_data']:
                    stats = depth_info['depth_stats']
                    print(f"  Depth: mean={stats['mean']:.1f}, std={stats['std']:.1f}, range={stats['range']:.1f}")
                    print(f"  Region size: {depth_info['region_size']} pixels")
                    grad_pattern = depth_info['gradient_analysis'].get('has_gradient_pattern', False)
                    print(f"  Gradient pattern detected: {grad_pattern}")
                else:
                    print(f"  No valid depth data")
            
            # Save debug visualization
            analyzer.save_debug_visualization(depth_analysis_results, test_folder)
            
            print("Depth analysis test completed successfully!")
        else:
            print("No shapes detected for depth analysis")
    else:
        print("No snapshot folders found for testing")


if __name__ == "__main__":
    test_depth_analysis() 