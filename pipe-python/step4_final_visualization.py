"""
Step 4: Final Visualization
This module creates the final visualization showing detected circular/elliptical planes
with their centers and normal vectors marked on the original RGB image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


class FinalVisualizer:
    def __init__(self):
        """Initialize the final visualizer."""
        pass
    
    def project_3d_to_image(self, point_3d: np.ndarray, normal_3d: np.ndarray, 
                           depth_shape: Tuple[int, int], rgb_shape: Tuple[int, int, int],
                           normal_length: float = 50.0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Project 3D center and normal vector back to image coordinates.
        
        Args:
            point_3d: 3D center point
            normal_3d: 3D normal vector
            depth_shape: Shape of depth image
            rgb_shape: Shape of RGB image
            normal_length: Length of normal vector in image pixels
            
        Returns:
            Tuple of (center_2d, normal_end_2d) in RGB image coordinates
        """
        # Convert 3D point back to depth image coordinates
        # This is a simplified back-projection (reverse of step 3)
        x_3d, y_3d, z_3d = point_3d
        
        # Project to depth image coordinates
        depth_x = int(x_3d + depth_shape[1] / 2)
        depth_y = int(y_3d + depth_shape[0] / 2)
        
        # Scale to RGB image coordinates
        rgb_x = int(depth_x * rgb_shape[1] / depth_shape[1])
        rgb_y = int(depth_y * rgb_shape[0] / depth_shape[0])
        
        # Ensure within bounds
        rgb_x = max(0, min(rgb_x, rgb_shape[1] - 1))
        rgb_y = max(0, min(rgb_y, rgb_shape[0] - 1))
        
        center_2d = (rgb_x, rgb_y)
        
        # Calculate normal vector end point in image coordinates
        # Project normal vector to image plane
        normal_2d_x = normal_3d[0] * normal_length
        normal_2d_y = normal_3d[1] * normal_length
        
        normal_end_x = int(rgb_x + normal_2d_x)
        normal_end_y = int(rgb_y + normal_2d_y)
        
        # Ensure within bounds
        normal_end_x = max(0, min(normal_end_x, rgb_shape[1] - 1))
        normal_end_y = max(0, min(normal_end_y, rgb_shape[0] - 1))
        
        normal_end_2d = (normal_end_x, normal_end_y)
        
        return center_2d, normal_end_2d
    
    def draw_plane_with_normal(self, image: np.ndarray, plane_props: Dict, 
                             depth_shape: Tuple[int, int], color_center: Tuple[int, int, int] = (0, 255, 0),
                             color_normal: Tuple[int, int, int] = (255, 0, 0),
                             color_boundary: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw a detected plane with its normal vector on the image.
        
        Args:
            image: RGB image array
            plane_props: Plane properties from step 3
            depth_shape: Shape of depth image
            color_center: Color for center point (B, G, R)
            color_normal: Color for normal vector (B, G, R)
            color_boundary: Color for shape boundary (B, G, R)
            
        Returns:
            Image with plane and normal vector drawn
        """
        result_image = image.copy()
        rgb_shape = image.shape
        
        # Get original shape parameters (in RGB coordinates)
        original_center = plane_props['original_center']
        
        # Project 3D center and normal back to image
        center_3d = plane_props['center_3d']
        normal_3d = plane_props['normal_vector']
        
        center_2d, normal_end_2d = self.project_3d_to_image(
            center_3d, normal_3d, depth_shape, rgb_shape, normal_length=80)
        
        # Draw original shape boundary
        if plane_props['shape_type'] == 'circle':
            # Get radius from original detection data stored in depth_stats or calculate from area
            if 'area' in plane_props:
                # Estimate radius from area: area = pi * r^2
                radius = int(np.sqrt(plane_props['area'] / np.pi))
            else:
                radius = 50  # Default radius fallback
            cv2.circle(result_image, tuple(map(int, original_center)), radius, color_boundary, 3)
        else:  # ellipse
            # Draw ellipse boundary using stored parameters
            if 'original_axes' in plane_props and 'original_angle' in plane_props:
                original_axes = plane_props['original_axes']
                original_angle = plane_props['original_angle']
                ellipse_params = (tuple(map(int, original_center)), 
                                tuple(map(int, original_axes)), original_angle)
                cv2.ellipse(result_image, ellipse_params, color_boundary, 3)
        
        # Draw center point
        cv2.circle(result_image, center_2d, 8, color_center, -1)
        cv2.circle(result_image, center_2d, 10, (0, 0, 0), 2)  # Black border
        
        # Draw normal vector
        cv2.arrowedLine(result_image, center_2d, normal_end_2d, color_normal, 3, tipLength=0.1)
        
        # Add text labels
        label_text = f"P{plane_props['shape_index']}"
        cv2.putText(result_image, label_text, (center_2d[0] + 15, center_2d[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, label_text, (center_2d[0] + 15, center_2d[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add normal vector info
        normal_text = f"N({normal_3d[0]:.2f},{normal_3d[1]:.2f},{normal_3d[2]:.2f})"
        cv2.putText(result_image, normal_text, (normal_end_2d[0] + 5, normal_end_2d[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(result_image, normal_text, (normal_end_2d[0] + 5, normal_end_2d[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_normal, 1)
        
        return result_image
    
    def create_comprehensive_visualization(self, rgb_image: np.ndarray, plane_properties_list: List[Dict],
                                         depth_image: np.ndarray, output_path: str) -> None:
        """
        Create a comprehensive visualization with multiple views.
        
        Args:
            rgb_image: Original RGB image
            plane_properties_list: List of detected plane properties
            depth_image: Depth image array
            output_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Main RGB image with annotations
        ax1 = fig.add_subplot(2, 3, (1, 2))
        
        result_image = rgb_image.copy()
        depth_shape = depth_image.shape
        
        # Draw all detected planes
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, plane_props in enumerate(plane_properties_list):
            if plane_props['valid']:
                color_idx = i % len(colors)
                color_center = colors[color_idx]
                color_normal = (255, 128, 0)  # Orange for all normal vectors
                color_boundary = (0, 255, 255)  # Cyan for boundaries
                
                result_image = self.draw_plane_with_normal(
                    result_image, plane_props, depth_shape, 
                    color_center, color_normal, color_boundary)
        
        ax1.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Detected Circular Planes with Normal Vectors ({len(plane_properties_list)} found)', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Original RGB image
        ax2 = fig.add_subplot(2, 3, 3)
        ax2.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        ax2.set_title('Original RGB Image')
        ax2.axis('off')
        
        # Depth image
        ax3 = fig.add_subplot(2, 3, 4)
        ax3.imshow(depth_image, cmap='viridis')
        ax3.set_title('Depth Image')
        ax3.axis('off')
        
        # 3D visualization
        ax4 = fig.add_subplot(2, 3, 5, projection='3d')
        
        if plane_properties_list:
            centers = np.array([p['center_3d'] for p in plane_properties_list])
            normals = np.array([p['normal_vector'] for p in plane_properties_list])
            
            # Plot centers
            ax4.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                       c='red', s=100, alpha=0.8, label='Plane Centers')
            
            # Plot normal vectors
            for i, (center, normal) in enumerate(zip(centers, normals)):
                normal_scaled = normal * 50
                ax4.quiver(center[0], center[1], center[2],
                          normal_scaled[0], normal_scaled[1], normal_scaled[2],
                          color='blue', alpha=0.7, arrow_length_ratio=0.1)
                ax4.text(center[0], center[1], center[2], f'P{i}',
                        fontsize=8, color='black')
            
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z (Depth)')
            ax4.set_title('3D Plane Geometry')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 0.5, 'No planes detected', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('No Data Available')
        
        # Statistics table
        ax5 = fig.add_subplot(2, 3, 6)
        ax5.axis('off')
        
        if plane_properties_list:
            # Create table data
            table_data = []
            for i, plane_props in enumerate(plane_properties_list):
                center = plane_props['center_3d']
                normal = plane_props['normal_vector']
                confidence = plane_props['planarity_confidence']
                area = plane_props['area']
                
                table_data.append([
                    f"P{i}",
                    f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})",
                    f"({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})",
                    f"{confidence:.3f}",
                    f"{area:.0f}"
                ])
            
            headers = ['Plane', 'Center (3D)', 'Normal Vector', 'Confidence', 'Area (px²)']
            
            table = ax5.table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center',
                             bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax5.set_title('Plane Properties Summary', fontsize=12, fontweight='bold', pad=20)
        else:
            ax5.text(0.5, 0.5, 'No planes detected', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title('No Data Available')
        
        plt.suptitle('Circular Plane Detection Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_simple_result_image(self, rgb_image: np.ndarray, plane_properties_list: List[Dict],
                                  depth_shape: Tuple[int, int], output_path: str) -> np.ndarray:
        """
        Create a simple result image showing only the detected planes on original RGB.
        
        Args:
            rgb_image: Original RGB image
            plane_properties_list: List of detected plane properties
            depth_shape: Shape of depth image
            output_path: Path to save the result image
            
        Returns:
            Result image with annotations
        """
        result_image = rgb_image.copy()
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, plane_props in enumerate(plane_properties_list):
            if plane_props['valid']:
                color_idx = i % len(colors)
                color_center = colors[color_idx]
                color_normal = (255, 128, 0)
                color_boundary = (0, 255, 255)
                
                result_image = self.draw_plane_with_normal(
                    result_image, plane_props, depth_shape, 
                    color_center, color_normal, color_boundary)
        
        # Save result
        cv2.imwrite(output_path, result_image)
        
        return result_image


def create_final_visualization(rgb_image_path: str, plane_properties_list: List[Dict],
                             depth_image_path: str, output_dir: str, 
                             filename_prefix: str = "") -> None:
    """
    Create final visualization showing detected planes and normal vectors.
    
    Args:
        rgb_image_path: Path to original RGB image
        plane_properties_list: List of plane properties from step 3
        depth_image_path: Path to depth image
        output_dir: Directory to save results
        filename_prefix: Prefix for output filenames
    """
    # Load images
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth_image[depth_image == 0] = np.nan
    depth_image[depth_image > 5000] = np.nan
    
    # Initialize visualizer
    visualizer = FinalVisualizer()
    
    print(f"Creating final visualization for {len(plane_properties_list)} detected planes...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive visualization
    comprehensive_path = os.path.join(output_dir, f"{filename_prefix}step4_comprehensive_results.png")
    visualizer.create_comprehensive_visualization(
        rgb_image, plane_properties_list, depth_image, comprehensive_path)
    
    # Create simple result image
    simple_path = os.path.join(output_dir, f"{filename_prefix}step4_final_result.jpg")
    result_image = visualizer.create_simple_result_image(
        rgb_image, plane_properties_list, depth_image.shape, simple_path)
    
    print(f"Final visualization saved to:")
    print(f"  Comprehensive: {comprehensive_path}")
    print(f"  Simple result: {simple_path}")
    
    return result_image


if __name__ == "__main__":
    # This would be called after step 3
    print("Step 4: Final Visualization module")
    print("This module should be called after step 3 plane geometry calculation.") 