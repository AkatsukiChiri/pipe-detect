"""
Step 3: Plane Center and Normal Vector Calculation
This module calculates the precise center and normal vector for detected planar circular/elliptical surfaces.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import os
from mpl_toolkits.mplot3d import Axes3D


class PlaneGeometryCalculator:
    def __init__(self, camera_params: Optional[Dict] = None):
        """
        Initialize the plane geometry calculator.
        
        Args:
            camera_params: Camera intrinsic parameters (if available)
        """
        self.camera_params = camera_params
        
    def calculate_plane_center_3d(self, analysis_result: Dict, depth_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Calculate the 3D center of a planar region.
        
        Args:
            analysis_result: Analysis result from step 2
            depth_image: Depth image array
            
        Returns:
            Tuple of (3D center coordinates, center info dictionary)
        """
        if not analysis_result['plane_info'] or not analysis_result['plane_info']['is_planar']:
            return None, {'valid': False, 'reason': 'Not a planar surface'}
        
        # Get plane parameters
        plane_info = analysis_result['plane_info']
        a, b, c = plane_info['coefficients']
        
        # Get the region center in image coordinates
        if analysis_result['shape_type'] == 'circle':
            center_x, center_y = analysis_result['scaled_center']
        else:  # ellipse
            center_x, center_y = analysis_result['scaled_center']
        
        # Calculate the depth at the center using the fitted plane equation
        center_depth = a * center_x + b * center_y + c
        
        # Convert to 3D coordinates (camera coordinate system)
        # Assuming standard camera model: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        if self.camera_params:
            fx, fy = self.camera_params.get('focal_length', (500, 500))
            cx, cy = self.camera_params.get('principal_point', (320, 240))
            
            x_3d = (center_x - cx) * center_depth / fx
            y_3d = (center_y - cy) * center_depth / fy
            z_3d = center_depth
        else:
            # Use simplified model if camera parameters not available
            x_3d = center_x - depth_image.shape[1] / 2
            y_3d = center_y - depth_image.shape[0] / 2
            z_3d = center_depth
        
        center_3d = np.array([x_3d, y_3d, z_3d])
        
        center_info = {
            'valid': True,
            'image_coordinates': (center_x, center_y),
            'depth': center_depth,
            'world_coordinates': center_3d,
            'fitted_center': True
        }
        
        return center_3d, center_info
    
    def refine_normal_vector(self, analysis_result: Dict, depth_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Refine the normal vector calculation using additional geometric constraints.
        
        Args:
            analysis_result: Analysis result from step 2
            depth_image: Depth image array
            
        Returns:
            Tuple of (refined normal vector, normal info dictionary)
        """
        if not analysis_result['plane_info'] or not analysis_result['plane_info']['is_planar']:
            return None, {'valid': False, 'reason': 'Not a planar surface'}
        
        plane_info = analysis_result['plane_info']
        
        # Get the normal vector from plane fitting (already normalized)
        normal_vector = plane_info['normal_vector']
        
        # For circular/elliptical planes, we can also estimate normal from depth gradient
        depth_values = analysis_result['depth_values']
        
        if analysis_result['shape_type'] == 'circle':
            center = analysis_result['scaled_center']
            radius = analysis_result['scaled_radius']
            
            # Extract depth region around the circle
            x_center, y_center = center
            h, w = depth_image.shape
            
            # Create coordinate grids
            y, x = np.ogrid[:h, :w]
            mask = (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2
            
            # Calculate gradient-based normal (alternative method)
            depth_region = depth_image.copy()
            depth_region[~mask] = np.nan
            
            # Calculate gradients
            grad_x = np.gradient(depth_region, axis=1)
            grad_y = np.gradient(depth_region, axis=0)
            
            # Average gradients in the circular region
            valid_mask = mask & ~np.isnan(depth_region)
            if np.sum(valid_mask) > 0:
                avg_grad_x = np.nanmean(grad_x[valid_mask])
                avg_grad_y = np.nanmean(grad_y[valid_mask])
                
                # Normal vector from gradient
                gradient_normal = np.array([-avg_grad_x, -avg_grad_y, 1])
                gradient_normal = gradient_normal / np.linalg.norm(gradient_normal)
                
                # Combine with fitted normal (weighted average)
                weight_fitted = plane_info['inlier_ratio']
                weight_gradient = 1 - weight_fitted
                
                combined_normal = (weight_fitted * normal_vector + 
                                 weight_gradient * gradient_normal)
                combined_normal = combined_normal / np.linalg.norm(combined_normal)
            else:
                combined_normal = normal_vector
                
        else:  # ellipse
            # For ellipses, use the fitted plane normal
            combined_normal = normal_vector
        
        # Ensure normal points towards camera (negative Z direction in camera coordinates)
        if combined_normal[2] > 0:
            combined_normal = -combined_normal
        
        normal_info = {
            'valid': True,
            'fitted_normal': normal_vector,
            'refined_normal': combined_normal,
            'confidence': plane_info['inlier_ratio'],
            'mean_residual': plane_info['mean_residual']
        }
        
        return combined_normal, normal_info
    
    def calculate_plane_properties(self, analysis_result: Dict, depth_image: np.ndarray) -> Dict:
        """
        Calculate comprehensive plane properties including center, normal, and geometric features.
        
        Args:
            analysis_result: Analysis result from step 2
            depth_image: Depth image array
            
        Returns:
            Dictionary with complete plane properties
        """
        if not analysis_result['plane_info'] or not analysis_result['plane_info']['is_planar']:
            return {'valid': False, 'reason': 'Not a planar surface'}
        
        # Calculate center
        center_3d, center_info = self.calculate_plane_center_3d(analysis_result, depth_image)
        
        # Calculate refined normal
        normal_vector, normal_info = self.refine_normal_vector(analysis_result, depth_image)
        
        # Calculate plane area and other geometric properties
        shape_area = self.calculate_shape_area(analysis_result)
        
        # Calculate plane orientation angles
        orientation = self.calculate_orientation_angles(normal_vector)
        
        plane_properties = {
            'valid': True,
            'shape_type': analysis_result['shape_type'],
            'shape_index': analysis_result['shape_index'],
            
            # Center information
            'center_3d': center_3d,
            'center_info': center_info,
            
            # Normal vector information
            'normal_vector': normal_vector,
            'normal_info': normal_info,
            
            # Geometric properties
            'area': shape_area,
            'orientation': orientation,
            
            # Quality metrics
            'planarity_confidence': analysis_result['plane_info']['inlier_ratio'],
            'fitting_residual': analysis_result['plane_info']['mean_residual'],
            
            # Original detection data
            'original_center': analysis_result['original_center'],
            'depth_stats': analysis_result['depth_stats']
        }
        
        return plane_properties
    
    def calculate_shape_area(self, analysis_result: Dict) -> float:
        """
        Calculate the area of the detected shape.
        
        Args:
            analysis_result: Analysis result from step 2
            
        Returns:
            Area in pixels squared
        """
        if analysis_result['shape_type'] == 'circle':
            radius = analysis_result['scaled_radius']
            area = np.pi * radius ** 2
        else:  # ellipse
            axes = analysis_result['scaled_axes']
            area = np.pi * axes[0] * axes[1] / 4  # Semi-major * semi-minor
        
        return area
    
    def calculate_orientation_angles(self, normal_vector: np.ndarray) -> Dict:
        """
        Calculate orientation angles of the plane normal vector.
        
        Args:
            normal_vector: 3D normal vector
            
        Returns:
            Dictionary with orientation angles in degrees
        """
        nx, ny, nz = normal_vector
        
        # Elevation angle (angle from XY plane)
        elevation = np.arcsin(abs(nz)) * 180 / np.pi
        
        # Azimuth angle (angle in XY plane)
        azimuth = np.arctan2(ny, nx) * 180 / np.pi
        
        # Tilt angles relative to each axis
        tilt_x = np.arccos(abs(nx)) * 180 / np.pi
        tilt_y = np.arccos(abs(ny)) * 180 / np.pi
        tilt_z = np.arccos(abs(nz)) * 180 / np.pi
        
        return {
            'elevation_deg': elevation,
            'azimuth_deg': azimuth,
            'tilt_x_deg': tilt_x,
            'tilt_y_deg': tilt_y,
            'tilt_z_deg': tilt_z
        }
    
    def visualize_plane_geometry(self, plane_properties_list: List[Dict], 
                               output_path: str) -> None:
        """
        Create 3D visualization of detected planes with centers and normal vectors.
        
        Args:
            plane_properties_list: List of plane properties from calculate_plane_properties
            output_path: Path to save visualization
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        
        valid_planes = [p for p in plane_properties_list if p['valid']]
        
        if valid_planes:
            centers = np.array([p['center_3d'] for p in valid_planes])
            normals = np.array([p['normal_vector'] for p in valid_planes])
            
            # Plot centers
            ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                       c='red', s=100, alpha=0.8, label='Plane Centers')
            
            # Plot normal vectors
            for i, (center, normal) in enumerate(zip(centers, normals)):
                # Scale normal vector for visualization
                normal_scaled = normal * 50  # Adjust scale as needed
                
                ax1.quiver(center[0], center[1], center[2],
                          normal_scaled[0], normal_scaled[1], normal_scaled[2],
                          color='blue', alpha=0.7, arrow_length_ratio=0.1)
                
                # Add labels
                ax1.text(center[0], center[1], center[2], f'P{i}',
                        fontsize=8, color='black')
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z (Depth)')
            ax1.set_title('3D Plane Centers and Normal Vectors')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 0.5, 'No valid planes detected',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('No Data Available')
        
        # Orientation analysis plot
        ax2 = fig.add_subplot(122)
        
        if valid_planes:
            elevations = [p['orientation']['elevation_deg'] for p in valid_planes]
            azimuths = [p['orientation']['azimuth_deg'] for p in valid_planes]
            confidences = [p['planarity_confidence'] for p in valid_planes]
            
            # Scatter plot of orientations
            scatter = ax2.scatter(azimuths, elevations, c=confidences, 
                                cmap='viridis', s=100, alpha=0.7)
            
            # Add labels for each plane
            for i, (az, el) in enumerate(zip(azimuths, elevations)):
                ax2.annotate(f'P{i}', (az, el), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Azimuth (degrees)')
            ax2.set_ylabel('Elevation (degrees)')
            ax2.set_title('Plane Orientations')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Planarity Confidence')
        else:
            ax2.text(0.5, 0.5, 'No valid planes detected',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('No Data Available')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def calculate_plane_geometry(analysis_results: List[Dict], depth_image_path: str,
                           output_dir: str, filename_prefix: str = "",
                           camera_params: Optional[Dict] = None) -> List[Dict]:
    """
    Calculate plane centers and normal vectors for all detected planar surfaces.
    
    Args:
        analysis_results: Results from step 2 depth analysis
        depth_image_path: Path to depth image
        output_dir: Directory to save results
        filename_prefix: Prefix for output filenames
        camera_params: Camera intrinsic parameters (optional)
        
    Returns:
        List of plane properties for each valid planar surface
    """
    # Initialize calculator
    calculator = PlaneGeometryCalculator(camera_params)
    
    # Load depth image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    # Filter out invalid depth values - only keep values in valid range (100-5000)
    depth_image[(depth_image < 100.0) | (depth_image > 5000.0)] = np.nan
    
    print(f"Calculating plane geometry for {len(analysis_results)} detected shapes...")
    
    plane_properties_list = []
    
    for result in analysis_results:
        if result['plane_info'] and result['plane_info']['is_planar']:
            print(f"Processing {result['shape_type']} {result['shape_index']}...")
            
            # Calculate plane properties
            plane_props = calculator.calculate_plane_properties(result, depth_image)
            
            if plane_props['valid']:
                plane_properties_list.append(plane_props)
                
                # Print summary
                center = plane_props['center_3d']
                normal = plane_props['normal_vector']
                print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
                print(f"  Normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
                print(f"  Confidence: {plane_props['planarity_confidence']:.3f}")
                print(f"  Area: {plane_props['area']:.1f} pixels²")
            else:
                print(f"  -> Failed to calculate geometry: {plane_props.get('reason', 'Unknown error')}")
        else:
            print(f"Skipping non-planar {result['shape_type']} {result['shape_index']}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize results
    viz_path = os.path.join(output_dir, f"{filename_prefix}step3_plane_geometry.png")
    calculator.visualize_plane_geometry(plane_properties_list, viz_path)
    
    print(f"Found {len(plane_properties_list)} valid planar surfaces")
    
    return plane_properties_list


if __name__ == "__main__":
    # This would be called after step 2
    print("Step 3: Plane Geometry Calculation module")
    print("This module should be called after step 2 depth analysis.") 