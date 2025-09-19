#!/usr/bin/env python3
"""
Step 4: Pipe Detection Module
Detect pipe structures and calculate their orientation/direction
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from sklearn.decomposition import PCA


class PipeDetector:
    def __init__(self):
        """Initialize the pipe detector"""
        self.pipe_detection_params = {
            'min_depth_variation': 20,  # Minimum depth variation for pipe detection
            'gradient_threshold': 5.0,   # Minimum gradient for edge detection
            'symmetry_threshold': 0.7,   # Threshold for circular symmetry
            'depth_consistency_threshold': 0.3  # Relative threshold for depth consistency
        }
    
    def detect_pipe(self, depth_info, circle):
        """
        Detect if the analyzed region contains a pipe structure
        
        Args:
            depth_info: Depth analysis results from step 3
            circle: Original circle detection results
            
        Returns:
            dict: Pipe detection results including direction if detected
        """
        if not depth_info['has_valid_data']:
            return {
                'is_pipe': False,
                'confidence': 0.0,
                'direction': None,
                'pipe_axis': None,
                'detection_details': 'No valid depth data'
            }
        
        # Analyze pipe characteristics
        pipe_analysis = self._analyze_pipe_characteristics(depth_info)
        
        # Make pipe detection decision
        is_pipe, confidence, details = self._decide_pipe_detection(pipe_analysis)
        
        # Calculate pipe direction if detected
        direction = None
        pipe_axis = None
        if is_pipe:
            direction, pipe_axis = self._calculate_pipe_direction(depth_info, pipe_analysis)
        
        result = {
            'is_pipe': is_pipe,
            'confidence': confidence,
            'direction': direction,
            'pipe_axis': pipe_axis,
            'detection_details': details,
            'analysis': pipe_analysis
        }
        
        return result
    
    def _analyze_pipe_characteristics(self, depth_info):
        """
        Analyze characteristics that indicate a pipe structure
        """
        depth_region = depth_info['depth_region']
        depth_stats = depth_info['depth_stats']
        gradient_analysis = depth_info['gradient_analysis']
        
        analysis = {}
        
        # 1. Depth variation analysis
        depth_variation = depth_stats.get('range', 0)
        analysis['depth_variation'] = depth_variation
        analysis['has_sufficient_variation'] = depth_variation > self.pipe_detection_params['min_depth_variation']
        
        # 2. Gradient pattern analysis
        if 'gradient_stats' in gradient_analysis:
            grad_stats = gradient_analysis['gradient_stats']
            analysis['gradient_strength'] = grad_stats.get('mean_gradient', 0)
            analysis['has_strong_gradients'] = (
                grad_stats.get('mean_gradient', 0) > self.pipe_detection_params['gradient_threshold']
            )
        else:
            analysis['gradient_strength'] = 0
            analysis['has_strong_gradients'] = False
        
        # 3. Circular pattern analysis
        analysis['has_circular_pattern'] = gradient_analysis.get('has_gradient_pattern', False)
        
        # 4. Depth consistency analysis (for cylindrical shape)
        if 'cleaned_depth' in gradient_analysis:
            consistency_score = self._analyze_depth_consistency(gradient_analysis['cleaned_depth'])
            analysis['depth_consistency'] = consistency_score
            analysis['has_consistent_depth'] = (
                consistency_score > self.pipe_detection_params['depth_consistency_threshold']
            )
        else:
            analysis['depth_consistency'] = 0
            analysis['has_consistent_depth'] = False
        
        # 5. Cylindrical shape analysis
        if 'cleaned_depth' in gradient_analysis:
            cylindrical_score = self._analyze_cylindrical_shape(gradient_analysis['cleaned_depth'])
            analysis['cylindrical_score'] = cylindrical_score
        else:
            analysis['cylindrical_score'] = 0
        
        return analysis
    
    def _analyze_depth_consistency(self, cleaned_depth):
        """
        Analyze depth consistency to detect cylindrical structures
        """
        if np.sum(~np.isnan(cleaned_depth)) < 20:
            return 0
        
        valid_mask = ~np.isnan(cleaned_depth)
        # Apply depth validity check (100-5000 range)
        depth_validity_mask = (cleaned_depth >= 100) & (cleaned_depth <= 5000)
        valid_mask = valid_mask & depth_validity_mask
        
        if np.sum(valid_mask) < 20:
            return 0
        
        height, width = cleaned_depth.shape
        center_x, center_y = width // 2, height // 2
        
        # Analyze radial depth profiles
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 16 radial directions
        radial_profiles = []
        
        max_radius = min(center_x, center_y, width - center_x, height - center_y) - 2
        if max_radius < 5:
            return 0
        
        for angle in angles:
            profile = []
            for r in range(1, max_radius):
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))
                
                if (0 <= x < width and 0 <= y < height and 
                    valid_mask[y, x]):
                    profile.append(cleaned_depth[y, x])
            
            if len(profile) > 3:
                radial_profiles.append(np.array(profile))
        
        if len(radial_profiles) < 8:
            return 0
        
        # Calculate consistency between radial profiles
        # For a cylinder, profiles should be similar
        consistency_scores = []
        for i in range(len(radial_profiles)):
            for j in range(i+1, len(radial_profiles)):
                if len(radial_profiles[i]) > 0 and len(radial_profiles[j]) > 0:
                    min_len = min(len(radial_profiles[i]), len(radial_profiles[j]))
                    if min_len > 3:
                        corr = np.corrcoef(radial_profiles[i][:min_len], 
                                         radial_profiles[j][:min_len])[0, 1]
                        if not np.isnan(corr):
                            consistency_scores.append(abs(corr))
        
        return np.mean(consistency_scores) if consistency_scores else 0
    
    def _analyze_cylindrical_shape(self, cleaned_depth):
        """
        Analyze if the depth pattern suggests a cylindrical shape
        """
        if np.sum(~np.isnan(cleaned_depth)) < 20:
            return 0
        
        valid_mask = ~np.isnan(cleaned_depth)
        # Apply depth validity check (100-5000 range)
        depth_validity_mask = (cleaned_depth >= 100) & (cleaned_depth <= 5000)
        valid_mask = valid_mask & depth_validity_mask
        
        if np.sum(valid_mask) < 20:
            return 0
        
        height, width = cleaned_depth.shape
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        # Calculate distances from center
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Analyze depth as a function of distance from center
        max_dist = min(center_x, center_y, width - center_x, height - center_y) - 2
        if max_dist < 5:
            return 0
        
        # Sample depth at different distances
        distance_depths = []
        for d in range(1, max_dist, 2):
            ring_mask = ((distances >= d-1) & (distances < d+1) & valid_mask)
            if np.sum(ring_mask) > 0:
                mean_depth = np.mean(cleaned_depth[ring_mask])
                distance_depths.append((d, mean_depth))
        
        if len(distance_depths) < 3:
            return 0
        
        # Check if depth changes monotonically with distance (cylinder characteristic)
        distances_arr = np.array([d[0] for d in distance_depths])
        depths_arr = np.array([d[1] for d in distance_depths])
        
        # Calculate correlation between distance and depth
        correlation = np.corrcoef(distances_arr, depths_arr)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0
    
    def _decide_pipe_detection(self, pipe_analysis):
        """
        Make the final decision on whether this is a pipe
        """
        confidence_factors = []
        detection_details = []
        
        # Check each criteria
        if pipe_analysis.get('has_sufficient_variation', False):
            confidence_factors.append(0.25)
            detection_details.append("Sufficient depth variation")
        
        if pipe_analysis.get('has_strong_gradients', False):
            confidence_factors.append(0.25)
            detection_details.append("Strong depth gradients")
        
        if pipe_analysis.get('has_circular_pattern', False):
            confidence_factors.append(0.3)
            detection_details.append("Circular gradient pattern")
        
        if pipe_analysis.get('has_consistent_depth', False):
            confidence_factors.append(0.2)
            detection_details.append("Consistent radial depth")
        
        # Additional scoring from cylindrical analysis
        cylindrical_score = pipe_analysis.get('cylindrical_score', 0)
        if cylindrical_score > 0.5:
            confidence_factors.append(cylindrical_score * 0.3)
            detection_details.append(f"Cylindrical shape (score: {cylindrical_score:.2f})")
        
        # Calculate overall confidence
        confidence = min(1.0, sum(confidence_factors))
        
        # Decision threshold
        is_pipe = confidence > 0.6
        
        if not detection_details:
            detection_details.append("No pipe characteristics detected")
        
        return is_pipe, confidence, "; ".join(detection_details)
    
    def _calculate_pipe_direction(self, depth_info, pipe_analysis):
        """
        Calculate the 3D pipe axis direction using depth information
        """
        if 'cleaned_depth' not in depth_info['gradient_analysis']:
            return None, None
        
        cleaned_depth = depth_info['gradient_analysis']['cleaned_depth']
        valid_mask = ~np.isnan(cleaned_depth)
        
        # Apply depth validity check (100-5000 range)
        depth_validity_mask = (cleaned_depth >= 100) & (cleaned_depth <= 5000)
        valid_mask = valid_mask & depth_validity_mask
        
        if np.sum(valid_mask) < 20:
            return None, None
        
        # Convert 2D image coordinates + depth to 3D points
        height, width = cleaned_depth.shape
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Get valid 3D points
        valid_y = y_coords[valid_mask]
        valid_x = x_coords[valid_mask]
        valid_z = cleaned_depth[valid_mask]
        
        # Create 3D point cloud
        points_3d = np.column_stack([valid_x, valid_y, valid_z])
        
        if len(points_3d) < 20:
            return None, None
        
        # Apply 3D PCA to find the pipe axis direction
        try:
            # Center the points
            centroid = np.mean(points_3d, axis=0)
            centered_points = points_3d - centroid
            
            # Apply PCA in 3D
            pca = PCA(n_components=3)
            pca.fit(centered_points)
            
            # The first principal component is the pipe axis direction in 3D
            pipe_axis_3d = pca.components_[0]
            
            # Calculate the direction angles
            # For a pipe, we want the direction that points most into/out of the image plane
            direction_angles = self._calculate_3d_direction_angles(pipe_axis_3d)
            
            # Also calculate the projection on the image plane for visualization
            image_plane_projection = self._project_to_image_plane(pipe_axis_3d)
            
            return direction_angles, {
                'axis_3d': pipe_axis_3d,
                'centroid_3d': centroid,
                'image_projection': image_plane_projection,
                'explained_variance': pca.explained_variance_ratio_
            }
            
        except Exception as e:
            print(f"Error in 3D pipe direction calculation: {e}")
            return None, None
    
    def _calculate_3d_direction_angles(self, axis_3d):
        """
        Calculate 3D direction angles from the axis vector
        
        Args:
            axis_3d: 3D axis vector [x, y, z]
            
        Returns:
            dict: Direction angles in different representations
        """
        x, y, z = axis_3d
        
        # Normalize the vector
        norm = np.linalg.norm(axis_3d)
        if norm == 0:
            return None
        
        x_norm, y_norm, z_norm = axis_3d / norm
        
        # Calculate spherical coordinates
        # Azimuth angle (rotation around Z axis, in image plane)
        azimuth = np.degrees(np.arctan2(y_norm, x_norm))
        
        # Elevation angle (angle from image plane)
        elevation = np.degrees(np.arcsin(z_norm))
        
        # For pipe direction, we care most about the direction into/out of the image
        # Z-component indicates how much the pipe points toward/away from camera
        depth_direction = "into_image" if z_norm > 0 else "out_of_image"
        
        # Calculate the angle between pipe axis and image plane
        image_plane_angle = np.degrees(np.arccos(abs(z_norm)))
        
        return {
            'azimuth_deg': azimuth,           # Direction in image plane (-180 to 180)
            'elevation_deg': elevation,       # Angle from image plane (-90 to 90)
            'depth_direction': depth_direction,
            'image_plane_angle_deg': image_plane_angle,  # 0=perpendicular to image, 90=parallel to image
            'axis_3d_normalized': [x_norm, y_norm, z_norm]
        }
    
    def _project_to_image_plane(self, axis_3d):
        """
        Project 3D axis to image plane for visualization
        """
        x, y, z = axis_3d
        
        # Project to image plane (ignore Z component)
        projected_vector = np.array([x, y])
        
        if np.linalg.norm(projected_vector) == 0:
            return None
        
        # Normalize
        projected_vector = projected_vector / np.linalg.norm(projected_vector)
        
        # Calculate angle in image plane
        angle_rad = np.arctan2(projected_vector[1], projected_vector[0])
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to [0, 180] for visualization
        if angle_deg < 0:
            angle_deg += 180
        elif angle_deg > 180:
            angle_deg -= 180
        
        return {
            'vector_2d': projected_vector,
            'angle_deg': angle_deg
        }


def test_pipe_detection():
    """Test the pipe detection with sample data"""
    from step1_data_loading import DataLoader
    from step2_circle_detection import CircleDetector
    from step3_depth_analysis import DepthAnalyzer
    
    snapshot_base = Path("../snapshot")
    available_folders = [f.name for f in snapshot_base.iterdir() if f.is_dir()]
    
    if available_folders:
        test_folder = available_folders[0]
        print(f"Testing pipe detection with folder: {test_folder}")
        
        # Load data and run previous steps
        loader = DataLoader()
        ir_image, depth_image, params = loader.load_data(snapshot_base / test_folder)
        
        detector = CircleDetector()
        enhanced_ir, circles = detector.detect_circles(ir_image)
        
        if circles:
            analyzer = DepthAnalyzer()
            pipe_detector = PipeDetector()
            
            for i, circle in enumerate(circles):
                print(f"\nAnalyzing Circle {i+1} at {circle['center']} (r={circle['radius']}):")
                
                # Get depth analysis
                depth_info = analyzer.analyze_circle_depth(depth_image, circle, radius_multiplier=3)
                
                # Detect pipe
                pipe_result = pipe_detector.detect_pipe(depth_info, circle)
                
                print(f"  Is pipe: {pipe_result['is_pipe']}")
                print(f"  Confidence: {pipe_result['confidence']:.2f}")
                print(f"  Details: {pipe_result['detection_details']}")
                
                if pipe_result['direction'] is not None:
                    direction_info = pipe_result['direction']
                    print(f"  3D Pipe Direction:")
                    print(f"    Azimuth: {direction_info['azimuth_deg']:.1f}° (in image plane)")
                    print(f"    Elevation: {direction_info['elevation_deg']:.1f}° (from image plane)")
                    print(f"    Depth direction: {direction_info['depth_direction']}")
                    print(f"    Image plane angle: {direction_info['image_plane_angle_deg']:.1f}°")
                    print(f"    3D axis: [{direction_info['axis_3d_normalized'][0]:.3f}, {direction_info['axis_3d_normalized'][1]:.3f}, {direction_info['axis_3d_normalized'][2]:.3f}]")
                    
                    if pipe_result['pipe_axis'] and 'explained_variance' in pipe_result['pipe_axis']:
                        variance_ratio = pipe_result['pipe_axis']['explained_variance']
                        print(f"    PCA variance explained: {variance_ratio[0]:.3f} (1st), {variance_ratio[1]:.3f} (2nd), {variance_ratio[2]:.3f} (3rd)")
            
            print("\nPipe detection test completed successfully!")
        else:
            print("No circles detected for pipe analysis")
    else:
        print("No snapshot folders found for testing")


if __name__ == "__main__":
    test_pipe_detection() 