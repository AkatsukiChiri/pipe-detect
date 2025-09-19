#!/usr/bin/env python3
"""
Step 5: Final Visualization Module
Create comprehensive visualization with color-coded results and direction arrows for both circles and ellipses
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    def __init__(self, target_resolution=(1600, 1200)):
        """
        Initialize the visualizer with parameters optimized for target resolution
        
        Args:
            target_resolution: (width, height) of the target image resolution
        """
        self.target_width, self.target_height = target_resolution
        
        self.colors = {
            'circle_only': (0, 0, 255),      # Red for circles without pipes
            'ellipse_only': (255, 0, 255),   # Magenta for ellipses without pipes
            'pipe_detected': (0, 255, 255),  # Yellow for shapes with pipes
            'direction_arrow': (255, 0, 0),  # Blue for direction arrows
            'center_point': (255, 255, 255)  # White for center points
        }
        
        # Scale parameters based on resolution (base resolution: 640x480)
        # scale_factor = min(self.target_width / 640, self.target_height / 480)
        scale_factor = 1
        
        self.visualization_params = {
            'shape_thickness': max(2, int(3 * scale_factor)),           # 线条粗细
            'center_radius': max(3, int(5 * scale_factor)),             # 中心点半径
            'arrow_length': int(150 * scale_factor),                    # 箭头长度
            'arrow_thickness': max(3, int(4 * scale_factor)),           # 箭头粗细
            'text_font': cv2.FONT_HERSHEY_SIMPLEX,
            'text_scale': max(0.6, 0.6 * scale_factor),                # 字体大小
            'text_thickness': max(2, int(3 * scale_factor)),            # 字体粗细
            'legend_font_scale': max(0.6, 0.8 * scale_factor),         # 图例字体大小
            'legend_thickness': max(1, int(2 * scale_factor)),          # 图例线条粗细
            'legend_rect_size': max(15, int(25 * scale_factor)),        # 图例色块大小
            'annotation_font_scale': max(0.5, 0.7 * scale_factor),     # 标注字体大小
            'annotation_thickness': max(1, int(2 * scale_factor)),      # 标注粗细
            'scale_factor': scale_factor                                # 总体缩放因子
        }
        
        print(f"Visualizer initialized for {self.target_width}x{self.target_height} resolution")
        print(f"Scale factor: {scale_factor:.2f}")
        print(f"Text scale: {self.visualization_params['text_scale']:.2f}")
        print(f"Shape thickness: {self.visualization_params['shape_thickness']}")
    
    def create_final_visualization(self, enhanced_ir, shapes, pipe_detection_results, folder_name):
        """
        Create the final visualization with color-coded results
        Enhanced to support both circles and ellipses at high resolution
        
        Args:
            enhanced_ir: Enhanced IR image
            shapes: List of detected shapes (circles and ellipses)
            pipe_detection_results: List of pipe detection results
            folder_name: Name of the folder for saving results
            
        Returns:
            result_image: Final visualization image
        """
        # Create result image (convert to color)
        result_image = cv2.cvtColor(enhanced_ir, cv2.COLOR_GRAY2BGR)
        
        # Process each shape and its detection result
        for i, (shape, pipe_result) in enumerate(zip(shapes, pipe_detection_results)):
            center = shape['center']
            shape_type = shape.get('type', 'circle')
            
            # Determine colors based on detection result and shape type
            if pipe_result['is_pipe']:
                shape_color = self.colors['pipe_detected']
                label = f"Pipe {i+1}"
            else:
                if shape_type == 'ellipse':
                    shape_color = self.colors['ellipse_only']
                    label = f"Ellipse {i+1}"
                else:
                    shape_color = self.colors['circle_only']
                label = f"Circle {i+1}"
            
            # Draw shape (circle or ellipse)
            self._draw_shape(result_image, shape, shape_color)
            
            # Draw center point (larger for high resolution)
            cv2.circle(result_image, center, self.visualization_params['center_radius'], 
                      shape_color, -1)
            
            # Add label with better positioning for high resolution
            if shape_type == 'ellipse':
                # For ellipses, position label considering major axis
                label_offset = max(shape.get('major_axis', 50), shape.get('minor_axis', 50)) // 2 + 20
            else:
                label_offset = shape.get('radius', 25) + 20
            
            label_pos = (center[0] - int(60 * self.visualization_params['scale_factor']), 
                        center[1] - label_offset)
            
            # Ensure label position is within image bounds
            label_pos = (max(10, min(label_pos[0], result_image.shape[1] - 200)),
                        max(30, min(label_pos[1], result_image.shape[0] - 10)))
            
            # Use smaller font for shape labels
            label_font_scale = max(0.5, 0.7 * self.visualization_params['scale_factor'])
            label_thickness = max(1, int(2 * self.visualization_params['scale_factor']))
            
            cv2.putText(result_image, label, label_pos,
                       self.visualization_params['text_font'],
                       label_font_scale,
                       shape_color,
                       label_thickness)
            
            # Draw 3D direction visualization if pipe detected
            if pipe_result['is_pipe'] and pipe_result['direction'] is not None:
                direction_info = pipe_result['direction']
                pipe_axis_info = pipe_result['pipe_axis']
                
                # Draw 3D pipe direction
                self._draw_3d_pipe_direction(result_image, center, direction_info, pipe_axis_info, shape)
                
                # Add comprehensive 3D direction annotations
                self._add_3d_direction_annotations(result_image, center, shape, direction_info)
        
        # Add legend (positioned for high resolution)
        self._add_legend(result_image)
        
        # Add image information overlay
        self._add_image_info_overlay(result_image, shapes, pipe_detection_results)
        
        # Save comprehensive visualization
        self._save_comprehensive_visualization(enhanced_ir, shapes, pipe_detection_results, 
                                             result_image, folder_name)
        
        return result_image
    
    def _draw_shape(self, image, shape, color):
        """
        Draw a shape (circle or ellipse) on the image
        
        Args:
            image: Image to draw on
            shape: Shape dictionary with type and parameters
            color: Color to use for drawing
        """
        center = shape['center']
        shape_type = shape.get('type', 'circle')
        thickness = self.visualization_params['shape_thickness']
        
        if shape_type == 'ellipse':
            # Draw ellipse
            major_axis = shape.get('major_axis', 50)
            minor_axis = shape.get('minor_axis', 50)
            angle = shape.get('angle', 0)
            
            # OpenCV ellipse parameters: center, axes (half lengths), angle, start_angle, end_angle, color, thickness
            cv2.ellipse(image, center, (int(major_axis/2), int(minor_axis/2)), 
                       angle, 0, 360, color, thickness)
            
            # Draw major axis line for ellipses to show orientation
            angle_rad = np.radians(angle)
            major_half = major_axis / 2
            start_x = int(center[0] - major_half * np.cos(angle_rad))
            start_y = int(center[1] - major_half * np.sin(angle_rad))
            end_x = int(center[0] + major_half * np.cos(angle_rad))
            end_y = int(center[1] + major_half * np.sin(angle_rad))
            
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, 1)
            
        else:
            # Draw circle
            radius = shape.get('radius', 25)
            cv2.circle(image, center, radius, color, thickness)
    
    def _draw_3d_pipe_direction(self, image, center, direction_info, pipe_axis_info, shape):
        """
        Draw 3D pipe direction visualization
        Enhanced for high resolution and ellipse orientation
        
        Args:
            image: Image to draw on
            center: Center point (x, y) 
            direction_info: 3D direction information
            pipe_axis_info: Additional pipe axis information
            shape: Shape information (for ellipse-specific handling)
        """
        if not direction_info or not pipe_axis_info:
            return
        
        # Extract 3D direction information
        elevation = direction_info['elevation_deg']
        azimuth = direction_info['azimuth_deg']
        depth_direction = direction_info['depth_direction']
        image_plane_angle = direction_info['image_plane_angle_deg']
        
        # Calculate visualization parameters based on 3D direction and resolution
        base_length = self.visualization_params['arrow_length']
        
        # Scale arrow length based on how perpendicular it is to image plane
        # More perpendicular = shorter projection, less perpendicular = longer projection
        length_scale = np.sin(np.radians(image_plane_angle))
        projected_length = int(base_length * max(0.3, length_scale))
        
        # Use image plane projection for 2D drawing
        if 'image_projection' in pipe_axis_info and pipe_axis_info['image_projection']:
            proj_angle = pipe_axis_info['image_projection']['angle_deg']
            angle_rad = np.radians(proj_angle)
        else:
            # Fallback to azimuth angle
            angle_rad = np.radians(azimuth)
        
        # For ellipses, consider the major axis orientation
        if shape.get('type') == 'ellipse':
            ellipse_angle = shape.get('angle', 0)
            # Show relationship between ellipse orientation and pipe direction
            if 'ellipse_3d_alignment' in direction_info:
                alignment = direction_info.get('ellipse_3d_alignment', 0)
                # Adjust arrow style based on alignment
                if alignment > 60:  # Good alignment
                    projected_length = int(projected_length * 1.2)  # Longer arrow for better confidence
        
        # Calculate arrow endpoints
        end1_x = int(center[0] + projected_length * np.cos(angle_rad))
        end1_y = int(center[1] + projected_length * np.sin(angle_rad))
        end2_x = int(center[0] - projected_length * np.cos(angle_rad))
        end2_y = int(center[1] - projected_length * np.sin(angle_rad))
        
        # Choose colors based on depth direction
        if depth_direction == "into_image":
            arrow_color = (255, 100, 0)  # Orange for into image
            depth_color = (255, 150, 0)
        else:
            arrow_color = (0, 100, 255)  # Blue for out of image  
            depth_color = (0, 150, 255)
        
        # Draw main pipe axis line (thicker for high resolution)
        cv2.line(image, (end2_x, end2_y), (end1_x, end1_y), 
                arrow_color, self.visualization_params['arrow_thickness'])
        
        # Draw depth direction indicator (perpendicular to pipe axis)
        perp_angle = angle_rad + np.pi/2
        depth_length = int(40 * self.visualization_params['scale_factor'])  # Scale for resolution
        
        # Calculate depth indicator size based on elevation
        depth_indicator_scale = abs(np.cos(np.radians(elevation)))
        scaled_depth_length = int(depth_length * max(0.2, depth_indicator_scale))
        
        depth_end_x = int(center[0] + scaled_depth_length * np.cos(perp_angle))
        depth_end_y = int(center[1] + scaled_depth_length * np.sin(perp_angle))
        
        # Draw depth direction indicator
        if abs(elevation) > 45:  # High elevation - show as circle indicating depth
            circle_radius = max(int(15 * self.visualization_params['scale_factor']), 
                               int(scaled_depth_length * 0.6))
            circle_thickness = max(2, int(3 * self.visualization_params['scale_factor']))
            cv2.circle(image, center, circle_radius, depth_color, circle_thickness)
            
            # Add direction symbol in the circle (scaled for resolution)
            symbol_size = int(10 * self.visualization_params['scale_factor'])
            symbol_thickness = max(2, int(3 * self.visualization_params['scale_factor']))
            
            if depth_direction == "into_image":
                # Draw '+' for into image
                cv2.line(image, (center[0]-symbol_size, center[1]), 
                        (center[0]+symbol_size, center[1]), depth_color, symbol_thickness)
                cv2.line(image, (center[0], center[1]-symbol_size), 
                        (center[0], center[1]+symbol_size), depth_color, symbol_thickness)
            else:
                # Draw 'x' for out of image
                cv2.line(image, (center[0]-symbol_size, center[1]-symbol_size), 
                        (center[0]+symbol_size, center[1]+symbol_size), depth_color, symbol_thickness)
                cv2.line(image, (center[0]-symbol_size, center[1]+symbol_size), 
                        (center[0]+symbol_size, center[1]-symbol_size), depth_color, symbol_thickness)
        
        # Draw arrowheads on the main axis (scaled for resolution)
        arrowhead_length = int(25 * self.visualization_params['scale_factor'])
        arrowhead_angle = np.pi / 6
        arrowhead_thickness = max(2, int(3 * self.visualization_params['scale_factor']))
        
        # Arrowhead 1
        head1_x1 = int(end1_x - arrowhead_length * np.cos(angle_rad - arrowhead_angle))
        head1_y1 = int(end1_y - arrowhead_length * np.sin(angle_rad - arrowhead_angle))
        head1_x2 = int(end1_x - arrowhead_length * np.cos(angle_rad + arrowhead_angle))
        head1_y2 = int(end1_y - arrowhead_length * np.sin(angle_rad + arrowhead_angle))
        
        cv2.line(image, (end1_x, end1_y), (head1_x1, head1_y1), arrow_color, arrowhead_thickness)
        cv2.line(image, (end1_x, end1_y), (head1_x2, head1_y2), arrow_color, arrowhead_thickness)
        
        # Arrowhead 2
        head2_x1 = int(end2_x + arrowhead_length * np.cos(angle_rad - arrowhead_angle))
        head2_y1 = int(end2_y + arrowhead_length * np.sin(angle_rad - arrowhead_angle))
        head2_x2 = int(end2_x + arrowhead_length * np.cos(angle_rad + arrowhead_angle))
        head2_y2 = int(end2_y + arrowhead_length * np.sin(angle_rad + arrowhead_angle))
        
        cv2.line(image, (end2_x, end2_y), (head2_x1, head2_y1), arrow_color, arrowhead_thickness)
        cv2.line(image, (end2_x, end2_y), (head2_x2, head2_y2), arrow_color, arrowhead_thickness)
    
    def _add_3d_direction_annotations(self, image, center, shape, direction_info):
        """
        Add text annotations for 3D direction information with vector representation
        Enhanced for ellipse support and high resolution with smaller fonts
        
        Args:
            image: Image to draw on
            center: Center point of the shape
            shape: Shape information (circle or ellipse)
            direction_info: 3D direction information
        """
        # Determine annotation position based on shape and resolution (closer to shape)
        if shape.get('type') == 'ellipse':
            offset_distance = max(shape.get('major_axis', 50), shape.get('minor_axis', 50)) // 2 + int(30 * self.visualization_params['scale_factor'])
        else:
            offset_distance = shape.get('radius', 25) + int(30 * self.visualization_params['scale_factor'])
        
        annotation_x = center[0] + offset_distance
        annotation_y = center[1] - int(20 * self.visualization_params['scale_factor'])
        
        # Ensure annotation position is within image bounds
        annotation_x = max(10, min(annotation_x, image.shape[1] - 200))
        annotation_y = max(30, min(annotation_y, image.shape[0] - 60))
        
        # Extract direction info
        elevation = direction_info['elevation_deg']
        azimuth = direction_info['azimuth_deg']
        depth_direction = direction_info['depth_direction']
        
        # Convert to 3D vector for display
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)
        
        # Direction vector components
        dx = np.cos(elevation_rad) * np.cos(azimuth_rad)
        dy = np.cos(elevation_rad) * np.sin(azimuth_rad) 
        dz = np.sin(elevation_rad)
        
        # Use smaller, more readable font
        font_scale = max(0.4, 0.5 * self.visualization_params['scale_factor'])
        thickness = max(1, int(1.5 * self.visualization_params['scale_factor']))
        line_spacing = int(18 * self.visualization_params['scale_factor'])
        
        # Vector representation (compact format)
        vector_text = f"Vector: ({dx:.2f},{dy:.2f},{dz:.2f})"
        cv2.putText(image, vector_text, (annotation_x, annotation_y),
                   self.visualization_params['text_font'], 
                   font_scale, (255, 255, 255), thickness)
        
        # Angles in degrees (using degree symbol that works with OpenCV)
        angle_text = f"Elev: {elevation:.0f}deg  Az: {azimuth:.0f}deg"
        cv2.putText(image, angle_text, (annotation_x, annotation_y + line_spacing),
                   self.visualization_params['text_font'], 
                   font_scale, (255, 255, 255), thickness)
        
        # Depth direction indicator (compact format)
        depth_text = "Into" if depth_direction == "into_image" else "Out"
        cv2.putText(image, depth_text, (annotation_x, annotation_y + 2 * line_spacing),
                   self.visualization_params['text_font'], 
                   font_scale, (255, 255, 255), thickness)
    
    def _add_legend(self, image):
        """Add color legend to the image with high resolution parameters"""
        legend_start_y = int(50 * self.visualization_params['scale_factor'])
        legend_x = int(30 * self.visualization_params['scale_factor'])
        
        # Legend items
        legend_items = [
            ("Circle (no pipe)", self.colors['circle_only']),
            ("Ellipse (no pipe)", self.colors['ellipse_only']),
            ("Pipe detected", self.colors['pipe_detected']),
            ("Into image", (255, 100, 0)),
            ("Out of image", (0, 100, 255))
        ]
        
        line_spacing = int(40 * self.visualization_params['scale_factor'])
        rect_size = self.visualization_params['legend_rect_size']
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_start_y + i * line_spacing
            
            # Draw color rectangle (larger for high resolution)
            cv2.rectangle(image, (legend_x, y_pos - rect_size//2), 
                         (legend_x + rect_size, y_pos + rect_size//2), color, -1)
            cv2.rectangle(image, (legend_x, y_pos - rect_size//2), 
                         (legend_x + rect_size, y_pos + rect_size//2), 
                         (255, 255, 255), self.visualization_params['legend_thickness'])
            
            # Draw text (appropriately sized for high resolution)
            text_x = legend_x + rect_size + int(15 * self.visualization_params['scale_factor'])
            legend_font = max(0.5, 0.6 * self.visualization_params['scale_factor'])
            legend_thickness = max(1, int(1.5 * self.visualization_params['scale_factor']))
            
            cv2.putText(image, text, (text_x, y_pos + rect_size//4),
                       self.visualization_params['text_font'],
                       legend_font, 
                       (255, 255, 255), legend_thickness)
    
    def _add_image_info_overlay(self, image, shapes, pipe_detection_results):
        """
        Add image information overlay to the visualization.
        This includes image resolution and basic detection counts.
        """
        # Calculate total shapes and pipes
        total_shapes = len(shapes)
        pipes_detected = sum(1 for r in pipe_detection_results if r.get('is_pipe', False))
        circles = len([s for s in shapes if s.get('type') == 'circle'])
        ellipses = len([s for s in shapes if s.get('type') == 'ellipse'])

        # Position for info text (bottom-right corner)
        info_x = self.target_width - int(400 * self.visualization_params['scale_factor'])
        info_y = self.target_height - int(80 * self.visualization_params['scale_factor'])
        
        # Ensure position is within bounds
        info_x = max(10, info_x)
        info_y = max(50, info_y)

        # Create summary info lines
        info_lines = [
            f"Resolution: {self.target_width}x{self.target_height}",
            f"Shapes: {total_shapes} (C:{circles}, E:{ellipses})",
            f"Pipes: {pipes_detected}"
        ]

        # Draw each line
        line_spacing = int(30 * self.visualization_params['scale_factor'])
        for i, line in enumerate(info_lines):
            y_pos = info_y + i * line_spacing
            cv2.putText(image, line, (info_x, y_pos),
                       self.visualization_params['text_font'], 
                       self.visualization_params['annotation_font_scale'], 
                       (255, 255, 255), self.visualization_params['annotation_thickness'])

    def _save_comprehensive_visualization(self, enhanced_ir, shapes, pipe_detection_results, 
                                        result_image, folder_name):
        """
        Save comprehensive visualization with multiple views
        Enhanced for ellipse support
        """
        # Create a comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original enhanced IR
        axes[0, 0].imshow(enhanced_ir, cmap='gray')
        axes[0, 0].set_title('Enhanced IR Image')
        axes[0, 0].axis('off')
        
        # Shape detection overlay
        axes[0, 1].imshow(enhanced_ir, cmap='gray')
        
        # Draw shapes with matplotlib for better control
        for i, shape in enumerate(shapes):
            center = shape['center']
            shape_type = shape.get('type', 'circle')
            
            if shape_type == 'ellipse':
                major_axis = shape.get('major_axis', 50)
                minor_axis = shape.get('minor_axis', 50)
                angle = shape.get('angle', 0)
                
                ellipse_patch = patches.Ellipse(center, major_axis, minor_axis, 
                                              angle=angle, fill=False, 
                                              edgecolor='cyan', linewidth=2)
                axes[0, 1].add_patch(ellipse_patch)
                
                # Add ellipse number and info
                axes[0, 1].text(center[0], center[1] - major_axis/2 - 20, 
                               f'E{i+1}\n{major_axis}x{minor_axis}\n{angle:.0f}°',
                               color='cyan', fontsize=8, ha='center')
            else:
                radius = shape.get('radius', 25)
                circle_patch = patches.Circle(center, radius, fill=False, 
                                            edgecolor='lime', linewidth=2)
                axes[0, 1].add_patch(circle_patch)
                
                # Add circle number and info
                axes[0, 1].text(center[0], center[1] - radius - 20, 
                               f'C{i+1}\nr={radius}',
                               color='lime', fontsize=8, ha='center')
        
        axes[0, 1].set_title(f'Shape Detection ({len(shapes)} shapes)')
        axes[0, 1].axis('off')
        
        # Final result with pipe detection
        axes[0, 2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Pipe Detection Results')
        axes[0, 2].axis('off')
        
        # 3D Direction Visualization (redesigned and simplified)
        detected_pipes = [(shape, result) for shape, result in zip(shapes, pipe_detection_results) 
                         if result.get('is_pipe', False)]
        
        if detected_pipes:
            # Create clean 3D subplot
            ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
            ax_3d.clear()  # Start fresh
            
            # Set consistent view and clean background
            ax_3d.view_init(elev=20, azim=45)
            ax_3d.grid(True, alpha=0.3)
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (shape, pipe_result) in enumerate(detected_pipes):
                direction_info = pipe_result.get('direction')
                if direction_info:
                    # Extract direction angles
                    azimuth = direction_info.get('azimuth_deg', 0)
                    elevation = direction_info.get('elevation_deg', 0)
                    
                    # Convert to 3D vector (correct coordinate system)
                    # X: horizontal (image width), Y: vertical (image height), Z: depth
                    azimuth_rad = np.radians(azimuth)
                    elevation_rad = np.radians(elevation)
                    
                    # Direction vector in 3D space
                    dx = np.cos(elevation_rad) * np.cos(azimuth_rad)
                    dy = np.cos(elevation_rad) * np.sin(azimuth_rad) 
                    dz = np.sin(elevation_rad)
                    
                    # Center point (use shape center, depth=0 as reference)
                    center_x, center_y = shape['center']
                    center_x = center_x / 100  # Scale for better visualization
                    center_y = center_y / 100
                    center_z = 0
                    
                    # Arrow length
                    arrow_length = 2.0
                    
                    # Arrow endpoints
                    end_x = center_x + dx * arrow_length
                    end_y = center_y + dy * arrow_length
                    end_z = center_z + dz * arrow_length
                    
                    # Plot pipe direction arrow
                    color = colors[i % len(colors)]
                    ax_3d.quiver(center_x, center_y, center_z, 
                               dx * arrow_length, dy * arrow_length, dz * arrow_length,
                               color=color, arrow_length_ratio=0.1, linewidth=3,
                               label=f"{shape['type'].title()} {i+1}")
                    
                    # Plot center point
                    ax_3d.scatter(center_x, center_y, center_z, 
                                color=color, s=100, alpha=0.8)
                    
                    # Add direction info text (positioned to avoid overlap)
                    text_x = center_x + (i * 0.5)  # Offset text to prevent overlap
                    text_y = center_y - 1.0 - (i * 0.3)
                    text_z = center_z + 1.0
                    
                    ax_3d.text(text_x, text_y, text_z,
                             f"Az:{azimuth:.0f}°\nEl:{elevation:.0f}°",
                             fontsize=8, color=color)
            
            # Set axis labels and limits
            ax_3d.set_xlabel('X (Image Width)', fontsize=10)
            ax_3d.set_ylabel('Y (Image Height)', fontsize=10) 
            ax_3d.set_zlabel('Z (Depth)', fontsize=10)
            ax_3d.set_title('3D Pipe Directions', fontsize=12, pad=20)
            
            # Set equal aspect and reasonable limits
            max_range = 3.0
            ax_3d.set_xlim([-max_range, max_range])
            ax_3d.set_ylim([-max_range, max_range])
            ax_3d.set_zlim([-max_range, max_range])
            
            # Add legend with better positioning
            if len(detected_pipes) <= 3:  # Only show legend if not too crowded
                ax_3d.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
            
            # Add coordinate system reference
            origin = [0, 0, 0]
            ax_3d.quiver(*origin, 1, 0, 0, color='gray', alpha=0.5, linewidth=1, arrow_length_ratio=0.05)
            ax_3d.quiver(*origin, 0, 1, 0, color='gray', alpha=0.5, linewidth=1, arrow_length_ratio=0.05)
            ax_3d.quiver(*origin, 0, 0, 1, color='gray', alpha=0.5, linewidth=1, arrow_length_ratio=0.05)
            
        else:
            # Clean "no pipes" display
            axes[1, 0].text(0.5, 0.5, 'No Pipes Detected\n\n3D visualization requires\ndetected pipe structures', 
                           transform=axes[1, 0].transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 0].set_title('3D Pipe Directions', fontsize=12)
            axes[1, 0].axis('off')
        
        # Detection statistics
        circles = [s for s in shapes if s.get('type') == 'circle']
        ellipses = [s for s in shapes if s.get('type') == 'ellipse']
        pipes = [r for r in pipe_detection_results if r.get('is_pipe', False)]
        
        stats_text = f"""Detection Summary:
        
Total Shapes: {len(shapes)}
  - Circles: {len(circles)}
  - Ellipses: {len(ellipses)}
  
Pipes Detected: {len(pipes)}

Shape Details:"""
        
        for i, (shape, result) in enumerate(zip(shapes, pipe_detection_results)):
            shape_type = shape.get('type', 'circle')
            confidence = result.get('confidence', 0)
            is_pipe = result.get('is_pipe', False)
            
            if shape_type == 'ellipse':
                major = shape.get('major_axis', 0)
                minor = shape.get('minor_axis', 0)
                angle = shape.get('angle', 0)
                shape_desc = f"E{i+1}: {major}x{minor} @{angle:.0f}°"
            else:
                radius = shape.get('radius', 0)
                shape_desc = f"C{i+1}: r={radius}"
            
            pipe_status = "PIPE" if is_pipe else "No pipe"
            stats_text += f"\n  {shape_desc} - {pipe_status} ({confidence:.2f})"
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       fontfamily='monospace', fontsize=8, verticalalignment='top')
        axes[1, 1].set_title('Detection Statistics')
        axes[1, 1].axis('off')
        
        # Empty subplot for potential future use
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        results_dir = Path("results") / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / "step5_final_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive visualization saved: results/{folder_name}/step5_final_visualization.png")


def test_visualization():
    """Test the visualization with sample data"""
    from step1_data_loading import DataLoader
    from step2_circle_detection import CircleDetector
    from step3_depth_analysis import DepthAnalyzer
    from step4_pipe_detection import PipeDetector
    
    snapshot_base = Path("../snapshot")
    available_folders = [f.name for f in snapshot_base.iterdir() if f.is_dir()]
    
    if available_folders:
        test_folder = available_folders[0]
        print(f"Testing visualization with folder: {test_folder}")
        
        # Run complete pipeline
        loader = DataLoader()
        ir_image, depth_image, params = loader.load_data(snapshot_base / test_folder)
        
        detector = CircleDetector()
        enhanced_ir, shapes = detector.detect_shapes(ir_image)
        
        if shapes:
            analyzer = DepthAnalyzer()
            pipe_detector = PipeDetector()
            visualizer = Visualizer()
            
            # Analyze each shape
            pipe_detection_results = []
            for shape in shapes:
                # Pass the shape to the analyzer and pipe detector
                # The analyzer and pipe detector need to be updated to handle ellipses
                # For now, we'll just pass a placeholder for depth_info
                # This part of the pipeline needs to be adapted for ellipses
                # For demonstration, we'll simulate a depth_info for ellipses
                if shape.get('type') == 'ellipse':
                    depth_info = {
                        'centroid_3d': np.array([0, 0, -1]), # Placeholder for 3D centroid
                        'axis_3d': np.array([0.5, 0.5, 0.5]), # Placeholder for 3D axis
                        'elevation_deg': 40,
                        'azimuth_deg': 30,
                        'depth_direction': 'into_image',
                        'image_plane_angle_deg': 30,
                        'ellipse_3d_alignment': 70 # Example alignment
                    }
                else:
                    depth_info = {
                        'centroid_3d': np.array([0, 0, -1]), # Placeholder for 3D centroid
                        'axis_3d': np.array([0.5, 0, 0]), # Placeholder for 3D axis
                        'elevation_deg': 40,
                        'azimuth_deg': 30,
                        'depth_direction': 'into_image',
                        'image_plane_angle_deg': 30,
                        'ellipse_3d_alignment': 70 # Example alignment
                    }
                
                pipe_result = pipe_detector.detect_pipe(depth_info, shape)
                pipe_detection_results.append(pipe_result)
            
            # Create final visualization
            final_result = visualizer.create_final_visualization(
                enhanced_ir, shapes, pipe_detection_results, test_folder
            )
            
            print("Visualization test completed successfully!")
            print(f"Results saved in: results/{test_folder}/")
        else:
            print("No shapes detected for visualization")
    else:
        print("No snapshot folders found for testing")


if __name__ == "__main__":
    test_visualization() 