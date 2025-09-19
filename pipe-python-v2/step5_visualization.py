#!/usr/bin/env python3
"""
Step 5: Final Visualization Module
Create comprehensive visualization with color-coded results and direction arrows
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    def __init__(self):
        """Initialize the visualizer"""
        self.colors = {
            'circle_only': (0, 0, 255),      # Red for circles without pipes
            'pipe_detected': (0, 255, 255),  # Yellow for circles with pipes
            'direction_arrow': (255, 0, 0),  # Blue for direction arrows
            'center_point': (255, 255, 255)  # White for center points
        }
        
        self.visualization_params = {
            'circle_thickness': 2,
            'center_radius': 3,
            'arrow_length': 100,
            'arrow_thickness': 3,
            'text_font': cv2.FONT_HERSHEY_SIMPLEX,
            'text_scale': 0.6,
            'text_thickness': 2
        }
    
    def create_final_visualization(self, enhanced_ir, circles, pipe_detection_results, folder_name):
        """
        Create the final visualization with color-coded results
        
        Args:
            enhanced_ir: Enhanced IR image
            circles: List of detected circles
            pipe_detection_results: List of pipe detection results
            folder_name: Name of the folder for saving results
            
        Returns:
            result_image: Final visualization image
        """
        # Create result image (convert to color)
        result_image = cv2.cvtColor(enhanced_ir, cv2.COLOR_GRAY2BGR)
        
        # Process each circle and its detection result
        for i, (circle, pipe_result) in enumerate(zip(circles, pipe_detection_results)):
            center = circle['center']
            radius = circle['radius']
            
            # Determine colors based on detection result
            if pipe_result['is_pipe']:
                circle_color = self.colors['pipe_detected']
                label = f"Pipe {i+1}"
            else:
                circle_color = self.colors['circle_only']
                label = f"Circle {i+1}"
            
            # Draw circle
            cv2.circle(result_image, center, radius, circle_color, 
                      self.visualization_params['circle_thickness'])
            
            # Draw center point
            cv2.circle(result_image, center, self.visualization_params['center_radius'], 
                      circle_color, -1)
            
            # Add label
            label_pos = (center[0] - 30, center[1] - radius - 10)
            cv2.putText(result_image, label, label_pos,
                       self.visualization_params['text_font'],
                       self.visualization_params['text_scale'],
                       circle_color,
                       self.visualization_params['text_thickness'])
            
            # Draw 3D direction visualization if pipe detected
            if pipe_result['is_pipe'] and pipe_result['direction'] is not None:
                direction_info = pipe_result['direction']
                pipe_axis_info = pipe_result['pipe_axis']
                
                # Draw 3D pipe direction
                self._draw_3d_pipe_direction(result_image, center, direction_info, pipe_axis_info)
                
                # Add comprehensive 3D direction annotations
                self._add_3d_direction_annotations(result_image, center, radius, direction_info)
        
        # Add legend
        self._add_legend(result_image)
        
        # Save comprehensive visualization
        self._save_comprehensive_visualization(enhanced_ir, circles, pipe_detection_results, 
                                             result_image, folder_name)
        
        return result_image
    
    def _draw_3d_pipe_direction(self, image, center, direction_info, pipe_axis_info):
        """
        Draw 3D pipe direction visualization
        
        Args:
            image: Image to draw on
            center: Center point (x, y) 
            direction_info: 3D direction information
            pipe_axis_info: Additional pipe axis information
        """
        if not direction_info or not pipe_axis_info:
            return
        
        # Extract 3D direction information
        elevation = direction_info['elevation_deg']
        azimuth = direction_info['azimuth_deg']
        depth_direction = direction_info['depth_direction']
        image_plane_angle = direction_info['image_plane_angle_deg']
        
        # Calculate visualization parameters based on 3D direction
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
        
        # Draw main pipe axis line
        cv2.line(image, (end2_x, end2_y), (end1_x, end1_y), 
                arrow_color, self.visualization_params['arrow_thickness'])
        
        # Draw depth direction indicator (perpendicular to pipe axis)
        perp_angle = angle_rad + np.pi/2
        depth_length = 20
        
        # Calculate depth indicator size based on elevation
        depth_indicator_scale = abs(np.cos(np.radians(elevation)))
        scaled_depth_length = int(depth_length * max(0.2, depth_indicator_scale))
        
        depth_end_x = int(center[0] + scaled_depth_length * np.cos(perp_angle))
        depth_end_y = int(center[1] + scaled_depth_length * np.sin(perp_angle))
        
        # Draw depth direction indicator
        if abs(elevation) > 45:  # High elevation - show as circle indicating depth
            circle_radius = max(8, int(scaled_depth_length * 0.6))
            cv2.circle(image, center, circle_radius, depth_color, 2)
            
            # Add direction symbol in the circle
            if depth_direction == "into_image":
                # Draw '+' for into image
                cv2.line(image, (center[0]-5, center[1]), (center[0]+5, center[1]), depth_color, 2)
                cv2.line(image, (center[0], center[1]-5), (center[0], center[1]+5), depth_color, 2)
            else:
                # Draw 'x' for out of image
                cv2.line(image, (center[0]-4, center[1]-4), (center[0]+4, center[1]+4), depth_color, 2)
                cv2.line(image, (center[0]-4, center[1]+4), (center[0]+4, center[1]-4), depth_color, 2)
        
        # Draw arrowheads on the main axis
        arrowhead_length = 12
        arrowhead_angle = np.pi / 6
        
        # Arrowhead 1
        head1_x1 = int(end1_x - arrowhead_length * np.cos(angle_rad - arrowhead_angle))
        head1_y1 = int(end1_y - arrowhead_length * np.sin(angle_rad - arrowhead_angle))
        head1_x2 = int(end1_x - arrowhead_length * np.cos(angle_rad + arrowhead_angle))
        head1_y2 = int(end1_y - arrowhead_length * np.sin(angle_rad + arrowhead_angle))
        
        cv2.line(image, (end1_x, end1_y), (head1_x1, head1_y1), arrow_color, 2)
        cv2.line(image, (end1_x, end1_y), (head1_x2, head1_y2), arrow_color, 2)
        
        # Arrowhead 2
        head2_x1 = int(end2_x + arrowhead_length * np.cos(angle_rad - arrowhead_angle))
        head2_y1 = int(end2_y + arrowhead_length * np.sin(angle_rad - arrowhead_angle))
        head2_x2 = int(end2_x + arrowhead_length * np.cos(angle_rad + arrowhead_angle))
        head2_y2 = int(end2_y + arrowhead_length * np.sin(angle_rad + arrowhead_angle))
        
        cv2.line(image, (end2_x, end2_y), (head2_x1, head2_y1), arrow_color, 2)
        cv2.line(image, (end2_x, end2_y), (head2_x2, head2_y2), arrow_color, 2)
    
    def _add_3d_direction_annotations(self, image, center, radius, direction_info):
        """
        Add comprehensive 3D direction annotations to the image
        """
        if not direction_info:
            return
        
        # Extract direction information
        elevation = direction_info['elevation_deg']
        azimuth = direction_info['azimuth_deg']
        depth_direction = direction_info['depth_direction']
        image_plane_angle = direction_info['image_plane_angle_deg']
        axis_3d = direction_info['axis_3d_normalized']
        
        # Position for annotations (to the right of the circle)
        base_x = center[0] + radius + 15
        base_y = center[1] - radius
        
        # Create a semi-transparent background for text
        text_bg_color = (0, 0, 0)
        text_color = (255, 255, 255)
        font = self.visualization_params['text_font']
        scale = 0.45
        thickness = 1
        
        # Prepare annotation lines
        annotations = [
            f"3D PIPE DIRECTION:",
            f"Azimuth: {azimuth:.1f}°",
            f"Elevation: {elevation:.1f}°", 
            f"Depth: {depth_direction}",
            f"Image∠: {image_plane_angle:.1f}°",
            f"Axis: [{axis_3d[0]:.2f},{axis_3d[1]:.2f},{axis_3d[2]:.2f}]"
        ]
        
        # Calculate text background size
        line_height = 15
        max_width = 0
        for text in annotations:
            (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
            max_width = max(max_width, text_width)
        
        # Draw background rectangle
        bg_top_left = (base_x - 5, base_y - 5)
        bg_bottom_right = (base_x + max_width + 10, base_y + len(annotations) * line_height + 5)
        cv2.rectangle(image, bg_top_left, bg_bottom_right, text_bg_color, -1)
        cv2.rectangle(image, bg_top_left, bg_bottom_right, (100, 100, 100), 1)
        
        # Draw annotations
        for i, text in enumerate(annotations):
            y_pos = base_y + (i + 1) * line_height
            
            # Use different colors for different types of information
            if i == 0:  # Title
                color = (0, 255, 255)  # Cyan
                thickness = 2
            elif "Elevation" in text:
                color = (0, 255, 0)   # Green
            elif "Azimuth" in text:
                color = (255, 255, 0) # Yellow
            elif "Depth" in text:
                if "into_image" in text:
                    color = (255, 100, 0)  # Orange
                else:
                    color = (0, 100, 255)  # Blue
            else:
                color = text_color     # White
            
            cv2.putText(image, text, (base_x, y_pos), font, scale, color, thickness)
        
        # Add interpretation text
        interpretation_y = base_y + len(annotations) * line_height + 25
        
        # Determine pipe orientation description
        if abs(elevation) > 70:
            if depth_direction == "into_image":
                orientation = "→ Camera"
            else:
                orientation = "← Camera"
        elif abs(elevation) < 20:
            orientation = "|| Image plane"
        else:
            if depth_direction == "into_image":
                orientation = "↗ Toward camera"
            else:
                orientation = "↙ Away camera"
        
        interpretation_text = f"Interpretation: {orientation}"
        cv2.putText(image, interpretation_text, (base_x, interpretation_y), 
                   font, scale, (255, 255, 255), thickness)
    
    def _add_legend(self, image):
        """Add legend to the image"""
        legend_x = 10
        legend_y = image.shape[0] - 140
        
        # Background for legend
        cv2.rectangle(image, (legend_x - 5, legend_y - 25), 
                     (legend_x + 220, legend_y + 110), (0, 0, 0), -1)
        cv2.rectangle(image, (legend_x - 5, legend_y - 25), 
                     (legend_x + 220, legend_y + 110), (255, 255, 255), 1)
        
        # Legend items
        legend_items = [
            ("Red: Circle only", self.colors['circle_only']),
            ("Yellow: Circle + Pipe", self.colors['pipe_detected']),
            ("Orange: Into image", (255, 100, 0)),
            ("Blue: Out of image", (0, 100, 255)),
            ("+/x: Depth direction", (255, 255, 255))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            
            # Draw color sample
            cv2.circle(image, (legend_x + 10, y_pos), 5, color, -1)
            
            # Draw text
            cv2.putText(image, text, (legend_x + 25, y_pos + 5),
                       self.visualization_params['text_font'],
                       self.visualization_params['text_scale'] * 0.8,
                       (255, 255, 255),
                       self.visualization_params['text_thickness'])
    
    def _save_comprehensive_visualization(self, enhanced_ir, circles, pipe_detection_results, 
                                        result_image, folder_name):
        """Save comprehensive visualization with all analysis steps"""
        # Create a large figure with multiple subplots including 3D visualization
        fig = plt.figure(figsize=(24, 14))
        
        # Main result (large subplot)
        ax_main = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2)
        ax_main.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax_main.set_title('Final Detection Result', fontsize=14, fontweight='bold')
        ax_main.axis('off')
        
        # 3D Direction Visualization (new!)
        ax_3d = plt.subplot2grid((3, 5), (0, 2), rowspan=2, projection='3d')
        self._create_3d_direction_plot(ax_3d, circles, pipe_detection_results)
        
        # Enhanced IR image
        ax_ir = plt.subplot2grid((3, 5), (0, 3))
        ax_ir.imshow(enhanced_ir, cmap='gray')
        ax_ir.set_title('Enhanced IR Image')
        ax_ir.axis('off')
        
        # Detection summary
        ax_summary = plt.subplot2grid((3, 5), (0, 4))
        self._create_detection_summary(ax_summary, circles, pipe_detection_results)
        
        # Overall statistics
        ax_stats = plt.subplot2grid((3, 5), (1, 3), colspan=2)
        self._create_overall_statistics(ax_stats, circles, pipe_detection_results)
        
        # Individual circle analysis
        n_circles = len(circles)
        if n_circles > 0:
            # Show first 3 circles in detail in bottom row
            for i in range(min(3, n_circles)):
                ax_circle = plt.subplot2grid((3, 5), (2, i))
                self._visualize_single_circle_analysis(ax_circle, circles[i], 
                                                     pipe_detection_results[i], i+1)
        
        # 3D coordinate explanation
        ax_coord_explain = plt.subplot2grid((3, 5), (2, 3), colspan=2)
        self._create_3d_coordinate_explanation(ax_coord_explain)
        
        plt.tight_layout()
        
        # Save result
        results_dir = Path("results") / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / "step5_final_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save just the result image
        cv2.imwrite(str(results_dir / "final_result.png"), result_image)
        
        print(f"Final visualization saved: results/{folder_name}/step5_final_visualization.png")
        print(f"Result image saved: results/{folder_name}/final_result.png")
    
    def _create_detection_summary(self, ax, circles, pipe_detection_results):
        """Create detection summary text"""
        total_circles = len(circles)
        total_pipes = sum(1 for result in pipe_detection_results if result['is_pipe'])
        
        summary_text = f"Detection Summary\n\n"
        summary_text += f"Total Circles: {total_circles}\n"
        summary_text += f"Detected Pipes: {total_pipes}\n"
        summary_text += f"Circle Only: {total_circles - total_pipes}\n\n"
        
        summary_text += "Details:\n"
        for i, (circle, result) in enumerate(zip(circles, pipe_detection_results)):
            status = "Pipe" if result['is_pipe'] else "Circle"
            confidence = result['confidence']
            summary_text += f"#{i+1}: {status} ({confidence:.2f})\n"
            
            if result['is_pipe'] and result['direction'] is not None:
                direction_info = result['direction']
                summary_text += f"     Elev: {direction_info['elevation_deg']:.1f}°\n"
                summary_text += f"     {direction_info['depth_direction']}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace', fontsize=9)
        ax.set_title('Detection Summary')
        ax.axis('off')
    
    def _visualize_single_circle_analysis(self, ax, circle, pipe_result, circle_num):
        """Visualize analysis for a single circle"""
        center = circle['center']
        radius = circle['radius']
        
        # Create a small visualization
        title = f"Circle {circle_num}: "
        if pipe_result['is_pipe']:
            title += f"PIPE (conf: {pipe_result['confidence']:.2f})"
            color = 'yellow'
        else:
            title += f"Circle Only (conf: {pipe_result['confidence']:.2f})"
            color = 'red'
        
        # Simple text summary
        details = f"Center: ({center[0]}, {center[1]})\n"
        details += f"Radius: {radius}\n"
        details += f"Status: {'Pipe' if pipe_result['is_pipe'] else 'Circle'}\n"
        details += f"Confidence: {pipe_result['confidence']:.2f}\n"
        
        if pipe_result['is_pipe'] and pipe_result['direction'] is not None:
            direction_info = pipe_result['direction']
            details += f"Elevation: {direction_info['elevation_deg']:.1f}°\n"
            details += f"Depth: {direction_info['depth_direction']}\n"
        
        ax.text(0.05, 0.95, details, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace', fontsize=8)
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    def _create_overall_statistics(self, ax, circles, pipe_detection_results):
        """Create overall statistics visualization"""
        if not circles:
            ax.text(0.5, 0.5, 'No circles detected', ha='center', va='center')
            ax.set_title('Overall Statistics')
            ax.axis('off')
            return
        
        # Calculate statistics
        confidences = [result['confidence'] for result in pipe_detection_results]
        pipe_confidences = [result['confidence'] for result in pipe_detection_results if result['is_pipe']]
        circle_confidences = [result['confidence'] for result in pipe_detection_results if not result['is_pipe']]
        
        # Create bar chart
        categories = ['All Detections', 'Pipes Only', 'Circles Only']
        mean_confidences = [
            np.mean(confidences) if confidences else 0,
            np.mean(pipe_confidences) if pipe_confidences else 0,
            np.mean(circle_confidences) if circle_confidences else 0
        ]
        
        bars = ax.bar(categories, mean_confidences, color=['blue', 'yellow', 'red'], alpha=0.7)
        ax.set_ylabel('Mean Confidence')
        ax.set_title('Detection Confidence Statistics')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, mean_confidences):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _create_3d_direction_plot(self, ax, circles, pipe_detection_results):
        """Create 3D plot showing pipe directions"""
        ax.set_title('3D Pipe Directions', fontsize=12, fontweight='bold')
        
        # Set up 3D coordinate system
        ax.set_xlabel('X (Image Right)', fontsize=10)
        ax.set_ylabel('Y (Image Down)', fontsize=10) 
        ax.set_zlabel('Z (Depth)', fontsize=10)
        
        # Set equal aspect ratio and reasonable limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        # Draw coordinate system reference
        # X axis (red)
        ax.quiver(0, 0, 0, 0.8, 0, 0, color='red', arrow_length_ratio=0.1, alpha=0.6, linewidth=2)
        ax.text(0.9, 0, 0, 'X', color='red', fontsize=10)
        
        # Y axis (green)  
        ax.quiver(0, 0, 0, 0, 0.8, 0, color='green', arrow_length_ratio=0.1, alpha=0.6, linewidth=2)
        ax.text(0, 0.9, 0, 'Y', color='green', fontsize=10)
        
        # Z axis (blue)
        ax.quiver(0, 0, 0, 0, 0, 0.8, color='blue', arrow_length_ratio=0.1, alpha=0.6, linewidth=2)
        ax.text(0, 0, 0.9, 'Z', color='blue', fontsize=10)
        
        # Draw detected pipes
        colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        for i, (circle, result) in enumerate(zip(circles, pipe_detection_results)):
            if result['is_pipe'] and result['direction'] is not None:
                direction_info = result['direction']
                axis_3d = direction_info['axis_3d_normalized']
                depth_direction = direction_info['depth_direction']
                
                # Choose color
                color = colors[i % len(colors)]
                
                # Draw pipe direction vector (both directions)
                scale = 0.7
                ax.quiver(0, 0, 0, 
                         axis_3d[0] * scale, axis_3d[1] * scale, axis_3d[2] * scale,
                         color=color, arrow_length_ratio=0.15, linewidth=3, alpha=0.8)
                ax.quiver(0, 0, 0,
                         -axis_3d[0] * scale, -axis_3d[1] * scale, -axis_3d[2] * scale, 
                         color=color, arrow_length_ratio=0.15, linewidth=3, alpha=0.8)
                
                # Add text label
                label_pos = np.array(axis_3d) * 0.8
                ax.text(label_pos[0], label_pos[1], label_pos[2], 
                       f'Pipe {i+1}', color=color, fontsize=9, fontweight='bold')
                
                # Add direction info
                elev = direction_info['elevation_deg']
                azim = direction_info['azimuth_deg']
                info_text = f'E:{elev:.0f}° A:{azim:.0f}°'
                info_pos = np.array(axis_3d) * 0.6
                ax.text(info_pos[0], info_pos[1], info_pos[2], 
                       info_text, color=color, fontsize=8)
        
        # Add camera position indicator
        ax.scatter([0], [0], [-0.9], color='black', s=100, marker='^', alpha=0.7)
        ax.text(0, 0, -0.8, 'Camera', color='black', fontsize=9, ha='center')
        
        # Add image plane
        xx, yy = np.meshgrid(np.linspace(-0.8, 0.8, 5), np.linspace(-0.8, 0.8, 5))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        ax.text(0, 0, 0.1, 'Image Plane', color='gray', fontsize=9, ha='center')
        
        # Set view angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def _create_3d_coordinate_explanation(self, ax):
        """Create explanation of 3D coordinate system"""
        ax.set_title('3D Coordinate System Explanation', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        explanation_text = """
3D Coordinate System:
• X-axis (Red): Image horizontal (left ← → right)
• Y-axis (Green): Image vertical (up ← → down)  
• Z-axis (Blue): Depth (camera ← → scene)

Pipe Direction Interpretation:
• Elevation > 70°: Pipe points toward/away from camera
• Elevation < 20°: Pipe lies parallel to image plane
• Azimuth: Direction within image plane

Color Coding:
• Orange: Pipe pointing toward camera (into_image)
• Blue: Pipe pointing away from camera (out_of_image)

Direction Vector [x,y,z]:
• Large |z|: Strong depth component
• Large |x|,|y|: Strong image plane component
        """
        
        ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


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
        enhanced_ir, circles = detector.detect_circles(ir_image)
        
        if circles:
            analyzer = DepthAnalyzer()
            pipe_detector = PipeDetector()
            visualizer = Visualizer()
            
            # Analyze each circle
            pipe_detection_results = []
            for circle in circles:
                depth_info = analyzer.analyze_circle_depth(depth_image, circle, radius_multiplier=3)
                pipe_result = pipe_detector.detect_pipe(depth_info, circle)
                pipe_detection_results.append(pipe_result)
            
            # Create final visualization
            final_result = visualizer.create_final_visualization(
                enhanced_ir, circles, pipe_detection_results, test_folder
            )
            
            print("Visualization test completed successfully!")
            print(f"Results saved in: results/{test_folder}/")
        else:
            print("No circles detected for visualization")
    else:
        print("No snapshot folders found for testing")


if __name__ == "__main__":
    test_visualization() 