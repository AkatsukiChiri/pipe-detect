#!/usr/bin/env python3
"""
Step 2: Circle Detection Module
Enhance IR image and detect circular pipe cross-sections
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class CircleDetector:
    def __init__(self):
        """Initialize the circle detector"""
        self.enhancement_params = {
            'clahe_clip_limit': 3.0,
            'clahe_tile_size': (8, 8),
            'brightness_boost': 50,
            'contrast_alpha': 1.5
        }
        
        self.detection_params = {
            'dp': 1,
            'min_dist': 50,
            'param1': 50,
            'param2': 30,
            'min_radius': 20,
            'max_radius': 200
        }
    
    def enhance_ir_image(self, ir_image):
        """
        Enhance the IR image for better circle detection
        
        Args:
            ir_image: Input IR image (grayscale)
            
        Returns:
            enhanced_image: Enhanced IR image
        """
        # Convert to uint8 if needed
        if ir_image.dtype != np.uint8:
            ir_image = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.enhancement_params['clahe_clip_limit'],
            tileGridSize=self.enhancement_params['clahe_tile_size']
        )
        enhanced = clahe.apply(ir_image)
        
        # Brightness and contrast adjustment
        enhanced = cv2.convertScaleAbs(
            enhanced,
            alpha=self.enhancement_params['contrast_alpha'],
            beta=self.enhancement_params['brightness_boost']
        )
        
        # Additional gamma correction for dark images
        gamma = 0.8
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lookup_table)
        
        return enhanced
    
    def detect_circles(self, ir_image):
        """
        Detect circles in the IR image after enhancement
        
        Args:
            ir_image: Input IR image
            
        Returns:
            tuple: (enhanced_image, detected_circles)
        """
        # Step 1: Enhance the IR image
        enhanced_ir = self.enhance_ir_image(ir_image)
        
        # Step 2: Apply edge detection
        blurred = cv2.GaussianBlur(enhanced_ir, (9, 9), 2)
        edges = cv2.Canny(blurred, 30, 120, apertureSize=3)
        
        # Step 3: Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=self.detection_params['dp'],
            minDist=self.detection_params['min_dist'],
            param1=self.detection_params['param1'],
            param2=self.detection_params['param2'],
            minRadius=self.detection_params['min_radius'],
            maxRadius=self.detection_params['max_radius']
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Filter circles that are within image bounds
                if (x - r >= 0 and y - r >= 0 and 
                    x + r < enhanced_ir.shape[1] and y + r < enhanced_ir.shape[0]):
                    detected_circles.append({'center': (x, y), 'radius': r})
        
        print(f"Detected {len(detected_circles)} circles")
        
        return enhanced_ir, detected_circles
    
    def save_debug_visualization(self, ir_image, enhanced_ir, circles, folder_name):
        """Save debug visualization of circle detection process"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original IR image
        axes[0, 0].imshow(ir_image, cmap='gray')
        axes[0, 0].set_title('Original IR Image')
        axes[0, 0].axis('off')
        
        # Enhanced IR image
        axes[0, 1].imshow(enhanced_ir, cmap='gray')
        axes[0, 1].set_title('Enhanced IR Image')
        axes[0, 1].axis('off')
        
        # Edge detection result
        blurred = cv2.GaussianBlur(enhanced_ir, (9, 9), 2)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection Result')
        axes[1, 0].axis('off')
        
        # Circle detection result
        result_image = cv2.cvtColor(enhanced_ir, cv2.COLOR_GRAY2RGB)
        for i, circle in enumerate(circles):
            center = circle['center']
            radius = circle['radius']
            
            # Draw circle and center
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)
            cv2.circle(result_image, center, 2, (0, 255, 0), 3)
            
            # Add circle number
            cv2.putText(result_image, f'Circle {i+1}', 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        axes[1, 1].imshow(result_image)
        axes[1, 1].set_title(f'Detected Circles ({len(circles)} found)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save result
        results_dir = Path("results") / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / "step2_circle_detection.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug output saved: results/{folder_name}/step2_circle_detection.png")


def test_circle_detection():
    """Test the circle detection with sample data"""
    from step1_data_loading import DataLoader
    
    snapshot_base = Path("../snapshot")
    available_folders = [f.name for f in snapshot_base.iterdir() if f.is_dir()]
    
    if available_folders:
        test_folder = available_folders[0]
        print(f"Testing circle detection with folder: {test_folder}")
        
        # Load data
        loader = DataLoader()
        ir_image, depth_image, params = loader.load_data(snapshot_base / test_folder)
        
        # Detect circles
        detector = CircleDetector()
        enhanced_ir, circles = detector.detect_circles(ir_image)
        
        # Save debug visualization
        detector.save_debug_visualization(ir_image, enhanced_ir, circles, test_folder)
        
        print("Circle detection test completed successfully!")
        print(f"Found {len(circles)} circles:")
        for i, circle in enumerate(circles):
            print(f"  Circle {i+1}: center={circle['center']}, radius={circle['radius']}")
    else:
        print("No snapshot folders found for testing")


if __name__ == "__main__":
    test_circle_detection() 