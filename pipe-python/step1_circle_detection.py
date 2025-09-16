"""
Step 1: Circle and Ellipse Detection in RGB Images
This module detects circular and elliptical shapes in RGB images using various OpenCV techniques.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os


class CircleEllipseDetector:
    def __init__(self, min_radius: int = 20, max_radius: int = 200):
        """
        Initialize the circle and ellipse detector.
        
        Args:
            min_radius: Minimum radius for circle detection
            max_radius: Maximum radius for circle detection
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better circle/ellipse detection.

        Args:
            image: Input RGB image

        Returns:
            Preprocessed contour image (binary edge map)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # # Apply Gaussian blur to reduce noise
        # blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Use Canny edge detector to get contour image
        # Raise the thresholds for stricter edge detection
        edges = cv2.Canny(gray, 50, 150)

        # 展示处理后的灰度图 (注释掉以避免在无GUI环境中出错)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(edges, cmap='gray')
        # plt.title('Preprocessed Grayscale (Edge Map)')
        # plt.axis('off')
        # plt.show()

        return edges
    
    def detect_circles_hough(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect circles using Hough Circle Transform, with stricter filtering to reduce false positives.
        
        Args:
            image: Input RGB image
            
        Returns:
            List of detected circles as (x, y, radius)
        """
        # Preprocess image to get edge map
        edge_img = self.preprocess_image(image)

        # Use a more restrictive HoughCircles configuration
        circles = cv2.HoughCircles(
            edge_img,
            cv2.HOUGH_GRADIENT,
            dp=1.2,                # Slightly increase dp for more robust detection
            minDist=80,            # Increase minDist to avoid multiple detections of same circle
            param1=100,            # Higher Canny high threshold for stricter edge detection
            param2=60,             # Increase accumulator threshold to require stronger evidence
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Additional filtering: ignore circles near the image border
                h, w = edge_img.shape
                if (r < self.min_radius or r > self.max_radius):
                    continue
                if x - r < 0 or x + r > w or y - r < 0 or y + r > h:
                    continue
                # Optionally: filter by mean edge strength in the annulus
                mask = np.zeros_like(edge_img, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, 2)
                edge_strength = cv2.mean(edge_img, mask=mask)[0]
                if edge_strength < 30:  # Require strong edge in the circle perimeter
                    continue
                detected_circles.append((x, y, r))
        return detected_circles
    
    def detect_ellipses_contour(self, image: np.ndarray) -> List[Dict]:
        """
        Detect ellipses using contour detection and ellipse fitting.
        改进：减少误检，优先找到最明显的椭圆（面积最大、长宽比合适、边缘清晰）。
        """
        gray = self.preprocess_image(image)

        # Edge detection with slightly higher thresholds for less clutter
        edges = cv2.Canny(gray, 80, 200)

        # Morphological closing to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        detected_ellipses = []
        ellipse_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if len(contour) >= 20 and area > 1000:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0

                    # 只保留近似圆或合理椭圆（长宽比不过分极端）
                    if (self.min_radius * 2 <= major_axis <= self.max_radius * 2 and
                        self.min_radius * 2 <= minor_axis <= self.max_radius * 2 and
                        0.6 <= aspect_ratio <= 1.5):

                        # 计算轮廓与椭圆的拟合程度（IoU 或轮廓点到椭圆的平均距离）
                        ellipse_mask = np.zeros_like(gray, dtype=np.uint8)
                        cv2.ellipse(ellipse_mask, (tuple(map(int, center)), tuple(map(int, axes)), angle), 255, -1)
                        contour_mask = np.zeros_like(gray, dtype=np.uint8)
                        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                        intersection = np.logical_and(ellipse_mask, contour_mask).sum()
                        union = np.logical_or(ellipse_mask, contour_mask).sum()
                        iou = intersection / union if union > 0 else 0

                        # 只保留IoU较高的椭圆
                        if iou > 0.5:
                            ellipse_candidates.append({
                                'center': center,
                                'axes': axes,
                                'angle': angle,
                                'contour': contour,
                                'area': area,
                                'iou': iou,
                                'aspect_ratio': aspect_ratio
                            })
                except cv2.error:
                    continue

        # 优先返回面积最大、IoU最高的前几个椭圆（比如最多3个）
        ellipse_candidates = sorted(ellipse_candidates, key=lambda e: (e['area'] * e['iou']), reverse=True)
        detected_ellipses = ellipse_candidates[:3]

        return detected_ellipses
    
    def visualize_detections(self, image: np.ndarray, circles: List[Tuple[int, int, int]], 
                           ellipses: List[Dict], output_path: str) -> np.ndarray:
        """
        Visualize detected circles and ellipses on the original image.
        
        Args:
            image: Original RGB image
            circles: List of detected circles
            ellipses: List of detected ellipses
            output_path: Path to save the visualization
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        # Draw circles
        for (x, y, r) in circles:
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result_image, (x, y), 2, (0, 255, 0), 3)
            cv2.putText(result_image, f'C({x},{y})', (x-30, y-r-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw ellipses
        for i, ellipse_data in enumerate(ellipses):
            ellipse_params = (tuple(map(int, ellipse_data['center'])), 
                            tuple(map(int, ellipse_data['axes'])), 
                            ellipse_data['angle'])
            cv2.ellipse(result_image, ellipse_params, (255, 0, 0), 2)
            
            center_x, center_y = map(int, ellipse_data['center'])
            cv2.circle(result_image, (center_x, center_y), 2, (255, 0, 0), 3)
            cv2.putText(result_image, f'E{i}({center_x},{center_y})', 
                       (center_x-30, center_y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Save visualization
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        return result_image


def process_rgb_image(image_path: str, output_dir: str, filename_prefix: str = ""):
    """
    Process a single RGB image to detect circles and ellipses.
    
    Args:
        image_path: Path to the input RGB image
        output_dir: Directory to save results
        filename_prefix: Prefix for output filenames
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize detector
    detector = CircleEllipseDetector(min_radius=20, max_radius=200)
    
    # Detect circles and ellipses
    circles = detector.detect_circles_hough(image_rgb)
    ellipses = detector.detect_ellipses_contour(image_rgb)
    
    print(f"Detected {len(circles)} circles and {len(ellipses)} ellipses")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize results
    output_path = os.path.join(output_dir, f"{filename_prefix}step1_detections.jpg")
    result_image = detector.visualize_detections(image_rgb, circles, ellipses, output_path)
    
    # Save detection data
    detection_data = {
        'circles': circles,
        'ellipses': ellipses,
        'image_shape': image_rgb.shape
    }
    
    return detection_data, result_image


if __name__ == "__main__":
    # Example usage
    image_path = "../snapshot/DS87_2025_09_16_19_58_53_0268/Color_00000000.jpg"
    output_dir = "../results/step1_output"
    
    if os.path.exists(image_path):
        detection_data, result_image = process_rgb_image(image_path, output_dir, "test_")
        print("Step 1 completed successfully!")
        print(f"Results saved to: {output_dir}")
    else:
        print(f"Image not found: {image_path}") 