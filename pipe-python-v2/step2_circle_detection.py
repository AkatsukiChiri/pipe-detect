#!/usr/bin/env python3
"""
Step 2: Circle and Ellipse Detection Module
Enhance IR image and detect circular and elliptical pipe cross-sections
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class CircleEllipseDetector:
    def __init__(self):
        """Initialize the circle and ellipse detector"""
        self.enhancement_params = {
            'clahe_clip_limit': 3.0,
            'clahe_tile_size': (8, 8),
            'brightness_boost': 50,
            'contrast_alpha': 1.5
        }
        
        self.circle_detection_params = {
            'dp': 1,
            'min_dist': 50,
            'param1': 50,
            'param2': 30,
            'min_radius': 20,
            'max_radius': 200
        }
        
        self.ellipse_detection_params = {
            'min_contour_area': 300,    # Minimum contour area (降低以检测更多椭圆)
            'max_contour_area': 50000,  # Maximum contour area (增大以包含更大椭圆)
            'min_axis_ratio': 0.2,      # Minimum ratio of minor to major axis (降低以检测更扁的椭圆)
            'max_axis_ratio': 1.0,      # Maximum ratio (1.0 = circle)
            'quality_threshold': 0.25,  # 椭圆质量阈值 (降低以包含更多候选)
            'canny_low': 30,            # Canny边缘检测低阈值 (降低以检测更多边缘)
            'canny_high': 120,          # Canny边缘检测高阈值
            'sobel_threshold': 60,      # Sobel阈值 (降低以检测更多边缘)
            'laplacian_threshold': 30,  # Laplacian阈值 (降低以检测更多边缘)
        }
    
    def enhance_ir_image(self, ir_image):
        """
        准备IR图像用于形状检测（移除提亮处理）
        
        Args:
            ir_image: Input IR image (grayscale)
            
        Returns:
            processed_image: 处理后的IR图像
        """
        # Convert to uint8 if needed
        if ir_image.dtype != np.uint8:
            ir_image = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 不进行提亮处理，直接返回原始图像
        return ir_image
    
    def detect_circles_and_ellipses(self, ir_image):
        """
        同时检测圆形和椭圆，椭圆检测前进行边缘检测
        
        Args:
            ir_image: Input IR image
            
        Returns:
            tuple: (processed_image, detected_shapes) where detected_shapes contains both circles and ellipses
        """
        # Step 1: 处理IR图像（不进行提亮）
        processed_ir = self.enhance_ir_image(ir_image)
        
        # Step 2: 检测圆形（使用HoughCircles）
        circles = self._detect_circles(processed_ir)
        
        # Step 3: 检测椭圆（使用边缘检测+轮廓分析）
        ellipses = self._detect_ellipses_with_edge_detection(processed_ir)
        
        # Step 4: 合并和过滤结果
        all_shapes = self._combine_and_filter_shapes(circles, ellipses, processed_ir.shape)
        
        print(f"Detected {len(circles)} circles and {len(ellipses)} ellipses")
        print(f"Total unique shapes after filtering: {len(all_shapes)}")
        
        return processed_ir, all_shapes
    
    def _detect_circles(self, processed_ir):
        """使用HoughCircles检测圆形"""
        # Apply edge detection for circle detection
        blurred = cv2.GaussianBlur(processed_ir, (9, 9), 2)
        edges = cv2.Canny(blurred, 30, 120, apertureSize=3)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_detection_params['dp'],
            minDist=self.circle_detection_params['min_dist'],
            param1=self.circle_detection_params['param1'],
            param2=self.circle_detection_params['param2'],
            minRadius=self.circle_detection_params['min_radius'],
            maxRadius=self.circle_detection_params['max_radius']
        )
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Filter circles that are within image bounds
                if (x - r >= 0 and y - r >= 0 and 
                    x + r < processed_ir.shape[1] and y + r < processed_ir.shape[0]):
                    detected_circles.append({
                        'type': 'circle',
                        'center': (x, y), 
                        'radius': r,
                        'major_axis': r * 2,  # For compatibility with ellipse format
                        'minor_axis': r * 2,
                        'angle': 0  # Circles have no rotation
                    })
        
        return detected_circles
    
    def _detect_ellipses_with_edge_detection(self, processed_ir):
        """
        改进的椭圆检测算法 - 更稳健，能检测圆形和椭圆
        使用多阶段检测策略：阈值分割 + 边缘检测 + 轮廓分析
        """
        detected_ellipses = []
        height, width = processed_ir.shape
        
        # Step 1: 多阈值分割方法
        binary_images = []
        
        # 方法1: 自适应阈值
        adaptive_thresh = cv2.adaptiveThreshold(processed_ir, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 15, 2)
        binary_images.append(adaptive_thresh)
        
        # 方法2: Otsu阈值
        _, otsu_thresh = cv2.threshold(processed_ir, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_images.append(otsu_thresh)
        
        # 方法3: 多级阈值（用于不同亮度的目标）
        hist = cv2.calcHist([processed_ir], [0], None, [256], [0, 256])
        # 使用直方图找到可能的阈值
        threshold_values = [
            int(np.percentile(processed_ir[processed_ir > 0], 25)),
            int(np.percentile(processed_ir[processed_ir > 0], 50)),
            int(np.percentile(processed_ir[processed_ir > 0], 75))
        ]
        
        for thresh_val in threshold_values:
            if thresh_val > 10:  # 避免过低的阈值
                _, binary = cv2.threshold(processed_ir, thresh_val, 255, cv2.THRESH_BINARY)
                binary_images.append(binary)
        
        # Step 2: 边缘检测方法
        # 简化但有效的Canny边缘检测
        blurred = cv2.GaussianBlur(processed_ir, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Step 3: 对每个二值图像检测轮廓
        all_contours = []
        
        for binary_img in binary_images:
            # 形态学操作清理噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        # 也从边缘图像中寻找轮廓
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(edge_contours)
        
        # Step 4: 分析每个轮廓
        for contour in all_contours:
            area = cv2.contourArea(contour)
            
            # 更宽松的面积过滤
            min_area = 100  # 降低最小面积要求
            max_area = (width * height) // 4  # 最大面积不超过图像的1/4
            
            if area < min_area or area > max_area:
                continue
            
            # 轮廓需要至少5个点来拟合椭圆
            if len(contour) < 5:
                continue
            
            try:
                # 拟合椭圆
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (axis1, axis2), angle = ellipse
                
                # 确保major_axis >= minor_axis
                if axis1 > axis2:
                    major_axis, minor_axis = axis1, axis2
                else:
                    major_axis, minor_axis = axis2, axis1
                    angle = (angle + 90) % 180
                
                center_x, center_y = int(center_x), int(center_y)
                
                # 检查椭圆中心是否在图像内
                if not (0 <= center_x < width and 0 <= center_y < height):
                    continue
                
                # 检查椭圆是否合理（不要太大）
                max_radius = max(major_axis, minor_axis) / 2
                if max_radius > min(width, height) / 3:  # 椭圆不能太大
                    continue
                
                # 计算轴比
                axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
                
                # 更宽松的轴比限制（接受更多椭圆形状）
                if axis_ratio < 0.2 or axis_ratio > 1.0:  # 0.2到1.0之间
                    continue
                
                # 简化的质量评估
                quality_score = self._simple_ellipse_quality(contour, ellipse, processed_ir)
                
                # 降低质量阈值，更容易检测
                if quality_score > 0.3:  # 从之前的高阈值降低到0.3
                    # 检查是否与已检测的椭圆重复
                    is_duplicate = False
                    for existing in detected_ellipses:
                        existing_center = existing['center']
                        distance = np.sqrt((center_x - existing_center[0])**2 + 
                                         (center_y - existing_center[1])**2)
                        if distance < max_radius * 0.8:  # 如果中心距离很近，认为是重复
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        detected_ellipses.append({
                            'type': 'ellipse',
                            'center': (center_x, center_y),
                            'major_axis': int(major_axis),
                            'minor_axis': int(minor_axis),
                            'angle': angle,
                            'radius': int(max_radius),  # For backward compatibility
                            'axis_ratio': axis_ratio,
                            'area': area,
                            'quality_score': quality_score
                        })
                        
            except cv2.error as e:
                # 椭圆拟合失败时跳过
                continue
        
        # 按质量分数排序，保留最好的结果
        detected_ellipses.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return detected_ellipses[:10]  # 最多返回10个最好的椭圆
    
    def _simple_ellipse_quality(self, contour, ellipse, image):
        """
        简化的椭圆质量评估
        
        Args:
            contour: 轮廓点
            ellipse: 椭圆参数
            image: 原始图像
            
        Returns:
            float: 质量分数 (0-1)
        """
        try:
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            center_x, center_y = int(center_x), int(center_y)
            
            # 1. 轮廓面积 vs 椭圆面积比较
            contour_area = cv2.contourArea(contour)
            ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)
            
            if ellipse_area == 0:
                return 0.0
            
            area_ratio = min(contour_area / ellipse_area, ellipse_area / contour_area)
            area_score = max(0, area_ratio) * 0.4  # 40%权重
            
            # 2. 轮廓的紧密度（周长^2 / 面积，圆形的值最小）
            perimeter = cv2.arcLength(contour, True)
            if contour_area > 0:
                compactness = (perimeter * perimeter) / (4 * np.pi * contour_area)
                compactness_score = max(0, 1.0 - min(compactness - 1.0, 2.0) / 2.0) * 0.3  # 30%权重
            else:
                compactness_score = 0
            
            # 3. 椭圆在图像中的位置合理性
            height, width = image.shape
            position_score = 0.3  # 30%权重
            
            # 检查椭圆是否太靠近边界
            margin = max(major_axis, minor_axis) / 2
            if (margin < center_x < width - margin and 
                margin < center_y < height - margin):
                position_score = 0.3
            else:
                position_score = 0.1
            
            total_score = area_score + compactness_score + position_score
            return min(1.0, total_score)
            
        except Exception as e:
            return 0.0
    
    def _evaluate_ellipse_quality(self, contour, ellipse, edge_image):
        """
        评估椭圆拟合质量
        
        Args:
            contour: 原始轮廓
            ellipse: 拟合的椭圆参数
            edge_image: 边缘图像
            
        Returns:
            float: 质量评分 (0-1，越高越好)
        """
        (center_x, center_y), (minor_axis, major_axis), angle = ellipse
        
        # 1. 轮廓点到椭圆的平均距离
        contour_points = contour.reshape(-1, 2)
        
        # 创建椭圆mask
        mask = np.zeros(edge_image.shape, dtype=np.uint8)
        cv2.ellipse(mask, ((int(center_x), int(center_y)), (int(major_axis/2), int(minor_axis/2))), 
                   angle, 0, 360, 255, 2)
        
        # 2. 椭圆周长与轮廓点的匹配度
        ellipse_perimeter = cv2.arcLength(contour, True)
        expected_perimeter = np.pi * (3 * (major_axis/2 + minor_axis/2) - 
                                     np.sqrt((3 * major_axis/2 + minor_axis/2) * (major_axis/2 + 3 * minor_axis/2)))
        
        perimeter_ratio = min(ellipse_perimeter, expected_perimeter) / max(ellipse_perimeter, expected_perimeter)
        
        # 3. 边缘响应强度
        edge_response = np.mean(edge_image[mask > 0]) / 255.0 if np.sum(mask > 0) > 0 else 0
        
        # 4. 轮廓填充度（轮廓应该相对完整）
        contour_bbox = cv2.boundingRect(contour)
        bbox_area = contour_bbox[2] * contour_bbox[3]
        fill_ratio = cv2.contourArea(contour) / bbox_area if bbox_area > 0 else 0
        
        # 综合评分
        quality_score = (perimeter_ratio * 0.4 + 
                        edge_response * 0.3 + 
                        fill_ratio * 0.3)
        
        return quality_score
    
    def _combine_and_filter_shapes(self, circles, ellipses, image_shape):
        """
        Combine circles and ellipses, removing duplicates and overlapping shapes
        """
        all_shapes = circles + ellipses
        
        if len(all_shapes) <= 1:
            return all_shapes
        
        # Remove overlapping shapes (prefer ellipses over circles when overlapping)
        filtered_shapes = []
        
        for i, shape1 in enumerate(all_shapes):
            is_duplicate = False
            
            for j, shape2 in enumerate(filtered_shapes):
                if self._shapes_overlap(shape1, shape2):
                    # If they overlap, prefer ellipse over circle
                    if shape1['type'] == 'ellipse' and shape2['type'] == 'circle':
                        # Remove the circle and add the ellipse
                        filtered_shapes.remove(shape2)
                        break
                    elif shape1['type'] == 'circle' and shape2['type'] == 'ellipse':
                        # Skip the circle (keep the ellipse)
                        is_duplicate = True
                        break
                    elif shape1['type'] == shape2['type']:
                        # If same type, keep the one with better quality metrics
                        if self._compare_shape_quality(shape1, shape2) <= 0:
                            is_duplicate = True
                            break
                        else:
                            filtered_shapes.remove(shape2)
                            break
            
            if not is_duplicate:
                filtered_shapes.append(shape1)
        
        return filtered_shapes
    
    def _shapes_overlap(self, shape1, shape2, overlap_threshold=0.5):
        """Check if two shapes overlap significantly"""
        center1 = shape1['center']
        center2 = shape2['center']
        
        # Calculate distance between centers
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Get effective radius for each shape
        radius1 = max(shape1.get('major_axis', shape1.get('radius', 0)) / 2, 
                     shape1.get('minor_axis', shape1.get('radius', 0)) / 2)
        radius2 = max(shape2.get('major_axis', shape2.get('radius', 0)) / 2,
                     shape2.get('minor_axis', shape2.get('radius', 0)) / 2)
        
        # Check if they overlap more than threshold
        overlap_distance = radius1 + radius2
        return distance < overlap_distance * overlap_threshold
    
    def _compare_shape_quality(self, shape1, shape2):
        """Compare quality of two shapes (lower is better)"""
        # For circles, prefer larger radius
        if shape1['type'] == 'circle' and shape2['type'] == 'circle':
            return shape2['radius'] - shape1['radius']
        
        # For ellipses, prefer better axis ratio (closer to 1.0 = more circular)
        if shape1['type'] == 'ellipse' and shape2['type'] == 'ellipse':
            ratio1 = shape1.get('axis_ratio', 0)
            ratio2 = shape2.get('axis_ratio', 0)
            # Prefer ratio closer to 0.7 (typical pipe cross-section)
            target_ratio = 0.7
            diff1 = abs(ratio1 - target_ratio)
            diff2 = abs(ratio2 - target_ratio)
            return diff1 - diff2
        
        return 0
    
    def save_debug_visualization(self, ir_image, processed_ir, shapes, folder_name):
        """Save debug visualization of circle and ellipse detection process"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original IR image
        axes[0, 0].imshow(ir_image, cmap='gray')
        axes[0, 0].set_title('Original IR Image (No Enhancement)')
        axes[0, 0].axis('off')
        
        # Edge detection result for ellipse detection
        # 重现椭圆检测中的边缘检测过程
        blurred1 = cv2.GaussianBlur(processed_ir, (5, 5), 1)
        edges_canny = cv2.Canny(blurred1, self.ellipse_detection_params['canny_low'], 
                               self.ellipse_detection_params['canny_high'], apertureSize=3)
        
        grad_x = cv2.Sobel(processed_ir, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed_ir, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(grad_x**2 + grad_y**2)
        edges_sobel = np.uint8(255 * edges_sobel / np.max(edges_sobel))
        _, edges_sobel_thresh = cv2.threshold(edges_sobel, self.ellipse_detection_params['sobel_threshold'], 255, cv2.THRESH_BINARY)
        
        edges_combined = cv2.bitwise_or(edges_canny, edges_sobel_thresh)
        
        axes[0, 1].imshow(edges_combined, cmap='gray')
        axes[0, 1].set_title('Combined Edge Detection (Canny + Sobel)')
        axes[0, 1].axis('off')
        
        # Circle detection edge result
        blurred = cv2.GaussianBlur(processed_ir, (9, 9), 2)
        edges_circle = cv2.Canny(blurred, 30, 120, apertureSize=3)
        axes[1, 0].imshow(edges_circle, cmap='gray')
        axes[1, 0].set_title('Edge Detection for Circle Detection')
        axes[1, 0].axis('off')
        
        # Shape detection result
        result_image = cv2.cvtColor(processed_ir, cv2.COLOR_GRAY2RGB)
        circle_count = 0
        ellipse_count = 0
        
        for i, shape in enumerate(shapes):
            center = shape['center']
            
            if shape['type'] == 'circle':
                circle_count += 1
                radius = shape['radius']
                # Draw circle in green
                cv2.circle(result_image, center, radius, (0, 255, 0), 2)
                cv2.circle(result_image, center, 2, (0, 255, 0), 3)
                cv2.putText(result_image, f'C{circle_count}', 
                           (center[0] - 15, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            elif shape['type'] == 'ellipse':
                ellipse_count += 1
                major_axis = shape['major_axis']
                minor_axis = shape['minor_axis']
                angle = shape['angle']
                
                # Draw ellipse in blue
                cv2.ellipse(result_image, center, (int(major_axis/2), int(minor_axis/2)), 
                           angle, 0, 360, (255, 0, 0), 2)
                cv2.circle(result_image, center, 2, (255, 0, 0), 3)
                cv2.putText(result_image, f'E{ellipse_count}', 
                           (center[0] - 15, center[1] - max(major_axis, minor_axis)//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw major axis line
                major_axis_end_x = int(center[0] + (major_axis/2) * np.cos(np.radians(angle)))
                major_axis_end_y = int(center[1] + (major_axis/2) * np.sin(np.radians(angle)))
                cv2.line(result_image, center, (major_axis_end_x, major_axis_end_y), (255, 0, 0), 1)
        
        axes[1, 1].imshow(result_image)
        axes[1, 1].set_title(f'Detected Shapes (C:{circle_count}, E:{ellipse_count})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save result
        results_dir = Path("results") / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / "step2_shape_detection.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug output saved: results/{folder_name}/step2_shape_detection.png")

    # Keep backward compatibility
    def detect_circles(self, ir_image):
        """Backward compatibility method"""
        return self.detect_circles_and_ellipses(ir_image)


# Backward compatibility alias
CircleDetector = CircleEllipseDetector


def test_circle_ellipse_detection():
    """Test the circle and ellipse detection with sample data"""
    from step1_data_loading import DataLoader
    
    snapshot_base = Path("../snapshot")
    available_folders = [f.name for f in snapshot_base.iterdir() if f.is_dir()]
    
    if available_folders:
        test_folder = available_folders[0]
        print(f"Testing circle and ellipse detection with folder: {test_folder}")
        
        # Load data
        loader = DataLoader()
        ir_image, depth_image, params = loader.load_data(snapshot_base / test_folder)
        
        # Detect shapes
        detector = CircleEllipseDetector()
        enhanced_ir, shapes = detector.detect_circles_and_ellipses(ir_image)
        
        # Save debug visualization
        detector.save_debug_visualization(ir_image, enhanced_ir, shapes, test_folder)
        
        print("Shape detection test completed successfully!")
        print(f"Found {len(shapes)} shapes:")
        for i, shape in enumerate(shapes):
            if shape['type'] == 'circle':
                print(f"  Circle {i+1}: center={shape['center']}, radius={shape['radius']}")
            else:
                print(f"  Ellipse {i+1}: center={shape['center']}, major={shape['major_axis']}, minor={shape['minor_axis']}, angle={shape['angle']:.1f}°")
    else:
        print("No snapshot folders found for testing")


if __name__ == "__main__":
    test_circle_ellipse_detection() 