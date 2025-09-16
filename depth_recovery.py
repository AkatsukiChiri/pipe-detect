#!/usr/bin/env python3
"""
深度图像恢复工具
用于从16位深度PNG图像中恢复真实的深度信息（毫米单位）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from pathlib import Path


class DepthImageProcessor:
    def __init__(self, params_file=None):
        """
        初始化深度图像处理器
        
        Args:
            params_file (str): 包含相机参数的JSON文件路径
        """
        self.depth_min = 1      # 默认最小深度值（mm）
        self.depth_max = 3049   # 默认最大深度值（mm）
        
        if params_file and os.path.exists(params_file):
            self.load_params(params_file)
    
    def load_params(self, params_file):
        """
        从JSON文件加载相机参数
        
        Args:
            params_file (str): 参数文件路径
        """
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            control = params.get('Control', {})
            self.depth_min = control.get('depthColorMapMin', 1)
            self.depth_max = control.get('depthColorMapMax', 3049)
            
            print(f"从参数文件加载深度范围: {self.depth_min}-{self.depth_max} mm")
            
        except Exception as e:
            print(f"加载参数文件失败: {e}")
            print(f"使用默认深度范围: {self.depth_min}-{self.depth_max} mm")
    
    def load_depth_image(self, depth_image_path):
        """
        加载16位深度PNG图像
        
        Args:
            depth_image_path (str): 深度图像路径
            
        Returns:
            numpy.ndarray: 原始16位深度数据
        """
        # 使用CV2_LOAD_IMAGE_ANYDEPTH标志来保持16位深度
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        
        if depth_image is None:
            raise ValueError(f"无法加载深度图像: {depth_image_path}")
        
        print(f"深度图像形状: {depth_image.shape}")
        print(f"数据类型: {depth_image.dtype}")
        print(f"像素值范围: {np.min(depth_image)} - {np.max(depth_image)}")
        
        return depth_image
    
    def convert_to_real_depth(self, depth_image_raw):
        """
        将原始深度数据转换为真实深度值（毫米）
        
        Args:
            depth_image_raw (numpy.ndarray): 原始16位深度数据
            
        Returns:
            numpy.ndarray: 真实深度值（毫米）
        """
        # 深度相机通常直接输出深度值，但可能需要缩放
        # 对于大多数深度相机，16位PNG中的值直接对应毫米深度
        depth_real = depth_image_raw.astype(np.float32)
        
        # 过滤无效深度值（通常为0）
        valid_mask = depth_real > 0
        
        # 应用深度范围限制
        depth_real[depth_real < self.depth_min] = 0
        depth_real[depth_real > self.depth_max] = 0
        
        print(f"有效深度像素数量: {np.sum(valid_mask)} / {depth_real.size}")
        print(f"有效深度范围: {np.min(depth_real[valid_mask])} - {np.max(depth_real[valid_mask])} mm")
        
        return depth_real
    
    def visualize_depth(self, depth_real, save_path=None, interactive=True):
        """
        Visualize depth image with interactive pixel depth display
        
        Args:
            depth_real (numpy.ndarray): Real depth values
            save_path (str): Save path (optional)
            interactive (bool): Enable interactive pixel depth display
        """
        # Create output directory for images
        if save_path:
            output_dir = os.path.dirname(save_path)
            os.makedirs(output_dir, exist_ok=True)
        
        # Create mask for valid depth
        valid_mask = depth_real > 0
        
        # Create main visualization figure
        fig = plt.figure(figsize=(16, 12))
        
        # Subplot 1: Color-mapped depth image with interactive cursor
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(depth_real, cmap='jet', vmin=self.depth_min, vmax=self.depth_max)
        cbar1 = plt.colorbar(im1, label='Depth (mm)')
        ax1.set_title('Depth Image (Color Map)')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        
        # Subplot 2: Grayscale depth image
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(depth_real, cmap='gray')
        cbar2 = plt.colorbar(im2, label='Depth (mm)')
        ax2.set_title('Depth Image (Grayscale)')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        
        # Subplot 3: Depth histogram
        ax3 = plt.subplot(2, 3, 3)
        valid_depths = depth_real[valid_mask]
        n, bins, patches = ax3.hist(valid_depths, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Depth (mm)')
        ax3.set_ylabel('Pixel Count')
        ax3.set_title('Depth Value Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: 3D depth surface (sampled)
        ax4 = plt.subplot(2, 3, 4)
        h, w = depth_real.shape
        step = max(1, min(h, w) // 50)  # Sampling step
        
        y_coords, x_coords = np.meshgrid(
            range(0, h, step),
            range(0, w, step),
            indexing='ij'
        )
        z_coords = depth_real[::step, ::step]
        
        # Only show valid depth points
        valid_3d = z_coords > 0
        if np.any(valid_3d):
            scatter = ax4.scatter(
                x_coords[valid_3d], 
                y_coords[valid_3d], 
                c=z_coords[valid_3d], 
                cmap='jet', 
                s=1
            )
            plt.colorbar(scatter, label='Depth (mm)')
        
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        ax4.set_title('Depth Point Cloud Projection')
        ax4.invert_yaxis()  # Invert Y-axis to match image coordinates
        
        # Subplot 5: Statistics display
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        stats = self.get_depth_statistics(depth_real)
        stats_text = "Depth Statistics:\n\n"
        for key, value in stats.items():
            if isinstance(value, float):
                if 'ratio' in key.lower() or 'percentage' in key.lower():
                    stats_text += f"{key}: {value:.2f}%\n"
                else:
                    stats_text += f"{key}: {value:.2f} mm\n"
            else:
                stats_text += f"{key}: {value}\n"
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        # Subplot 6: Interactive depth display area
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        cursor_info = ax6.text(0.1, 0.5, 'Move cursor over depth image\nto see pixel depth values', 
                              transform=ax6.transAxes, fontsize=12,
                              verticalalignment='center', fontfamily='monospace')
        
        # Add interactive cursor functionality
        if interactive:
            def on_mouse_move(event):
                if event.inaxes == ax1:  # Only respond to main depth image
                    if event.xdata is not None and event.ydata is not None:
                        x, y = int(event.xdata), int(event.ydata)
                        if 0 <= x < depth_real.shape[1] and 0 <= y < depth_real.shape[0]:
                            depth_value = depth_real[y, x]
                            if depth_value > 0:
                                info_text = f"Position: ({x}, {y})\nDepth: {depth_value:.2f} mm"
                            else:
                                info_text = f"Position: ({x}, {y})\nDepth: Invalid/No Data"
                            cursor_info.set_text(info_text)
                            fig.canvas.draw_idle()
            
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        
        plt.tight_layout()
        
        # Save visualization if path provided
        if save_path:
            # Save main visualization
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
            
            # Save individual images
            base_path = os.path.splitext(save_path)[0]
            
            # Save color-mapped depth image
            fig_color = plt.figure(figsize=(10, 8))
            plt.imshow(depth_real, cmap='jet', vmin=self.depth_min, vmax=self.depth_max)
            plt.colorbar(label='Depth (mm)')
            plt.title('Depth Image (Color Map)')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.savefig(f"{base_path}_color_map.png", dpi=300, bbox_inches='tight')
            plt.close(fig_color)
            
            # Save grayscale depth image
            fig_gray = plt.figure(figsize=(10, 8))
            plt.imshow(depth_real, cmap='gray')
            plt.colorbar(label='Depth (mm)')
            plt.title('Depth Image (Grayscale)')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.savefig(f"{base_path}_grayscale.png", dpi=300, bbox_inches='tight')
            plt.close(fig_gray)
            
            # Save histogram
            fig_hist = plt.figure(figsize=(10, 6))
            plt.hist(valid_depths, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Depth (mm)')
            plt.ylabel('Pixel Count')
            plt.title('Depth Value Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{base_path}_histogram.png", dpi=300, bbox_inches='tight')
            plt.close(fig_hist)
            
            print(f"Individual images saved:")
            print(f"  - Color map: {base_path}_color_map.png")
            print(f"  - Grayscale: {base_path}_grayscale.png")
            print(f"  - Histogram: {base_path}_histogram.png")
        
        plt.show()
    
    def save_depth_data(self, depth_real, output_path):
        """
        保存深度数据到不同格式
        
        Args:
            depth_real (numpy.ndarray): 真实深度值
            output_path (str): 输出文件路径（不含扩展名）
        """
        # 保存为numpy数组
        np.save(f"{output_path}_depth.npy", depth_real)
        
        # 保存为CSV（稀疏格式）
        valid_mask = depth_real > 0
        if np.any(valid_mask):
            y_coords, x_coords = np.where(valid_mask)
            depths = depth_real[valid_mask]
            
            depth_data = np.column_stack([x_coords, y_coords, depths])
            np.savetxt(
                f"{output_path}_depth.csv",
                depth_data,
                delimiter=',',
                header='x,y,depth_mm',
                comments='',
                fmt='%d,%d,%.2f'
            )
        
        # 保存为标准化的16位PNG（用于后续处理）
        depth_normalized = np.zeros_like(depth_real, dtype=np.uint16)
        valid_mask = depth_real > 0
        if np.any(valid_mask):
            # 将深度值映射到16位范围
            depth_valid = depth_real[valid_mask]
            depth_normalized[valid_mask] = depth_valid.astype(np.uint16)
        
        cv2.imwrite(f"{output_path}_depth_processed.png", depth_normalized)
        
        print(f"Depth data saved:")
        print(f"  - NumPy array: {output_path}_depth.npy")
        print(f"  - CSV file: {output_path}_depth.csv")
        print(f"  - Processed PNG: {output_path}_depth_processed.png")
    
    def get_depth_statistics(self, depth_real):
        """
        Get depth statistics
        
        Args:
            depth_real (numpy.ndarray): Real depth values
            
        Returns:
            dict: Statistics information
        """
        valid_mask = depth_real > 0
        valid_depths = depth_real[valid_mask]
        
        if len(valid_depths) == 0:
            return {"error": "No valid depth data"}
        
        stats = {
            "Total Pixels": depth_real.size,
            "Valid Pixels": len(valid_depths),
            "Valid Pixel Ratio": len(valid_depths) / depth_real.size * 100,
            "Min Depth": np.min(valid_depths),
            "Max Depth": np.max(valid_depths),
            "Mean Depth": np.mean(valid_depths),
            "Median Depth": np.median(valid_depths),
            "Std Deviation": np.std(valid_depths)
        }
        
        return stats
    
    def process_depth_image(self, depth_image_path, output_dir=None, visualize=True):
        """
        Complete depth image processing workflow
        
        Args:
            depth_image_path (str): Depth image path
            output_dir (str): Output directory (optional)
            visualize (bool): Whether to show visualization
            
        Returns:
            numpy.ndarray: Processed depth data
        """
        print(f"Processing depth image: {depth_image_path}")
        
        # Load depth image
        depth_raw = self.load_depth_image(depth_image_path)
        
        # Convert to real depth
        depth_real = self.convert_to_real_depth(depth_raw)
        
        # Get statistics
        stats = self.get_depth_statistics(depth_real)
        print("\nDepth Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                if 'ratio' in key.lower() or 'percentage' in key.lower():
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value:.2f} mm")
            else:
                print(f"  {key}: {value}")
        
        # Set output path - create organized output structure
        if output_dir is None:
            # Create output directory in current working directory
            output_dir = "depth_analysis_output"
        
        # Create timestamp-based subdirectory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get original filename without extension
        base_name = Path(depth_image_path).stem
        final_output_dir = os.path.join(output_dir, f"{base_name}_{timestamp}")
        
        os.makedirs(final_output_dir, exist_ok=True)
        
        output_path = os.path.join(final_output_dir, base_name)
        
        # Save depth data
        self.save_depth_data(depth_real, output_path)
        
        # Visualization
        if visualize:
            viz_path = f"{output_path}_analysis"
            self.visualize_depth(depth_real, f"{viz_path}.png")
        
        print(f"\nAll outputs saved to: {final_output_dir}")
        
        return depth_real


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Depth Image Recovery Tool')
    parser.add_argument('depth_image', help='16-bit depth PNG image path')
    parser.add_argument('--params', help='Camera parameters JSON file path')
    parser.add_argument('--output', help='Output directory path')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create processor
    processor = DepthImageProcessor(args.params)
    
    # Process depth image
    try:
        depth_real = processor.process_depth_image(
            args.depth_image,
            args.output,
            not args.no_viz
        )
        print(f"\nDepth image processing completed!")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # If running script directly, can process sample data
    import sys
    
    if len(sys.argv) == 1:
        # When no command line arguments, try to process sample data in current directory
        sample_dirs = [
            "snapshot/DS87_2025_09_15_16_40_50_0268",
            "snapshot/DS87_2025_09_15_16_40_54_0268"
        ]
        
        for sample_dir in sample_dirs:
            if os.path.exists(sample_dir):
                depth_file = os.path.join(sample_dir, "Depth_00000000.png")
                params_file = os.path.join(sample_dir, "params.json")
                
                if os.path.exists(depth_file):
                    print(f"\nProcessing sample data: {depth_file}")
                    processor = DepthImageProcessor(params_file if os.path.exists(params_file) else None)
                    try:
                        depth_real = processor.process_depth_image(depth_file)
                        print(f"Sample processing completed: {sample_dir}")
                    except Exception as e:
                        print(f"Sample processing failed: {e}")
                break
    else:
        # When command line arguments provided, process normally
        sys.exit(main()) 