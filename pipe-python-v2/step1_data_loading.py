#!/usr/bin/env python3
"""
Step 1: Data Loading Module
Load IR and Depth images from snapshot folder
"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path


class DataLoader:
    def __init__(self):
        """Initialize the data loader"""
        pass
    
    def load_data(self, folder_path):
        """
        Load IR image, Depth image, and parameters from a snapshot folder
        
        Args:
            folder_path: Path to the snapshot folder
            
        Returns:
            tuple: (ir_image, depth_image, params)
        """
        folder_path = Path(folder_path)
        
        # Find IR and Depth images
        ir_files = list(folder_path.glob("IR_*.png"))
        depth_files = list(folder_path.glob("Depth_*.png"))
        param_file = folder_path / "params.json"
        
        if not ir_files or not depth_files:
            raise FileNotFoundError(f"IR or Depth images not found in {folder_path}")
        
        # Load the first IR and Depth images
        ir_path = ir_files[0]
        depth_path = depth_files[0]
        
        print(f"Loading IR image: {ir_path.name}")
        print(f"Loading Depth image: {depth_path.name}")
        
        # Load IR image (grayscale)
        ir_image = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        if ir_image is None:
            raise ValueError(f"Failed to load IR image: {ir_path}")
        
        # Load Depth image
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_image is None:
            raise ValueError(f"Failed to load Depth image: {depth_path}")
        
        # Load parameters if available
        params = {}
        if param_file.exists():
            with open(param_file, 'r') as f:
                params = json.load(f)
        
        # Create debug visualization
        self._save_debug_visualization(ir_image, depth_image, folder_path.name)
        
        print(f"IR Image shape: {ir_image.shape}")
        print(f"Depth Image shape: {depth_image.shape}")
        print(f"IR Image range: [{ir_image.min()}, {ir_image.max()}]")
        print(f"Depth Image range: [{depth_image.min()}, {depth_image.max()}]")
        
        return ir_image, depth_image, params
    
    def _save_debug_visualization(self, ir_image, depth_image, folder_name):
        """Save debug visualization of loaded images"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # IR Image
        axes[0].imshow(ir_image, cmap='gray')
        axes[0].set_title('Original IR Image')
        axes[0].axis('off')
        
        # Depth Image (with colormap for better visualization)
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        axes[1].imshow(depth_normalized, cmap='jet')
        axes[1].set_title('Original Depth Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Create results directory
        results_dir = Path("results") / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(results_dir / "step1_loaded_images.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Debug output saved: results/{folder_name}/step1_loaded_images.png")


def test_data_loader():
    """Test the data loader with available snapshot"""
    snapshot_base = Path("../snapshot")
    available_folders = [f.name for f in snapshot_base.iterdir() if f.is_dir()]
    
    if available_folders:
        test_folder = available_folders[0]
        print(f"Testing with folder: {test_folder}")
        
        loader = DataLoader()
        ir_image, depth_image, params = loader.load_data(snapshot_base / test_folder)
        
        print("Data loading test completed successfully!")
    else:
        print("No snapshot folders found for testing")


if __name__ == "__main__":
    test_data_loader() 