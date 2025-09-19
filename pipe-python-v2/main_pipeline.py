#!/usr/bin/env python3
"""
Main pipeline for pipe detection and localization using IR and Depth images.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from step1_data_loading import DataLoader
from step2_circle_detection import CircleDetector
from step3_depth_analysis import DepthAnalyzer
from step4_pipe_detection import PipeDetector
from step5_visualization import Visualizer


class PipeDetectionPipeline:
    def __init__(self, snapshot_path):
        """Initialize the pipeline with snapshot data path"""
        self.snapshot_path = Path(snapshot_path)
        self.output_path = Path("results")
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize all components
        self.data_loader = DataLoader()
        self.circle_detector = CircleDetector()
        self.depth_analyzer = DepthAnalyzer()
        self.pipe_detector = PipeDetector()
        self.visualizer = Visualizer()
        
    def run_full_pipeline(self, folder_name):
        """Run the complete pipeline on a specific data folder"""
        print(f"Starting pipe detection pipeline for folder: {folder_name}")
        
        # Step 1: Load IR and Depth images
        print("Step 1: Loading IR and Depth images...")
        ir_image, depth_image, params = self.data_loader.load_data(
            self.snapshot_path / folder_name
        )
        
        # Step 2: Detect circles in IR image
        print("Step 2: Detecting circles in IR image...")
        enhanced_ir, circles = self.circle_detector.detect_circles(ir_image)
        
        # Save circle detection debug visualization
        self.circle_detector.save_debug_visualization(ir_image, enhanced_ir, circles, folder_name)
        
        # Step 3: Analyze depth information for each circle
        print("Step 3: Analyzing depth information...")
        depth_analysis_results = []
        for circle in circles:
            depth_info = self.depth_analyzer.analyze_circle_depth(
                depth_image, circle, radius_multiplier=2
            )
            depth_analysis_results.append(depth_info)
        
        # Save depth analysis debug visualization
        self.depth_analyzer.save_debug_visualization(depth_analysis_results, folder_name)
        
        # Step 4: Detect pipe structures
        print("Step 4: Detecting pipe structures...")
        pipe_detection_results = []
        for i, (circle, depth_info) in enumerate(zip(circles, depth_analysis_results)):
            pipe_info = self.pipe_detector.detect_pipe(depth_info, circle)
            pipe_detection_results.append(pipe_info)
        
        # Step 5: Create final visualization
        print("Step 5: Creating final visualization...")
        final_result = self.visualizer.create_final_visualization(
            enhanced_ir, circles, pipe_detection_results, folder_name
        )
        
        # Save results
        output_folder = self.output_path / folder_name
        output_folder.mkdir(exist_ok=True)
        
        print(f"Pipeline completed! Results saved to: {output_folder}")
        return final_result


def main():
    """Main function to run the pipeline"""
    # Choose a snapshot folder
    snapshot_base = Path("../snapshot")
    available_folders = [f.name for f in snapshot_base.iterdir() if f.is_dir()]
    
    print("Available snapshot folders:")
    for i, folder in enumerate(available_folders):
        print(f"{i}. {folder}")
    
        # For this demo, we'll use the first available folder
        selected_folder = available_folders[i]
        print(f"\nUsing folder: {selected_folder}")
        
        # Initialize and run pipeline
        pipeline = PipeDetectionPipeline(snapshot_base)
        result = pipeline.run_full_pipeline(selected_folder)


if __name__ == "__main__":
    main() 