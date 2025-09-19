#!/usr/bin/env python3
"""
Main pipeline for pipe detection and localization using IR and Depth images.
Enhanced to detect both circular and elliptical pipe cross-sections.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from step1_data_loading import DataLoader
from step2_circle_detection import CircleEllipseDetector
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
        self.shape_detector = CircleEllipseDetector()
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
        
        # Step 2: Detect circles and ellipses in IR image
        print("Step 2: Detecting shapes (circles and ellipses) in IR image...")
        enhanced_ir, shapes = self.shape_detector.detect_circles_and_ellipses(ir_image)
        
        # Save shape detection debug visualization
        self.shape_detector.save_debug_visualization(ir_image, enhanced_ir, shapes, folder_name)
        
        # Step 3: Analyze depth information for each shape
        print("Step 3: Analyzing depth information...")
        depth_analysis_results = []
        for shape in shapes:
            # Use major axis * 2 as square side for ellipses, radius * 2 for circles
            depth_info = self.depth_analyzer.analyze_shape_depth(
                depth_image, shape, radius_multiplier=2
            )
            depth_analysis_results.append(depth_info)
        
        # Save depth analysis debug visualization
        self.depth_analyzer.save_debug_visualization(depth_analysis_results, folder_name)
        
        # Step 4: Detect pipe structures
        print("Step 4: Detecting pipe structures...")
        pipe_detection_results = []
        for i, (shape, depth_info) in enumerate(zip(shapes, depth_analysis_results)):
            pipe_info = self.pipe_detector.detect_pipe(depth_info, shape)
            pipe_detection_results.append(pipe_info)
        
        # Step 5: Create final visualization
        print("Step 5: Creating final visualization...")
        final_result = self.visualizer.create_final_visualization(
            enhanced_ir, shapes, pipe_detection_results, folder_name
        )
        
        # Save results
        output_folder = self.output_path / folder_name
        output_folder.mkdir(exist_ok=True)
        
        # Print summary
        self._print_detection_summary(shapes, pipe_detection_results)
        
        print(f"Pipeline completed! Results saved to: {output_folder}")
        return final_result
    
    def _print_detection_summary(self, shapes, pipe_detection_results):
        """Print a summary of the detection results"""
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        
        total_shapes = len(shapes)
        circles = [s for s in shapes if s.get('type') == 'circle']
        ellipses = [s for s in shapes if s.get('type') == 'ellipse']
        detected_pipes = [r for r in pipe_detection_results if r.get('is_pipe', False)]
        
        print(f"Total shapes detected: {total_shapes}")
        print(f"  - Circles: {len(circles)}")
        print(f"  - Ellipses: {len(ellipses)}")
        print(f"Pipes detected: {len(detected_pipes)}")
        
        if detected_pipes:
            print("\nPipe Details:")
            for i, (shape, pipe_result) in enumerate(zip(shapes, pipe_detection_results)):
                if pipe_result.get('is_pipe', False):
                    shape_type = shape.get('type', 'unknown')
                    confidence = pipe_result.get('confidence', 0)
                    
                    if shape_type == 'ellipse':
                        shape_desc = (f"Ellipse at {shape['center']} "
                                    f"(major={shape['major_axis']}, minor={shape['minor_axis']}, "
                                    f"angle={shape['angle']:.1f}°)")
                    else:
                        shape_desc = f"Circle at {shape['center']} (r={shape['radius']})"
                    
                    print(f"  {i+1}. {shape_desc}")
                    print(f"     Confidence: {confidence:.2f}")
                    
                    if pipe_result.get('direction'):
                        direction = pipe_result['direction']
                        print(f"     3D Direction: azimuth={direction['azimuth_deg']:.1f}°, "
                              f"elevation={direction['elevation_deg']:.1f}°")
        
        print("="*60)


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