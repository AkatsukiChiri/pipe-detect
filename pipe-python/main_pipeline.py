"""
Main Pipeline for Circular Plane Detection
This script orchestrates the complete pipeline for detecting circular/elliptical planes
in RGB-D images and calculating their centers and normal vectors.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import json

# Import all step modules
from step1_circle_detection import process_rgb_image
from step2_depth_analysis import analyze_shapes_depth
from step3_plane_calculation import calculate_plane_geometry
from step4_final_visualization import create_final_visualization


class CircularPlaneDetectionPipeline:
    def __init__(self, output_base_dir: str = "results"):
        """
        Initialize the pipeline.
        
        Args:
            output_base_dir: Base directory for all output files
        """
        self.output_base_dir = output_base_dir
        self.session_dir = None
        
    def create_session_directory(self, session_name: Optional[str] = None) -> str:
        """
        Create a unique session directory for this run.
        
        Args:
            session_name: Optional custom session name
            
        Returns:
            Path to the created session directory
        """
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"plane_detection_{timestamp}"
        
        session_dir = os.path.join(self.output_base_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        self.session_dir = session_dir
        return session_dir
    
    def run_pipeline(self, rgb_image_path: str, depth_image_path: str,
                    session_name: Optional[str] = None, 
                    camera_params: Optional[Dict] = None,
                    ring_width: int = 3) -> Dict:
        """
        Run the complete pipeline on a single RGB-D image pair.
        
        Args:
            rgb_image_path: Path to RGB image
            depth_image_path: Path to corresponding depth image
            session_name: Optional session name for output organization
            camera_params: Optional camera intrinsic parameters
            ring_width: Width of the ring to extract for pipe detection (default: 3 pixels)
            
        Returns:
            Dictionary with complete pipeline results
        """
        print("="*60)
        print("CIRCULAR PLANE DETECTION PIPELINE")
        print("="*60)
        print(f"RGB Image: {rgb_image_path}")
        print(f"Depth Image: {depth_image_path}")
        print()
        
        # Create session directory
        session_dir = self.create_session_directory(session_name)
        print(f"Session directory: {session_dir}")
        print()
        
        # Initialize results dictionary
        results = {
            'session_dir': session_dir,
            'rgb_image_path': rgb_image_path,
            'depth_image_path': depth_image_path,
            'camera_params': camera_params,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'step_results': {}
        }
        
        try:
            # Step 1: Circle and Ellipse Detection
            print("STEP 1: Circle and Ellipse Detection in RGB Image")
            print("-" * 50)
            step1_start = time.time()
            
            step1_output_dir = os.path.join(session_dir, "step1_circle_detection")
            detection_data, result_image = process_rgb_image(
                rgb_image_path, step1_output_dir, "")
            
            step1_time = time.time() - step1_start
            results['step_results']['step1'] = {
                'detection_data': detection_data,
                'execution_time': step1_time,
                'output_dir': step1_output_dir
            }
            
            print(f"Step 1 completed in {step1_time:.2f}s")
            print(f"Found {len(detection_data['circles'])} circles and {len(detection_data['ellipses'])} ellipses")
            print()
            
            # Step 2: Depth Analysis and Plane Detection
            print("STEP 2: Depth Analysis and Plane Detection")
            print("-" * 50)
            step2_start = time.time()
            
            step2_output_dir = os.path.join(session_dir, "step2_depth_analysis")
            analysis_results = analyze_shapes_depth(
                detection_data, depth_image_path, step2_output_dir, "", ring_width=ring_width)
            
            step2_time = time.time() - step2_start
            results['step_results']['step2'] = {
                'analysis_results': analysis_results,
                'execution_time': step2_time,
                'output_dir': step2_output_dir
            }
            
            # Count planar surfaces
            planar_count = sum(1 for result in analysis_results 
                             if result.get('plane_info') and result['plane_info'].get('is_planar', False))
            
            print(f"Step 2 completed in {step2_time:.2f}s")
            print(f"Found {planar_count} planar surfaces out of {len(analysis_results)} analyzed shapes")
            print()
            
            # Step 3: Plane Center and Normal Vector Calculation
            print("STEP 3: Plane Center and Normal Vector Calculation")
            print("-" * 50)
            step3_start = time.time()
            
            step3_output_dir = os.path.join(session_dir, "step3_plane_calculation")
            plane_properties_list = calculate_plane_geometry(
                analysis_results, depth_image_path, step3_output_dir, "", camera_params)
            
            step3_time = time.time() - step3_start
            results['step_results']['step3'] = {
                'plane_properties_list': plane_properties_list,
                'execution_time': step3_time,
                'output_dir': step3_output_dir
            }
            
            print(f"Step 3 completed in {step3_time:.2f}s")
            print(f"Successfully calculated geometry for {len(plane_properties_list)} planes")
            print()
            
            # Step 4: Final Visualization
            print("STEP 4: Final Visualization")
            print("-" * 50)
            step4_start = time.time()
            
            step4_output_dir = os.path.join(session_dir, "step4_final_visualization")
            final_result_image = create_final_visualization(
                rgb_image_path, plane_properties_list, depth_image_path, step4_output_dir, "")
            
            step4_time = time.time() - step4_start
            results['step_results']['step4'] = {
                'output_dir': step4_output_dir,
                'execution_time': step4_time
            }
            
            print(f"Step 4 completed in {step4_time:.2f}s")
            print()
            
            # Pipeline summary
            total_time = sum(results['step_results'][step]['execution_time'] 
                           for step in results['step_results'])
            
            results['success'] = True
            results['total_execution_time'] = total_time
            results['summary'] = {
                'total_shapes_detected': len(detection_data['circles']) + len(detection_data['ellipses']),
                'circles_detected': len(detection_data['circles']),
                'ellipses_detected': len(detection_data['ellipses']),
                'planar_surfaces_found': len(plane_properties_list),
                'total_execution_time': total_time
            }
            
            print("PIPELINE SUMMARY")
            print("-" * 50)
            print(f"Total shapes detected: {results['summary']['total_shapes_detected']}")
            print(f"  - Circles: {results['summary']['circles_detected']}")
            print(f"  - Ellipses: {results['summary']['ellipses_detected']}")
            print(f"Planar surfaces found: {results['summary']['planar_surfaces_found']}")
            print(f"Total execution time: {total_time:.2f}s")
            print()
            
            # Print detailed results for each plane
            if plane_properties_list:
                print("DETECTED PLANES DETAILS:")
                print("-" * 50)
                for i, plane_props in enumerate(plane_properties_list):
                    center = plane_props['center_3d']
                    normal = plane_props['normal_vector']
                    confidence = plane_props['planarity_confidence']
                    print(f"Plane {i}:")
                    print(f"  Type: {plane_props['shape_type']}")
                    print(f"  Center (3D): ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
                    print(f"  Normal Vector: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
                    print(f"  Confidence: {confidence:.3f}")
                    print(f"  Area: {plane_props['area']:.1f} pixels²")
                    print()
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            print(f"ERROR: Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # Save results to JSON
        results_file = os.path.join(session_dir, "pipeline_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_json = self._serialize_results(results)
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        print("="*60)
        
        return results
    
    def _serialize_results(self, results: Dict) -> Dict:
        """
        Convert numpy arrays to lists for JSON serialization.
        
        Args:
            results: Results dictionary with numpy arrays
            
        Returns:
            Serializable results dictionary
        """
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            else:
                return obj
        
        return convert_numpy(results)
    
    def run_batch_processing(self, image_pairs: List[Dict], 
                           camera_params: Optional[Dict] = None) -> List[Dict]:
        """
        Run pipeline on multiple RGB-D image pairs.
        
        Args:
            image_pairs: List of dictionaries with 'rgb_path' and 'depth_path' keys
            camera_params: Optional camera intrinsic parameters
            
        Returns:
            List of results for each image pair
        """
        batch_results = []
        
        print(f"Starting batch processing of {len(image_pairs)} image pairs...")
        print()
        
        for i, pair in enumerate(image_pairs):
            print(f"Processing pair {i+1}/{len(image_pairs)}")
            
            session_name = f"batch_{i+1:03d}_{os.path.basename(pair['rgb_path']).split('.')[0]}"
            
            result = self.run_pipeline(
                pair['rgb_path'], 
                pair['depth_path'], 
                session_name, 
                camera_params
            )
            
            batch_results.append(result)
            print()
        
        # Save batch summary
        batch_summary_file = os.path.join(self.output_base_dir, "batch_summary.json")
        with open(batch_summary_file, 'w') as f:
            batch_summary = {
                'total_pairs': len(image_pairs),
                'successful_pairs': sum(1 for r in batch_results if r['success']),
                'failed_pairs': sum(1 for r in batch_results if not r['success']),
                'batch_results': self._serialize_results(batch_results)
            }
            json.dump(batch_summary, f, indent=2)
        
        print(f"Batch processing completed. Summary saved to: {batch_summary_file}")
        
        return batch_results


def main():
    """Main function to run the pipeline."""
    # Example usage
    pipeline = CircularPlaneDetectionPipeline(output_base_dir="../results")
    
    # Single image processing
    rgb_path = "../snapshot/DS87_2025_09_16_20_20_32_0268/Color_00000000.jpg"
    depth_path = "../snapshot/DS87_2025_09_16_20_20_32_0268/Depth_00000000.png"
    
    if os.path.exists(rgb_path) and os.path.exists(depth_path):
        results = pipeline.run_pipeline(rgb_path, depth_path, "example_run")
        
        if results['success']:
            print("Pipeline completed successfully!")
        else:
            print(f"Pipeline failed: {results['error']}")
    else:
        print("Example images not found. Please provide valid image paths.")
        print(f"Expected RGB image: {rgb_path}")
        print(f"Expected depth image: {depth_path}")


if __name__ == "__main__":
    main() 