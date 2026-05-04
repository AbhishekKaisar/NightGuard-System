import os
import cv2
import time
import csv
import itertools
from pathlib import Path
from tqdm import tqdm

# Import the core functions from the main pipeline
from main import (
    enhance_image_with_ensemble,
    detect_faces,
    detect_humans,
    detect_vehicles
)

def run_tuning():
    """
    Performs a grid search over enhancement and detection parameters 
    to find the optimal configuration for the pipeline.
    """
    # 1. Define the directory containing tuning images
    tuning_dir = "data/tuning_set/"
    if not os.path.exists(tuning_dir):
        print(f"Error: The directory '{tuning_dir}' does not exist.")
        print("Please create it and add some test images before running.")
        return
        
    image_files = [f for f in os.listdir(tuning_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Error: No images found in '{tuning_dir}'.")
        return
        
    print(f"Found {len(image_files)} images in '{tuning_dir}' for tuning.\n")

    # 2. Define the parameter grids separately to optimize inferences
    enhancement_params = {
        'tile_size': [256, 512],
        'tile_overlap': [16, 32]
    }
    
    detection_params = {
        'face_conf': [0.2, 0.3, 0.4],
        'human_conf': [0.3, 0.4, 0.5],
        'vehicle_conf': [0.4, 0.5, 0.6],
        'blur_kernel': [(3, 3), (5, 5)]
    }

    enh_keys = list(enhancement_params.keys())
    enh_values = list(enhancement_params.values())
    enh_combinations = list(itertools.product(*enh_values))

    det_keys = list(detection_params.keys())
    det_values = list(detection_params.values())
    det_combinations = list(itertools.product(*det_values))
    
    # 3. Prepare the CSV file for logging
    output_csv = "tuning_results.csv"
    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Tile Size", 
                "Tile Overlap", 
                "Face Conf",
                "Human Conf",
                "Vehicle Conf",
                "Blur Kernel", 
                "Total Faces", 
                "Total Humans", 
                "Total Vehicles", 
                "Enhance Time (s)",
                "Detection Time (s)",
                "Total Time (s)"
            ])

        total_runs = len(enh_combinations) * len(det_combinations)
        print(f"Starting grid search over {total_runs} total combinations...")
        print(f"({len(enh_combinations)} enhancement configs × {len(det_combinations)} detection configs)")
        
        # Load all original images into memory to save disk I/O time
        print("Preloading raw images...")
        raw_images = {img_name: cv2.imread(os.path.join(tuning_dir, img_name)) for img_name in image_files}
        raw_images = {k: v for k, v in raw_images.items() if v is not None}
        
        for e_idx, e_combo in enumerate(enh_combinations, 1):
            e_params = dict(zip(enh_keys, e_combo))
            tile_size = e_params['tile_size']
            tile_overlap = e_params['tile_overlap']
            
            print(f"\n[{e_idx}/{len(enh_combinations)}] Applying Enhancement: Tile={tile_size}, Overlap={tile_overlap}")
            
            # Step A: Cache the enhanced images for this specific enhancement configuration
            enhanced_cache = {}
            total_enhance_time = 0.0
            
            for img_name, img in tqdm(raw_images.items(), desc="Enhancing Images", leave=False):
                start_e = time.time()
                try:
                    enhanced = enhance_image_with_ensemble(
                        img, 
                        tile_size=tile_size, 
                        tile_overlap=tile_overlap
                    )
                    enhanced_cache[img_name] = enhanced
                except Exception as e:
                    print(f"  Error enhancing {img_name}: {e}")
                total_enhance_time += (time.time() - start_e)
                
            avg_enhance_time = total_enhance_time / len(raw_images) if raw_images else 0

            # Step B: Run all detection parameter combinations on the cached enhanced images
            print(f"  Running {len(det_combinations)} detection combinations on enhanced images...")
            for d_idx, d_combo in enumerate(det_combinations, 1):
                d_params = dict(zip(det_keys, d_combo))
                face_conf = d_params['face_conf']
                human_conf = d_params['human_conf']
                vehicle_conf = d_params['vehicle_conf']
                blur_kernel = d_params['blur_kernel']
                
                total_faces = 0
                total_humans = 0
                total_vehicles = 0
                total_det_time = 0.0
                
                for img_name, enhanced_img in enhanced_cache.items():
                    start_d = time.time()
                    try:
                        faces = detect_faces(enhanced_img, conf=face_conf)
                        humans = detect_humans(enhanced_img, conf=human_conf, blur_ksize=blur_kernel)
                        vehicles = detect_vehicles(enhanced_img, conf=vehicle_conf)
                        
                        total_faces += len(faces)
                        total_humans += len(humans)
                        total_vehicles += len(vehicles)
                    except Exception as e:
                        print(f"  Error detecting on {img_name}: {e}")
                        
                    total_det_time += (time.time() - start_d)
                
                avg_det_time = total_det_time / len(enhanced_cache) if enhanced_cache else 0
                avg_total_time = avg_enhance_time + avg_det_time
                
                writer.writerow([
                    tile_size,
                    tile_overlap,
                    face_conf,
                    human_conf,
                    vehicle_conf,
                    f"{blur_kernel[0]}x{blur_kernel[1]}",
                    total_faces,
                    total_humans,
                    total_vehicles,
                    f"{avg_enhance_time:.2f}",
                    f"{avg_det_time:.2f}",
                    f"{avg_total_time:.2f}"
                ])
                file.flush() 

    print(f"\nGrid search complete! Results saved to '{output_csv}'.")

if __name__ == "__main__":
    run_tuning()