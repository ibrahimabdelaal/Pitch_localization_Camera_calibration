import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'  # Fix OpenMP error

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.data.dataloader import SoccerHomographyDataset
from src.utils.homography_utils import (
    get_homography_from_annotations,
    verify_homography,
    visualize_homography_results,
    calculate_homography
)
from src.utils.pitch_utils import get_standard_pitch_points

def test_homography_calculation():
    """Test homography calculation on soccer field images."""
    # Create output directory
    output_dir = Path("data/processed/homography_vis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = SoccerHomographyDataset(
        seq_folder="data/raw/SNGS-060",
        input_size=(224, 224),
        sequence_length=1
    )
    
    # Load the full annotation file
    json_path = os.path.join(dataset.seq_folder, "Labels-GameState.json")
    with open(json_path, 'r') as f:
        full_annotations = json.load(f)
    
    # Test on first few images
    for i in range(3):
        sample = dataset[i]
        image = sample['sequence'][0]['image']
        image_id = sample['image_ids'][0]
        
        try:
            print(f"\n{'='*50}")
            print(f"Processing image {i} (ID: {image_id}):")
            print(f"{'='*50}")
            
            # Get points for this specific image
            points_src, points_dst, weights = get_homography_from_annotations(
                full_annotations, 
                image_id
            )
            
            if len(points_src) >= 8:  # Require more points
                # Use weights in homography calculation
                H, mask = calculate_homography(
                    points_src.reshape(-1, 1, 2), 
                    points_dst.reshape(-1, 1, 2),
                    weights=weights,
                    method=cv2.RANSAC
                )
                
                # Pass weights to verify_homography
                metrics = verify_homography(H, points_src, points_dst, mask, weights)
                
                print(f"\nHomography matrix:\n{H}")
                print("\nQuality metrics:")
                for k, v in metrics.items():
                    if k == 'quality':
                        print(f"{k}: {v}")  # Print string as-is
                    elif k != 'errors':  # Skip printing the error array
                        print(f"{k}: {v:.3f}")  # Format numbers with 3 decimal places
                
                # Print point-wise errors with weights
                print("\nPoint-wise errors:")
                for j, (err, is_inlier, w) in enumerate(zip(metrics['errors'], mask, weights)):
                    point_type = "Player" if w == 0.3 else "Line"
                    print(f"{point_type} Point {j}: error={err:.2f} pixels, {'inlier' if is_inlier else 'outlier'}, weight={w:.1f}")
                    print(f"  Image: {points_src[j]}")
                    print(f"  Pitch: {points_dst[j]}")
                
                # Visualize results
                visualize_homography_results(
                    image.permute(1,2,0).numpy(),
                    H,
                    points_src,
                    points_dst,
                    mask,
                    save_path=output_dir / f"image_{i}.png"
                )
            else:
                print(f"Not enough points for homography (need 8, got {len(points_src)})")
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_homography_calculation() 