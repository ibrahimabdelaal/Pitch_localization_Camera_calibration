import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.utils.pitch_utils import get_standard_pitch_points, map_image_to_pitch_points

def calculate_homography(points_src, points_dst, weights=None, method=cv2.RANSAC):
    """Calculate homography matrix using weighted player points."""
    points_src = np.float32(points_src).reshape(-1, 1, 2)
    points_dst = np.float32(points_dst).reshape(-1, 1, 2)
    
    # RANSAC parameters
    ransacReprojThreshold = 2.0  # Reduced threshold for better accuracy
    confidence = 0.99
    
    if weights is not None:
        # Sort points by weight and use the most reliable ones first
        indices = np.argsort(weights)[::-1]
        points_src = points_src[indices]
        points_dst = points_dst[indices]
        
        # Use top 70% of points for initial estimation
        num_reliable = max(4, int(len(weights) * 0.7))
        
        # Calculate initial homography with best points
        H, mask = cv2.findHomography(
            points_src[:num_reliable],
            points_dst[:num_reliable],
            method,
            ransacReprojThreshold,
            confidence
        )
        
        # Refine with all points if initial estimate is good
        if H is not None:
            H, mask = cv2.findHomography(
                points_src,
                points_dst,
                method,
                ransacReprojThreshold,
                confidence
            )
    else:
        H, mask = cv2.findHomography(points_src, points_dst, method, ransacReprojThreshold)
    
    if H is None:
        raise ValueError("Could not compute homography - not enough good matches")
    
    return H, mask

def get_homography_from_annotations(annotations, image_id):
    """Extract point correspondences primarily from player positions."""
    player_points_src = []
    player_points_dst = []
    player_confidences = []
    
    # Find all annotations for this image
    image_annotations = []
    for ann in annotations['annotations']:
        if ann.get('image_id') == image_id:
            image_annotations.append(ann)
    
    # Process player positions
    for ann in image_annotations:
        if ann.get('category_id') in [1, 2]:  # Players and goalkeepers
            if 'bbox_image' in ann and 'bbox_pitch' in ann:
                # Get player's feet position in image (bottom center of bbox)
                img_x = ann['bbox_image']['x_center']
                img_y = ann['bbox_image']['y'] + ann['bbox_image']['h']  # Bottom of bbox
                
                # Get corresponding pitch position
                pitch_x = ann['bbox_pitch']['x_bottom_middle']
                pitch_y = ann['bbox_pitch']['y_bottom_middle']
                
                # Calculate confidence based on several factors
                confidence = 1.0  # Base confidence (increased from 0.3)
                
                # Adjust confidence based on role
                if ann.get('attributes', {}).get('role') == 'goalkeeper':
                    confidence *= 1.2  # Goalkeepers are usually more static
                
                # Adjust confidence based on visibility
                if ann['bbox_image']['h'] > 100:  # Larger players are closer/more visible
                    confidence *= 1.1
                
                # Adjust confidence based on field position
                if abs(pitch_y) < 30 and abs(pitch_x) < 40:  # Players not too close to boundaries
                    confidence *= 1.1
                
                player_points_src.append([img_x, img_y])
                player_points_dst.append([pitch_x, pitch_y])
                player_confidences.append(confidence)
                
                print(f"\nAdded player point:")
                print(f"Image: [{img_x}, {img_y}]")
                print(f"Pitch: [{pitch_x}, {pitch_y}]")
                print(f"Role: {ann.get('attributes', {}).get('role')}")
                print(f"Team: {ann.get('attributes', {}).get('team')}")
                print(f"Confidence: {confidence:.2f}")
    
    points_src = np.array(player_points_src, dtype=np.float32)
    points_dst = np.array(player_points_dst, dtype=np.float32)
    weights = np.array(player_confidences, dtype=np.float32)
    
    print(f"\nTotal player points collected: {len(points_src)}")
    
    if len(points_src) < 4:
        raise ValueError(f"Not enough player points for homography (need at least 4, got {len(points_src)})")
    
    return points_src, points_dst, weights

def verify_homography(H, points_src, points_dst, mask, weights, threshold=2.0):
    """Verify homography quality using player points."""
    # Calculate reprojection errors
    points_h = np.hstack([points_src, np.ones((points_src.shape[0], 1))])
    warped_h = (H @ points_h.T).T
    warped = warped_h[:, :2] / warped_h[:, 2:]
    errors = np.linalg.norm(warped - points_dst, axis=1)
    
    # Calculate metrics
    metrics = {
        'mean_error': errors.mean(),
        'max_error': errors.max(),
        'inlier_ratio': mask.mean(),
        'num_points': len(points_src),
        'num_inliers': mask.sum(),
        'errors': errors
    }
    
    # Add quality assessment
    metrics['quality'] = 'good' if (
        metrics['mean_error'] < 3.0 and
        metrics['max_error'] < 10.0 and
        metrics['inlier_ratio'] > 0.7
    ) else 'poor'
    
    return metrics

def visualize_homography_results(image, H, points_src, points_dst, mask, save_path=None):
    """
    Visualize homography results with point correspondences.
    
    Args:
        image: Original image
        H: Homography matrix
        points_src: Source points
        points_dst: Destination points
        mask: Inlier mask
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original image with source points
    ax1.imshow(image)
    ax1.scatter(points_src[mask.ravel()==1][:,0], 
               points_src[mask.ravel()==1][:,1], 
               c='g', marker='o', label='Inliers')
    ax1.scatter(points_src[mask.ravel()==0][:,0], 
               points_src[mask.ravel()==0][:,1], 
               c='r', marker='x', label='Outliers')
    ax1.set_title('Original Image')
    ax1.legend()
    
    # Plot pitch with destination points
    ax2.scatter(points_dst[mask.ravel()==1][:,0], 
               points_dst[mask.ravel()==1][:,1], 
               c='g', marker='o', label='Inliers')
    ax2.scatter(points_dst[mask.ravel()==0][:,0], 
               points_dst[mask.ravel()==0][:,1], 
               c='r', marker='x', label='Outliers')
    ax2.set_title('Pitch Coordinates')
    ax2.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_points_from_annotations(annotations):
    """
    Extract and process point correspondences from annotations.
    
    Args:
        annotations: Dictionary containing line annotations
    
    Returns:
        points_src: Image points (denormalized)
        points_dst: Corresponding pitch points
    """
    points_src = []  # Image coordinates
    points_dst = []  # Pitch coordinates
    
    if 'lines' in annotations:
        for line_name, points in annotations['lines'].items():
            print(f"\nProcessing line: {line_name}")
            
            for point in points:
                # Denormalize image coordinates
                x_img = point['x'] * 1920  # Original image width
                y_img = point['y'] * 1080  # Original image height
                points_src.append([x_img, y_img])
                
                # Get corresponding pitch coordinates based on line type
                if line_name == "Circle central":
                    # Center circle points
                    # TODO: Map to actual pitch coordinates
                    pass
                elif line_name == "Middle line":
                    # Middle line points
                    # TODO: Map to actual pitch coordinates
                    pass
                elif "Side line" in line_name:
                    # Side line points
                    # TODO: Map to actual pitch coordinates
                    pass
                
    points_src = np.array(points_src)
    points_dst = np.array(points_dst)
    
    print(f"\nExtracted {len(points_src)} point correspondences")
    if len(points_src) > 0:
        print("Sample points:")
        print("Image points (denormalized):", points_src[0])
        print("Pitch points:", points_dst[0])
    
    return points_src, points_dst 