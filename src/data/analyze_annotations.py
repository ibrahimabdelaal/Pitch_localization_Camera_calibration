import os
import json
from collections import defaultdict
from pprint import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

def analyze_json_structure(json_path):
    """Analyze the structure and content of the Labels-GameState.json file."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n=== JSON Structure Analysis ===\n")
    
    # 1. Basic Information
    print("1. Basic Structure:")
    print(f"Main keys in JSON: {list(data.keys())}")
    print(f"Total number of images: {len(data['images'])}")
    print(f"Total number of annotations: {len(data['annotations'])}")
    
    # 2. Image Information
    print("\n2. Image Entry Example:")
    print("First image entry:")
    pprint(data['images'][0], indent=2)
    
    # 3. Annotation Analysis
    print("\n3. Annotation Statistics:")
    
    # Analyze roles
    roles = defaultdict(int)
    teams = defaultdict(int)
    categories = defaultdict(int)
    
    # First, let's examine the structure of the first annotation
    print("\nFirst annotation structure:")
    pprint(data['annotations'][0], indent=2)
    
    for ann in data['annotations']:
        # Safely get attributes
        attributes = ann.get('attributes', {})
        if isinstance(attributes, dict):
            roles[attributes.get('role', 'unknown')] += 1
            teams[attributes.get('team', 'unknown')] += 1
        categories[ann.get('category_id', 'unknown')] += 1
    
    print("\nRoles distribution:")
    pprint(dict(roles))
    
    print("\nTeams distribution:")
    pprint(dict(teams))
    
    print("\nCategory IDs distribution:")
    pprint(dict(categories))
    
    # 4. Detailed Annotation Example
    print("\n4. Annotation Examples by Category:")
    for category_id in categories.keys():
        print(f"\nExample annotation for category_id {category_id}:")
        example_ann = next((ann for ann in data['annotations'] 
                          if ann.get('category_id') == category_id), None)
        if example_ann:
            pprint(example_ann, indent=2)
    
    # 5. Bounding Box Analysis
    print("\n5. Bounding Box Information:")
    bbox_types = set()
    bbox_keys = defaultdict(set)
    
    for ann in data['annotations']:
        # Find all bbox-related keys
        for key in ann.keys():
            if 'bbox' in key.lower():
                bbox_types.add(key)
                bbox_value = ann.get(key)
                # Only process if bbox value exists and is a dictionary
                if isinstance(bbox_value, dict):
                    bbox_keys[key].update(bbox_value.keys())
    
    print("\nTypes of bounding boxes found:")
    print(list(bbox_types))
    
    print("\nKeys in each bbox type:")
    for bbox_type, keys in bbox_keys.items():
        print(f"\n{bbox_type}:")
        pprint(list(keys))

    # 6. Additional Analysis
    print("\n6. Sample Counts:")
    
    # Count annotations by type
    bbox_counts = defaultdict(int)
    for ann in data['annotations']:
        for bbox_type in bbox_types:
            if ann.get(bbox_type):
                bbox_counts[bbox_type] += 1
    
    print("\nNumber of annotations by bbox type:")
    for bbox_type, count in bbox_counts.items():
        print(f"{bbox_type}: {count}")
    
    # Count annotations by role
    print("\nNumber of annotations by role:")
    role_counts = defaultdict(int)
    for ann in data['annotations']:
        role = ann.get('attributes', {}).get('role', 'unknown')
        role_counts[role] += 1
    
    for role, count in role_counts.items():
        print(f"{role}: {count}")

def load_and_analyze_annotations(seq_folder):
    """
    Load and analyze annotation files to understand available points.
    """
    annotations_path = Path(seq_folder) / "annotations.json"
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"\nAnalyzing annotations from: {annotations_path}")
    print(f"Number of frames: {len(annotations)}")
    
    # Analyze first frame in detail
    first_frame = list(annotations.keys())[0]
    frame_data = annotations[first_frame]
    
    print("\nFrame data structure:")
    for key in frame_data.keys():
        print(f"- {key}")
    
    # Analyze point annotations
    if 'points' in frame_data:
        points = frame_data['points']
        print("\nPoint annotations:")
        for key, value in points.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    if isinstance(subval, list):
                        print(f"  {subkey}: {len(subval)} points")
                        if len(subval) > 0:
                            print(f"  First point: {subval[0]}")
    
    # Plot points if available
    if 'points' in frame_data:
        plt.figure(figsize=(12, 6))
        
        # Plot image points
        if 'image' in frame_data['points']:
            image_points = np.array(frame_data['points']['image'])
            plt.subplot(1, 2, 1)
            plt.scatter(image_points[:, 0], image_points[:, 1], c='b', marker='o')
            plt.title('Image Points')
            plt.axis('equal')
        
        # Plot pitch points
        if 'pitch' in frame_data['points']:
            pitch_points = np.array(frame_data['points']['pitch'])
            plt.subplot(1, 2, 2)
            plt.scatter(pitch_points[:, 0], pitch_points[:, 1], c='r', marker='o')
            plt.title('Pitch Points')
            plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    return annotations

def find_good_frames(annotations):
    """
    Find frames that have good point correspondences for homography calculation.
    """
    good_frames = []
    
    for frame_id, frame_data in annotations.items():
        if 'points' in frame_data:
            points = frame_data['points']
            if 'image' in points and 'pitch' in points:
                image_points = points['image']
                pitch_points = points['pitch']
                if len(image_points) >= 4 and len(pitch_points) >= 4:
                    good_frames.append({
                        'frame_id': frame_id,
                        'num_points': len(image_points),
                        'image_points': image_points,
                        'pitch_points': pitch_points
                    })
    
    print(f"\nFound {len(good_frames)} frames with sufficient point correspondences")
    if good_frames:
        print("\nExample frame details:")
        frame = good_frames[0]
        print(f"Frame ID: {frame['frame_id']}")
        print(f"Number of points: {frame['num_points']}")
        print(f"First image point: {frame['image_points'][0]}")
        print(f"First pitch point: {frame['pitch_points'][0]}")
    
    return good_frames

def analyze_point_annotations(seq_folder):
    """
    Analyze point annotations from the dataset to check homography feasibility.
    """
    annotations_path = Path(seq_folder) / "Labels-GameState.json"
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nAnalyzing annotations from: {annotations_path}")
    
    # Collect points per frame
    frame_points = {}
    for ann in data['annotations']:
        frame_id = ann['image_id']
        
        # Initialize frame data if not exists
        if frame_id not in frame_points:
            frame_points[frame_id] = {
                'image_points': [],
                'pitch_points': [],
                'point_types': []
            }
        
        # Extract points if they exist
        if 'point' in ann:
            image_point = ann['point']
            if 'pitch_coordinate' in ann:
                pitch_point = ann['pitch_coordinate']
                frame_points[frame_id]['image_points'].append(image_point)
                frame_points[frame_id]['pitch_points'].append(pitch_point)
                frame_points[frame_id]['point_types'].append(ann.get('attributes', {}).get('role', 'unknown'))
    
    # Analyze points
    frames_with_enough_points = []
    for frame_id, points in frame_points.items():
        n_points = len(points['image_points'])
        if n_points >= 4:  # Minimum points needed for homography
            frames_with_enough_points.append({
                'frame_id': frame_id,
                'num_points': n_points,
                'image_points': points['image_points'],
                'pitch_points': points['pitch_points'],
                'point_types': points['point_types']
            })
    
    print(f"\nTotal frames analyzed: {len(frame_points)}")
    print(f"Frames with 4+ point correspondences: {len(frames_with_enough_points)}")
    
    if frames_with_enough_points:
        # Show example of a good frame
        example = frames_with_enough_points[0]
        print(f"\nExample frame {example['frame_id']}:")
        print(f"Number of points: {example['num_points']}")
        print("\nPoint correspondences:")
        for i in range(min(5, example['num_points'])):
            print(f"\nPoint {i} ({example['point_types'][i]}):")
            print(f"Image: {example['image_points'][i]}")
            print(f"Pitch: {example['pitch_points'][i]}")
        
        # Visualize points
        visualize_points(example)
    
    return frames_with_enough_points

def visualize_points(frame_data):
    """Visualize point correspondences."""
    image_points = np.array(frame_data['image_points'])
    pitch_points = np.array(frame_data['pitch_points'])
    
    plt.figure(figsize=(15, 6))
    
    # Plot image points
    plt.subplot(121)
    plt.scatter(image_points[:, 0], image_points[:, 1], c='b', marker='o')
    for i, (x, y) in enumerate(image_points):
        plt.annotate(f"{i}", (x, y), xytext=(5, 5), textcoords='offset points')
    plt.title('Image Points')
    plt.axis('equal')
    
    # Plot pitch points
    plt.subplot(122)
    plt.scatter(pitch_points[:, 0], pitch_points[:, 1], c='r', marker='o')
    for i, (x, y) in enumerate(pitch_points):
        plt.annotate(f"{i}", (x, y), xytext=(5, 5), textcoords='offset points')
    plt.title('Pitch Points')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    # Adjust this path to your dataset location
    seq_folder = "data/raw/SNGS-060"
    json_path = os.path.join(seq_folder, "Labels-GameState.json")
    
    if not os.path.exists(json_path):
        print(f"Error: Cannot find annotation file at {json_path}")
        return
        
    analyze_json_structure(json_path)

if __name__ == "__main__":
    main() 