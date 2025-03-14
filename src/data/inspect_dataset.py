import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.data.dataloader import SoccerHomographyDataset

def inspect_dataset():
    # Load dataset
    dataset = SoccerHomographyDataset(
        seq_folder="data/raw/SNGS-060",
        input_size=(224, 224),
        sequence_length=1
    )
    
    # Inspect first sample in detail
    sample = dataset[0]
    print("\nSample keys:", sample.keys())
    
    # Inspect sequence data
    seq = sample['sequence'][0]
    print("\nSequence keys:", seq.keys())
    
    # Inspect pitch points
    pitch_points = seq['pitch_points']
    print("\nPitch points data:")
    for key, value in pitch_points.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"\n{key}:")
            print(f"Type: {type(value)}")
            print(f"Length: {len(value)}")
            if len(value) > 0:
                print(f"First element: {value[0]}")
                if hasattr(value[0], 'shape'):
                    print(f"Element shape: {value[0].shape}")
    
    # Check if we have corresponding points
    if 'image' in pitch_points and 'pitch' in pitch_points:
        img_points = pitch_points['image']
        pitch_coords = pitch_points['pitch']
        print("\nCorrespondence check:")
        print(f"Number of image points: {len(img_points)}")
        print(f"Number of pitch points: {len(pitch_coords)}")
        
        if len(img_points) > 0 and len(pitch_coords) > 0:
            print("\nExample correspondence:")
            print(f"Image point: {img_points[0]}")
            print(f"Pitch point: {pitch_coords[0]}")

if __name__ == "__main__":
    inspect_dataset() 