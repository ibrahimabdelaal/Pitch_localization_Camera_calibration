import os
import numpy as np
import torch
from pathlib import Path

def inspect_structure(obj):
    """Inspect the structure of an object."""
    if isinstance(obj, (list, tuple)):
        print(f"Type: {type(obj)}, Length: {len(obj)}")
        if len(obj) > 0:
            print("First element type:", type(obj[0]))
            if hasattr(obj[0], 'shape'):
                print("First element shape:", obj[0].shape)
    elif hasattr(obj, 'shape'):
        print(f"Type: {type(obj)}, Shape: {obj.shape}")
    else:
        print(f"Type: {type(obj)}")

# Example usage
nested_data = (
    [ 
        {"a": 1, "b": [2, 3]}, 
        {"x": {"y": "hello", "z": 42}} 
    ],
    ({"key": [1, 2, 3]}, [4, 5])
)

def get_device():
    """Get the available device (CUDA or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

def save_features(features_dict, metadata, save_dir, prefix=""):
    """
    Save extracted features and metadata to disk.
    Features are moved to CPU before saving.
    
    Args:
        features_dict: Dictionary of features from different layers
        metadata: Dictionary containing image info (filenames, etc.)
        save_dir: Directory to save features
        prefix: Optional prefix for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert features to numpy and save
    for level, features in features_dict.items():
        features_np = features.detach().cpu().numpy()
        filename = save_dir / f"{prefix}_{level}_features.npy"
        np.save(filename, features_np)
    
    # Save metadata
    metadata_file = save_dir / f"{prefix}_metadata.npy"
    np.save(metadata_file, metadata)
    
def load_features(load_dir, prefix=""):
    """
    Load saved features and metadata from disk.
    
    Args:
        load_dir: Directory containing saved features
        prefix: Optional prefix used when saving
    
    Returns:
        features_dict: Dictionary of loaded features
        metadata: Dictionary of metadata
    """
    load_dir = Path(load_dir)
    features_dict = {}
    
    # Load features
    for level in ['early', 'middle', 'deep']:
        filename = load_dir / f"{prefix}_{level}_features.npy"
        if filename.exists():
            features = np.load(filename)
            features_dict[level] = torch.from_numpy(features)
    
    # Load metadata
    metadata_file = load_dir / f"{prefix}_metadata.npy"
    metadata = np.load(metadata_file, allow_pickle=True).item()
    
    return features_dict, metadata


