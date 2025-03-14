import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
import torch
torch.set_num_threads(1)  # Limit OpenMP threads
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.models.feature_extractor import SoccerFeatureExtractor
from src.data.dataloader import SoccerHomographyDataset
from src.utils.utils_func import get_device, save_features, load_features

def collate_fn(batch):
    """Custom collate function to handle our data structure."""
    return {
        'sequence': [{
            'image': torch.stack([item['sequence'][0]['image'] for item in batch]),
            'transform_params': [item['sequence'][0]['transform_params'] for item in batch],
            'pitch_points': [item['sequence'][0]['pitch_points'] for item in batch],
            'player_points': [item['sequence'][0]['player_points'] for item in batch],
        }],
        'temporal_info': [{  # Convert sets to lists for batching
            'consistent_players': list(item['temporal_info']['consistent_players']),
            'player_tracks': item['temporal_info']['player_tracks'],
            'ball_track': item['temporal_info']['ball_track']
        } for item in batch],
        'image_ids': [item['image_ids'][0] for item in batch],
        'file_names': [item['file_names'][0] for item in batch],
        'num_frames': torch.tensor([item['num_frames'] for item in batch]),
        'is_full_sequence': torch.tensor([item['is_full_sequence'] for item in batch])
    }

def test_mini_extraction():
    """Test feature extraction on a small subset of images."""
    
    device = get_device()
    
    # Initialize feature extractor
    feature_extractor = SoccerFeatureExtractor(
        model_size='huge',
        pretrained=True
    )
    feature_extractor.eval()
    
    # Load dataset
    dataset = SoccerHomographyDataset(
        seq_folder="data/raw/SNGS-060",
        input_size=(224, 224),
        sequence_length=1
    )
    
    # Select specific indices to test (e.g., first 3 images)
    test_indices = [0, 1, 2]
    subset_dataset = Subset(dataset, test_indices)
    
    # Create mini dataloader with custom collate_fn
    dataloader = DataLoader(
        subset_dataset, 
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn  # Add custom collate function
    )
    
    # Create features directory for mini test
    features_dir = Path("data/processed/features_mini_test")
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Process and save features
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get images (already stacked in collate_fn)
            frames = batch['sequence'][0]['image'].to(device)
            
            # Extract features
            features = feature_extractor(frames)
            
            # Display basic information
            print("\nProcessing test images:")
            print(f"Image filenames: {batch['file_names']}")
            print("\nFeature shapes:")
            for level, feat in features.items():
                print(f"{level}: {feat.shape}")
            
            # Save features with metadata
            metadata = {
                'file_names': batch['file_names'],
                'image_ids': batch['image_ids'],
                'transform_params': batch['sequence'][0]['transform_params'],
                'pitch_points': batch['sequence'][0]['pitch_points'],
                'player_points': batch['sequence'][0]['player_points']
            }
            
            save_features(features, metadata, features_dir, prefix="mini_test")
            
            # Visualize original images and features
            for img_idx in range(len(batch['file_names'])):
                # Show original image
                img = frames[img_idx].cpu().permute(1, 2, 0)
                img = torch.clamp(img * torch.tensor([0.229, 0.224, 0.225]) + 
                                torch.tensor([0.485, 0.456, 0.406]), 0, 1)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(img)
                plt.title(f"Original Image: {batch['file_names'][img_idx]}")
                plt.axis('off')
                plt.show()
                
                # Show feature maps for each level
                for level, feat_maps in features.items():
                    # Move to CPU for visualization
                    feat_map = feat_maps[img_idx].cpu()
                    # Select first 8 channels for visualization
                    num_channels = min(8, feat_map.shape[0])
                    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
                    fig.suptitle(f"{level} Features - {batch['file_names'][img_idx]}")
                    
                    for i in range(num_channels):
                        ax = axes[i//4, i%4]
                        ax.imshow(feat_map[i], cmap='viridis')
                        ax.axis('off')
                    
                    plt.show()

def verify_saved_features():
    """Verify that saved features can be loaded and are correct."""
    
    features_dir = Path("data/processed/features_mini_test")
    features, metadata = load_features(features_dir, prefix="mini_test")
    
    print("\nLoaded feature information:")
    print(f"Files processed: {metadata['file_names']}")
    print("\nFeature shapes:")
    for level, feat in features.items():
        print(f"{level}: {feat.shape}")
        
    return features, metadata

if __name__ == "__main__":
    print("Running mini feature extraction test...")
    test_mini_extraction()
    
    print("\nVerifying saved features...")
    features, metadata = verify_saved_features() 