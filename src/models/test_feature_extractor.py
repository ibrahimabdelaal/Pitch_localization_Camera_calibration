import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.models.feature_extractor import SoccerFeatureExtractor
from src.data.dataloader import SoccerHomographyDataset
from src.utils.utils_func import save_features, load_features

def collate_fn(batch):
    """Custom collate function to handle our data structure."""
    return {
        'sequence': [{
            'image': torch.stack([item['sequence'][0]['image'] for item in batch]),
            'transform_params': [item['sequence'][0]['transform_params'] for item in batch],
            'pitch_points': [item['sequence'][0]['pitch_points'] for item in batch],
            'player_points': [item['sequence'][0]['player_points'] for item in batch],
        }],
        'image_ids': [item['image_ids'][0] for item in batch],
        'file_names': [item['file_names'][0] for item in batch],
        'num_frames': torch.tensor([item['num_frames'] for item in batch]),
        'is_full_sequence': torch.tensor([item['is_full_sequence'] for item in batch])
    }

def extract_and_save_features():
    # Initialize feature extractor
    feature_extractor = SoccerFeatureExtractor(
        model_size='huge',
        pretrained=True
    )
    feature_extractor.eval()
    
    # Load data with batch processing
    dataset = SoccerHomographyDataset(
        seq_folder="data/raw/SNGS-060",
        input_size=(224, 224),
        sequence_length=1
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create features directory
    features_dir = Path("data/processed/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and save features
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get batch of images
            frames = batch['sequence'][0]['image']
            
            # Extract features
            features = feature_extractor(frames)
            
            # Prepare metadata
            metadata = {
                'file_names': batch['file_names'],
                'image_ids': batch['image_ids'],
                'transform_params': batch['sequence'][0]['transform_params'],
                'pitch_points': batch['sequence'][0]['pitch_points'],
                'player_points': batch['sequence'][0]['player_points']
            }
            
            # Save features and metadata
            save_features(
                features,
                metadata,
                features_dir,
                prefix=f"batch_{batch_idx:04d}"
            )
            
            print(f"Saved features for batch {batch_idx}, images: {batch['file_names']}")
            
            if batch_idx == 0:  # Test loading
                # Load and verify features
                loaded_features, loaded_metadata = load_features(
                    features_dir,
                    prefix=f"batch_{batch_idx:04d}"
                )
                
                print("\nVerifying loaded features:")
                for level in features:
                    print(f"\n{level} features:")
                    print(f"Original shape: {features[level].shape}")
                    print(f"Loaded shape: {loaded_features[level].shape}")
                    print(f"Match: {torch.allclose(features[level], loaded_features[level])}")

def visualize_features(batch_idx=0):
    """Visualize saved features for a specific batch."""
    import matplotlib.pyplot as plt
    
    # Load features
    features_dir = Path("data/processed/features")
    features, metadata = load_features(features_dir, prefix=f"batch_{batch_idx:04d}")
    
    # Visualize features for each image in the batch
    for img_idx, filename in enumerate(metadata['file_names']):
        print(f"\nVisualizing features for {filename}")
        
        # Plot feature maps for each level
        for level in features:
            feat_maps = features[level][img_idx]
            
            # Plot first 16 channels
            num_channels = min(16, feat_maps.shape[0])
            fig, axes = plt.subplots(4, 4, figsize=(15, 15))
            fig.suptitle(f"{level} features - {filename}")
            
            for i in range(num_channels):
                ax = axes[i//4, i%4]
                ax.imshow(feat_maps[i].cpu(), cmap='viridis')
                ax.axis('off')
            
            plt.show()

if __name__ == "__main__":
    # Extract and save features
    extract_and_save_features()
    
    # Visualize features for first batch
    visualize_features(batch_idx=0) 