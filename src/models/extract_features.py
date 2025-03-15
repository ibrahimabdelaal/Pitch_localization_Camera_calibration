import os
import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.data.dataloader import SoccerHomographyDataset
    from src.models.feature_extractor import SoccerFeatureExtractor
except ImportError:
    # Fallback for direct script execution
    import sys
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.data.dataloader import SoccerHomographyDataset
    from src.models.feature_extractor import SoccerFeatureExtractor

def extract_and_save_features(
    data_root="data/raw",  # Root folder containing SNGS-XXX folders
    save_dir="data/processed/features",
    batch_size=16  # Larger batch size for single images
):
    """Extract features from all clips."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all clip folders
    clip_folders = [f for f in Path(data_root).glob("SNGS-*") if f.is_dir()]
    print(f"Found {len(clip_folders)} clips")
    
    # Initialize feature extractor
    feature_extractor = SoccerFeatureExtractor(
        model_size='huge',
        pretrained=True
    ).eval().cuda()
    
    # Main progress bar for clips
    for clip_folder in tqdm(clip_folders, desc="Processing clips", position=0):
        clip_name = clip_folder.name
        print(f"\nProcessing clip: {clip_name}")
        
        # Initialize dataset for this clip
        dataset = SoccerHomographyDataset(
            seq_folder=str(clip_folder),
            sequence_length=1  # Single images
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Create save directory for this clip
        clip_save_dir = save_dir / clip_name
        clip_save_dir.mkdir(exist_ok=True)
        
        # Progress bar for batches within clip
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Extracting features", position=1, leave=False)
            for batch_idx, batch in enumerate(pbar):
                # Get images
                images = batch['sequence'][0]['image'].cuda()
                image_ids = batch['image_ids']
                
                # Extract features
                features = feature_extractor(images)
                
                # Update progress bar with batch info
                pbar.set_postfix({
                    'batch_size': len(images),
                    'image_ids': f"{image_ids[0]}...{image_ids[-1]}"
                })
                
                # Save features and metadata
                save_batch_features(
                    features=features,
                    metadata={
                        'image_ids': image_ids,
                        'file_names': batch['file_names']
                    },
                    save_dir=clip_save_dir,
                    batch_idx=batch_idx
                )

def save_batch_features(features, metadata, save_dir, batch_idx):
    """Save features and metadata for a batch."""
    batch_dir = save_dir / f"batch_{batch_idx:04d}"
    batch_dir.mkdir(exist_ok=True)
    
    # Save each feature level
    for level, feat in features.items():
        feat_path = batch_dir / f"{level}_features.npy"
        np.save(feat_path, feat.cpu().numpy())
    
    # Save metadata
    meta_path = batch_dir / "metadata.npy"
    np.save(meta_path, metadata)

if __name__ == "__main__":
    extract_and_save_features() 