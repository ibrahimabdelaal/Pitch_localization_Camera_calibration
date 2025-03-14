import torch
from pathlib import Path
from src.models.feature_preparation import TimeSformerFeaturePrep
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class FeatureSequenceDataset(Dataset):
    def __init__(self, features_dir, sequence_length=8):
        self.features_dir = Path(features_dir)
        self.sequence_length = sequence_length
        
        # Get all clip folders
        self.clip_folders = sorted(f for f in self.features_dir.glob("SNGS-*"))
        
        # Get all batch folders for each clip
        self.sequences = []
        for clip_folder in self.clip_folders:
            batch_folders = sorted(clip_folder.glob("batch_*"))
            
            # Create sequences
            for i in range(0, len(batch_folders) - sequence_length + 1):
                self.sequences.append(batch_folders[i:i + sequence_length])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        batch_folders = self.sequences[idx]
        
        # Load features and metadata for sequence
        sequence_features = {
            'early': [], 'middle': [], 'deep': []
        }
        
        metadata = []
        
        for batch_folder in batch_folders:
            # Load features
            for level in ['early', 'middle', 'deep']:
                feat_path = batch_folder / f"{level}_features.npy"
                features = np.load(feat_path)
                sequence_features[level].append(features)
            
            # Load metadata
            meta_path = batch_folder / "metadata.npy"
            metadata.append(np.load(meta_path, allow_pickle=True).item())
        
        # Stack features
        sequence_features = {
            k: np.stack(v) for k, v in sequence_features.items()
        }
        
        return {
            'features': sequence_features,
            'metadata': metadata
        }

def prepare_timesformer_features(
    features_dir="data/processed/features",
    sequence_length=8,
    batch_size=4
):
    """Prepare features for TimeSformer in batches."""
    dataset = FeatureSequenceDataset(features_dir, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    feature_prep = TimeSformerFeaturePrep()
    
    # Progress bar for sequence preparation
    pbar = tqdm(dataloader, desc="Preparing sequences", total=len(dataloader))
    for batch in pbar:
        # Prepare features for TimeSformer
        features = batch['features']
        timesformer_input = feature_prep(features, sequence_length)
        
        # Update progress info
        pbar.set_postfix({
            'batch_size': features['early'].size(0),
            'sequence_length': sequence_length,
            'feature_shape': f"{timesformer_input.shape[2:]}",
        })
        
        # Now timesformer_input has shape [B, T, 768, H, W]
        yield timesformer_input, batch['metadata']

def test_pipeline():
    total_sequences = 0
    total_frames = 0
    
    print("Testing TimeSformer feature preparation pipeline...")
    for features, metadata in prepare_timesformer_features():
        total_sequences += 1
        total_frames += len(metadata)
        
        if total_sequences == 1:  # First batch details
            print("\nFeature shapes:")
            print(f"TimeSformer input: {features.shape}")
            print("\nSequence metadata:")
            print(f"Number of frames: {len(metadata)}")
    
    print(f"\nProcessed {total_sequences} sequences")
    print(f"Total frames: {total_frames}")

if __name__ == "__main__":
    test_pipeline() 