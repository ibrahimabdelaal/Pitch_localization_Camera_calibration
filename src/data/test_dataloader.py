import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
torch.set_num_threads(1)  # Limit OpenMP threads
from dataloader import SoccerHomographyDataset, transform
import matplotlib.pyplot as plt

def test_dataloader():
    # Initialize dataset with sequence parameters
    seq_folder = "data/raw/SNGS-060"
    dataset = SoccerHomographyDataset(
    seq_folder="data/raw/SNGS-060",
    input_size=(224, 224),
    preserve_aspect_ratio=True,
    sequence_length=1
)
    
    print(f"Dataset size: {len(dataset)} sequences")
    
    # Test loading initial frames and middle sequence
    for i in [0, 1, 2, len(dataset)//2]:  # Test first few frames and a middle sequence
        sample = dataset[i]
        print(f"\nSequence {i}:")
        print(f"Number of frames in sequence: {sample['num_frames']}")
        print(f"Is full sequence: {sample['is_full_sequence']}")
        print(f"Sequence file names: {sample['file_names']}")
        
        # Print temporal information
        print("\nTemporal Information:")
        print(f"Number of consistent players: {len(sample['temporal_info']['consistent_players'])}")
        print(f"Ball trajectory length: {len(sample['temporal_info']['ball_track'])}")
        
        # Visualize all frames in the sequence
        for j, frame_data in enumerate(sample['sequence']):
            img = frame_data['image']
            
            # Convert tensor to image for visualization
            img_display = img.permute(1, 2, 0)
            img_display = torch.clamp(img_display * torch.tensor([0.229, 0.224, 0.225]) + 
                                    torch.tensor([0.485, 0.456, 0.406]), 0, 1)
            
            # Display the image
            plt.figure(figsize=(10, 5))
            plt.imshow(img_display)
            plt.title(f"Sequence {i}, Frame {j}/{sample['num_frames']-1}: {sample['file_names'][j]}")
            plt.axis('off')
            plt.show()
            
            # Print frame-specific information
            print(f"\nFrame {j}:")
            print(f"Number of pitch points: {len(frame_data['pitch_points']['image'])}")
            print(f"Number of player points: {len(frame_data['player_points']['image'])}")

if __name__ == "__main__":
    test_dataloader() 