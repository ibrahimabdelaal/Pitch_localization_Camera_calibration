import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SoccerHomographyDataset(Dataset):
    def __init__(self, seq_folder, transform=None, sequence_length=1, stride=1, 
                 input_size=(224, 224), preserve_aspect_ratio=True):
        """
        Args:
            seq_folder: Path to sequence folder
            transform: Custom transformations (if None, will use default)
            sequence_length: Number of consecutive frames
            stride: Step size between sequences
            input_size: Tuple of (height, width) for model input
            preserve_aspect_ratio: If True, maintains aspect ratio and pads
        """
        self.seq_folder = seq_folder
        self.sequence_length = sequence_length
        self.stride = stride
        self.input_size = input_size
        self.preserve_aspect_ratio = preserve_aspect_ratio

        # Define default transformations if none provided
        if transform is None:
            if preserve_aspect_ratio:
                self.transform = transforms.Compose([
                    transforms.Resize(input_size[0]),  # Resize shorter side
                    transforms.CenterCrop(input_size),  # Crop center
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(input_size),  # Direct resize
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        # Load the JSON file with annotations
        json_path = os.path.join(seq_folder, "Labels-GameState.json")
        with open(json_path, "r") as f:
            self.annotations = json.load(f)

        # Extract image directory and list of image file names
        self.img_dir = os.path.join(seq_folder, self.annotations["info"]["im_dir"])
        self.images = sorted(self.annotations["images"], key=lambda x: x["file_name"])
        
        # Create sequences of frame indices (using previous frames when available)
        self.sequences = []
        for i in range(len(self.images)):
            # Calculate how many previous frames are available
            num_prev_available = min(i, sequence_length - 1)
            # Start index will be 0 if not enough previous frames
            start_idx = max(0, i - (sequence_length - 1))
            # Create sequence with available frames
            if i % stride == 0:  # Only create sequences at stride intervals
                sequence = list(range(start_idx, i + 1))
                # Pad sequence info for tracking how many frames we actually have
                self.sequences.append({
                    'indices': sequence,
                    'num_frames': len(sequence)  # This will be < sequence_length for initial frames
                })
        
        # Create a mapping from image_id to all annotations
        self.id_to_annotations = {}
        # Track temporal information
        self.track_history = {}  # Store player/ball tracking information
        
        for ann in self.annotations["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.id_to_annotations:
                self.id_to_annotations[image_id] = {
                    'players': [],
                    'ball': None,
                    'pitch_lines': None
                }
            
            # Categorize annotations and track objects
            if ann['category_id'] == 5:  # Pitch lines
                self.id_to_annotations[image_id]['pitch_lines'] = ann['lines']
            elif ann.get('attributes', {}).get('role') == 'ball':
                self.id_to_annotations[image_id]['ball'] = ann
                self._update_track_history('ball', image_id, ann)
            elif ann.get('attributes', {}).get('role') in ['player', 'goalkeeper']:
                self.id_to_annotations[image_id]['players'].append(ann)
                self._update_track_history(f"player_{ann['track_id']}", image_id, ann)

    def _update_track_history(self, track_key, image_id, annotation):
        """Update tracking history for temporal analysis."""
        if track_key not in self.track_history:
            self.track_history[track_key] = {}
        self.track_history[track_key][image_id] = annotation

    def get_temporal_info(self, image_ids):
        """Get temporal information for a sequence of frames."""
        temporal_info = {
            'player_tracks': [],
            'ball_track': [],
            'consistent_players': set()  # Players visible throughout sequence
        }
        
        # Find players that appear in all frames
        first_frame = self.id_to_annotations[image_ids[0]]
        player_tracks = set(p['track_id'] for p in first_frame['players'])
        
        for img_id in image_ids[1:]:
            current_players = set(p['track_id'] for p in self.id_to_annotations[img_id]['players'])
            player_tracks &= current_players
        
        temporal_info['consistent_players'] = player_tracks
        
        # Get temporal trajectories
        for track_id in player_tracks:
            track_key = f"player_{track_id}"
            trajectory = []
            for img_id in image_ids:
                if img_id in self.track_history[track_key]:
                    ann = self.track_history[track_key][img_id]
                    trajectory.append({
                        'image_pos': [
                            ann['bbox_image']['x_center'],
                            ann['bbox_image']['y'] + ann['bbox_image']['h']
                        ],
                        'pitch_pos': [
                            ann['bbox_pitch']['x_bottom_middle'],
                            ann['bbox_pitch']['y_bottom_middle']
                        ]
                    })
            temporal_info['player_tracks'].append({
                'track_id': track_id,
                'trajectory': trajectory
            })
        
        # Get ball trajectory
        ball_trajectory = []
        for img_id in image_ids:
            ball_ann = self.id_to_annotations[img_id]['ball']
            if ball_ann:
                ball_trajectory.append({
                    'image_pos': [
                        ball_ann['bbox_image']['x_center'],
                        ball_ann['bbox_image']['y_center']
                    ],
                    'pitch_pos': [
                        ball_ann['bbox_pitch']['x_bottom_middle'],
                        ball_ann['bbox_pitch']['y_bottom_middle']
                    ]
                })
        temporal_info['ball_track'] = ball_trajectory
        
        return temporal_info

    def __len__(self):
        return len(self.sequences)

    def get_pitch_points(self, pitch_lines):
        """Extract pitch line points for homography estimation."""
        points_image = []
        points_pitch = []  # You'll need to define the actual pitch coordinates
        
        if pitch_lines:
            for line_name, points in pitch_lines.items():
                for point in points:
                    points_image.append([point['x'] * 1920, point['y'] * 1080])  # Denormalize coordinates
                    # Add corresponding pitch coordinates (you'll need to define these)
                    # points_pitch.append([x_pitch, y_pitch])
        
        return np.array(points_image), np.array(points_pitch)

    def get_player_points(self, players):
        """Extract player foot points for additional homography constraints."""
        points_image = []
        points_pitch = []
        
        for player in players:
            if 'bbox_image' in player and 'bbox_pitch' in player:
                # Get bottom center point in image coordinates
                bbox = player['bbox_image']
                image_point = [
                    bbox['x'] + bbox['w']/2,  # x center
                    bbox['y'] + bbox['h']     # bottom y
                ]
                points_image.append(image_point)
                
                # Get corresponding pitch point
                pitch_bbox = player['bbox_pitch']
                pitch_point = [
                    pitch_bbox['x_bottom_middle'],
                    pitch_bbox['y_bottom_middle']
                ]
                points_pitch.append(pitch_point)
        
        return np.array(points_image), np.array(points_pitch)

    def _process_image(self, image, original_size):
        """Process image while tracking scale factors for coordinate conversion."""
        w, h = original_size
        
        if self.preserve_aspect_ratio:
            # Calculate scaling factors
            scale = self.input_size[0] / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize image
            image = transforms.Resize((new_h, new_w))(image)
            
            # Calculate padding
            pad_h = max(0, self.input_size[0] - new_h)
            pad_w = max(0, self.input_size[1] - new_w)
            padding = (pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2)
            
            # Pad image
            image = transforms.Pad(padding)(image)
            
            # Store transformation parameters
            transform_params = {
                'scale': scale,
                'padding': padding,
                'original_size': original_size,
                'resized_size': (new_h, new_w)
            }
        else:
            # Direct resize
            transform_params = {
                'scale_h': self.input_size[0] / h,
                'scale_w': self.input_size[1] / w,
                'original_size': original_size
            }
        
        # Apply remaining transformations
        if self.transform:
            image = self.transform(image)
            
        return image, transform_params

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        sequence_indices = sequence_info['indices']
        sequence_data = []
        image_ids = []
        
        for frame_idx in sequence_indices:
            img_info = self.images[frame_idx]
            image_id = img_info["image_id"]
            image_ids.append(image_id)
            
            # Load and process image
            img_path = os.path.join(self.img_dir, img_info["file_name"])
            image = Image.open(img_path).convert("RGB")
            original_size = (img_info['width'], img_info['height'])
            
            # Process image and get transformation parameters
            processed_image, transform_params = self._process_image(image, original_size)
            
            # Get annotations and points
            annotations = self.id_to_annotations.get(image_id, {})
            pitch_points_img, pitch_points_pitch = self.get_pitch_points(
                annotations.get('pitch_lines', {})
            )
            player_points_img, player_points_pitch = self.get_player_points(
                annotations.get('players', [])
            )
            
            sequence_data.append({
                'image': processed_image,
                'transform_params': transform_params,
                'pitch_points': {
                    'image': pitch_points_img,
                    'pitch': pitch_points_pitch
                },
                'player_points': {
                    'image': player_points_img,
                    'pitch': player_points_pitch
                }
            })

        return {
            'sequence': sequence_data,
            'temporal_info': self.get_temporal_info(image_ids),
            'image_ids': image_ids,
            'file_names': [self.images[i]["file_name"] for i in sequence_indices],
            'num_frames': sequence_info['num_frames'],
            'is_full_sequence': sequence_info['num_frames'] == self.sequence_length
        }

# Define transformations
transform = transforms.Compose([
    transforms.Resize((720, 1280)),  # Maintain aspect ratio
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use one sequence for initial testing
seq_folder = "data/raw/SNGS-060"  # Adjust path as needed
dataset = SoccerHomographyDataset(seq_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
