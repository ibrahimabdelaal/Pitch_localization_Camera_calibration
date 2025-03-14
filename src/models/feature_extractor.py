import torch
import torch.nn as nn
from transformers import ConvNextV2Model, ConvNextV2Config
from src.utils.utils_func import inspect_structure, get_device

class SoccerFeatureExtractor(nn.Module):
    def __init__(self, model_size='huge', pretrained=True, output_layers=None):
        """
        Args:
            model_size: ConvNeXt V2 variant ('huge' recommended for best features)
            pretrained: Whether to use pretrained weights
            output_layers: Which layers to extract features from. If None, uses default layers
        """
        super().__init__()
        self.device = get_device()
        # Initialize ConvNeXt V2
        if pretrained:
            self.backbone = ConvNextV2Model.from_pretrained(f"facebook/convnextv2-{model_size}-22k-512")
        else:
            config = ConvNextV2Config.from_pretrained(f"facebook/convnextv2-{model_size}-22k-512")
            self.backbone = ConvNextV2Model(config)
        
        # Move model to appropriate device
        self.backbone = self.backbone.to(self.device)
        # Freeze backbone (optional)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Default layer indices based on the actual number of hidden states
        if output_layers is None:
            # For 5 hidden states: use indices 0, 2, 4 (early, middle, late layers)
            self.output_layers = [0, 2, 4]
        else:
            self.output_layers = output_layers
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)
        Returns:
            dict: Multi-scale features
                'early': High-resolution features (layer 0)
                'middle': Medium-resolution features (layer 2)
                'deep': Low-resolution, semantic features (layer 4)
        """
        # Move input to same device as model
        x = x.to(self.device)
        
        # Handle both single images and sequences
        B = x.shape[0]
        if len(x.shape) == 5:  # Sequence input
            T = x.shape[1]
            x = x.view(B * T, *x.shape[2:])  # Merge batch and sequence dims
            
        # Extract hierarchical features
        features = {}
        outputs = self.backbone(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Debug info
        print("\nHidden states structure:")
        inspect_structure(hidden_states)
        print("\nAvailable feature dimensions:")
        for i, state in enumerate(hidden_states):
            print(f"Layer {i}: {state.shape}")
        
        # Collect features from different layers
        features['early'] = hidden_states[self.output_layers[0]]    # Early features
        features['middle'] = hidden_states[self.output_layers[1]]   # Middle features
        features['deep'] = hidden_states[self.output_layers[2]]     # Deep features
        
        # Reshape back to sequence format if needed
        if len(x.shape) == 5:
            for k in features.keys():
                features[k] = features[k].view(B, T, *features[k].shape[1:])
                
        return features

    def get_feature_dims(self):
        """Returns the feature dimensions for each scale."""
        return {
            'early': self.backbone.config.hidden_sizes[self.output_layers[0]],
            'middle': self.backbone.config.hidden_sizes[self.output_layers[1]],
            'deep': self.backbone.config.hidden_sizes[self.output_layers[2]]
        } 