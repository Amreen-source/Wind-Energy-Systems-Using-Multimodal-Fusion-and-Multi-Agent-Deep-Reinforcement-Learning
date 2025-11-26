"""
Neural Network Encoders for Multimodal State Processing
Includes Vision, Text, Time-Series, and Graph encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel
import timm
from typing import Tuple, Optional

from config import config


class VisionEncoder(nn.Module):
    """Vision Transformer encoder for blade inspection images"""
    
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(config.VISION_MODEL_NAME)
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Additional processing layers
        self.projection = nn.Sequential(
            nn.Linear(config.VISION_EMBED_DIM + 2, 512),  # +2 for defect info
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            vision_features: Pre-extracted vision features (batch, vision_dim + 2)
        
        Returns:
            Encoded features (batch, 256)
        """
        # Vision features are pre-extracted from data pipeline
        # Just apply projection
        return self.projection(vision_features)


class TextEncoder(nn.Module):
    """BERT encoder for maintenance logs"""
    
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(config.TEXT_MODEL_NAME)
        
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(config.TEXT_EMBED_DIM + 1, 512),  # +1 for severity
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            text_features: Pre-extracted text features (batch, text_dim + 1)
        
        Returns:
            Encoded features (batch, 256)
        """
        # Text features are pre-extracted from data pipeline
        return self.projection(text_features)


class TimeSeriesEncoder(nn.Module):
    """Temporal Convolutional Network for SCADA time-series data"""
    
    def __init__(self):
        super().__init__()
        
        in_channels = len(config.SCADA_FEATURES)
        
        # TCN layers
        self.tcn = nn.ModuleList()
        prev_channels = in_channels
        
        for out_channels in config.TCN_CHANNELS:
            self.tcn.append(
                nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, 
                             kernel_size=config.TCN_KERNEL_SIZE,
                             padding=config.TCN_KERNEL_SIZE // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(config.TCN_DROPOUT)
                )
            )
            prev_channels = out_channels
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(config.TCN_CHANNELS[-1], config.TIME_SERIES_EMBED_DIM)
    
    def forward(self, scada_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            scada_data: Time series data (batch, sequence_length, features)
        
        Returns:
            Encoded features (batch, embed_dim)
        """
        # Transpose for conv1d: (batch, features, sequence)
        x = scada_data.transpose(1, 2)
        
        # Apply TCN layers
        for layer in self.tcn:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Final projection
        x = self.projection(x)
        
        return x


class GraphEncoder(nn.Module):
    """Graph Attention Network for turbine topology"""
    
    def __init__(self):
        super().__init__()
        
        # Simple GAT layers (using MLPs since we don't have pytorch-geometric scatter)
        self.node_embedding = nn.Linear(config.NUM_COMPONENTS_PER_TURBINE * 2, 
                                       config.GNN_HIDDEN_DIM)  # *2 for health + age
        
        self.gat_layers = nn.ModuleList()
        prev_dim = config.GNN_HIDDEN_DIM
        
        for _ in range(config.GNN_NUM_LAYERS):
            self.gat_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, config.GNN_HIDDEN_DIM),
                    nn.LayerNorm(config.GNN_HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(config.GNN_DROPOUT)
                )
            )
        
        self.projection = nn.Linear(config.GNN_HIDDEN_DIM, config.GNN_EMBED_DIM)
    
    def forward(self, component_health: torch.Tensor, 
                component_age: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            component_health: Component health values (batch, num_components)
            component_age: Component age values (batch, num_components)
        
        Returns:
            Encoded graph features (batch, embed_dim)
        """
        # Concatenate health and age as node features
        x = torch.cat([component_health, component_age], dim=1)
        
        # Initial embedding
        x = self.node_embedding(x)
        
        # Apply GAT layers (simplified without actual graph structure)
        for layer in self.gat_layers:
            x = layer(x) + x  # Residual connection
        
        # Final projection
        x = self.projection(x)
        
        return x


class MultimodalEncoder(nn.Module):
    """Complete multimodal encoder combining all modalities"""
    
    def __init__(self):
        super().__init__()
        
        # Individual encoders
        self.vision_encoder = VisionEncoder(freeze_backbone=config.VISION_FREEZE_BACKBONE)
        self.text_encoder = TextEncoder(freeze_backbone=config.TEXT_FREEZE_BACKBONE)
        self.timeseries_encoder = TimeSeriesEncoder()
        self.graph_encoder = GraphEncoder()
        
        # Calculate total dimension before fusion
        self.total_dim = (256 + 256 + config.TIME_SERIES_EMBED_DIM + 
                         config.GNN_EMBED_DIM + len(config.WEATHER_FEATURES) * 3)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(self.total_dim, config.FUSION_HIDDEN_DIM),
            nn.LayerNorm(config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.FUSION_DROPOUT),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.FINAL_STATE_DIM)
        )
    
    def forward(self, state_dict: dict, use_vision: bool = True,
                use_text: bool = True, use_graph: bool = True) -> torch.Tensor:
        """
        Forward pass through all encoders
        
        Args:
            state_dict: Dict with all modality inputs
            use_vision: Use vision encoder
            use_text: Use text encoder
            use_graph: Use graph encoder
        
        Returns:
            Fused state representation (batch, final_state_dim)
        """
        features = []
        
        # Time series (always used)
        ts_features = self.timeseries_encoder(state_dict['scada'])
        features.append(ts_features)
        
        # Vision
        if use_vision:
            vision_features = self.vision_encoder(state_dict['vision'])
            features.append(vision_features)
        else:
            features.append(torch.zeros(ts_features.shape[0], 256, 
                                       device=ts_features.device))
        
        # Text
        if use_text:
            text_features = self.text_encoder(state_dict['text'])
            features.append(text_features)
        else:
            features.append(torch.zeros(ts_features.shape[0], 256,
                                       device=ts_features.device))
        
        # Graph
        if use_graph:
            graph_features = self.graph_encoder(
                state_dict['component_health'],
                state_dict['component_age']
            )
            features.append(graph_features)
        else:
            features.append(torch.zeros(ts_features.shape[0], config.GNN_EMBED_DIM,
                                       device=ts_features.device))
        
        # Weather (simple concatenation)
        features.append(state_dict['weather'])
        
        # Concatenate all features
        fused = torch.cat(features, dim=1)
        
        # Apply fusion network
        output = self.fusion(fused)
        
        return output


# Test encoders
if __name__ == "__main__":
    print("Testing Neural Network Encoders...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    
    # Test Vision Encoder
    print("\n1. Testing Vision Encoder...")
    vision_encoder = VisionEncoder().to(device)
    vision_input = torch.randn(batch_size, config.VISION_EMBED_DIM + 2).to(device)
    vision_output = vision_encoder(vision_input)
    print(f"  Input shape: {vision_input.shape}")
    print(f"  Output shape: {vision_output.shape}")
    
    # Test Text Encoder
    print("\n2. Testing Text Encoder...")
    text_encoder = TextEncoder().to(device)
    text_input = torch.randn(batch_size, config.TEXT_EMBED_DIM + 1).to(device)
    text_output = text_encoder(text_input)
    print(f"  Input shape: {text_input.shape}")
    print(f"  Output shape: {text_output.shape}")
    
    # Test Time Series Encoder
    print("\n3. Testing Time Series Encoder...")
    ts_encoder = TimeSeriesEncoder().to(device)
    ts_input = torch.randn(batch_size, config.SCADA_SEQUENCE_LENGTH, 
                          len(config.SCADA_FEATURES)).to(device)
    ts_output = ts_encoder(ts_input)
    print(f"  Input shape: {ts_input.shape}")
    print(f"  Output shape: {ts_output.shape}")
    
    # Test Graph Encoder
    print("\n4. Testing Graph Encoder...")
    graph_encoder = GraphEncoder().to(device)
    health_input = torch.rand(batch_size, config.NUM_COMPONENTS_PER_TURBINE).to(device)
    age_input = torch.randint(0, 100, (batch_size, config.NUM_COMPONENTS_PER_TURBINE)).float().to(device)
    graph_output = graph_encoder(health_input, age_input)
    print(f"  Health shape: {health_input.shape}")
    print(f"  Age shape: {age_input.shape}")
    print(f"  Output shape: {graph_output.shape}")
    
    # Test Multimodal Encoder
    print("\n5. Testing Multimodal Encoder...")
    multimodal_encoder = MultimodalEncoder().to(device)
    
    state_dict = {
        'scada': ts_input,
        'vision': vision_input,
        'text': text_input,
        'component_health': health_input,
        'component_age': age_input,
        'weather': torch.randn(batch_size, len(config.WEATHER_FEATURES) * 3).to(device)
    }
    
    multimodal_output = multimodal_encoder(state_dict)
    print(f"  Output shape: {multimodal_output.shape}")
    print(f"  Expected: (batch={batch_size}, final_state_dim={config.FINAL_STATE_DIM})")
    
    # Test ablations
    print("\n6. Testing ablations...")
    output_no_vision = multimodal_encoder(state_dict, use_vision=False)
    output_no_text = multimodal_encoder(state_dict, use_text=False)
    print(f"  Without vision: {output_no_vision.shape}")
    print(f"  Without text: {output_no_text.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in multimodal_encoder.parameters())
    trainable_params = sum(p.numel() for p in multimodal_encoder.parameters() if p.requires_grad)
    print(f"\n7. Model statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n✓ All encoders test passed!")
