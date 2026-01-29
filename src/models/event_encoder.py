import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datetime import datetime
from typing import Dict, Any, Union, Optional

class EventContextEncoder(nn.Module):
    """
    Encodes event metadata (name, timestamp, location) into a fixed-size vector.
    Compatible with CLIP embedding space (512-dim).
    """
    def __init__(self, output_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        
        # Time encoding: sin/cos for hour of day (2) and day of week (2) = 4 features
        # Location encoding: lat, lon (2 features)
        # Event name: We'll assume it's pre-encoded by CLIP or we use a simple embedding.
        # For this implementation, we expect a 512-dim vector for the event name (from CLIP).
        
        self.metadata_mlp = nn.Sequential(
            nn.Linear(4 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 512, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
    def _encode_timestamp(self, ts: Union[str, float, int]) -> torch.Tensor:
        """Encodes timestamp into periodic sin/cos features."""
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            dt = datetime.fromtimestamp(ts)
            
        # Hour of day (0-23)
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        # Day of week (0-6)
        dow = dt.weekday()
        
        # Sin/Cos encoding
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        dow_sin = math.sin(2 * math.pi * dow / 7.0)
        dow_cos = math.cos(2 * math.pi * dow / 7.0)
        
        return torch.tensor([hour_sin, hour_cos, dow_sin, dow_cos], dtype=torch.float32)

    def _encode_location(self, lat: float, lon: float) -> torch.Tensor:
        """Normalizes lat/lon to [-1, 1] range."""
        # Lat: -90 to 90, Lon: -180 to 180
        norm_lat = lat / 90.0
        norm_lon = lon / 180.0
        return torch.tensor([norm_lat, norm_lon], dtype=torch.float32)

    def forward(self, event_name_emb: torch.Tensor, timestamp: Union[str, float], lat: float, lon: float) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            event_name_emb: (batch_size, 512) tensor from CLIP text encoder.
            timestamp: ISO string or unix timestamp.
            lat: Latitude.
            lon: Longitude.
            
        Returns:
            (batch_size, output_dim) tensor.
        """
        device = event_name_emb.device
        batch_size = event_name_emb.shape[0]
        
        # In a real batch scenario, we'd handle these as tensors. 
        # For simplicity in this module, we'll assume single inputs or convert.
        # If batch_size > 1, we expect these to be lists or tensors already.
        
        # For the purpose of this implementation, let's handle single item or broadcast.
        ts_feat = self._encode_timestamp(timestamp).to(device)
        loc_feat = self._encode_location(lat, lon).to(device)
        
        meta_feat = torch.cat([ts_feat, loc_feat], dim=-1).unsqueeze(0).repeat(batch_size, 1)
        meta_hidden = self.metadata_mlp(meta_feat)
        
        combined = torch.cat([meta_hidden, event_name_emb], dim=-1)
        output = self.fusion_layer(combined)
        
        return F.normalize(output, p=2, dim=-1)

if __name__ == "__main__":
    # Test
    encoder = EventContextEncoder()
    dummy_event_emb = torch.randn(2, 512)
    out = encoder(dummy_event_emb, "2023-10-27T10:30:00Z", 10.762622, 106.660172)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 512)
    print("EventContextEncoder test passed!")