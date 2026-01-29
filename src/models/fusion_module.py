import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultimodalFusion(nn.Module):
    """
    Fuses Visual, OCR, and Event Context embeddings into a unified representation.
    Supports Cross-Attention and Concatenation-based fusion.
    """
    def __init__(
        self, 
        embed_dim: int = 512, 
        num_heads: int = 8, 
        fusion_type: str = "cross_attention",
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "cross_attention":
            # Multi-head Cross-Attention
            # We treat Visual features as Query, and (OCR, Event) as Key/Value
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        elif fusion_type == "concat":
            self.projection = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self, 
        visual_emb: torch.Tensor, 
        ocr_emb: torch.Tensor, 
        event_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuses the input modalities.
        
        Args:
            visual_emb: (batch_size, 512)
            ocr_emb: (batch_size, 512)
            event_emb: (batch_size, 512)
            
        Returns:
            Unified embedding (batch_size, 512)
        """
        # Ensure correct shapes (batch_size, 1, embed_dim) for attention
        v = visual_emb.unsqueeze(1) if visual_emb.dim() == 2 else visual_emb
        o = ocr_emb.unsqueeze(1) if ocr_emb.dim() == 2 else ocr_emb
        e = event_emb.unsqueeze(1) if event_emb.dim() == 2 else event_emb

        if self.fusion_type == "cross_attention":
            # Visual as query, OCR and Event as context (key/value)
            context = torch.cat([o, e], dim=1)  # (batch_size, 2, 512)
            
            # attn_output: (batch_size, 1, 512)
            attn_output, _ = self.multihead_attn(v, context, context)
            
            # Residual connection + LayerNorm
            x = self.layer_norm(v + attn_output)
            
            # Feed-forward
            x = x + self.ffn(x)
            x = x.squeeze(1)
        else:
            # Concatenation + Linear Projection
            combined = torch.cat([visual_emb, ocr_emb, event_emb], dim=-1)
            x = self.projection(combined)

        return F.normalize(x, p=2, dim=-1)

if __name__ == "__main__":
    # Test
    fusion = MultimodalFusion(fusion_type="cross_attention")
    v = torch.randn(4, 512)
    o = torch.randn(4, 512)
    e = torch.randn(4, 512)
    
    out = fusion(v, o, e)
    print(f"Cross-Attention Output shape: {out.shape}")
    assert out.shape == (4, 512)
    
    fusion_concat = MultimodalFusion(fusion_type="concat")
    out_concat = fusion_concat(v, o, e)
    print(f"Concat Output shape: {out_concat.shape}")
    assert out_concat.shape == (4, 512)
    
    print("MultimodalFusion tests passed!")