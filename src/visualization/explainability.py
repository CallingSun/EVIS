import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Union, List, Optional, Tuple
from src.features.clip_extractor import CLIPExtractor

class AttentionVisualizer:
    """
    Provides Explainable AI (XAI) capabilities for CLIP-based searches.
    Uses attention maps from the Vision Transformer to visualize model focus.
    """
    def __init__(self, extractor: Optional[CLIPExtractor] = None, device: Optional[str] = None):
        if extractor is None:
            self.extractor = CLIPExtractor(device=device)
        else:
            self.extractor = extractor
        
        self.device = self.extractor.device
        self.model = self.extractor.model
        self.processor = self.extractor.processor

    def generate_heatmap(self, image: Union[str, Image.Image], query: str) -> np.ndarray:
        """
        Generates an attention heatmap for a given image and text query.
        
        Args:
            image: Path to image or PIL Image.
            query: Text query to explain.
            
        Returns:
            A 2D numpy array representing the heatmap (normalized 0-1).
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        # Prepare inputs
        inputs = self.processor(text=[query], images=[img], return_tensors="pt", padding=True).to(self.device)
        
        # Get text features for the query
        text_outputs = self.model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        
        # Handle transformers 5.x output objects
        if hasattr(text_outputs, "text_embeds"):
            text_features = text_outputs.text_embeds
        elif hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs

        text_features = F.normalize(text_features, p=2, dim=-1) # (1, 512)

        # Forward pass on vision model to get patch embeddings
        vision_outputs = self.model.vision_model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=True
        )
        
        # last_hidden_state shape: (1, 197, 768) for ViT-B/16 (1 CLS + 196 patches)
        last_hidden_state = vision_outputs.last_hidden_state
        patch_embeddings = last_hidden_state[:, 1:, :] # (1, 196, 768)
        
        # Project patch embeddings to the same space as text features
        # self.model.visual_projection is the linear layer
        projected_patches = self.model.visual_projection(patch_embeddings) # (1, 196, 512)
        projected_patches = F.normalize(projected_patches, p=2, dim=-1)
        
        # Compute similarity between text features and each patch
        # (1, 512) @ (1, 512, 196) -> (1, 196) or (1, 1, 196)
        similarity = torch.matmul(text_features.unsqueeze(1), projected_patches.transpose(1, 2))
        cls_to_patches = similarity.flatten() # (196,)
        
        # Determine grid size
        # For ViT-B/16 and 224x224 input, grid is 14x14
        num_patches = cls_to_patches.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        heatmap = cls_to_patches.reshape(grid_size, grid_size).detach().cpu().numpy()
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap

    def overlay_heatmap(self, image: Union[str, Image.Image], heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlays the heatmap on the original image.
        
        Args:
            image: Path or PIL Image.
            heatmap: 2D numpy array (normalized 0-1).
            alpha: Opacity of the heatmap.
            
        Returns:
            RGB numpy array of the overlaid image.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image.convert("RGB"))

        height, width = img.shape[:2]
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (width, height))
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlaid = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        
        return overlaid

    def save_visualization(self, overlaid: np.ndarray, output_path: str):
        """Saves the overlaid image to disk."""
        Image.fromarray(overlaid).save(output_path)
        print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Test
    import os
    os.makedirs("results", exist_ok=True)
    
    vis = AttentionVisualizer()
    test_img = "data/test_images/hanoi_street.jpg"
    if os.path.exists(test_img):
        query = "a motorbike"
        print(f"Generating heatmap for query: '{query}' on {test_img}")
        heatmap = vis.generate_heatmap(test_img, query)
        overlaid = vis.overlay_heatmap(test_img, heatmap)
        vis.save_visualization(overlaid, "results/heatmap_test.jpg")