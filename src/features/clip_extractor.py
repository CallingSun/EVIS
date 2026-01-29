import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from typing import Union, List, Optional
import numpy as np

class CLIPExtractor:
    """
    CLIP Feature Extractor for EVIS system.
    Handles image and text embedding extraction using OpenAI's CLIP model.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: Optional[str] = None):
        """
        Initialize the CLIP model and processor.
        
        Args:
            model_name: The HuggingFace model identifier.
            device: 'cuda' or 'cpu'. If None, automatically detects CUDA.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing CLIPExtractor with model {model_name} on {self.device}")
        # Explicitly set attn_implementation in config to avoid SDPA conflicts
        self.model = CLIPModel.from_pretrained(
            model_name,
            attn_implementation="eager",
            output_attentions=True
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract_image_features(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> torch.Tensor:
        """
        Extract normalized visual features from one or more images.
        
        Args:
            images: A single image path, PIL Image, or a list of them.
            
        Returns:
            A tensor of shape (batch_size, 512) containing normalized embeddings.
        """
        if isinstance(images, (str, Image.Image)):
            images = [images]
            
        processed_images = []
        for img in images:
            if isinstance(img, str):
                processed_images.append(Image.open(img).convert("RGB"))
            else:
                processed_images.append(img.convert("RGB"))

        inputs = self.processor(images=processed_images, return_tensors="pt", padding=True).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        
        # Ensure we have a tensor (transformers 5.x might return objects)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        elif hasattr(image_features, "image_embeds"):
            image_features = image_features.image_embeds

        # Normalize the embeddings
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    @torch.no_grad()
    def extract_text_features(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract normalized textual features from one or more text queries.
        
        Args:
            text: A single string or a list of strings.
            
        Returns:
            A tensor of shape (batch_size, 512) containing normalized embeddings.
        """
        if isinstance(text, str):
            text = [text]
            
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        
        # Ensure we have a tensor (transformers 5.x might return objects)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        elif hasattr(text_features, "text_embeds"):
            text_features = text_features.text_embeds

        # Normalize the embeddings
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

if __name__ == "__main__":
    # Simple verification script
    extractor = CLIPExtractor()
    
    # Test text similarity
    texts = ["a photo of a cat", "a dog in the park", "a feline animal"]
    text_embeddings = extractor.extract_text_features(texts)
    
    # Similarity between "a photo of a cat" and "a feline animal"
    sim_cat_feline = torch.matmul(text_embeddings[0], text_embeddings[2].unsqueeze(-1)).item()
    # Similarity between "a photo of a cat" and "a dog in the park"
    sim_cat_dog = torch.matmul(text_embeddings[0], text_embeddings[1].unsqueeze(-1)).item()
    
    print(f"Similarity ('cat', 'feline'): {sim_cat_feline:.4f}")
    print(f"Similarity ('cat', 'dog'): {sim_cat_dog:.4f}")
    
    assert sim_cat_feline > sim_cat_dog, "Feline should be more similar to cat than dog"
    print("Verification successful!")