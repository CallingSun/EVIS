import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Tuple
from src.features.clip_extractor import CLIPExtractor
from src.features.ocr_extractor import OCRExtractor
from src.models.event_encoder import EventContextEncoder
from src.models.fusion_module import MultimodalFusion
from src.indexing.vector_db import VectorIndex

class IndexingPipeline:
    """
    Orchestrates the flow: 
    Image -> CLIP Vision + OCR -> Event Metadata -> Cross-Attention Fusion -> Vector DB.
    """
    def __init__(
        self, 
        device: Optional[str] = None,
        vector_db: Optional[VectorIndex] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing IndexingPipeline on {self.device}")
        
        # Initialize modules
        self.clip_extractor = CLIPExtractor(device=self.device)
        self.ocr_extractor = OCRExtractor(gpu=(self.device == "cuda"), clip_extractor=self.clip_extractor)
        self.event_encoder = EventContextEncoder().to(self.device)
        self.fusion_module = MultimodalFusion(fusion_type="cross_attention").to(self.device)
        
        self.vector_db = vector_db if vector_db is not None else VectorIndex(dimension=512)

    def process_image(
        self,
        image_path: str,
        event_metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, str]:
        """
        Process a single image through the entire pipeline to get a fused embedding.
        
        Args:
            image_path: Path to the image.
            event_metadata: Dict containing 'event_name', 'timestamp', 'lat', 'lon'.
            
        Returns:
            A tuple of (fused embedding as numpy array, OCR text).
        """
        # 1. CLIP Vision
        visual_emb = self.clip_extractor.extract_image_features(image_path).to(self.device)
        
        # 2. OCR
        ocr_text, _ = self.ocr_extractor.extract_text(image_path)
        ocr_emb = self.ocr_extractor.get_ocr_embedding(ocr_text).to(self.device)
        
        # 3. Event Encoding
        # Get embedding for event name using CLIP
        event_name_text = event_metadata.get('event_name', 'unknown event')
        event_name_emb = self.clip_extractor.extract_text_features(event_name_text).to(self.device)
        
        event_emb = self.event_encoder(
            event_name_emb,
            event_metadata.get('timestamp', '2020-01-01T00:00:00Z'),
            event_metadata.get('lat', 0.0),
            event_metadata.get('lon', 0.0)
        )
        
        # 4. Fusion
        fused_emb = self.fusion_module(visual_emb, ocr_emb, event_emb)
        
        return fused_emb.detach().cpu().numpy(), ocr_text

    def index_images(self, image_data: List[Dict[str, Any]]):
        """
        Processes and indexes multiple images.
        
        Args:
            image_data: List of dicts with 'path' and 'metadata'.
        """
        all_embeddings = []
        all_metadata = []
        
        for item in image_data:
            path = item['path']
            meta = item['metadata']
            
            print(f"Processing {path}...")
            try:
                emb, ocr_text = self.process_image(path, meta)
                all_embeddings.append(emb)
                
                # Enrich metadata with path and OCR info
                full_meta = meta.copy()
                full_meta['image_path'] = path
                full_meta['ocr_text'] = ocr_text
                all_metadata.append(full_meta)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        if all_embeddings:
            embeddings_stack = np.vstack(all_embeddings)
            self.vector_db.add_vectors(embeddings_stack, all_metadata)
            print(f"Successfully indexed {len(all_embeddings)} images.")

    def save_index(self, path: str):
        self.vector_db.save(path)

if __name__ == "__main__":
    # Mock test (would need actual images to run fully)
    print("Initializing Pipeline test...")
    pipeline = IndexingPipeline()
    print("Pipeline initialized successfully.")