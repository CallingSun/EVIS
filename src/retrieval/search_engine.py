import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import os
from src.features.clip_extractor import CLIPExtractor
from src.indexing.vector_db import VectorIndex
from src.visualization.explainability import AttentionVisualizer

class SearchEngine:
    """
    Handles retrieval by converting text queries to embeddings 
    and searching the Vector DB.
    """
    def __init__(
        self, 
        vector_db: Optional[VectorIndex] = None,
        vector_db_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.clip_extractor = CLIPExtractor(device=self.device)
        self.visualizer = AttentionVisualizer(extractor=self.clip_extractor)
        
        if vector_db:
            self.vector_db = vector_db
        elif vector_db_path:
            self.vector_db = VectorIndex(dimension=512)
            self.vector_db.load(vector_db_path)
        else:
            self.vector_db = VectorIndex(dimension=512)

    def search(
        self, 
        query: str, 
        top_k: int = 5,
        conditional_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for images matching the text query.
        
        Args:
            query: Textual search query.
            top_k: Number of results.
            conditional_context: Optional context to refine search (TBD implementation).
            
        Returns:
            List of search results with metadata and scores.
        """
        # 1. Convert text query to CLIP embedding
        query_emb = self.clip_extractor.extract_text_features(query)
        query_np = query_emb.detach().cpu().numpy()
        
        # 2. Search in Vector DB
        # Currently, we focus on embedding similarity.
        # Future: Incorporate conditional_context (e.g., filtering or re-weighting)
        results = self.vector_db.search(query_np, top_k=top_k)
        
        return results

    def explain_result(self, result: Dict[str, Any], query: str, output_dir: str = "results") -> str:
        """
        Generates an explanation (heatmap) for a specific search result.
        
        Args:
            result: A single result dictionary from search().
            query: The query used for search.
            output_dir: Directory to save the heatmap.
            
        Returns:
            Path to the saved heatmap image.
        """
        image_path = result.get('image_path')
        if not image_path:
            raise ValueError("Result does not contain 'image_path'.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract filename for output
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"explain_{filename}")
        
        # Generate and save heatmap
        heatmap = self.visualizer.generate_heatmap(image_path, query)
        overlaid = self.visualizer.overlay_heatmap(image_path, heatmap)
        self.visualizer.save_visualization(overlaid, output_path)
        
        return output_path

if __name__ == "__main__":
    # Mock test
    print("Initializing SearchEngine test...")
    engine = SearchEngine()
    print("SearchEngine initialized.")