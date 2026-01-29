import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.search_engine import SearchEngine
from src.indexing.pipeline import IndexingPipeline

def setup_test_index():
    """Sets up a temporary index with test images."""
    pipeline = IndexingPipeline()
    
    test_images = [
        {
            "path": "data/test_images/hanoi_street.jpg",
            "metadata": {"event_name": "Hanoi Traffic", "timestamp": "2024-01-01T12:00:00Z", "lat": 21.0285, "lon": 105.8542}
        },
        {
            "path": "data/test_images/saigon_coffee.jpg",
            "metadata": {"event_name": "Saigon Coffee", "timestamp": "2024-01-02T09:00:00Z", "lat": 10.7769, "lon": 106.7009}
        }
    ]
    
    # Filter for existing images
    existing_images = [img for img in test_images if os.path.exists(img["path"])]
    if not existing_images:
        print("No test images found. Skipping index setup.")
        return None
        
    pipeline.index_images(existing_images)
    return pipeline.vector_db

def main():
    print("Starting XAI (Explainable AI) Test...")
    
    # 1. Setup Search Engine
    vector_db = setup_test_index()
    if vector_db is None:
        return
        
    engine = SearchEngine(vector_db=vector_db)
    
    # 2. Perform a search
    query = "a motorbike on the street"
    print(f"Searching for: '{query}'")
    results = engine.search(query, top_k=1)
    
    if not results:
        print("No search results found.")
        return
        
    # 3. Explain the top result
    top_result = results[0]
    print(f"Top result: {top_result['image_path']} (Score: {top_result['score']:.4f})")
    
    print("Generating explanation heatmap...")
    output_path = engine.explain_result(top_result, query, output_dir="results/test_xai")
    
    print(f"SUCCESS: Heatmap generated at {output_path}")
    print("Please check the 'results/test_xai' directory to view the explanation.")

if __name__ == "__main__":
    main()