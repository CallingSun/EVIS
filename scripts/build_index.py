import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.indexing.pipeline import IndexingPipeline
from src.retrieval.search_engine import SearchEngine

def create_dummy_data(data_dir: str):
    """Creates a few dummy images with text for testing."""
    os.makedirs(data_dir, exist_ok=True)
    
    images_info = [
        {
            "name": "hanoi_street.jpg",
            "text": "Phở Gia Truyền",
            "event": "Hanoi Trip",
            "ts": "2024-05-20T12:00:00",
            "lat": 21.0285,
            "lon": 105.8542
        },
        {
            "name": "saigon_coffee.jpg",
            "text": "Cà Phê Sữa Đá",
            "event": "Saigon Meeting",
            "ts": "2024-06-15T08:30:00",
            "lat": 10.7626,
            "lon": 106.6602
        },
        {
            "name": "office_work.jpg",
            "text": "Project Deadline: Friday",
            "event": "Work at Office",
            "ts": "2024-07-01T15:45:00",
            "lat": 10.7626,
            "lon": 106.6602
        }
    ]
    
    for info in images_info:
        img_path = os.path.join(data_dir, info["name"])
        if not os.path.exists(img_path):
            img = Image.new('RGB', (400, 300), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10, 10), info["text"], fill=(255, 255, 0))
            img.save(img_path)
            print(f"Created dummy image: {img_path}")
            
    return images_info

def main():
    data_dir = "data/test_images"
    index_path = "data/evis_index"
    os.makedirs("data", exist_ok=True)
    
    # 1. Prepare dummy data
    print("Preparing dummy data...")
    images_info = create_dummy_data(data_dir)
    
    # 2. Initialize Pipeline
    print("\nInitializing Indexing Pipeline...")
    pipeline = IndexingPipeline()
    
    # 3. Process and Index
    indexing_data = []
    for info in images_info:
        indexing_data.append({
            "path": os.path.join(data_dir, info["name"]),
            "metadata": {
                "event_name": info["event"],
                "timestamp": info["ts"],
                "lat": info["lat"],
                "lon": info["lon"]
            }
        })
        
    print("\nStarting indexing process...")
    pipeline.index_images(indexing_data)
    
    # 4. Save Index
    print(f"\nSaving index to {index_path}...")
    pipeline.save_index(index_path)
    
    # 5. Verify with Search Engine
    print("\nTesting Search Engine...")
    search_engine = SearchEngine(vector_db_path=index_path)
    
    queries = ["coffee in Saigon", "Hanoi phở", "deadline at office"]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = search_engine.search(q, top_k=2)
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['image_path']} (Score: {res['score']:.4f})")
            print(f"     Event: {res['event_name']}, OCR Text: {res.get('ocr_text', 'N/A')}")

if __name__ == "__main__":
    main()