import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.pipeline import IndexingPipeline
from src.retrieval.search_engine import SearchEngine

def create_demo_data(data_dir: str):
    """Creates a set of dummy images with text for the demo."""
    os.makedirs(data_dir, exist_ok=True)
    
    images_info = [
        {
            "name": "hanoi_street.jpg",
            "text": "Phở Gia Truyền - 49 Bát Đàn",
            "event": "Hanoi Trip",
            "ts": "2024-05-20T12:00:00",
            "lat": 21.0285,
            "lon": 105.8542
        },
        {
            "name": "saigon_coffee.jpg",
            "text": "Cà Phê Sữa Đá Sài Gòn",
            "event": "Saigon Morning",
            "ts": "2024-06-15T08:30:00",
            "lat": 10.7769,
            "lon": 106.7009
        },
        {
            "name": "office_work.jpg",
            "text": "Project Deadline: Friday",
            "event": "Working at Office",
            "ts": "2024-07-01T15:45:00",
            "lat": 10.7626,
            "lon": 106.6602
        },
        {
            "name": "samsung_store.jpg",
            "text": "SAMSUNG Galaxy S24 Ultra",
            "event": "Tech Expo",
            "ts": "2024-08-10T10:00:00",
            "lat": 10.7757,
            "lon": 106.7004
        }
    ]
    
    for info in images_info:
        img_path = os.path.join(data_dir, info["name"])
        # Always recreate or update to ensure text is there
        img = Image.new('RGB', (600, 400), color=(50, 50, 50))
        d = ImageDraw.Draw(img)
        # Try to draw text prominently
        try:
            # Simple text drawing
            d.text((20, 100), info["text"], fill=(255, 255, 255))
            d.text((20, 200), f"Event: {info['event']}", fill=(200, 200, 0))
        except Exception:
            pass
        img.save(img_path)
        print(f"Created/Updated demo image: {img_path}")
            
    return images_info

def main():
    print("="*60)
    print("EVIS SYSTEM END-TO-END DEMONSTRATION")
    print("="*60)

    data_dir = "data/demo_images"
    index_path = "data/demo_index"
    results_dir = "results/demo"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Prepare Data
    print("\n[Step 1] Preparing demo data...")
    images_info = create_demo_data(data_dir)
    
    # 2. Build Index
    print("\n[Step 2] Building Vector Index...")
    pipeline = IndexingPipeline()
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
    pipeline.index_images(indexing_data)
    pipeline.save_index(index_path)
    
    # 3. Initialize Search Engine
    print("\n[Step 3] Initializing Search Engine...")
    search_engine = SearchEngine(vector_db_path=index_path)
    
    # 4. Demonstrate Queries
    demo_queries = [
        {
            "type": "Text-Only/Semantic",
            "query": "coffee shop in the city",
            "desc": "Showcases CLIP's semantic understanding of visual scenes."
        },
        {
            "type": "OCR-Specific",
            "query": "Samsung banner",
            "desc": "Demonstrates fusion of OCR text into the searchable space."
        },
        {
            "type": "Contextual",
            "query": "Tech Expo event",
            "desc": "Shows how event metadata (event name) aids retrieval."
        }
    ]
    
    print("\n[Step 4] Running Complex Queries...")
    
    for q_info in demo_queries:
        query = q_info["query"]
        print(f"\n--- Query Type: {q_info['type']} ---")
        print(f"Query: '{query}'")
        print(f"Description: {q_info['desc']}")
        
        results = search_engine.search(query, top_k=2)
        
        if not results:
            print("No results found.")
            continue
            
        for i, res in enumerate(results):
            score = res['score']
            path = res['image_path']
            ocr = res.get('ocr_text', 'N/A').replace('\n', ' ')
            event = res.get('event_name', 'Unknown')
            
            print(f"\n  Result #{i+1} (Score: {score:.4f})")
            print(f"  Path: {path}")
            print(f"  Event: {event}")
            print(f"  OCR Detected: {ocr[:50]}...")
            
            # Generate XAI Heatmap for top result
            if i == 0:
                print(f"  Generating XAI Heatmap for top result...")
                try:
                    explanation_path = search_engine.explain_result(
                        res, 
                        query, 
                        output_dir=os.path.join(results_dir, q_info['type'].lower().replace('/', '_'))
                    )
                    print(f"  Heatmap saved to: {explanation_path}")
                except Exception as e:
                    print(f"  Error generating explanation: {e}")

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print(f"All results and visualizations are saved in: {results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()