import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple

class VectorIndex:
    """
    FAISS-based vector database for indexing and retrieval.
    Stores high-dimensional embeddings and associated metadata.
    """
    def __init__(self, dimension: int = 512, metric: str = "cosine"):
        """
        Initialize the FAISS index.
        
        Args:
            dimension: The dimensionality of the vectors.
            metric: Similarity metric ("cosine" or "l2").
        """
        self.dimension = dimension
        self.metric = metric
        
        if metric == "cosine":
            # Cosine similarity is inner product on normalized vectors
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
            
        self.metadata: List[Dict[str, Any]] = []

    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Add vectors and their corresponding metadata to the index.
        
        Args:
            vectors: Numpy array of shape (N, dimension).
            metadata_list: List of N dictionaries containing metadata.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata items must match.")
        
        # Ensure vectors are float32 (FAISS requirement)
        vectors = vectors.astype('float32')
        
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity if using IndexFlatIP
            faiss.normalize_L2(vectors)
            
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors.
        
        Args:
            query_vector: Numpy array of shape (1, dimension) or (dimension,).
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing search results (metadata + score).
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        query_vector = query_vector.astype('float32')
        
        if self.metric == "cosine":
            faiss.normalize_L2(query_vector)
            
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx].copy()
                res["score"] = float(dist)
                results.append(res)
                
        return results

    def save(self, path: str):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            path: Base path to save files.
        """
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        with open(f"{path}.metadata.pkl", "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "dimension": self.dimension,
                "metric": self.metric
            }, f)
        print(f"Index and metadata saved to {path}.index and {path}.metadata.pkl")

    def load(self, path: str):
        """
        Load the FAISS index and metadata from disk.
        
        Args:
            path: Base path to load files from.
        """
        if not os.path.exists(f"{path}.index") or not os.path.exists(f"{path}.metadata.pkl"):
            raise FileNotFoundError(f"Index or metadata file not found at {path}")
            
        self.index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
            self.metric = data["metric"]
        print(f"Index and metadata loaded from {path}")

if __name__ == "__main__":
    # Quick test
    db = VectorIndex(dimension=512, metric="cosine")
    
    # Dummy data
    vectors = np.random.randn(10, 512).astype('float32')
    metadata = [{"id": i, "path": f"image_{i}.jpg"} for i in range(10)]
    
    db.add_vectors(vectors, metadata)
    
    # Search
    query = vectors[0]
    results = db.search(query, top_k=3)
    
    print("Search results for first vector:")
    for r in results:
        print(f"ID: {r['id']}, Score: {r['score']:.4f}")
        
    assert results[0]['id'] == 0
    
    # Save/Load
    db.save("test_index")
    
    new_db = VectorIndex(dimension=512)
    new_db.load("test_index")
    
    results_new = new_db.search(query, top_k=3)
    assert results_new[0]['id'] == 0
    print("Save/Load test passed!")
    
    # Cleanup
    os.remove("test_index.index")
    os.remove("test_index.metadata.pkl")