import json
import faiss
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from PIL import Image

from src.config import DATA_DIR, EMBEDDING_MODEL

class MultimodalRetriever:
    def __init__(self, index_dir: Path = DATA_DIR / "faiss_index"):
        self.index_path = index_dir / "multimodal.index"
        self.metadata_path = index_dir / "metadata.json"

        self.model_name = EMBEDDING_MODEL 
        
        self._load_resources()

    def _load_resources(self):
        """Load FAISS index, Metadata, and Embedding Model."""
        print(f"Loading Index from {self.index_path}...")
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}. Run indexer.py first!")

        # Load the FAISS index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load the metadata (to know which ID corresponds to which article/image)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        print(f"Loading Model {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)

    @property
    def total_items(self) -> int:
        """Returns the total number of items in the index."""
        return self.index.ntotal

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for text or images using a text query.
        Returns a list of metadata dictionaries with scores.
        """
        # 1. Embed the query text
        # CLIP maps text queries to the same vector space as images
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # 2. Search FAISS
        # D = Distances (similarity scores), I = Indices (IDs in the DB)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        # indices[0] is the list of IDs found for the first (and only) query
        for rank, idx in enumerate(indices[0]):
            if idx == -1: continue # No result found
            
            # Retrieve metadata using the ID
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                # Add the similarity score to the result for display
                meta["score"] = float(distances[0][rank])
                results.append(meta)
            
        return results

if __name__ == "__main__":
    # --- INTERNAL TEST ---
    try:
        retriever = MultimodalRetriever()
        
        # Test Query
        test_query = "Artificial Intelligence"
        print(f"\nScanning database for: '{test_query}'...")
        
        results = retriever.search(test_query, k=3)
        
        print(f"\n--- Search Results ---")
        if not results:
            print("No results found.")
        
        for r in results:
            # Label as [IMG] or [TXT] so we know what we found
            type_label = "[IMG]" if r['type'] == 'image' else "[TXT]"
            print(f"{type_label} {r['title']} (Score: {r['score']:.4f})")
            
    except Exception as e:
        print(f"Error during test: {e}")