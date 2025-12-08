import json
import logging
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DIR, IMAGES_DIR, DATA_DIR

# ----------------------
# Configuration
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We use CLIP because it embeds Text and Images into the SAME vector space.
# This allows "Text Query" -> "Image Result" matching.
MODEL_NAME = "clip-ViT-B-32"
INDEX_DIR = DATA_DIR / "faiss_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_data(file_path: Path) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_multimodal_index():
    # 1. Load Model
    logger.info(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Load Processed Articles
    data_path = PROCESSED_DIR / "articles_clean.json"
    if not data_path.exists():
        logger.error(f"File not found: {data_path}. Did you run the scraper/processor?")
        return
        
    articles = load_data(data_path)
    
    # 3. Prepare Data Lists
    text_chunks = []
    image_paths = []
    metadata = []  # Stores mapping: Vector_ID -> {Type, Title, URL, Content}
    
    logger.info("Preparing data for embedding...")
    
    doc_id_counter = 0
    
    for art in articles:
        # --- Process Text (Title + Content) ---
        # We combine title and a snippet for a dense representation
        # (For a real production system, you'd chunk the full text loop here)
        combined_text = f"{art['title']}: {art['text'][:500]}" 
        text_chunks.append(combined_text)
        
        metadata.append({
            "id": doc_id_counter,
            "type": "text",
            "url": art["url"],
            "title": art["title"],
            "content": art["text"][:300] + "..." # Store preview
        })
        doc_id_counter += 1
        
        # --- Process Image ---
        if art.get("local_image_path"):
            img_p = Path(art["local_image_path"])
            if img_p.exists():
                image_paths.append(str(img_p))
                metadata.append({
                    "id": doc_id_counter,
                    "type": "image",
                    "url": art["url"],
                    "title": art["title"],
                    "image_path": str(img_p),
                    "content": "[Image Associated with Article]"
                })
                doc_id_counter += 1
            else:
                logger.warning(f"Image missing at {img_p}")

    # 4. Generate Embeddings
    logger.info(f"Embedding {len(text_chunks)} text items...")
    text_embeddings = model.encode(text_chunks, convert_to_numpy=True)
    
    logger.info(f"Embedding {len(image_paths)} images...")
    # Load images for CLIP
    images = [Image.open(p) for p in image_paths]
    image_embeddings = model.encode(images, convert_to_numpy=True)
    
    # 5. Combine & Normalize
    # FAISS works best with normalized vectors for Cosine Similarity (Inner Product)
    all_embeddings = np.vstack([text_embeddings, image_embeddings])
    faiss.normalize_L2(all_embeddings)
    
    # 6. Create FAISS Index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine similarity if normalized)
    index.add(all_embeddings)
    
    logger.info(f"Index built with {index.ntotal} vectors.")
    
    # 7. Save Index & Metadata
    faiss.write_index(index, str(INDEX_DIR / "multimodal.index"))
    
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Saved index and metadata to {INDEX_DIR}")

if __name__ == "__main__":
    build_multimodal_index()