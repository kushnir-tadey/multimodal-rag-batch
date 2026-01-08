import json
import logging
import os
import numpy as np
import faiss
import torch
import open_clip
from pathlib import Path
from typing import List, Dict
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# --- IMPORTS FROM CONFIG ---
from src.indexing.chunker import Chunker
from src.config import (
    RAW_DIR, 
    DATA_DIR, 
    INDEX_DIR, 
    TEXT_INDEX_PATH, 
    TEXT_METADATA_PATH, 
    IMAGE_INDEX_PATH, 
    IMAGE_METADATA_PATH,
    TEXT_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_PRETRAINED
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = RAW_DIR / "articles.json" 

def load_data(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text: str) -> str:
    if not text: return ""
    return text.strip()

def build_multimodal_index():
    # 1. Setup Models using Config
    logger.info("🔧 Initializing Hybrid Models from Config...")
    
    # Text Brain
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info(f"   - Text Model: {TEXT_EMBEDDING_MODEL}")
    
    # Image Brain
    logger.info(f"   - Image Model: {IMAGE_EMBEDDING_MODEL} ({IMAGE_EMBEDDING_PRETRAINED})")
    siglip_model, _, siglip_preprocess = open_clip.create_model_and_transforms(
        IMAGE_EMBEDDING_MODEL, 
        pretrained=IMAGE_EMBEDDING_PRETRAINED
    )
    siglip_tokenizer = open_clip.get_tokenizer(IMAGE_EMBEDDING_MODEL)
    
    chunker = Chunker(chunk_size=1000, chunk_overlap=200)

    # 2. Load Data
    raw_articles = load_data(RAW_DATA_PATH)
    logger.info(f"📚 Processing {len(raw_articles)} articles...")

    text_chunks = []
    text_metadata = []
    image_tensors = []
    image_metadata = []
    doc_id_counter = 0

    # 3. Processing Loop
    for art in raw_articles:
        # A. Process Text
        cleaned_text = clean_text(art.get("text", ""))
        chunks = chunker.chunk_text(cleaned_text)
        
        for chunk in chunks:
            text_chunks.append(chunk) 
            text_metadata.append({
                "id": doc_id_counter,
                "type": "text",
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "content": chunk
            })
            doc_id_counter += 1

        # B. Process Images
        raw_img_path = art.get("image_path") or art.get("local_image_path")
        if raw_img_path:
            clean_filename = str(raw_img_path).replace("\\", "/").split("/")[-1]
            clean_image_path = DATA_DIR / "images" / clean_filename
            
            if clean_image_path.exists():
                try:
                    img = Image.open(clean_image_path).convert("RGB")
                    img_tensor = siglip_preprocess(img)
                    image_tensors.append(img_tensor)
                    
                    image_metadata.append({
                        "id": doc_id_counter,
                        "type": "image",
                        "title": art.get("title", ""),
                        "url": art.get("url", ""),
                        "image_path": str(clean_image_path),
                        "content": "[Visual Content]"
                    })
                    doc_id_counter += 1
                except Exception as e:
                    logger.warning(f"Image error {clean_filename}: {e}")

    # 4. Indexing (Text)
    if text_chunks:
        logger.info(f"🧠 Embedding {len(text_chunks)} text chunks...")
        text_embeddings = []
        batch_size = 100
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i : i + batch_size]
            response = openai_client.embeddings.create(input=batch, model=TEXT_EMBEDDING_MODEL)
            batch_embs = [d.embedding for d in response.data]
            text_embeddings.extend(batch_embs)
            
        text_emb_np = np.array(text_embeddings, dtype='float32')
        faiss.normalize_L2(text_emb_np)
        
        d_text = text_emb_np.shape[1]
        text_index = faiss.IndexFlatIP(d_text)
        text_index.add(text_emb_np)
        
        # Use Config Paths
        faiss.write_index(text_index, str(TEXT_INDEX_PATH))
        with open(TEXT_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(text_metadata, f, indent=2)

    # 5. Indexing (Images)
    if image_tensors:
        logger.info(f"👁️ Embedding {len(image_tensors)} images...")
        image_batch = torch.stack(image_tensors)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_emb_torch = siglip_model.encode_image(image_batch)
            
        image_emb_np = image_emb_torch.cpu().numpy().astype('float32')
        faiss.normalize_L2(image_emb_np)
        
        d_img = image_emb_np.shape[1]
        image_index = faiss.IndexFlatIP(d_img)
        image_index.add(image_emb_np)
        
        # Use Config Paths
        faiss.write_index(image_index, str(IMAGE_INDEX_PATH))
        with open(IMAGE_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(image_metadata, f, indent=2)

    logger.info("✅ Hybrid Indexing Complete!")

if __name__ == "__main__":
    build_multimodal_index()