import json
import faiss
import numpy as np
import torch
import os
import open_clip
from openai import OpenAI
from dotenv import load_dotenv

# --- IMPORTS FROM CONFIG ---
from src.config import (
    TEXT_INDEX_PATH, 
    TEXT_METADATA_PATH, 
    IMAGE_INDEX_PATH, 
    IMAGE_METADATA_PATH,
    TEXT_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_PRETRAINED
)

load_dotenv()

class MultimodalRetriever:
    def __init__(self):
        self._load_resources()

    def _load_resources(self):
        print("Loading Hybrid Retriever Resources...")
        
        # 1. Text Resources (OpenAI)
        if TEXT_INDEX_PATH.exists():
            self.text_index = faiss.read_index(str(TEXT_INDEX_PATH))
            with open(TEXT_METADATA_PATH, "r", encoding="utf-8") as f:
                self.text_metadata = json.load(f)
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            print("⚠️ Text Index not found.")
            self.text_index = None

        # 2. Image Resources (SigLIP)
        if IMAGE_INDEX_PATH.exists():
            self.image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
            with open(IMAGE_METADATA_PATH, "r", encoding="utf-8") as f:
                self.image_metadata = json.load(f)
            
            # Load Model from Config
            print(f"Loading Image Model: {IMAGE_EMBEDDING_MODEL}...")
            self.siglip_model, _, _ = open_clip.create_model_and_transforms(
                IMAGE_EMBEDDING_MODEL, 
                pretrained=IMAGE_EMBEDDING_PRETRAINED
            )
            self.siglip_tokenizer = open_clip.get_tokenizer(IMAGE_EMBEDDING_MODEL)
        else:
            print("⚠️ Image Index not found.")
            self.image_index = None

    # --- THIS WAS MISSING ---
    @property
    def total_items(self) -> int:
        t = self.text_index.ntotal if self.text_index else 0
        i = self.image_index.ntotal if self.image_index else 0
        return t + i

    def search(self, query: str, k: int = 5):
        results = []

        # --- A. Text Search ---
        if self.text_index:
            try:
                resp = self.openai_client.embeddings.create(input=[query], model=TEXT_EMBEDDING_MODEL)
                query_vec_text = np.array([resp.data[0].embedding], dtype='float32')
                faiss.normalize_L2(query_vec_text)
                
                D, I = self.text_index.search(query_vec_text, k)
                for rank, idx in enumerate(I[0]):
                    if idx != -1 and idx < len(self.text_metadata):
                        meta = self.text_metadata[idx]
                        meta["score"] = float(D[0][rank])
                        results.append(meta)
            except Exception as e:
                print(f"Error searching text: {e}")

        # --- B. Image Search ---
        if self.image_index:
            try:
                with torch.no_grad():
                    text_tokens = self.siglip_tokenizer([query])
                    query_vec_img = self.siglip_model.encode_text(text_tokens)
                    
                query_vec_img_np = query_vec_img.cpu().numpy().astype('float32')
                faiss.normalize_L2(query_vec_img_np)
                
                D, I = self.image_index.search(query_vec_img_np, k=3)
                for rank, idx in enumerate(I[0]):
                    if idx != -1 and idx < len(self.image_metadata):
                        meta = self.image_metadata[idx]
                        meta["score"] = float(D[0][rank])
                        results.append(meta)
            except Exception as e:
                print(f"Error searching images: {e}")

        return results