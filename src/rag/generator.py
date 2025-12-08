import os
import base64
import logging
from typing import List, Dict, Optional
from pathlib import Path
from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_path: str) -> str:
    """Encodes a local image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""

def generate_answer(query: str, retrieved_items: List[Dict]) -> str:
    """
    Generates an answer using LLM based on retrieved text and images.
    """
    # 1. Prepare Context (Text)
    context_texts = []
    image_urls = []

    for item in retrieved_items:
        # Add text context
        if item.get("content"):
            context_texts.append(f"- [{item['title']}] {item['content']}")
        
        # Add images if available (Multimodal part)
        if item["type"] == "image" and item.get("image_path"):
            img_b64 = encode_image(item["image_path"])
            if img_b64:
                image_urls.append(img_b64)

    context_block = "\n".join(context_texts)

    # 2. Construct Messages for LLM
    # System Prompt
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant for 'The Batch' newsletter. "
            "Use the provided context and images to answer the user's question. "
            "If the answer is not in the context, say you don't know. "
            "Reference the article titles when possible."
        )
    }

    # User Prompt (Text + Images)
    user_content = [
        {
            "type": "text", 
            "text": f"Context:\n{context_block}\n\nQuestion: {query}"
        }
    ]

    # Attach images to the prompt (Vision API)
    for img_b64 in image_urls:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    messages = [system_message, {"role": "user", "content": user_content}]

    # 3. Call OpenAI API
    try:
        logger.info(f"Sending request to {LLM_MODEL} with {len(image_urls)} images...")
        response = client.chat.completions.create(
            model=LLM_MODEL or "gpt-4o-mini", # Fallback if env var is empty
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        return "I encountered an error generating the response."

if __name__ == "__main__":
    # --- INTERNAL TEST ---
    # We mock some retrieved items to test the LLM connection without running the full retriever
    print("Testing Generator...")
    
    mock_items = [
        {
            "type": "text", 
            "title": "Test Article", 
            "content": "Artificial Intelligence is evolving rapidly in 2024.",
            "image_path": None
        }
    ]
    
    answer = generate_answer("How is AI evolving?", mock_items)
    print("\n--- LLM Answer ---")
    print(answer)