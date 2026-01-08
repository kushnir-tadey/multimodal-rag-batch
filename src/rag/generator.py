import base64
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- KEY CHANGE: INLINE CITATION RULES ---
SYSTEM_PROMPT = """
You are a Senior AI News Analyst. Your task is to synthesize technical answers based STRICTLY on the provided retrieved context.

### 1. INPUT STRUCTURE
The user will provide:
- **Query:** The specific question.
- **Context:** A list of numbered text chunks (e.g., [1], [2]) and images.

### 2. STRICT CITATION RULES (CRITICAL)
- **Inline Citations:** You must cite the source **immediately** after the specific sentence or fact is stated. 
- **Format:** Use the format `[ID]`.
- **Bad Example:** "Qwen is fast and uses 512 experts. [1][2]" (Do NOT do this).
- **Good Example:** "Qwen uses 512 experts [1]. It also uses Gated DeltaNet layers for speed [2]."
- **Grouping:** If a sentence combines facts from multiple sources, cite both: "Qwen is fast and efficient [1][2]."

### 3. STRICT GUIDELINES
- **Grounding:** Answer ONLY using the provided context.
- **Refusal:** If the context is missing info, state "I cannot answer this."

### 4. IMAGE HANDLING
- Only mention images if they are technically relevant.
- Format: "*(See image 1)*".

### 5. OUTPUT FORMAT
- Use clean Markdown.
- Use bullet points for lists.
"""

def generate_answer(query, retrieved_items, temperature=0.0):
    print(f"DEBUG: Generating with Temperature: {temperature}")

    text_context = "### RETRIEVED TEXT CONTEXT\n"
    image_payloads = []
    
    text_items = [item for item in retrieved_items if item['type'] == 'text']
    image_items = [item for item in retrieved_items if item['type'] == 'image']

    if not text_items:
        text_context += "No text articles found.\n"
    else:
        for i, item in enumerate(text_items):
            # We map the Index (i+1) to the content.
            # The LLM sees "[1]" and associates it with this text.
            text_context += f"\n--- DOCUMENT [{i+1}] ---\nSource Title: {item['title']}\nContent: {item['content']}\n"

    # Process Images (Convert to base64)
    for item in image_items:
        if item.get('image_path'):
            try:
                with open(item['image_path'], "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    image_payloads.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    })
            except Exception as e:
                print(f"Error loading image {item['image_path']}: {e}")

    final_user_text = f"""
{text_context}

### USER QUERY
{query}
"""
    
    user_message_content = [{"type": "text", "text": final_user_text}]
    user_message_content.extend(image_payloads)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message_content}
            ],
            max_tokens=600, # Increased slightly to allow for citations
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"