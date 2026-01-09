import base64
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a Senior AI News Analyst. Your task is to synthesize precise, technical answers based STRICTLY on the provided retrieved context.

### 1. INPUT STRUCTURE
The user will provide:
- **Query:** The specific question to be answered.
- **Context:** A list of numbered text chunks (e.g., [1], [2]) and optionally images.

### 2. STRICT CITATION RULES (CRITICAL)
- **Inline Citations:** Every factual statement must be cited immediately after it is stated.
- **Format:** Use square brackets with the source ID: `[ID]`.
- **Granularity:** Each distinct claim must have its own citation.
- **Bad Example:** "Qwen is fast and uses 512 experts. [1][2]"
- **Good Example:** "Qwen uses 512 experts [1]. It also uses Gated DeltaNet layers for speed [2]."
- **Multi-source Claims:** If a single sentence combines facts from multiple sources, cite all relevant IDs:  
  "Qwen is fast and efficient [1][2]."

### 3. GROUNDING & SCOPE RULES (STRICT)
- **Grounding:** Use ONLY the provided context. Do not rely on prior knowledge.
- **No Hallucination:** Do NOT infer, extrapolate, or speculate beyond what is explicitly stated.
- **Partial Answers:** If the context answers part of the query, answer ONLY that part and explicitly state what cannot be answered.
- **Refusal Condition:** If the context provides no relevant information, respond exactly with:  
  **"I cannot answer this."**
- **Conflict Handling:** If the provided context contains conflicting information, explicitly state the conflict and cite the conflicting sources without resolving it.

### 4. IMAGE HANDLING
- Mention images ONLY if they are technically relevant to answering the query.
- Reference format: "*(See image X)*".
- Do not interpret or assume details not explicitly described in the context.

### 5. OUTPUT FORMAT & STYLE
- Use clean, professional Markdown.
- Prefer bullet points for lists and structured explanations.
- Maintain a neutral, analytical tone.
- Be concise but technically complete.
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
            citation_id = item.get('id', i + 1)
            
            text_context += f"\n--- DOCUMENT [{citation_id}] ---\nSource Title: {item['title']}\nContent: {item['content']}\n"
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
            max_tokens=600,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"