import base64
from openai import OpenAI
import os

# Initialize OpenAI Client
# Ensure your .env file has OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query, retrieved_items, temperature=0.0):
    #Adding debug print to test if the temperature works

    print(f"DEBUG: Generating with Temperature: {temperature}")
    """
    Generates an answer using GPT-4o based on retrieved text and images.
    """
    # 1. Prepare the System Prompt (THE FIX)
    # We explicitly tell the LLM to prioritize text and ignore irrelevant images.
    system_prompt = """
    You are a helpful and knowledgeable AI News Assistant. 
    You have access to a database of retrieved articles (text chunks) and images.

    CORE INSTRUCTIONS:
    1. **PRIORITY:** Your primary goal is to answer the User's Query using the provided TEXT chunks. 
    2. **IMAGES:** You will see images in the context. 
       - ONLY refer to them if they are directly relevant to the question (e.g., if the user asks for a chart, or the image shows the specific robot discussed).
       - If the images are generic stock photos, cartoons, or unrelated to the specific topic (like "Qwen3"), **IGNORE THEM**. Do not describe them.
    3. **FALLBACK:** If the text chunks contain the answer, use them! Do not say "I don't know" just because the images are irrelevant.
    4. **Formatting:** Use Markdown. Be concise but detailed.
    """

    # 2. Build the Message Payload
    # We combine text strings and image blobs into the format GPT-4o expects.
    user_content = []
    
    # Add the Query text first
    user_content.append({"type": "text", "text": f"User Question: {query}\n\nRetrieved Context:"})

    # Process items
    for item in retrieved_items:
        # A. Text Item
        if item['type'] == 'text':
            snippet = f"\n- [Source: {item['title']}] {item['content']}"
            user_content.append({"type": "text", "text": snippet})
        
        # B. Image Item
        elif item['type'] == 'image' and item.get('image_path'):
            try:
                # We must read the local file and encode it to base64
                with open(item['image_path'], "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Add image to payload
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low" # 'low' is cheaper and usually sufficient
                    }
                })
            except Exception as e:
                print(f"Error loading image {item['image_path']}: {e}")

    # 3. Send to LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # or "gpt-4o" if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=500,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"