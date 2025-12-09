import sys
import os
from pathlib import Path

# --- FIX: Add Project Root to Path ---
# This ensures Python can find 'src' modules regardless of where the script is run
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

import streamlit as st
import time

# Import our RAG modules
from src.rag.retriever import MultimodalRetriever
from src.rag.generator import generate_answer

# ----------------------
# Page Config
# ----------------------
st.set_page_config(page_title="The Batch RAG", layout="wide")

st.title("🤖 The Batch Multimodal RAG")
st.markdown("Ask questions about **AI news**, and I'll find answers from *The Batch* articles and images.")

# ----------------------
# Load Resources (Cached)
# ----------------------
@st.cache_resource
def load_retriever():
    """Load the retriever only once to save time."""
    try:
        return MultimodalRetriever()
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None

retriever = load_retriever()

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieval Count (Top K)", 1, 10, 3)
    st.info("This controls how many articles/images are sent to the LLM.")

# ----------------------
# Main Interaction
# ----------------------
query = st.text_input("Enter your question:", placeholder="e.g., What is new in robotics?")

if st.button("Search & Answer") and query:
    if not retriever:
        st.error("Retriever is not ready. Did you run the indexer?")
        st.stop()

    with st.spinner("🔍 Retrieving relevant context..."):
        # 1. Retrieve
        start_time = time.time()
        retrieved_items = retriever.search(query, k=top_k)
        retrieval_time = time.time() - start_time

    # 2. Display Retrieved Items (Context)
    with st.expander(f"📂 View Retrieved Context ({len(retrieved_items)} items)", expanded=False):
        st.caption(f"Retrieval took {retrieval_time:.2f}s")
        
        cols = st.columns(len(retrieved_items))
        for idx, item in enumerate(retrieved_items):
            with cols[idx]:
                score = item.get('score', 0)
                st.markdown(f"**{item['title']}**")
                st.caption(f"Score: {score:.4f}")
                
                # Show Image if available
                if item['type'] == 'image' and item.get('image_path'):
                    st.image(item['image_path'], use_container_width=True)
                # Show Text snippet
                else:
                    st.text(f"{item.get('content', '')[:150]}...")

    # 3. Generate Answer
    with st.spinner("💡 Generating answer with GPT-4o..."):
        answer = generate_answer(query, retrieved_items)

    # 4. Show Output
    st.markdown("### 📝 Answer")
    st.markdown(answer)

    # Success/Source info
    st.success("Answer generated based on retrieved multimodal data.")