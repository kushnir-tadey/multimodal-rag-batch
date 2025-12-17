import sys
import os
import math
import time
from pathlib import Path

# --- Path Setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

import streamlit as st

from src.rag.retriever import MultimodalRetriever
from src.rag.generator import generate_answer

# ----------------------
# Page Config
# ----------------------
st.set_page_config(page_title="The Batch RAG", layout="wide")

st.title("🤖 The Batch Multimodal RAG")
st.markdown("Ask questions about **AI news**, and I'll find answers from *The Batch* articles and images.")

# ----------------------
# Session State Setup
# ----------------------
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'generated_answer' not in st.session_state:
    st.session_state.generated_answer = ""
if 'retrieval_time' not in st.session_state:
    st.session_state.retrieval_time = 0.0

# ----------------------
# Load Resources (Cached)
# ----------------------
@st.cache_resource
def load_retriever():
    try:
        return MultimodalRetriever()
    except Exception as e:
        return None

retriever = load_retriever()

# Determine max limit safely
max_items = retriever.total_items if retriever else 10
if max_items < 1: max_items = 1

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.header("Settings")
    
    # Recall Control
    default_k = min(20, max_items)
    top_k = st.slider("Retrieval Context (Top K)", 1, max_items, default_k)
    
    if retriever:
        st.info(f"📚 Database contains {max_items} searchable chunks/images.")
    else:
        st.error("⚠️ Index not found. Please run `python -m src.indexing.indexer`.")
    
    st.divider()
    
    items_per_page = st.number_input("Grid items per page", min_value=3, max_value=12, value=6, step=3)

# ----------------------
# Main Search Interaction
# ----------------------
with st.form(key="search_form"):
    query = st.text_input("Enter your question:", placeholder="e.g., What is new in robotics?")
    submit_button = st.form_submit_button("Search & Answer")

if submit_button and query:
    if not retriever:
        st.error("❌ Retriever is not ready.")
        st.stop()

    with st.spinner("🔍 Retrieving & Generating..."):
        # 1. Retrieve
        start_time = time.time()
        results = retriever.search(query, k=top_k)
        retrieval_time = time.time() - start_time
        
        # --- KEYWORD BOOSTING (Re-Ranking) ---
        # Vector search finds "concepts" (e.g., AI performance), but might miss specific names (e.g., Qwen3).
        # We manually boost the score of any result that contains the exact keywords.
        query_terms = [term.lower() for term in query.split() if len(term) > 3] # Filter out short words
        
        for item in results:
            # Combine title and content to check for keywords
            text_to_check = (item.get('title', '') + " " + item.get('content', '')).lower()
            
            for term in query_terms:
                if term in text_to_check:
                    # Add a significant boost (0.2) to push these items to the top
                    item['score'] = item.get('score', 0) + 0.2
        
        # Re-sort the results based on the new boosted scores
        results.sort(key=lambda x: x['score'], reverse=True)
        # -------------------------------------

        # 2. Store in Session State
        st.session_state.search_results = results
        st.session_state.current_page = 1
        st.session_state.retrieval_time = retrieval_time
        
        # 3. Generate Answer (SMART SPLIT LIMIT)
        # Strategy: Send LOTS of text (to find deep stats) but FEW images (to prevent timeouts).
        
        # Separate the results by type
        text_results = [item for item in results if item['type'] == 'text']
        image_results = [item for item in results if item['type'] == 'image']
        
        # Limit Text to 75 chunks (~3500 words) -> Deep Context
        # Limit Images to 3 items -> Safe Payload size
        final_context = text_results[:75] + image_results[:3]
        
        try:
            st.session_state.generated_answer = generate_answer(query, final_context)
        except Exception as e:
            st.session_state.generated_answer = f"⚠️ Error generating answer: {str(e)}"

# ----------------------
# Display Results
# ----------------------
if st.session_state.search_results:
    results = st.session_state.search_results
    
    # --- 1. Answer Section ---
    st.markdown("### 📝 Answer")
    if st.session_state.generated_answer:
        st.markdown(st.session_state.generated_answer)
        st.caption(f"Answer generated using top {min(top_k, 75)} text chunks + top 3 images.")
    
    # --- 2. NEW: Top Visual Matches (The "Show Me" Section) ---
    # If we found images, show the best 3 immediately so the user sees them.
    top_images = [item for item in results if item['type'] == 'image'][:3]
    
    if top_images:
        st.divider()
        st.markdown("### 🖼️ Relevant Images")
        img_cols = st.columns(3)
        for i, img_item in enumerate(top_images):
            with img_cols[i]:
                # Use container to align captions nicely
                with st.container(border=True):
                    st.image(img_item['image_path'], width="stretch")
                    st.caption(img_item.get('title', '')[:60] + "...")

    st.divider()

    # --- 3. Pagination & Grid Logic ---
    total_items = len(results)
    total_pages = math.ceil(total_items / items_per_page)
    
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages

    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_batch = results[start_idx:end_idx]

    # Use an expander so the full grid is optional
    with st.expander(f"📂 View All Retrieved Context ({len(results)} items found)", expanded=False):
        st.caption(f"Page {st.session_state.current_page}/{total_pages} | Retrieval: {st.session_state.retrieval_time:.4f}s")
        
        cols = st.columns(3)
        for idx, item in enumerate(current_batch):
            col_idx = idx % 3
            with cols[col_idx]:
                with st.container(border=True):
                    # Show Title
                    title = item.get('title', 'No Title')
                    if len(title) > 60: title = title[:60] + "..."
                    st.markdown(f"**{title}**")
                    
                    st.caption(f"Score: {item.get('score', 0):.4f}")
                    
                    # Show Image or Text Chunk
                    if item['type'] == 'image' and item.get('image_path'):
                        st.image(item['image_path'], width="stretch")
                    else:
                        st.text(f"{item.get('content', '')[:120]}...")

        # Pagination Buttons
        if total_pages > 1:
            c1, c2, c3 = st.columns([1, 8, 1])
            with c1:
                if st.button("Previous"):
                    if st.session_state.current_page > 1:
                        st.session_state.current_page -= 1
                        st.rerun()
            with c3:
                if st.button("Next"):
                    if st.session_state.current_page < total_pages:
                        st.session_state.current_page += 1
                        st.rerun()