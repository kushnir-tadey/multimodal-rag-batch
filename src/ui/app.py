import sys
import os
import math
import time
from pathlib import Path

# --- FIX: Add Project Root to Path ---
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

# ----------------------
# Load Resources (Cached)
# ----------------------
@st.cache_resource
def load_retriever():
    try:
        return MultimodalRetriever()
    except Exception as e:
        st.error(f"Failed to load index: {e}")
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
    # Let user retrieve MANY items (e.g. 20 or 50) to see them in the UI
    top_k = st.slider("Retrieval Context (Top K)", 1, max_items, min(10, max_items))
    st.info(f"Database contains {max_items} items.")
    
    st.divider()
    
    # Pagination Settings
    items_per_page = st.number_input("Items per page", min_value=3, max_value=12, value=6, step=3)

# ----------------------
# Main Search Interaction
# ----------------------
with st.form(key="search_form"):
    query = st.text_input("Enter your question:", placeholder="e.g., What is new in robotics?")
    submit_button = st.form_submit_button("Search & Answer")

if submit_button and query:
    if not retriever:
        st.error("Retriever is not ready.")
        st.stop()

    with st.spinner("🔍 Retrieving & Generating..."):
        # 1. Retrieve
        start_time = time.time()
        results = retriever.search(query, k=top_k)
        retrieval_time = time.time() - start_time
        
        # 2. Store in Session State (Reset Page to 1)
        st.session_state.search_results = results
        st.session_state.current_page = 1
        st.session_state.retrieval_time = retrieval_time
        
        # 3. Generate Answer (OPTIMIZATION: Only send Top 5 to LLM)
        # This prevents 429 Rate Limit errors on Free Tier
        llm_context = results[:5] 
        st.session_state.generated_answer = generate_answer(query, llm_context)

# ----------------------
# Display Results (Paginated)
# ----------------------
if st.session_state.search_results:
    results = st.session_state.search_results
    
    # --- 1. Show the Answer First ---
    st.markdown("### 📝 Answer")
    st.markdown(st.session_state.generated_answer)
    st.divider()

    # --- 2. Pagination Logic ---
    total_items = len(results)
    total_pages = math.ceil(total_items / items_per_page)
    
    # Ensure current page is valid
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages

    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_batch = results[start_idx:end_idx]

    # --- 3. Context Expandable with Pagination ---
    with st.expander(f"📂 View Retrieved Context ({len(results)} items found)", expanded=True):
        st.caption(f"Showing items {start_idx + 1}-{min(end_idx, total_items)} of {total_items} | Page {st.session_state.current_page}/{total_pages}")
        
        # Display Grid
        cols = st.columns(3)
        for idx, item in enumerate(current_batch):
            col_idx = idx % 3
            with cols[col_idx]:
                with st.container(border=True):
                    # Truncate title
                    title = item['title']
                    if len(title) > 60: title = title[:60] + "..."
                    
                    st.markdown(f"**{title}**")
                    st.caption(f"Score: {item.get('score', 0):.4f}")
                    
                    if item['type'] == 'image' and item.get('image_path'):
                        # FIX: Use 'use_container_width' to fix warning
                        st.image(item['image_path'], use_container_width=True)
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