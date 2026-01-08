import sys
import os
import math
import time
import re
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
# Helper Function: Clickable Citations
# ----------------------
def make_citations_clickable(text, documents):
    """
    Finds '[1]', '[2]' in the text and replaces them with markdown links 
    like '[[1]](http://link_to_source)'.
    """
    def replace_match(match):
        try:
            doc_index = int(match.group(1)) - 1  # Convert '1' to index 0
            if 0 <= doc_index < len(documents):
                url = documents[doc_index].get('url', '#')
                # Return markdown link: [[1]](http://google.com)
                return f"[[{match.group(1)}]]({url})"
        except:
            pass
        return match.group(0)

    # Regex to find [1], [2], [10], etc.
    return re.sub(r'\[(\d+)\]', replace_match, text)

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
    slider_max = min(max_items, 100) 
    default_k = min(20, slider_max)
    
    top_k = st.slider("Retrieval Context (Top K)", 1, slider_max, default_k)
    
    if retriever:
        st.info(f"📚 Database contains {max_items} searchable chunks/images.")
    else:
        st.error("⚠️ Index not found. Please run `python -m src.indexing.indexer`.")
    
    st.divider()
    
    st.header("🤖 Model Behavior")
    temperature = st.slider(
        "Creativity (Temperature)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.1,
        help="0.0 = Precise/Factual. 1.0 = Creative/Random."
    )

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
        query_terms = [term.lower() for term in query.split() if len(term) > 3] 
        
        for item in results:
            text_to_check = (item.get('title', '') + " " + item.get('content', '')).lower()
            for term in query_terms:
                if term in text_to_check:
                    item['score'] = item.get('score', 0) + 0.2
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 2. Store in Session State
        st.session_state.search_results = results
        st.session_state.current_page = 1
        st.session_state.retrieval_time = retrieval_time
        
        # 3. Generate Answer
        text_results = [item for item in results if item['type'] == 'text']
        image_results = [item for item in results if item['type'] == 'image']
        
        # Use top 75 chunks for context
        final_context = text_results[:75] + image_results[:3]
        
        try:
            st.session_state.generated_answer = generate_answer(query, final_context, temperature=temperature)
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
        
        # A. Make Citations Clickable (The Perplexity Style)
        # We need the exact list of text docs passed to the LLM to map [1] -> URL
        used_text_docs = [item for item in results if item['type'] == 'text'][:75]
        
        clickable_answer = make_citations_clickable(st.session_state.generated_answer, used_text_docs)
        st.markdown(clickable_answer)
        
        # B. Source Details (Bottom Expanders)
        if used_text_docs:
            with st.expander("📚 Sources / References (Details)", expanded=False):
                st.caption("Detailed list of sources used in this answer:")
                for i, item in enumerate(used_text_docs, 1):
                    title = item.get('title', 'Unknown Article')
                    url = item.get('url', '#')
                    st.markdown(f"**[{i}]** [{title}]({url})")
                    
        st.caption(f"Answer generated using top {len(used_text_docs)} text chunks + top 3 images.")
    
    # --- 2. Top Visual Matches ---
    top_images = [item for item in results if item['type'] == 'image'][:3]
    
    if top_images:
        st.divider()
        st.markdown("### 🖼️ Relevant Images")
        img_cols = st.columns(3)
        for i, img_item in enumerate(top_images):
            with img_cols[i]:
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

    with st.expander(f"📂 View All Retrieved Context ({len(results)} items found)", expanded=False):
        st.caption(f"Page {st.session_state.current_page}/{total_pages} | Retrieval: {st.session_state.retrieval_time:.4f}s")
        
        cols = st.columns(3)
        for idx, item in enumerate(current_batch):
            col_idx = idx % 3
            with cols[col_idx]:
                with st.container(border=True):
                    score = item.get('score', 0)
                    title = item['title']
                    if len(title) > 50: title = title[:50] + "..."
                    
                    st.markdown(f"**{title}**")
                    st.caption(f"Score: {score:.4f}")
                    
                    if item['type'] == 'image' and item.get('image_path'):
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