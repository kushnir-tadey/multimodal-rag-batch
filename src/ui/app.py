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
# Helper: De-Duplicate & Aggregate
# ----------------------
def process_results_for_llm(results):
    """
    1. Identify Unique Docs.
    2. AGGREGATE chunks from the same doc into one big text block.
    """
    unique_docs_map = {} # Url -> {id, title, chunks}
    unique_docs_list = [] # For UI display
    
    text_results = [item for item in results if item['type'] == 'text']
    
    for item in text_results[:75]:
        url = item.get('url', '#')
        title = item.get('title', 'Unknown Title')
        content = item.get('content', '')
        
        if url not in unique_docs_map:
            new_id = len(unique_docs_list) + 1
            unique_docs_map[url] = {
                'id': new_id,
                'title': title,
                'url': url,
                'content_parts': []
            }
            unique_docs_list.append({'id': new_id, 'title': title, 'url': url})
            
        unique_docs_map[url]['content_parts'].append(content)

    # The LLM will receive 1 item per unique URL.
    llm_context = []
    for url, data in unique_docs_map.items():
        # Combine all chunks into one text
        full_text = "\n...\n".join(data['content_parts'])
        llm_context.append({
            "type": "text",
            "id": data['id'],       
            "title": data['title'],
            "content": full_text    
        })

    return unique_docs_list, llm_context

# ----------------------
# Helper: Clickable Citations
# ----------------------
def make_citations_clickable(text, unique_docs):
    def replace_match(match):
        try:
            doc_id = int(match.group(1))
            doc = next((d for d in unique_docs if d['id'] == doc_id), None)
            if doc:
                return f"[[{doc_id}]]({doc['url']})"
        except:
            pass
        return match.group(0)
    return re.sub(r'\[(\d+)\]', replace_match, text)

# ----------------------
# Session State
# ----------------------
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'generated_answer' not in st.session_state:
    st.session_state.generated_answer = ""
if 'unique_docs' not in st.session_state: 
    st.session_state.unique_docs = []

@st.cache_resource
def load_retriever():
    try:
        return MultimodalRetriever()
    except Exception as e:
        return None

retriever = load_retriever()
max_items = retriever.total_items if retriever else 10

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.header("Settings")
    slider_max = min(max_items, 100) if max_items > 0 else 10
    top_k = st.slider("Retrieval Context (Top K)", 1, slider_max, min(20, slider_max))
    
    if retriever:
        st.info(f"📚 Database contains {max_items} searchable items.")
    else:
        st.error("⚠️ Index not found.")
        
    st.divider()
    temperature = st.slider("Creativity", 0.0, 1.0, 0.0, 0.1)
    items_per_page = st.number_input("Grid items per page", 3, 12, 6)

# ----------------------
# Main Search
# ----------------------
with st.form("search_form"):
    query = st.text_input("Enter your question:", placeholder="e.g., What is new in robotics?")
    submit_button = st.form_submit_button("Search & Answer")

if submit_button and query:
    if not retriever:
        st.error("Retriever not ready.")
        st.stop()

    with st.spinner("🔍 Retrieving & Generating..."):
        # 1. Search
        results = retriever.search(query, k=top_k)
        
        # Keyword Boost
        query_terms = [t.lower() for t in query.split() if len(t) > 3]
        for item in results:
            content = (item.get('title', '') + " " + item.get('content', '')).lower()
            if any(t in content for t in query_terms):
                item['score'] = item.get('score', 0) + 0.2
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 2. Process for LLM (Aggregation Logic)
        unique_docs, text_context_for_llm = process_results_for_llm(results)
        
        # 3. Add Images
        image_results = [item for item in results if item['type'] == 'image'][:3]
        final_context = text_context_for_llm + image_results

        # 4. Generate
        ans = generate_answer(query, final_context, temperature=temperature)
        
        # 5. Save State
        st.session_state.search_results = results
        st.session_state.generated_answer = ans
        st.session_state.unique_docs = unique_docs
        st.session_state.current_page = 1

# ----------------------
# Display Results
# ----------------------
if st.session_state.search_results:
    # 1. Answer
    st.markdown("### 📝 Answer")
    if st.session_state.generated_answer:
        final_ans = make_citations_clickable(st.session_state.generated_answer, st.session_state.unique_docs)
        st.markdown(final_ans)

        # 2. Unique Sources
        if st.session_state.unique_docs:
            with st.expander("📚 Sources / References", expanded=False):
                st.caption("Detailed list of unique articles used:")
                for doc in st.session_state.unique_docs:
                    st.markdown(f"**[{doc['id']}]** [{doc['title']}]({doc['url']})")

    # 3. Images
    top_images = [item for item in st.session_state.search_results if item['type'] == 'image'][:3]
    if top_images:
        st.divider()
        st.markdown("### 🖼️ Relevant Images")
        cols = st.columns(3)
        for i, img in enumerate(top_images):
            with cols[i]:
                st.image(img['image_path'], width="stretch")

    st.divider()
    
    # 4. Grid (Chunks)
    results = st.session_state.search_results
    total_items = len(results)
    total_pages = math.ceil(total_items / items_per_page)
    
    if st.session_state.current_page > total_pages: st.session_state.current_page = total_pages
    start = (st.session_state.current_page - 1) * items_per_page
    batch = results[start : start + items_per_page]

    with st.expander(f"📂 View All Retrieved Context ({len(results)} items)", expanded=False):
        cols = st.columns(3)
        for idx, item in enumerate(batch):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**{item.get('title', '')[:50]}**")
                    st.caption(f"Score: {item.get('score', 0):.4f}")
                    if item['type'] == 'image':
                        st.image(item['image_path'])
                    else:
                        st.text(f"{item['content'][:100]}...")
                        
        if total_pages > 1:
            c1, _, c2 = st.columns([1, 8, 1])
            if c1.button("Prev") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()
            if c2.button("Next") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()