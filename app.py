# app.py

import os
import sys
import streamlit as st

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retriever import BM25Retriever
from src.reranker import BertReranker
from src.llm_refiner import LLMRefiner

# --- Initialize Models with Caching ---
@st.cache_resource
def load_models():
    """
    Loads and returns the models.
    """
    try:
        retriever = BM25Retriever()
        reranker = BertReranker()
        llm_refiner = LLMRefiner(api_key='AIzaSyCoF_qrgNdqIG1JXRZZ53YG1_3ciKBXFNI')
        return retriever, reranker, llm_refiner
    except Exception as e:
        st.error(f"Error during model initialization: {e}")
        st.warning("Please ensure you have run 'python scripts/download_msmarco.py' "
                   "and the 'src' folder contains the necessary files.")
        st.stop()

# --- Main Streamlit App ---

def main():
    """
    The main function that builds the Streamlit UI.
    """
    st.set_page_config(page_title="Neural Passage Ranker", page_icon="🧠")
    st.title("🧠 Neural Passage Ranker")
    st.markdown(
        """
        This is a multi-stage retrieval system. It first uses **BM25** to find candidates,
        then a **BERT Cross-Encoder** to re-rank them, and finally an **LLM** to refine the top answer.
        """
    )

    # Load the models using the cached function
    retriever, reranker, llm_refiner = load_models()
    
    # Text input for the user query
    query = st.text_input("Enter your search query:")
    
    if st.button("Search") and query:
        with st.spinner("Executing search pipeline..."):
            
            # --- Stage 1: Candidate Retrieval with BM25 ---
            st.header("Stage 1: BM25 Retrieval")
            st.info("Finding top 100 candidate passages...")
            candidate_passages = retriever.search(query, k=100)
            
            if not candidate_passages:
                st.warning("No passages found by BM25. Try a different query.")
                return

            st.markdown(f"Found **{len(candidate_passages)}** candidates.")
            
            # This is the section that displays the BM25 results
            st.text("Top 5 BM25-ranked candidates:")
            for i, (_, passage) in enumerate(candidate_passages[:5]):
                st.markdown(f"**{i+1}.** {passage[:150]}...")
            
            # --- Stage 2: Semantic Re-ranking with BERT ---
            st.header("Stage 2: BERT Re-ranking")
            st.info("Re-ranking candidates for semantic relevance...")
            reranked_results = reranker.rerank(query, candidate_passages)
            
            # --- Display Final Results ---
            st.header("Final Top 5 Ranked Passages")
            
            if reranked_results:
                # --- Stage 3: LLM Refinement ---
                top_pid, top_passage, top_score = reranked_results[0]
                
                with st.spinner("Refining top result with LLM..."):
                    top_passage_refined = llm_refiner.refine_passage(top_passage)
                
                st.subheader(f"🥇 **Top Result** (Score: {top_score:.4f})")
                
                st.markdown(f"**Original Passage:**")
                st.markdown(f"_{top_passage}_")
                st.markdown("---")
                
                st.markdown(f"**LLM Refined Passage:**")
                st.success(top_passage_refined)
                st.markdown("---")
            
                for i, (_, passage, score) in enumerate(reranked_results[1:5]):
                    st.markdown(f"**Rank {i+2}.** (Score: {score:.4f})")
                    st.markdown(passage[:150] + "...")
                    
                    with st.expander("Click to read full passage"):
                        st.markdown(passage)
                    st.markdown("---")
            else:
                st.warning("No re-ranked results found.")

if __name__ == '__main__':
    main()
