# app.py

import os
import sys
import streamlit as st

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retriever import BM25Retriever
from src.reranker import BertReranker

# --- Initialize Models with Caching ---
# We use st.cache_resource to load these heavy models only once,
# even when the app re-runs due to user input. This is critical for performance.
@st.cache_resource
def load_models():
    """
    Loads and returns the BM25Retriever and BertReranker models.
    """
    try:
        retriever = BM25Retriever()
        reranker = BertReranker()
        return retriever, reranker
    except Exception as e:
        st.error(f"Error during model initialization: {e}")
        st.warning("Please ensure you have run 'python scripts/download_msmarco.py' "
                   "and the 'src' folder contains retriever.py and reranker.py.")
        st.stop()

# --- Main Streamlit App ---

def main():
    """
    The main function that builds the Streamlit UI and handles user interaction.
    """
    st.set_page_config(page_title="Neural Passage Ranker", page_icon="🧠")
    st.title("🧠 Neural Passage Ranker")
    st.markdown(
        """
        This is a two-stage retrieval system. It first uses **BM25** to find candidate passages
        and then uses a **BERT Cross-Encoder** to re-rank the results for better relevance.
        """
    )

    # Load the models using the cached function
    retriever, reranker = load_models()
    
    # Text input for the user query
    query = st.text_input("Enter your search query:")
    
    # Only run the pipeline if a query is entered and the button is clicked
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
            st.text("Top 5 BM25-ranked candidates:")
            
            for i, (_, passage) in enumerate(candidate_passages[:5]):
                st.markdown(f"**{i+1}.** {passage[:150]}...")
            
            # --- Stage 2: Semantic Re-ranking with BERT ---
            st.header("Stage 2: BERT Re-ranking")
            st.info("Re-ranking candidates for semantic relevance...")
            reranked_results = reranker.rerank(query, candidate_passages)
            
            # --- Display Final Results ---
            st.header("Final Top 5 Ranked Passages")
            
            # Check if there are any results to display
            if reranked_results:
                # --- Display the top answer prominently ---
                top_pid, top_passage, top_score = reranked_results[0]
                st.subheader(f"🥇 **Top Result** (Score: {top_score:.4f})")
                st.markdown(f"_{top_passage}_")
                st.markdown("---") # Add a separator below the top result
            
                for i, (_, passage, score) in enumerate(reranked_results[1:5]):
                    st.markdown(
                        f"**Rank {i+2}.** (Score: {score:.4f})"
                    )
                    # Show a snippet of the passage before the expander
                    st.markdown(passage[:150] + "...")
                    
                    # Use an expander to show the full passage
                    with st.expander("Click to read full passage"):
                        st.markdown(passage)
                    st.markdown("---") # Separator between results
            else:
                st.warning("No re-ranked results found.")

if __name__ == '__main__':
    main()
