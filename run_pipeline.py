# run_pipeline.py

import os
import sys

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retriever import BM25Retriever
from src.reranker import BertReranker
from src.llm_refiner import LLMRefiner

def main():
    # --- Initialize all components ---
    retriever = BM25Retriever()
    reranker = BertReranker()
    llm_refiner = LLMRefiner(api_key='AIzaSyCoF_qrgNdqIG1JXRZZ53YG1_3ciKBXFNI')

    # --- Define a query ---
    query = "function of the human heart"

    print("\n" + "="*50)
    print(f"Executing search for query: '{query}'")
    print("="*50 + "\n")

    # --- Stage 1: Candidate Retrieval ---
    print("--- Stage 1: Retrieving candidates with BM25 ---")
    candidate_passages = retriever.search(query, k=100)
    print(f"Retrieved {len(candidate_passages)} candidates.")

    # --- Stage 2: Semantic Re-ranking ---
    print("\n--- Stage 2: Re-ranking with BERT Cross-Encoder ---")
    reranked_results = reranker.rerank(query, candidate_passages)
    print("Re-ranking complete.")

    # --- Stage 3: LLM Refinement ---
    if reranked_results:
        top_passage_original = reranked_results[0][1] # Get the top-ranked passage text
        print("\n--- Stage 3: LLM Refinement of Top Passage ---")
        top_passage_refined = llm_refiner.refine_passage(top_passage_original)
        
        print("\nOriginal Top Passage:")
        print(f"  {top_passage_original}")
        print("\nLLM Refined Passage:")
        print(f"  {top_passage_refined}")
    
    print("\n" + "="*50)
    print("Pipeline execution finished! 🎉")

if __name__ == '__main__':
    main()
