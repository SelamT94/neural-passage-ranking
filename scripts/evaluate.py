# scripts/evaluate.py

import os
import pandas as pd
from tqdm import tqdm
from src.retriever import BM25Retriever
from src.reranker import BertReranker

def main():
    """
    Main function to run the full evaluation pipeline and calculate MRR@10.
    """
    # --- Load Data ---
    # The evaluation data (queries and relevance judgments)
    output_dir = "data"
    queries_path = os.path.join(output_dir, "queries.dev.small.tsv")
    qrels_path = os.path.join(output_dir, "qrels.dev.small.tsv")

    print("Loading queries and qrels for evaluation...")
    queries_df = pd.read_csv(queries_path, sep='\t', names=['qid', 'query'])
    qrels_df = pd.read_csv(qrels_path, sep='\t', names=['qid', 'iter', 'pid', 'relevance'])

    # Create helper dictionaries for faster lookup
    query_map = dict(zip(queries_df['qid'], queries_df['query']))
    # The qrels file has a single relevant passage per query for MS MARCO v1.1 dev set
    qrels_map = dict(zip(qrels_df['qid'], qrels_df['pid']))
    
    # --- Initialize Retriever and Reranker ---
    print("\nInitializing retrieval and re-ranking models...")
    retriever = BM25Retriever()
    reranker = BertReranker()

    # --- Run Evaluation ---
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")
    
    reciprocal_ranks = []
    
    # We will evaluate on all queries in our small dev set
    for qid, query_text in tqdm(query_map.items(), desc="Evaluating Queries"):
        
        # --- Stage 1: Retrieve top-100 candidates with BM25 ---
        candidate_passages = retriever.search(query_text, k=100)
        
        # --- Stage 2: Rerank candidates with BERT Cross-Encoder ---
        reranked_results = reranker.rerank(query_text, candidate_passages)
        
        # --- Calculate Reciprocal Rank ---
        relevant_pid = qrels_map.get(qid)
        if relevant_pid is not None:
            found = False
            for rank, (pid, _, _) in enumerate(reranked_results):
                # We care about the rank of the *first* relevant document
                if pid == relevant_pid:
                    # Ranks are 1-based, list indices are 0-based
                    rank_position = rank + 1
                    
                    # We only consider hits within the top 10 for MRR@10
                    if rank_position <= 10:
                        reciprocal_ranks.append(1 / rank_position)
                    else:
                        reciprocal_ranks.append(0) # Not found in top 10
                        
                    found = True
                    break
            
            # If the relevant passage was not in the top 100 candidates from BM25,
            # it's not possible to rerank it, so the reciprocal rank is 0.
            if not found:
                reciprocal_ranks.append(0)
    
    # --- Calculate Final MRR@10 Score ---
    if reciprocal_ranks:
        mrr_at_10 = sum(reciprocal_ranks) / len(reciprocal_ranks)
        print("\n" + "="*50)
        print(f"Evaluation Complete! 🎉")
        print(f"Final MRR@10 Score: {mrr_at_10:.4f}")
        print("="*50)
    else:
        print("\nNo queries found for evaluation. Please check your data files.")

if __name__ == '__main__':
    main()
