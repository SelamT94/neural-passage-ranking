# run_pipeline.py

from src.retriever import BM25Retriever
from src.reranker import BertReranker

def main():
    # --- Initialize both components ---
    retriever = BM25Retriever()
    reranker = BertReranker()

    # --- Define a query ---
    # Try changing this to other queries!
    query = "function of the human heart"

    print("\n" + "="*50)
    print(f"Executing search for query: '{query}'")
    print("="*50 + "\n")

    # --- Stage 1: Candidate Retrieval ---
    # Get the top 100 candidate passages from BM25
    print("--- Stage 1: Retrieving candidates with BM25 ---")
    candidate_passages = retriever.search(query, k=100)

    print(f"Retrieved {len(candidate_passages)} candidates.")
    print("Top 5 candidates from BM25:")
    for i, (_, passage) in enumerate(candidate_passages[:5]):
        print(f"  {i+1}. {passage[:120]}...")

    # --- Stage 2: Semantic Re-ranking ---
    print("\n--- Stage 2: Re-ranking with BERT Cross-Encoder ---")
    reranked_passages = reranker.rerank(query, candidate_passages)

    print("Re-ranking complete.")
    print("\nFinal Top 5 Ranked Passages:")
    for i, (_, passage, score) in enumerate(reranked_passages[:5]):
        print(f"  {i+1}. (Score: {score:.4f}) {passage[:120]}...")

    print("\nPipeline execution finished! 🎉")

if __name__ == '__main__':
    main()