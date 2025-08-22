# src/reranker.py

from sentence_transformers import CrossEncoder

class BertReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print("Initializing BERT Reranker...")
        # Load the pre-trained Cross-Encoder model
        self.model = CrossEncoder(model_name)
        print("BERT Reranker loaded successfully. ✅")

    def rerank(self, query, passages):
        """
        Reranks a list of passages for a given query.
        'passages' is expected to be a list of tuples: (pid, passage_text)

        Returns a sorted list of tuples: (pid, passage_text, score)
        """
        # Create pairs of [query, passage_text] for the model
        query_passage_pairs = [[query, p[1]] for p in passages]

        # Get the scores from the model
        scores = self.model.predict(query_passage_pairs)

        # Combine pids, passages, and scores
        results_with_scores = []
        for i in range(len(passages)):
            pid = passages[i][0]
            passage_text = passages[i][1]
            score = scores[i]
            results_with_scores.append((pid, passage_text, score))

        # Sort the results by score in descending order
        results_with_scores.sort(key=lambda x: x[2], reverse=True)

        return results_with_scores

# This block allows us to test the reranker directly
if __name__ == '__main__':
    reranker = BertReranker()

    test_query = "what is the capital of france"
    # Let's create some dummy candidate passages from BM25
    dummy_passages = [
        (101, "Paris is known for the Eiffel Tower."),
        (102, "The capital of France is Paris, which is a major European city."),
        (103, "What is the primary function of a capital city?"),
        (104, "London is the capital of the United Kingdom.")
    ]

    print(f"\nPerforming re-ranking for query: '{test_query}'")

    reranked_results = reranker.rerank(test_query, dummy_passages)

    print("\nTop re-ranked results:")
    for i, (pid, passage, score) in enumerate(reranked_results):
        print(f"{i+1}. PID: {pid}, Score: {score:.4f}")
        print(f"   Passage: {passage[:100]}...")