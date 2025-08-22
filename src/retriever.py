# # src/retriever.py

# import pandas as pd
# from rank_bm25 import BM25Okapi
# from tqdm import tqdm

# class BM25Retriever:
#     def __init__(self, collection_path="data/collection.tsv"):
#         print("Initializing BM25 Retriever...")
#         self.collection_df = pd.read_csv(collection_path, sep='\t', names=['pid', 'passage'])

#         # For faster lookup
#         self.pid_to_passage = dict(zip(self.collection_df.pid, self.collection_df.passage))

#         print("Tokenizing the corpus for BM25... This might take a moment.")
#         # Simple whitespace tokenization
#         tokenized_corpus = [doc.split(" ") for doc in tqdm(self.collection_df.passage)]

#         self.bm25 = BM25Okapi(tokenized_corpus)
#         print("BM25 index built successfully. ✅")

#     def search(self, query, k=100):
#         """
#         Search for a query and return top k passages.
#         Returns a list of tuples: (pid, passage_text)
#         """
#         tokenized_query = query.split(" ")

#         # get_top_n returns scores, but we need pids. We'll get them from the doc_scores method.
#         doc_scores = self.bm25.get_scores(tokenized_query)

#         # Get the indices of the top-k scores
#         top_k_indices = doc_scores.argsort()[::-1][:k]

#         # Get the corresponding pids for these indices
#         top_k_pids = self.collection_df.iloc[top_k_indices]['pid'].tolist()

#         # Retrieve the passage text for the top pids
#         results = [(pid, self.pid_to_passage[pid]) for pid in top_k_pids]

#         return results

# # This block allows us to test the retriever directly
# if __name__ == '__main__':
#     retriever = BM25Retriever()

#     test_query = "what is the capital of france"
#     print(f"\nPerforming search for query: '{test_query}'")

#     search_results = retriever.search(test_query, k=5)

#     print("\nTop 5 BM25 search results:")
#     for i, (pid, passage) in enumerate(search_results):
#         print(f"{i+1}. PID: {pid}")
#         # Print only the first 100 chars of the passage
#         print(f"   Passage: {passage[:100]}...")

# src/retriever.py

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

class BM25Retriever:
    def __init__(self, collection_path="data/collection_clean.tsv", original_collection_path="data/collection.tsv"):
        print("Initializing BM25 Retriever...")

        # Load cleaned passages for indexing
        self.collection_df = pd.read_csv(collection_path, sep='\t', names=['pid', 'clean_passage'])

        # Load original passages for display
        original_df = pd.read_csv(original_collection_path, sep='\t', names=['pid', 'passage'])
        self.pid_to_original = dict(zip(original_df.pid, original_df.passage))

        print("Tokenizing the cleaned corpus for BM25... This might take a moment.")
        tokenized_corpus = [doc.split(" ") for doc in tqdm(self.collection_df.clean_passage)]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully. ✅")

    def search(self, query, k=100):
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = doc_scores.argsort()[::-1][:k]
        top_k_pids = self.collection_df.iloc[top_k_indices]['pid'].tolist()

        # Return original passages for display
        results = [(pid, self.pid_to_original[pid]) for pid in top_k_pids]
        return results
    
if __name__ == '__main__':
    retriever = BM25Retriever()

    test_query = "what is the capital of france"
    print(f"\nPerforming search for query: '{test_query}'")

    search_results = retriever.search(test_query, k=5)

    print("\nTop 5 BM25 search results:")
    for i, (pid, passage) in enumerate(search_results):
        print(f"{i+1}. PID: {pid}")
        # Print only the first 100 chars of the passage
        print(f"   Passage: {passage[:100]}...")
