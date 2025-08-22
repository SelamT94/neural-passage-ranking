# scripts/download_msmarco.py

import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

print("Using Hugging Face datasets to download and process MS MARCO.")

# --- Configuration ---
# Let's create a smaller, manageable collection for this project.
# 500,000 is a good number for development.
MAX_PASSAGES = 500_000
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load dataset from Hugging Face ---
# This will download and cache the data.
print("Loading MS MARCO from Hugging Face (v1.1)...")
# Note: The 'dev' split is not available for this dataset version.
# The correct split for development is 'validation'.
dataset = load_dataset("ms_marco", "v1.1")
print("Dataset loaded successfully.")

# --- Part 1: Create the Passage Collection and a Lookup Map ---
print(f"\nProcessing passages to create a collection of size {MAX_PASSAGES}...")

passages_to_write = []
passage_to_pid_map = {} # A helper to find the PID of a passage text later
seen_passages = set() # To avoid duplicates in our collection

# We iterate through the 'train' set which contains a large set of passages
for record in tqdm(dataset['train'], desc="Building Collection"):
    # The 'passages' field is a dictionary itself
    for passage_text in record['passages']['passage_text']:
        if passage_text not in seen_passages:
            seen_passages.add(passage_text)
            
            # Assign a new, sequential passage ID (pid)
            pid = len(passages_to_write)
            passages_to_write.append([pid, passage_text])
            passage_to_pid_map[passage_text] = pid
            
            # Stop once we've collected enough passages
            if len(passages_to_write) >= MAX_PASSAGES:
                break
    if len(passages_to_write) >= MAX_PASSAGES:
        break

# Save the collection to a TSV file
collection_df = pd.DataFrame(passages_to_write, columns=['pid', 'passage'])
collection_path = os.path.join(OUTPUT_DIR, "collection.tsv")
collection_df.to_csv(collection_path, sep='\t', index=False, header=False)
print(f"Saved {len(collection_df)} unique passages to {collection_path}")


# --- Part 2: Create the Development Set Queries ---
print("\nProcessing dev set queries...")
dev_queries_to_write = []
# FIX: The 'dev' key does not exist. Using 'validation' instead.
for record in tqdm(dataset['validation'], desc="Processing Dev Queries"):
    qid = record['query_id']
    query_text = record['query']
    dev_queries_to_write.append([qid, query_text])

# Save the dev queries to a TSV file
queries_df = pd.DataFrame(dev_queries_to_write, columns=['qid', 'query'])
queries_path = os.path.join(OUTPUT_DIR, "queries.dev.small.tsv")
queries_df.to_csv(queries_path, sep='\t', index=False, header=False)
print(f"Saved {len(queries_df)} dev queries to {queries_path}")


# --- Part 3: Create Relevance Judgments (Qrels) for the Dev Set ---
print("\nProcessing dev set relevance judgments (qrels)...")
qrels_to_write = []
# FIX: The 'dev' key does not exist. Using 'validation' instead.
for record in tqdm(dataset['validation'], desc="Processing Qrels"):
    qid = record['query_id']
    
    # Find the single relevant passage using the 'is_selected' flag
    for i, is_selected in enumerate(record['passages']['is_selected']):
        if is_selected == 1:
            relevant_passage_text = record['passages']['passage_text'][i]
            
            # Look up the PID from the map we created earlier
            pid = passage_to_pid_map.get(relevant_passage_text)
            
            # Only include the qrel if the relevant passage is in our collection
            if pid is not None:
                # Format: qid, 0, pid, 1 (standard TREC format)
                qrels_to_write.append([qid, 0, pid, 1])
            break # MS MARCO v1.1 has only one relevant passage per query in dev

# Save the qrels to a TSV file
qrels_df = pd.DataFrame(qrels_to_write, columns=['qid', 'iter', 'pid', 'relevance'])
qrels_path = os.path.join(OUTPUT_DIR, "qrels.dev.small.tsv")
qrels_df.to_csv(qrels_path, sep='\t', index=False, header=False)
print(f"Saved {len(qrels_df)} qrels to {qrels_path}")

print("\nData preparation complete! ✨")
