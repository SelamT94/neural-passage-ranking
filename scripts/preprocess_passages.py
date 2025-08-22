# scripts/preprocess_passages.py

import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import os

# --- Download NLTK stopwords if not already ---
nltk.download('stopwords')

# --- Preprocessing functions ---
def lowercase_text(text):
    return text.lower()

def remove_html(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Optionally use BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    return text

def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_special_chars(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])

def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_html(text)
    text = normalize_whitespace(text)
    text = remove_special_chars(text)
    # Optional: remove stopwords
    # text = remove_stopwords(text)
    return text

# --- Main preprocessing pipeline ---
def main():
    DATA_DIR = "data"
    input_file = os.path.join(DATA_DIR, "collection.tsv")
    output_file = os.path.join(DATA_DIR, "collection_clean.tsv")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist. Please run download_msmarco.py first.")
        return

    print("Loading raw passages...")
    collection_df = pd.read_csv(input_file, sep='\t', names=['pid', 'passage'])

    print(f"Cleaning and preprocessing {len(collection_df)} passages...")
    collection_df['clean_passage'] = collection_df['passage'].apply(preprocess_text)

    print(f"Saving cleaned passages to {output_file} ...")
    collection_df[['pid', 'clean_passage']].to_csv(output_file, sep='\t', index=False, header=False)

    print("Preprocessing complete! ✅")

if __name__ == "__main__":
    main()
