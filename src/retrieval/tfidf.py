"""TF-IDF based retriever for philosophical text passages."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle
from typing import List, Tuple


class TFIDFRetriever:
    """TF-IDF based document retriever with cosine similarity."""

    def __init__(self, corpus_df=None, max_features=5000, min_df=1, max_df=0.95):
        self.corpus_df = corpus_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, lowercase=True,
            min_df=min_df, max_df=max_df, ngram_range=(1, 2),
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit(self, texts):
        """Fit the vectorizer on a collection of texts."""
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant passages for a query."""
        if not self.is_fitted:
            raise ValueError("Retriever has not been fitted. Call fit() first.")
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            text = self.corpus_df.iloc[idx]['text']
            score = float(similarities[idx])
            results.append((idx, text, score))
        return results

    def get_source(self, idx):
        """Get metadata (philosopher, work) for a passage by index."""
        if self.corpus_df is not None:
            row = self.corpus_df.iloc[idx]
            return {
                'philosopher': row.get('philosopher', 'Unknown'),
                'work': row.get('work', 'Unknown')
            }
        return {'philosopher': 'Unknown', 'work': 'Unknown'}

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'corpus_df': self.corpus_df
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.corpus_df = data['corpus_df']
            self.is_fitted = True
