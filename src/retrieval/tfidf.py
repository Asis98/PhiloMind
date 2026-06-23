"""TF-IDF based retriever for philosophical text passages with subject filtering and fallback."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle
from typing import List, Tuple, Optional


class TFIDFRetriever:
    """TF-IDF based document retriever with cosine similarity, subject filter and source tracking."""

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
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def retrieve(self, query: str, top_k: int = 3, subject_filter: Optional[str] = None) -> List[Tuple[int, str, float]]:
        if not self.is_fitted:
            raise ValueError("Retriever has not been fitted. Call fit() first.")
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        if subject_filter and self.corpus_df is not None and 'subject' in self.corpus_df.columns:
            mask = self.corpus_df['subject'].str.lower() == subject_filter.lower()
            similarities[~mask] = -1.0
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] < 0:
                continue
            text = self.corpus_df.iloc[idx]['text']
            score = float(similarities[idx])
            results.append((idx, text, score))
        return results

    def retrieve_with_boost(self, query: str, top_k: int = 3,
                            teacher_retriever: Optional['TFIDFRetriever'] = None,
                            boost_factor: float = 1.5) -> List[Tuple[int, str, float, str]]:
        teacher_results = []
        general_results = []
        if teacher_retriever and teacher_retriever.is_fitted:
            teacher_results = teacher_retriever.retrieve(query, top_k=top_k)
        general_results = self.retrieve(query, top_k=top_k)
        seen_texts = set()
        combined = []
        for idx, text, score in teacher_results:
            if text not in seen_texts:
                combined.append((idx, text, score * boost_factor, 'teacher'))
                seen_texts.add(text)
        for idx, text, score in general_results:
            if text not in seen_texts:
                combined.append((idx, text, score, 'general'))
                seen_texts.add(text)
        combined.sort(key=lambda x: x[2], reverse=True)
        return combined[:top_k]

    def get_source(self, idx):
        if self.corpus_df is not None:
            row = self.corpus_df.iloc[idx]
            source = {
                'philosopher': row.get('philosopher', 'Unknown'),
                'work': row.get('work', 'Unknown'),
            }
            if 'subject' in self.corpus_df.columns:
                source['subject'] = row.get('subject', '')
            if 'source_file' in self.corpus_df.columns:
                source['source_file'] = row.get('source_file', '')
            return source
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
