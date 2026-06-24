"""TF-IDF retriever with entity extraction, query expansion, and reranking."""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle
from typing import List, Tuple, Optional


PHILOSOPHERS = [
    "Plato", "Platonic", "Platonism",
    "Aristotle", "Aristotelian",
    "Socrates", "Socratic",
    "Descartes", "Cartesian",
    "Kant", "Kantian",
    "Nietzsche", "Nietzschean",
    "Hegel", "Hegelian",
    "Hegel", "Marx", "Hume", "Locke", "Rousseau", "Spinoza", "Leibniz",
    "Schopenhauer", "Heidegger", "Wittgenstein", "Husserl", "Sartre",
    "Foucault", "Arendt", "Kierkegaard", "Russell", "Frege", "Rawls",
    "Nozick", "Singer", "Deleuze", "Derrida", "Habermas", "Adorno",
    "Bacon", "Hobbes", "Berkeley", "Plotinus", "Epicurus", "Zeno",
    "Thales", "Pythagoras", "Heraclitus", "Parmenides", "Anaximander"
]

CONCEPTS = [
    "forms", "theory of forms", "categorical imperative", "social contract",
    "will to power", "eternal return", "cogito", "veil of ignorance",
    "state of nature", "ubermensch", "overman", "dialectic", "dialectics",
    "the self", "the other", "biopolitics", "phenomenology", "invisible hand",
    "general will", "utility", "catharsis", "mimesis", "the unconscious",
    "death of god", "end of history", "iron cage", "panopticon",
    "tabula rasa", "a priori", "a posteriori", "noumenon", "phenomenon",
    "being", "becoming", "substance", "essence", "existence",
    "transcendental", "empiricism", "rationalism", "stoicism", "epicureanism",
    "utilitarianism", "deontology", "virtue ethics", "contractarianism",
    "existentialism", "phenomenology", "structuralism", "poststructuralism",
    "ontology", "epistemology", "metaphysics", "ethics", "aesthetics",
    "logic", "dialectical materialism", "historical materialism"
]


def extract_entities(text: str) -> dict:
    text_lower = text.lower()
    found_philosophers = [p for p in PHILOSOPHERS if p.lower() in text_lower]
    found_concepts = [c for c in CONCEPTS if c in text_lower]
    return {'philosophers': found_philosophers, 'concepts': found_concepts}


def expand_query(text: str) -> str:
    entities = extract_entities(text)
    expanded = text
    for p in entities['philosophers']:
        expanded += f" {p} philosophy {p} theory"
    for c in entities['concepts']:
        expanded += f" {c} concept {c} philosophical"
    return expanded


def rerank_by_entity_overlap(query: str, passages: List[Tuple[int, str, float]]) -> List[Tuple[int, str, float]]:
    if not passages:
        return passages
    query_entities = extract_entities(query)
    query_phils = set(p.lower() for p in query_entities['philosophers'])
    query_concepts = set(query_entities['concepts'])
    scored = []
    for idx, text, score in passages:
        bonus = 0.0
        text_lower = text.lower()
        for p in query_phils:
            if p in text_lower:
                bonus += 0.15
        for c in query_concepts:
            if c in text_lower:
                bonus += 0.10
        scored.append((idx, text, score + bonus, bonus))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(idx, text, score) for idx, text, score, _ in scored]


class TFIDFRetriever:
    """TF-IDF retriever with entity extraction, query expansion, and reranking."""

    def __init__(self, corpus_df=None, max_features=10000, min_df=1, max_df=0.95):
        self.corpus_df = corpus_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, lowercase=True,
            min_df=min_df, max_df=max_df, ngram_range=(1, 3),
            stop_words='english', sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit(self, texts):
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True

    def retrieve(self, query: str, top_k: int = 3, subject_filter: Optional[str] = None) -> List[Tuple[int, str, float]]:
        if not self.is_fitted:
            raise ValueError("Retriever has not been fitted. Call fit() first.")
        expanded = expand_query(query)
        query_vec = self.vectorizer.transform([expanded])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        if subject_filter and self.corpus_df is not None and 'subject' in self.corpus_df.columns:
            mask = self.corpus_df['subject'].str.lower() == subject_filter.lower()
            similarities[~mask] = -1.0
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        results = []
        for idx in top_indices:
            if similarities[idx] < 0:
                continue
            text = self.corpus_df.iloc[idx]['text']
            score = float(similarities[idx])
            results.append((idx, text, score))
        results = rerank_by_entity_overlap(query, results)
        return results[:top_k]

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
