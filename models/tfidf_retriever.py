"""
Agent 2: Retriever TF-IDF per il recupero di brani filosofici.
Recupera i brani più rilevanti dal corpus il base alla query (domanda).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle
from typing import List, Tuple

BASE_DIR = Path("models-utils")
BASE_DIR.mkdir(parents=True, exist_ok=True)

class TFIDFRetriever:
    """Retriever basato su TF-IDF per documenti filosofici."""

    def __init__(self, corpus_df=None, min_df=1, max_df=0.95):
        """
        Args:
            corpus_df: DataFrame con colonne ['text', 'philosopher', 'work']
            min_df, max_df: parametri TF-IDF
        """
        self.corpus_df = corpus_df
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            lowercase=True,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),  # unigrams e bigrams
            stop_words='english'  # o italiana
        )
        self.tfidf_matrix = None
        self.is_fitted = False

    def fit(self, texts):
        """Allena il vectorizer su una collezione di testi."""
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        print(f" TF-IDF fitted su {len(texts)} documenti")
        print(f"   Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Recupera i top_k brani più rilevanti per una query.

        Returns:
            Lista di tuple (testo, similarità_score)
        """
        if not self.is_fitted:
            raise ValueError("Retriever non è stato trainato. Chiama fit() prima.")

        # Vectoriza la query
        query_vec = self.vectorizer.transform([query])

        # Calcola similarità coseno
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            text = self.corpus_df.iloc[idx]['text']
            score = similarities[idx]
            results.append((text, float(score)))

        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 3):
        """Recupera per multiple queries."""
        return [self.retrieve(q, top_k) for q in queries]

    def save(self, path: str):
        """Salva il retriever."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'corpus_df': self.corpus_df
            }, f)
        print(f" Retriever salvato a {path}")

    def load(self, path: str):
        """Carica il retriever."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.corpus_df = data['corpus_df']
            self.is_fitted = True
        print(f" Retriever caricato da {path}")


class CorpusPreparer:
    """Prepara il corpus filosofico per il retriever."""

    @staticmethod
    def parse_kaggle_corpus(csv_path: str) -> pd.DataFrame:
        """
        Carica il corpus Kaggle filosofico e lo preprocessa.
        Assume structure dalle PhilosophyData Kaggle: [author, title, sentence_str, ...]
        """
        print(f"  Caricamento {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)

        # Normalizza nomi colonne nel formato Kaggle
        if 'sentence_str' in df.columns:
            df['text'] = df['sentence_str']
        elif 'sentence' in df.columns:
            df['text'] = df['sentence']
        elif 'text' not in df.columns:
            # Prova a usare la prima colonna testuale
            text_cols = [c for c in df.columns if df[c].dtype == 'object']
            if text_cols:
                df['text'] = df[text_cols[0]]

        # Philosopher column
        if 'author' in df.columns:
            df['philosopher'] = df['author']
        elif 'philosopher' not in df.columns:
            df['philosopher'] = 'Unknown'

        # Work column
        if 'title' in df.columns and 'work' not in df.columns:
            df['work'] = df['title']
        elif 'work' not in df.columns:
            df['work'] = 'Unknown'

        # Rimuovi null
        df = df.dropna(subset=['text'])

        # Pulizia testo
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 10]  # Solo testi significativi

        print(f"   Corpus caricato: {len(df)} documenti")
        return df.reset_index(drop=True)

    @staticmethod
    def create_semantic_chunks(df: pd.DataFrame, chunk_size: int = 200) -> pd.DataFrame:
        """
        Divide i testi in chunks semantici (per phrase/sentence).
        """
        chunks = []

        for idx, row in df.iterrows():
            text = row['text']
            # Split per periodi (!)
            sentences = text.split('.')

            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    word_count = len(sentence.split())

                    if current_length + word_count > chunk_size and current_chunk:
                        # Salva chunk
                        chunks.append({
                            'text': '. '.join(current_chunk) + '.',
                            'philosopher': row.get('philosopher', 'Unknown'),
                            'work': row.get('work', 'Unknown'),
                            'original_idx': idx
                        })
                        current_chunk = [sentence]
                        current_length = word_count
                    else:
                        current_chunk.append(sentence)
                        current_length += word_count

            # Chunk rimanente
            if current_chunk:
                chunks.append({
                    'text': '. '.join(current_chunk) + '.',
                    'philosopher': row.get('philosopher', 'Unknown'),
                    'work': row.get('work', 'Unknown'),
                    'original_idx': idx
                })

        print(f" {len(chunks)} chunks semantici creati")
        return pd.DataFrame(chunks)


def build_corpus_index(raw_corpus_path: str, output_dir: Path = BASE_DIR, sample_size: int = 5000):
    """
    Costruisce l'indice TF-IDF completo del corpus.

    Args:
        raw_corpus_path: path al CSV del corpus
        output_dir: directory dove salvare
        sample_size: numero massimo di documenti da usare (per performance)
    """
    print("\n"+"="*60)
    print("Building TF-IDF Corpus Index")
    print("="*60)

    # Carica corpus
    print(f"\n Caricamento corpus da {raw_corpus_path}...")
    raw_corpus = CorpusPreparer.parse_kaggle_corpus(raw_corpus_path)

    # Se il corpus è troppo grande, campiona
    if len(raw_corpus) > sample_size:
        print(f"  ⚠️  Corpus grande ({len(raw_corpus)} docs). Campio {sample_size} per velocità...")
        raw_corpus = raw_corpus.sample(n=sample_size, random_state=42)

    # Chunking
    print("\n✂  Chunking semantico...")
    corpus_chunks = CorpusPreparer.create_semantic_chunks(raw_corpus, chunk_size=100)

    #FILTER NOISE CHUNKS
    corpus_chunks = corpus_chunks[corpus_chunks["text"].str.len() > 80]
    corpus_chunks = corpus_chunks[
        ~corpus_chunks["text"].str.contains(
            "defense|military|weapon",
            case=False,
            na=False
        )
    ]

    # Costruisci retriever
    print("\n Costruzione TF-IDF retriever...")
    retriever = TFIDFRetriever(corpus_df=corpus_chunks)
    retriever.fit(corpus_chunks['text'].values)

    # Salva
    output_dir = BASE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever.save(output_dir / "tfidf_retriever.pkl")
    corpus_chunks.to_csv(output_dir / "corpus_chunks.csv", index=False)

    return retriever, corpus_chunks


if __name__ == '__main__':
    # Paths
    raw_corpus_path = '../data/raw/philosophy_data.csv'
    output_dir = str(BASE_DIR)

    # Verifica se il corpus esiste
    if not Path(raw_corpus_path).exists():
        print(f"⚠️  {raw_corpus_path} non trovato!")
        print("Creando corpus di esempio...")

        # Crea corpus di ejemplo
        example_corpus = pd.DataFrame({
            'text': [
                'In Plato, the Republic represents the most famous dialogue on justice. Socrates discusses with his interlocutors to define virtue.',
                'Aristotle criticizes Plato’s theory of Forms and develops the concept of primary substance.',
                'Descartes begins his method of doubt with the famous proposition “Cogito ergo sum.”',
                'Nietzsche introduces the concept of the Übermensch as a overcoming of traditional morality.',
                'Marx analyzes the capitalist mode of production in *Capital*.',
                'Kant argues that space and time are a priori forms of sensibility.',
                'In Hegel, dialectics is used to understand the movement of the absolute spirit in history.',
                'Rousseau theorizes the social contract based on the general will of the people.',
                'Spinoza conceives God as nature, equivalent to universal determinism.',
                'Schopenhauer sees the will as the fundamental principle of reality.',
            ],
            'philosopher': [
                'Plato', 'Aristotle', 'Descartes', 'Nietzsche', 'Marx',
                'Kant', 'Hegel', 'Rousseau', 'Spinoza', 'Schopenhauer'
            ],
            'work': [
                'Republic', 'Metaphysics', 'Meditations', 'Thus Spoke Zarathustra', 'Capital',
                'Critique of Pure Reason', 'Phenomenology of Spirit', 'The Social Contract', 'Ethics',
                'The World as Will and Representation'
            ]
        })

        Path('../data/raw').mkdir(parents=True, exist_ok=True)
        example_corpus.to_csv(raw_corpus_path, index=False)
        print(f" Corpus di esempio creato a {raw_corpus_path}")

    # Build retriever
    retriever, corpus_chunks = build_corpus_index(raw_corpus_path, BASE_DIR)

    print("\n" + "="*60)
    print("Test Retriever")
    print("="*60)

    # Test queries
    test_queries = [
        "What is Plato’s theory of Forms?",
        "How do Plato and Aristotle differ?",
        "Who is Nietzsche and what is the will to power?"
    ]

    for query in test_queries:
        print(f"\n Query: {query}")
        results = retriever.retrieve(query, top_k=2)

        for i, (text, score) in enumerate(results, 1):
            print(f"   [{i}] (score: {score:.4f})")
            print(f"       {text[:100]}...")

    print("\n Corpus index pronto!")



