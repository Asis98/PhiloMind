"""Dense embeddings, GloVe query expansion, and cross-encoder reranking.

Reinforces course concepts:
- GloVeExpander -> Word embeddings / distributed representations (static)
- DenseRetriever -> Contextual embeddings / Transformer / cosine similarity
- CrossEncoderReranker -> Attention mechanism (full cross-attention)
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# GloVe-based query expansion — Word Embeddings / Distributed Representations
# ---------------------------------------------------------------------------
# Pure numpy implementation: loads GloVe from a .txt file (standard format),
# caches as .npz, and finds nearest neighbours via cosine similarity.
# No external dependency beyond numpy + sklearn (already in requirements).
#
# If no GloVe file is available, the expander gracefully falls back to the
# existing entity-based expansion — demonstrating that classical word
# embeddings are optional but beneficial.
#
# Course connection: distributed representations where semantically related
# words occupy nearby positions in vector space.

GLOVE_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'glove_cache.npz'
GLOVE_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'glove.6B.50d.txt'


class GloVeQueryExpander:
    """Expands a query with semantically similar words using GloVe embeddings.

    Loads GloVe from a standard text file (word vec1 ... vecN per line),
    caches as compressed numpy for fast reload.  Falls back gracefully.
    """

    def __init__(self, glove_path=None, top_n: int = 3):
        self.glove_path = Path(glove_path) if glove_path else GLOVE_DEFAULT_PATH
        self.top_n = top_n
        self._vectors = None
        self._words = None
        self._word2idx = None

    def _load(self):
        """Load GloVe from cache (.npz) or text file."""
        # Try cache first
        if GLOVE_CACHE_PATH.exists():
            try:
                data = np.load(str(GLOVE_CACHE_PATH))
                self._words = data['words'].tolist() if hasattr(data['words'], 'tolist') else list(data['words'])
                self._vectors = data['vectors']
                self._word2idx = {w: i for i, w in enumerate(self._words)}
                return
            except Exception:
                pass

        # Load from text file
        if not self.glove_path.exists():
            return

        print(f"Loading GloVe from {self.glove_path}...")
        words = []
        vectors = []
        with open(str(self.glove_path), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                word = parts[0].lower()
                vec = np.array([float(x) for x in parts[1:]])
                words.append(word)
                vectors.append(vec)

        self._words = words
        self._vectors = np.array(vectors, dtype=np.float32)
        self._word2idx = {w: i for i, w in enumerate(self._words)}

        # Cache
        GLOVE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(GLOVE_CACHE_PATH),
                            words=np.array(self._words, dtype=object),
                            vectors=self._vectors)
        print(f"  Loaded {len(self._words)} words, dim={self._vectors.shape[1]}")

    def expand(self, text: str) -> str:
        """Expand query by appending nearest GloVe neighbours for each content word."""
        if self._vectors is None:
            self._load()
        if self._vectors is None:
            return text

        words = text.lower().split()
        expanded_terms = []

        for word in words:
            word = word.strip("?.,!;:'\"()[]{}")
            if len(word) < 3:
                continue
            if word not in self._word2idx:
                continue

            idx = self._word2idx[word]
            query_vec = self._vectors[idx:idx+1]
            sims = cosine_similarity(query_vec, self._vectors)[0]
            nearest = np.argsort(sims)[::-1][1:self.top_n + 1]  # skip self

            for ni in nearest:
                similar_word = self._words[ni]
                if similar_word not in expanded_terms and similar_word != word:
                    expanded_terms.append(similar_word)

        if not expanded_terms:
            return text

        return text + " " + " ".join(expanded_terms)


# ---------------------------------------------------------------------------
# Dense embedding retriever — Contextual Embeddings / Transformer / Cosine
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DenseRetriever:
    """Dense (neural) retriever using SentenceTransformer contextual embeddings.

    Course connections:
    1. Distributed representations: 384-dim dense vectors vs. 10K-dim sparse TF-IDF.
    2. Contextual embeddings: MiniLM is a distilled BERT — "bank" in "river bank"
       and "bank" in "investment bank" receive different embeddings.
    3. Transformer: MiniLM-L6 is a 6-layer Transformer encoder.  The [CLS]
       pooling produces a fixed-size sentence vector.
    4. Cosine similarity: same metric as TF-IDF, now applied to learned vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.embeddings: Optional[np.ndarray] = None
        self.corpus_df = None

    @property
    def model(self):
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"Loading SentenceTransformer ({self.model_name})...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: List[str]):
        if self.model is None:
            raise RuntimeError("SentenceTransformers not available.")
        print(f"Encoding {len(texts)} passages with {self.model_name}...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, str, float]]:
        if self.embeddings is None or self.corpus_df is None:
            raise ValueError("DenseRetriever has not been fitted. Call fit() first.")
        if self.model is None:
            raise RuntimeError("SentenceTransformers not available.")
        query_emb = self.model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            text = self.corpus_df.iloc[idx]['text']
            score = float(similarities[idx])
            results.append((idx, text, score))
        return results

    def get_source(self, idx: int) -> dict:
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
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.embeddings)

    def load(self, path: str):
        self.embeddings = np.load(path + ".npy" if not path.endswith(".npy") else path)


# ---------------------------------------------------------------------------
# Cross-encoder reranker  —  Attention mechanism
# ---------------------------------------------------------------------------

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    """Reranks candidate passages using a cross-encoder Transformer.

    Course connection: Full attention mechanism.
    Unlike a bi-encoder (DenseRetriever) which encodes query and passage
    independently, the cross-encoder applies attention between every query
    token and every passage token (cross-attention).  This yields more
    accurate relevance scores but is O(n*m) per pair, illustrating the
    efficiency-accuracy trade-off central to Transformer architectures.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None and CROSS_ENCODER_AVAILABLE:
            print(f"Loading CrossEncoder ({self.model_name})...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str,
               candidates: List[Tuple[int, str, float]],
               top_k: int = 3) -> List[Tuple[int, str, float]]:
        if not candidates or self.model is None:
            return candidates[:top_k]

        pairs = [(query, text) for _, text, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = []
        for (idx, text, orig_score), ce_score in zip(candidates, scores):
            reranked.append((idx, text, float(ce_score)))

        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked[:top_k]
