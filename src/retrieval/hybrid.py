"""Hybrid retrieval — sparse (TF-IDF) + dense (Transformer embeddings).

Course connections:
1. Combines BoW/N-gram (TF-IDF) and distributed representations (dense).
2. Demonstrates complementarity: sparse excels at exact term matching
   (philosopher names, work titles), dense excels at semantic similarity.
3. The combination weight α is a simple learnable hyperparameter —
   introduces the concept of ensemble / feature combination.
4. Cross-encoder reranking at the top adds full attention between
   query and passage (attention mechanism).
"""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.preprocessing import minmax_scale

from .tfidf import TFIDFRetriever
from .embeddings import DenseRetriever, CrossEncoderReranker, GloVeQueryExpander


class HybridRetriever:
    """Orchestrates sparse + dense retrieval with reranking and type-aware weighting.

    Flow:
        query
          → GloVe expansion (word embeddings)
          → TF-IDF retrieval + Dense retrieval (parallel)
          → Score normalization + weighted combination (hybrid)
          → Entity-overlap & type-aware reranking
          → Cross-encoder reranking (attention)
          → Top-K results
    """

    def __init__(self,
                 tfidf_retriever: TFIDFRetriever,
                 dense_retriever: Optional[DenseRetriever] = None,
                 cross_encoder: Optional[CrossEncoderReranker] = None,
                 glove_expander: Optional[GloVeQueryExpander] = None,
                 alpha: float = 0.3):
        """Initialise hybrid retriever.

        Args:
            tfidf_retriever: Fitted TF-IDF retriever (sparse).
            dense_retriever: Fitted dense retriever (neural).
            cross_encoder: Optional cross-encoder reranker.
            glove_expander: Optional GloVe query expander.
            alpha: Weight for TF-IDF in hybrid score (1-α for dense).
                   Default 0.3 favours dense (more semantic signal).
        """
        self.tfidf = tfidf_retriever
        self.dense = dense_retriever
        self.cross_encoder = cross_encoder
        self.glove = glove_expander
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 3,
                 subject_filter: Optional[str] = None,
                 question_type: Optional[str] = None) -> List[Tuple[int, str, float, str]]:
        """Hybrid retrieval with optional cross-encoder reranking.

        For "what is X" queries, the pipeline automatically:
        1. Extracts the concept ("categorical imperative" from "What is…?")
        2. Engineers the retrieval query for definition matching
        3. Applies a strong definition boost during reranking

        Returns:
            List of (index, text, score, source_type) tuples.
            source_type is 'sparse', 'dense', or 'hybrid'.
        """
        dense_ok = self.dense is not None and self.dense.embeddings is not None

        # ---- stage 0: definition anchoring ----
        from .tfidf import is_definition_query, extract_definition_concept
        definition_concept = None
        if is_definition_query(query):
            definition_concept = extract_definition_concept(query)

        # For definition queries, engineer the query to favour definition patterns
        # Course connection: query rewriting for intent-specific retrieval
        tfidf_query = query
        dense_query = query
        if definition_concept:
            # Resolve concept→author to inject the author name into both queries
            from .tfidf import CONCEPT_AUTHOR_MAP
            concept_author = None
            for c_name, phil in CONCEPT_AUTHOR_MAP.items():
                if c_name in definition_concept:
                    concept_author = phil
                    break
            author_tag = f' {concept_author}' if concept_author else ''

            tfidf_query = (
                f'{query} definition of {definition_concept} '
                f'"{definition_concept} is" refers to defined_as '
                f'what_is {definition_concept} concept of {definition_concept}'
                f'{author_tag}'
            )
            dense_query = f'definition of {definition_concept} {query}{author_tag}'

        # ---- stage 1: parallel retrieval ----
        tfidf_results = self.tfidf.retrieve(
            tfidf_query, top_k=top_k * 3,
            subject_filter=subject_filter,
            question_type=question_type,
            glove_expander=self.glove,
        )

        dense_results = []
        if dense_ok:
            dense_results = self.dense.retrieve(dense_query, top_k=top_k * 3)

        # ---- stage 2: hybrid scoring ----
        merged = self._hybrid_merge(tfidf_results, dense_results, top_k, dense_ok)

        # ---- stage 3: cross-encoder reranking ----
        # The cross-encoder applies full attention between query and passage
        # tokens (unlike the bi-encoder's independent encoding).  Raw logits
        # can be negative/unbounded, so we normalise to [0, 1] within the
        # candidate set before combining with the hybrid score.
        if self.cross_encoder is not None and len(merged) > top_k:
            ce_input = [(idx, text, score) for idx, text, score, _ in merged]
            ce_reranked = self.cross_encoder.rerank(query, ce_input, top_k=top_k * 2)

            # Normalise CE scores to [0, 1]
            ce_scores = np.array([s for _, _, s in ce_reranked])
            if ce_scores.max() > ce_scores.min():
                ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min())
            else:
                ce_norm = np.zeros_like(ce_scores)

            ce_score_map = {idx: float(ce_norm[i]) for i, (idx, _, _) in enumerate(ce_reranked)}
            ce_order = {idx: i for i, (idx, _, _) in enumerate(ce_reranked)}

            # Blend: 60 % cross-encoder confidence + 40 % original hybrid score
            for i, (idx, text, orig_score, st) in enumerate(merged):
                ce_s = ce_score_map.get(idx, 0.0)
                orig_norm = orig_score  # already in [0,1] from _hybrid_merge
                merged[i] = (idx, text, 0.6 * ce_s + 0.4 * orig_norm, st)

            # Re-sort by blended score
            merged.sort(key=lambda x: x[2], reverse=True)

        # ---- stage 4: type-aware entity reranking (with definition boost) ----
        merged_no_st = [(idx, text, score) for idx, text, score, _ in merged]
        from .tfidf import rerank_by_entity_overlap, score_definition_match
        reranked = rerank_by_entity_overlap(
            query, merged_no_st,
            question_type=question_type,
            definition_concept=definition_concept,
        )

        seen_texts = set()
        final = []
        for idx, text, score in reranked:
            if text not in seen_texts:
                seen_texts.add(text)
                final.append((idx, text, score, 'hybrid'))

        # ---- stage 5: concept→author metadata boost ----
        # For "what is X" queries where X maps to a known philosopher
        # (e.g. "categorical imperative" → Kant), boost passages where
        # the *metadata* philosopher matches — this catches cases where
        # a philosopher's own writing doesn't mention their own name.
        if definition_concept:
            from .tfidf import CONCEPT_AUTHOR_MAP
            concept_author = None
            for c_name, phil in CONCEPT_AUTHOR_MAP.items():
                if c_name in definition_concept:
                    concept_author = phil.lower()
                    break
            if concept_author:
                final_with_author = []
                for idx, text, score, st in final:
                    row = self.tfidf.corpus_df.iloc[idx]
                    row_phil = str(row.get('philosopher', '')).lower()
                    if concept_author in row_phil:
                        score += 0.25  # metadata author boost
                    final_with_author.append((idx, text, score, st))
                final_with_author.sort(key=lambda x: x[2], reverse=True)
                final = final_with_author

        # Guarantee: for "what is X" queries, at least one definition chunk
        # must survive.  If the reranking boost wasn't enough, inject one.
        if definition_concept and not any(
            score_definition_match(definition_concept, t) > 0
            for _, t, _, _ in final
        ):
            # Re-retrieve with an aggressively definition-focused query
            fallback = self.tfidf.retrieve(
                f'definition of {definition_concept} "{definition_concept} is"',
                top_k=top_k,
                question_type='definition',
            )
            # Replace the last result with the best definition chunk
            for idx, text, score in fallback:
                if text not in seen_texts and score_definition_match(definition_concept, text) > 0:
                    final[-1] = (idx, text, score, 'hybrid')
                    break

        return final[:top_k]

    def _hybrid_merge(self, tfidf_results, dense_results, top_k, dense_ok):
        """Normalise and combine sparse + dense scores."""
        score_map = {}  # idx -> (text, sparse_score, dense_score)

        for idx, text, score in tfidf_results:
            if idx not in score_map:
                score_map[idx] = [text, score, 0.0]
            else:
                score_map[idx][1] = max(score_map[idx][1], score)

        if dense_ok:
            for idx, text, score in dense_results:
                if idx not in score_map:
                    score_map[idx] = [text, 0.0, score]
                else:
                    score_map[idx][2] = max(score_map[idx][2], score)

        if not score_map:
            return []

        indices = list(score_map.keys())
        sparse_scores = np.array([score_map[i][1] for i in indices])
        dense_scores = np.array([score_map[i][2] for i in indices])

        # normalise each to [0, 1] for comparability
        if sparse_scores.max() > sparse_scores.min():
            sparse_norm = (sparse_scores - sparse_scores.min()) / (
                sparse_scores.max() - sparse_scores.min()
            )
        else:
            sparse_norm = np.zeros_like(sparse_scores)

        if dense_ok and dense_scores.max() > dense_scores.min():
            dense_norm = (dense_scores - dense_scores.min()) / (
                dense_scores.max() - dense_scores.min()
            )
        else:
            dense_norm = np.zeros_like(dense_scores)

        # weighted combination
        if dense_ok:
            hybrid = self.alpha * sparse_norm + (1 - self.alpha) * dense_norm
        else:
            hybrid = sparse_norm

        # build result list
        combined = []
        for i, idx in enumerate(indices):
            text = score_map[idx][0]
            combined.append((idx, text, float(hybrid[i]), 'hybrid'))

        combined.sort(key=lambda x: x[2], reverse=True)
        return combined[:top_k * 3]

    def get_source(self, idx: int) -> dict:
        return self.tfidf.get_source(idx)
