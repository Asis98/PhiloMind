"""TF-IDF retriever with entity extraction, query expansion, and reranking.

Classification-aware weighting reinforces the connection between the
classification and retrieval modules of the project.
"""

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

# Map concepts to their primary philosopher/originator.
# When a definition query asks about a concept, passages by this author
# get a strong boost — often the only signal when the term is absent from
# the corpus (e.g. "categorical imperative" appears in 0/1640 chunks).
# Course connection: symbolic knowledge representation, entity linking.
CONCEPT_AUTHOR_MAP = {
    "categorical imperative": "Kant",
    "Cartesian dualism": "Descartes",
    "dualism": "Descartes",
    "mind-body": "Descartes",
    "cogito": "Descartes",
    "forms": "Plato",
    "theory of forms": "Plato",
    "will to power": "Nietzsche",
    "eternal return": "Nietzsche",
    "ubermensch": "Nietzsche",
    "overman": "Nietzsche",
    "death of god": "Nietzsche",
    "social contract": "Rousseau",
    "state of nature": "Hobbes",
    "general will": "Rousseau",
    "veil of ignorance": "Rawls",
    "biopolitics": "Foucault",
    "panopticon": "Foucault",
    "tabula rasa": "Locke",
    "noumenon": "Kant",
    "phenomenon": "Kant",
    "a priori": "Kant",
    "a posteriori": "Kant",
    "transcendental": "Kant",
    "deontology": "Kant",
    "dialectic": "Hegel",
    "dialectics": "Hegel",
    "dialectical materialism": "Marx",
    "historical materialism": "Marx",
    "utilitarianism": "Mill",
    "existentialism": "Sartre",
    "phenomenology": "Husserl",
    "stoicism": "Epictetus",
    "epicureanism": "Epicurus",
    "invisible hand": "Smith",
    "being": "Heidegger",
    "dasein": "Heidegger",
    "the other": "Sartre",
    "the self": "Hume",
    "substance": "Spinoza",
    "empiricism": "Hume",
    "rationalism": "Descartes",
}


# ---------------------------------------------------------------------------
# Definition anchoring — find chunks that actually *define* a term
# Course connection: query parsing + pattern-based information extraction
# ---------------------------------------------------------------------------

DEFINITION_QUERY_PATTERNS = [
    re.compile(r'^what\s+is\s+(?:the\s+)?([^.?,!]{3,60}?)\??\s*$', re.I),
    re.compile(r'^what\s+are\s+([^.?,!]{3,60}?)\??\s*$', re.I),
    re.compile(r'^define\s+([^.?,!]{3,60}?)\??\s*$', re.I),
    re.compile(r'^explain\s+what\s+([^.?,!]{3,60}?)\s+is\??\s*$', re.I),
    re.compile(r'^definition\s+of\s+([^.?,!]{3,60}?)\??\s*$', re.I),
]


def is_definition_query(question: str) -> bool:
    """True if the question asks for a definition."""
    return any(p.match(question.strip()) for p in DEFINITION_QUERY_PATTERNS)


def extract_definition_concept(question: str) -> Optional[str]:
    """Extract 'X' from 'What is X?', 'Define X', etc."""
    q = question.strip()
    for p in DEFINITION_QUERY_PATTERNS:
        m = p.match(q)
        if m:
            return m.group(1).strip()
    return None


DEFINITION_PATTERNS = [
    r'\b{concept}\s+is\b',
    r'\b{concept}\s+refers?\s+to\b',
    r'\b{concept}\s+is\s+defined\s+as\b',
    r'\bdefinition\s+of\s+{concept}\b',
    r'\b{concept}\s+means?\b',
    r'\b{concept}\s+consists?\s+in\b',
    r'\bconcept\s+of\s+{concept}\b',
    r'\bis\s+called\s+{concept}\b',
    r'\bknown\s+as\s+{concept}\b',
]

DEFINITION_BOOST = 0.5  # per pattern — 2 matches = 1.0, easily beats entity overlap


def score_definition_match(concept: str, passage_text: str) -> float:
    """Score how strongly a passage defines *concept*.

    Returns total boost from all matching definition patterns.
    """
    text_lower = passage_text.lower()
    escaped = re.escape(concept)
    matches = 0
    for tmpl in DEFINITION_PATTERNS:
        pattern = tmpl.replace(r'\{concept\}', escaped)
        if re.search(pattern, text_lower):
            matches += 1
    return matches * DEFINITION_BOOST


# ---------------------------------------------------------------------------
# Classification-aware weighting — connects classification and retrieval
# ---------------------------------------------------------------------------
# Each question type benefits from passage characteristics that align with
# the type's communicative goal.  We encode this as small score boosts.
# Course connection: task-specific inductive bias, feature engineering.

TYPE_BOOST_WORDS = {
    'definition': {'is', 'are', 'refers to', 'defined as', 'means', 'concept',
                   'essence', 'nature of', 'consists in'},
    'comparison': {'unlike', 'however', 'in contrast', 'both', 'different',
                   'similarly', 'whereas', 'although', 'on the other hand',
                   'distinction', 'difference', 'common', 'shared'},
    'example':    {'for example', 'for instance', 'consider', 'suppose',
                   'illustrate', 'such as', 'specifically', 'namely'},
    'deepening':  set(),    # no special lexical signal
    'quiz':       {'is', 'was', 'are', 'were', 'the', 'a', 'an'},
}
TYPE_BIAS_AMOUNT = 0.08

QUESTION_WORDS = {'what', 'how', 'why', 'which', 'explain', 'compare',
                  'does', 'who', 'whom', 'whose', 'where', 'when'}


def classify_type_bias(question_type: Optional[str], passage_text: str) -> float:
    """Return a score bonus based on question type and passage content."""
    if question_type is None:
        return 0.0
    qtype = question_type.lower().strip()
    boost_words = TYPE_BOOST_WORDS.get(qtype)
    if not boost_words:
        return 0.0
    text_lower = passage_text.lower()
    matches = sum(1 for w in boost_words if w in text_lower)
    return matches * TYPE_BIAS_AMOUNT


def extract_entities(text: str) -> dict:
    text_lower = text.lower()
    found_philosophers = [p for p in PHILOSOPHERS if p.lower() in text_lower]
    found_concepts = [c for c in CONCEPTS if c in text_lower]
    return {'philosophers': found_philosophers, 'concepts': found_concepts}


def expand_query(text: str, glove_expander=None) -> str:
    """Expand query with entity terms and optionally with GloVe semantic neighbours.

    Course connection: when glove_expander is used, this demonstrates
    word embeddings (distributed representations) discovering related
    terms automatically — a contrast to the manual entity list above.
    """
    entities = extract_entities(text)
    expanded = text
    for p in entities['philosophers']:
        expanded += f" {p} philosophy {p} theory"
    for c in entities['concepts']:
        expanded += f" {c} concept {c} philosophical"
        # Add the concept's author if known (e.g. "categorical imperative" → "Kant")
        author = CONCEPT_AUTHOR_MAP.get(c)
        if author and author.lower() not in expanded.lower():
            expanded += f" {author}"

    if glove_expander is not None:
        glove_terms = glove_expander.expand(text)
        if glove_terms != text:
            expanded += " " + " ".join(
                w for w in glove_terms.split()
                if w.lower() not in expanded.lower()
            )
    return expanded


def rerank_by_entity_overlap(query: str, passages: List[Tuple[int, str, float]],
                              question_type: Optional[str] = None,
                              definition_concept: Optional[str] = None,
                              ) -> List[Tuple[int, str, float]]:
    """Rerank passages: entity overlap + classification bias + definition boost.

    The definition boost (0.5 per matching pattern) is intentionally much
    stronger than entity overlap (0.15) — for "what is X" queries, a passage
    that says "X is…" must beat one that merely mentions X.

    When the definition concept maps to a known philosopher (e.g. "categorical
    imperative" → Kant), passages by that author get an additional boost,
    compensating for corpus chunks that rarely contain the exact concept phrase.
    """
    if not passages:
        return passages
    query_entities = extract_entities(query)
    query_phils = set(p.lower() for p in query_entities['philosophers'])
    query_concepts = set(query_entities['concepts'])

    # Resolve concept→author for definition queries: if the user asks
    # "What is categorical imperative?" and the term never appears in
    # any chunk, we still boost Kant's passages via this symbolic link.
    concept_author_lower = None
    if definition_concept:
        for concept_name, philosopher in CONCEPT_AUTHOR_MAP.items():
            if concept_name in definition_concept:
                concept_author_lower = philosopher.lower()
                break

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
        bonus += classify_type_bias(question_type, text)
        if definition_concept:
            bonus += score_definition_match(definition_concept, text)
            # Author boost: if concept maps to known philosopher,
            # boost all passages that mention them
            if concept_author_lower and concept_author_lower in text_lower:
                bonus += 0.30
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

    def retrieve(self, query: str, top_k: int = 3,
                 subject_filter: Optional[str] = None,
                 question_type: Optional[str] = None,
                 glove_expander=None) -> List[Tuple[int, str, float]]:
        """Retrieve passages using TF-IDF with expansion and type-aware reranking.

        Args:
            query: User question.
            top_k: Number of passages to return.
            subject_filter: Optional subject to restrict to.
            question_type: Classification label (definition/comparison/etc.) for
                           type-aware reranking.
            glove_expander: Optional GloVeQueryExpander for semantic expansion.
        """
        if not self.is_fitted:
            raise ValueError("Retriever has not been fitted. Call fit() first.")
        expanded = expand_query(query, glove_expander=glove_expander)
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
        results = rerank_by_entity_overlap(query, results, question_type=question_type)
        return results[:top_k]

    def retrieve_with_boost(self, query: str, top_k: int = 3,
                            teacher_retriever: Optional['TFIDFRetriever'] = None,
                            boost_factor: float = 1.5,
                            question_type: Optional[str] = None,
                            glove_expander=None) -> List[Tuple[int, str, float, str]]:
        teacher_results = []
        general_results = []
        if teacher_retriever and teacher_retriever.is_fitted:
            teacher_results = teacher_retriever.retrieve(query, top_k=top_k, question_type=question_type)
        general_results = self.retrieve(query, top_k=top_k, question_type=question_type, glove_expander=glove_expander)
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
