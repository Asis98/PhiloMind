"""Pipeline core — orchestrates classification, parallel retrieval and response/quiz generation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.classification.base import BaseClassifier
from src.classification.bilstm import BiLSTMClassifier
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.embeddings import DenseRetriever, CrossEncoderReranker, GloVeQueryExpander
from src.retrieval.embeddings import SENTENCE_TRANSFORMERS_AVAILABLE, CROSS_ENCODER_AVAILABLE
from src.retrieval.hybrid import HybridRetriever
from src.agents.registry import AgentRegistry
from src.agents.response import ResponseGenerator
from src.agents.quiz import QuizGenerator
from .dataclasses import ClassificationResult, RetrievalResult, Passage, PipelineOutput

try:
    from src.classification.distilbert import DistilBERTClassifier
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
    DISTILBERT_AVAILABLE = True
except ImportError:
    DISTILBERT_AVAILABLE = False


class PhiloMindPipeline:
    """Integrated pipeline: classification → hybrid retrieval → response generation.

    Uses a HybridRetriever (TF-IDF + dense embeddings + cross-encoder)
    when neural components are available, with graceful fallback to pure
    TF-IDF.  The classification label feeds into retrieval for type-aware
    weighting, connecting the two course modules.
    """

    def __init__(self, classifier: BaseClassifier = None,
                 retriever_path: str = None,
                 config_path: str = None,
                 use_distilbert: bool = False,
                 teacher_retriever_path: Optional[str] = None):
        self.classifier = classifier or self._load_default_classifier(use_distilbert)
        self.general_retriever = self._load_retriever(retriever_path)
        self.teacher_retriever = self._load_retriever(teacher_retriever_path) if teacher_retriever_path else None

        # Neural retrieval components (loaded lazily)
        base = Path(__file__).resolve().parent.parent.parent
        self.dense_retriever = self._init_dense_retriever(base)
        self.cross_encoder = self._init_cross_encoder()
        self.glove_expander = self._init_glove_expander()

        # Hybrid orchestrator (falls back to TF-IDF if dense unavailable)
        self.hybrid_retriever = self._build_hybrid()

        self.registry = AgentRegistry(config_path or self._resolve('config/disciplines.json'))
        self.response_agent = ResponseGenerator(self.registry)
        self.quiz_agent = QuizGenerator()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve(self, path):
        base = Path(__file__).resolve().parent.parent.parent
        return str(base / path)

    def _load_default_classifier(self, use_distilbert):
        if use_distilbert and DISTILBERT_AVAILABLE:
            clf = DistilBERTClassifier()
            model_dir = Path(self._resolve('models/distilbert'))
            if (model_dir / 'model.safetensors').exists() or (model_dir / 'pytorch_model.bin').exists():
                clf.model = DistilBertForSequenceClassification.from_pretrained(str(model_dir)).to(clf.device)
                clf.tokenizer = DistilBertTokenizer.from_pretrained(str(model_dir))
            return clf
        vocab_path = self._resolve('models/bilstm/vocab.pkl')
        model_path = self._resolve('models/bilstm/final.pt')
        label_path = self._resolve('models/bilstm/label2idx.json')
        clf = BiLSTMClassifier(vocab_size=1)
        if Path(model_path).exists():
            clf.load(model_path, vocab_path, label_path)
        return clf

    def _load_retriever(self, retriever_path):
        path = retriever_path or self._resolve('models/retrieval/tfidf.pkl')
        retriever = TFIDFRetriever()
        if Path(path).exists():
            retriever.load(path)
        return retriever

    def _init_dense_retriever(self, base) -> Optional[DenseRetriever]:
        emb_path = base / 'models' / 'retrieval' / 'dense_embeddings.npy'
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        if emb_path.exists():
            dr = DenseRetriever()
            dr.load(str(emb_path))
            corpus_path = base / 'models' / 'retrieval' / 'corpus_chunks.csv'
            if corpus_path.exists():
                import pandas as pd
                dr.corpus_df = pd.read_csv(corpus_path)
            return dr
        return None

    def _init_cross_encoder(self) -> Optional[CrossEncoderReranker]:
        if CROSS_ENCODER_AVAILABLE:
            try:
                return CrossEncoderReranker()
            except Exception:
                return None
        return None

    def _init_glove_expander(self) -> Optional[GloVeQueryExpander]:
        try:
            expander = GloVeQueryExpander()
            glove_path = Path(self._resolve('data/glove.6B.50d.txt'))
            cache_path = Path(self._resolve('data/glove_cache.npz'))
            if glove_path.exists() or cache_path.exists():
                return expander
        except Exception:
            pass
        return None

    def _build_hybrid(self) -> HybridRetriever:
        return HybridRetriever(
            tfidf_retriever=self.general_retriever,
            dense_retriever=self.dense_retriever,
            cross_encoder=self.cross_encoder,
            glove_expander=self.glove_expander,
            alpha=0.3,
        )

    # ------------------------------------------------------------------
    # Topic extraction
    # ------------------------------------------------------------------

    def _extract_topic(self, question: str) -> str:
        q = question.strip().rstrip('?.,!;')
        import re

        patterns = [
            (r'(?i)^what\s+is\s+(?:the\s+)?(.+)$', 1),
            (r'(?i)^what\s+does\s+(?:.+?\s+)?mean\s+by\s+(.+)$', 1),
            (r'(?i)^what\s+does\s+(.+?)\s+mean$', 1),
            (r'(?i)^how\s+do\s+(.+?)\s+differ', 1),
            (r'(?i)^how\s+does\s+(.+?)\s+relate', 1),
            (r'(?i)^how\s+(?:does|is|are)\s+(.+?)\s+(?:understood|connected|related|defined)', 1),
            (r'(?i)^give\s+an?\s+example\s+of\s+(.+)$', 1),
            (r'(?i)^explain\s+(.+)$', 1),
            (r'(?i)^compare\s+(.+)$', 1),
            (r'(?i)^test\s+me\s+(?:on|about)\s+(.+)$', 1),
            (r'(?i)^quiz\s+me\s+(?:on|about)\s+(.+)$', 1),
        ]

        for pat, grp in patterns:
            m = re.match(pat, q)
            if m:
                topic = m.group(grp).strip()
                topic = re.split(r'\s+according\s+to\s+', topic)[0].strip()
                if topic:
                    topic = re.sub(r'^a\s+|^an\s+|^the\s+', '', topic, count=1).strip()
                    return topic

        clean = lambda w: w.strip('?.,!;\'"()[]{}').rstrip("'s")
        words = q.split()
        skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                      'by', 'of', 'to', 'in', 'for', 'on', 'at', 'with', 'from',
                      'do', 'does', 'me', 'give', 'explain', 'compare',
                      'what', 'how', 'why', 'which', 'who', 'whom', 'whose',
                      'test', 'quiz', 'does', 'did', 'can', 'will', 'would',
                      'could', 'should', 'may', 'might', 'has', 'have', 'had',
                      'been', 'being', 'not', 'no', 'nor', 'but', 'or', 'and'}
        for w in reversed(words):
            cw = clean(w)
            if cw and cw[0].isupper() and cw.lower() not in skip_words:
                return cw
        for w in reversed(words):
            cw = clean(w)
            if cw and cw.lower() not in skip_words and len(cw) > 3:
                return cw
        for w in reversed(words):
            cw = clean(w)
            if cw and cw.lower() not in skip_words:
                return cw
        return clean(words[-1]) if words else 'concept'

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def _retrieve_passages(self, question: str, top_k: int,
                           question_type: Optional[str]) -> list:
        """Run hybrid retrieval (with TF-IDF fallback)."""
        raw = self.hybrid_retriever.retrieve(
            question, top_k=top_k, question_type=question_type
        )
        passages = []
        for item in raw:
            idx, text, score, source_type = item
            source = self.hybrid_retriever.get_source(idx)
            passages.append(Passage(text=text, source=source, score=score, source_type=source_type))
        return passages

    def _passthrough(self, question: str, output: PipelineOutput) -> PipelineOutput:
        return output

    def process(self, question: str, top_k: int = 3) -> PipelineOutput:
        pred_label, confidence, top_3 = self.classifier.predict(question)
        classification = ClassificationResult(question, pred_label, confidence, top_3)

        topic = self._extract_topic(question)

        with ThreadPoolExecutor(max_workers=2) as pool:
            main_future = pool.submit(self._run_main_pipeline, question, top_k, pred_label)
            quiz_future = pool.submit(self.quiz_agent.generate, question, topic, pred_label)

            retrieval, response = main_future.result()
            quiz = quiz_future.result()

        return PipelineOutput(question, classification, retrieval, response, quiz)

    def _run_main_pipeline(self, question: str, top_k: int,
                           pred_label: str):
        """Run retrieval then response generation (sequential dependency)."""
        passages = self._retrieve_passages(question, top_k, question_type=pred_label)
        retrieval = RetrievalResult(passages=passages)

        passages_with_sources = [
            {'text': p.text, 'source': p.source, 'source_type': p.source_type}
            for p in passages
        ]

        response = self.response_agent.generate(
            question, pred_label, passages_with_sources,
            philosopher=self.registry.list_disciplines()[0]
        )
        return retrieval, response

    def process_with_teacher_materials(self, question: str, top_k: int = 3) -> PipelineOutput:
        pred_label, confidence, top_3 = self.classifier.predict(question)
        classification = ClassificationResult(question, pred_label, confidence, top_3)

        topic = self._extract_topic(question)

        with ThreadPoolExecutor(max_workers=2) as pool:
            main_future = pool.submit(
                self._run_teacher_pipeline, question, top_k, pred_label
            )
            quiz_future = pool.submit(self.quiz_agent.generate, question, topic, pred_label)

            retrieval, response = main_future.result()
            quiz = quiz_future.result()

        return PipelineOutput(question, classification, retrieval, response, quiz)

    def _run_teacher_pipeline(self, question: str, top_k: int,
                              pred_label: str):
        if self.teacher_retriever and self.teacher_retriever.is_fitted:
            combined = self.general_retriever.retrieve_with_boost(
                question, top_k=top_k,
                teacher_retriever=self.teacher_retriever,
                boost_factor=2.0,
                question_type=pred_label,
                glove_expander=self.glove_expander,
            )
            passages = []
            for idx, text, score, stype in combined:
                retriever = self.teacher_retriever if stype == 'teacher' else self.general_retriever
                source = retriever.get_source(idx)
                passages.append(Passage(text=text, source=source, score=score, source_type=stype))
        else:
            passages = self._retrieve_passages(question, top_k, question_type=pred_label)

        retrieval = RetrievalResult(passages=passages)

        passages_with_sources = [
            {'text': p.text, 'source': p.source, 'source_type': p.source_type}
            for p in passages
        ]

        response = self.response_agent.generate(
            question, pred_label, passages_with_sources,
            philosopher=self.registry.list_disciplines()[0]
        )
        return retrieval, response

