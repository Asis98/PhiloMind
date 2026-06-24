"""Pipeline core - orchestrates all agents with teacher material support."""

from pathlib import Path
from typing import Optional

from src.classification.base import BaseClassifier
from src.classification.bilstm import BiLSTMClassifier
from src.retrieval.tfidf import TFIDFRetriever
from src.agents.registry import AgentRegistry
from src.agents.response import ResponseGenerator
from src.agents.quiz import QuizGenerator
from .dataclasses import ClassificationResult, RetrievalResult, Passage, PipelineOutput

try:
    from src.classification.distilbert import DistilBERTClassifier
    DISTILBERT_AVAILABLE = True
except ImportError:
    DISTILBERT_AVAILABLE = False


class PhiloMindPipeline:
    """Integrated pipeline connecting all agents with teacher material fallback."""

    def __init__(self, classifier: BaseClassifier = None,
                 retriever_path: str = None,
                 config_path: str = None,
                 use_distilbert: bool = False,
                 teacher_retriever_path: Optional[str] = None):
        self.classifier = classifier or self._load_default_classifier(use_distilbert)
        self.general_retriever = self._load_retriever(retriever_path)
        self.teacher_retriever = self._load_retriever(teacher_retriever_path) if teacher_retriever_path else None
        self.registry = AgentRegistry(config_path or self._resolve('config/disciplines.json'))
        self.response_agent = ResponseGenerator(self.registry)
        self.quiz_agent = QuizGenerator()

    def _resolve(self, path):
        base = Path(__file__).resolve().parent.parent.parent
        return str(base / path)

    def _load_default_classifier(self, use_distilbert):
        if use_distilbert and DISTILBERT_AVAILABLE:
            clf = DistilBERTClassifier()
            model_path = self._resolve('models/distilbert/pytorch_model.bin')
            if Path(model_path).exists():
                clf.load(self._resolve('models/distilbert/model.pt'))
            return clf
        vocab_path = self._resolve('models/bilstm/vocab.pkl')
        model_path = self._resolve('models/bilstm/final.pt')
        label_path = self._resolve('models/bilstm/label2idx.json')
        clf = BiLSTMClassifier(vocab_size=1)
        if Path(model_path).exists():
            clf.load(model_path, vocab_path, label_path)
        return clf

    def _extract_topic(self, question: str) -> str:
        question_words = {'what', 'how', 'why', 'which', 'explain', 'compare', 'does', 'who', 'whom', 'whose', 'where', 'when'}
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'by', 'of', 'to', 'in', 'for', 'on', 'at', 'with', 'from'}
        clean = lambda w: w.strip('?.,!;\'\"()[]{}').rstrip("'s")
        words = question.split()
        for w in words[1:]:
            cw = clean(w)
            if cw and cw[0].isupper() and cw.lower() not in question_words and cw.lower() not in stop_words:
                return cw
        for w in words:
            cw = clean(w)
            if cw and cw.lower() not in question_words and cw.lower() not in stop_words:
                return cw
        return clean(words[-1]) if words else 'concept'

    def _load_retriever(self, retriever_path):
        path = retriever_path or self._resolve('models/retrieval/tfidf.pkl')
        retriever = TFIDFRetriever()
        if Path(path).exists():
            retriever.load(path)
        return retriever

    def process(self, question: str, top_k: int = 3) -> PipelineOutput:
        pred_label, confidence, top_3 = self.classifier.predict(question)
        classification = ClassificationResult(question, pred_label, confidence, top_3)

        topic = self._extract_topic(question)

        results = self.general_retriever.retrieve(question, top_k=top_k)
        passages = []
        for idx, text, score in results:
            source = self.general_retriever.get_source(idx)
            passages.append(Passage(text=text, source=source, score=score, source_type='general'))

        retrieval = RetrievalResult(passages=passages)

        passages_with_sources = [{'text': p.text, 'source': p.source, 'source_type': p.source_type}
                                  for p in passages]

        response = self.response_agent.generate(
            question, pred_label, passages_with_sources,
            philosopher=self.registry.list_disciplines()[0]
        )
        quiz = self.quiz_agent.generate(question, topic, pred_label)

        return PipelineOutput(question, classification, retrieval, response, quiz)

    def process_with_teacher_materials(self, question: str, top_k: int = 3) -> PipelineOutput:
        pred_label, confidence, top_3 = self.classifier.predict(question)
        classification = ClassificationResult(question, pred_label, confidence, top_3)

        topic = self._extract_topic(question)

        combined = self.general_retriever.retrieve_with_boost(
            question, top_k=top_k, teacher_retriever=self.teacher_retriever, boost_factor=2.0
        )
        passages = []
        for idx, text, score, stype in combined:
            retriever = self.teacher_retriever if stype == 'teacher' else self.general_retriever
            source = retriever.get_source(idx)
            passages.append(Passage(text=text, source=source, score=score, source_type=stype))

        retrieval = RetrievalResult(passages=passages)

        passages_with_sources = [{'text': p.text, 'source': p.source, 'source_type': p.source_type}
                                  for p in passages]

        response = self.response_agent.generate(
            question, pred_label, passages_with_sources,
            philosopher=self.registry.list_disciplines()[0]
        )
        quiz = self.quiz_agent.generate(question, topic, pred_label)

        return PipelineOutput(question, classification, retrieval, response, quiz)
