"""Pipeline core - orchestrates all agents."""

from pathlib import Path

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
    """Integrated pipeline connecting all agents."""

    def __init__(self, classifier: BaseClassifier = None,
                 retriever_path: str = None,
                 config_path: str = None,
                 use_distilbert: bool = False):
        self.classifier = classifier or self._load_default_classifier(use_distilbert)
        self.retriever = self._load_retriever(retriever_path)
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

    def _load_retriever(self, retriever_path):
        path = retriever_path or self._resolve('models/retrieval/tfidf.pkl')
        retriever = TFIDFRetriever()
        if Path(path).exists():
            retriever.load(path)
        return retriever

    def process(self, question: str, top_k: int = 3) -> PipelineOutput:
        pred_label, confidence, top_3 = self.classifier.predict(question)
        classification = ClassificationResult(question, pred_label, confidence, top_3)

        topic = question.split()[:3][-1] if len(question.split()) > 2 else 'concept'
        raw_results = self.retriever.retrieve(question, top_k=top_k)
        passages = []
        for idx, text, score in raw_results:
            source = self.retriever.get_source(idx)
            passages.append(Passage(text=text, source=source, score=score))
        retrieval = RetrievalResult(passages=passages)

        response = self.response_agent.generate(
            question, pred_label, [p.text for p in passages],
            philosopher=self.registry.list_disciplines()[0]
        )
        quiz = self.quiz_agent.generate(question, topic, pred_label)

        return PipelineOutput(question, classification, retrieval, response, quiz)
