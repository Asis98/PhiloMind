"""Data classes for pipeline input/output with JSON serialization."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple


@dataclass
class ClassificationResult:
    question: str
    predicted_label: str
    confidence: float
    top_3_labels: List[Tuple[str, float]]

    def to_dict(self):
        return {
            'question': self.question,
            'predicted_label': self.predicted_label,
            'confidence': round(self.confidence, 4),
            'top_3_labels': [{'label': l, 'score': round(s, 4)} for l, s in self.top_3_labels]
        }


@dataclass
class Passage:
    text: str
    source: Dict[str, str]
    score: float
    source_type: str = 'general'

    def to_dict(self):
        return {
            'text': self.text[:300],
            'source': self.source,
            'score': round(self.score, 4),
            'source_type': self.source_type
        }


@dataclass
class RetrievalResult:
    passages: List[Passage] = field(default_factory=list)

    def to_dict(self):
        return {'passages': [p.to_dict() for p in self.passages]}


@dataclass
class PipelineOutput:
    question: str
    classification: ClassificationResult
    retrieval: RetrievalResult
    response: str
    quiz: Dict[str, str]

    def to_dict(self):
        return {
            'question': self.question,
            'classification': self.classification.to_dict(),
            'retrieval': self.retrieval.to_dict(),
            'response': self.response,
            'quiz': self.quiz
        }
