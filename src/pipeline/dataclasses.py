"""Data classes for pipeline input/output."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class ClassificationResult:
    question: str
    predicted_label: str
    confidence: float
    top_3_labels: List[Tuple[str, float]]


@dataclass
class Passage:
    text: str
    source: Dict[str, str]
    score: float


@dataclass
class RetrievalResult:
    passages: List[Passage] = field(default_factory=list)


@dataclass
class PipelineOutput:
    question: str
    classification: ClassificationResult
    retrieval: RetrievalResult
    response: str
    quiz: str
