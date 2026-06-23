"""Abstract base class for question classifiers."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseClassifier(ABC):
    """Abstract interface for question classification models."""

    @abstractmethod
    def predict(self, question: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict the class label for a question.

        Returns:
            Tuple of (predicted_label, confidence, top_3_labels)
            where top_3_labels is a list of (label, score) tuples.
        """
        pass
