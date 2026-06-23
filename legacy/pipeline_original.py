"""
Integrated pipeline for PhiloMind.
Connects:
- Agent 1: BiLSTM Classifier
- Agent 2: TF-IDF Retriever
- Agent 3: Response Generator (stub)
- Agent 4: Quiz Generator (stub)
"""

import torch
import pickle
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Import developed modules
from models.bilstm_classifier import QuestionClassifier, QuestionDataset
from models.tfidf_retriever import TFIDFRetriever
from agents.base_agents import ExpertAgent
from agents.registry import AgentRegistry

@dataclass
class ClassificationResult:
    """Result of question classification."""
    question: str
    predicted_label: str
    confidence: float
    top_3_labels: List[Tuple[str, float]]


@dataclass
class RetrievalResult:
    """Result of philosophical passage retrieval."""
    passages: List[str]
    sources: List[Dict]  # philosopher, work
    scores: List[float]


@dataclass
class PipelineOutput:
    """Complete pipeline output."""
    question: str
    classification: ClassificationResult
    retrieval: RetrievalResult
    response: str  # generato da Agent 3 (stub)
    quiz: str  # generato da Agent 4 (stub)


class QuestionClassificationAgent:
    """Agent 1: Question classification."""

    def __init__(self, model_path: str, vocab_path: str, label2idx_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: path to saved BiLSTM model
            vocab_path: path to vocabulary
            label2idx_path: path to label->index mapping
        """
        self.device = torch.device(device)

        # Carica vocabolario e mapping
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        with open(label2idx_path, 'r') as f:
            self.label2idx = json.load(f)

        self.idx2label = {v: k for k, v in self.label2idx.items()}

        # Recreate the model
        self.classifier = QuestionClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=100,
            hidden_dim=64,
            n_classes=5,
            n_layers=2,
            dropout=0.3,
            device=device
        )

        # Carica i pesi
        if Path(model_path).exists():
            self.classifier.load_model(model_path)
            print(f"✅ Classificatore caricato da {model_path}")
        else:
            print(f"⚠️  Model not found at {model_path}")

    def classify(self, question: str) -> ClassificationResult:
        """Classify a question."""
        # Tokenizza
        tokens = question.lower().split()[:50]
        token_indices = []

        for token in tokens:
            clean_token = token.strip(',.!?;:\'"')
            if clean_token in self.vocab:
                token_indices.append(self.vocab[clean_token])
            else:
                token_indices.append(self.vocab.get('<UNK>', 1))

        # Padding
        if len(token_indices) < 50:
            token_indices += [self.vocab.get('<PAD>', 0)] * (50 - len(token_indices))

        # Predizione
        tokens_tensor = torch.LongTensor([token_indices]).to(self.device)
        lengths = torch.LongTensor([min(len(tokens), 50)]).to(self.device)

        preds, probs = self.classifier.predict(tokens_tensor, lengths)

        pred_idx = preds[0]
        pred_label = self.idx2label[pred_idx]
        confidence = float(probs[0, pred_idx])

        # Top 3
        top_3_indices = np.argsort(probs[0])[::-1][:3]
        top_3_labels = [
            (self.idx2label[idx], float(probs[0, idx]))
            for idx in top_3_indices
        ]

        return ClassificationResult(
            question=question,
            predicted_label=pred_label,
            confidence=confidence,
            top_3_labels=top_3_labels
        )


class DocumentRetrievalAgent:
    """Agent 2: Philosophical passage retrieval."""

    def __init__(self, retriever_path: str):
        """
        Args:
            retriever_path: path to saved TF-IDF retriever
        """
        self.retriever = TFIDFRetriever()
        self.retriever.load(retriever_path)
        print(f"✅ Retriever caricato da {retriever_path}")

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve relevant passages."""
        results = self.retriever.retrieve(query, top_k=top_k)

        passages = [text for text, _ in results]
        scores = [score for _, score in results]

        # Estrai metadata
        sources = []
        for text, score in results:
            # Search for the passage in the corpus to get metadata
            matching_rows = self.retriever.corpus_df[self.retriever.corpus_df['text'] == text]
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                sources.append({
                    'philosopher': row.get('philosopher', 'Unknown'),
                    'work': row.get('work', 'Unknown')
                })
            else:
                sources.append({'philosopher': 'Unknown', 'work': 'Unknown'})

        return RetrievalResult(
            passages=passages,
            sources=sources,
            scores=scores
        )


class ResponseGeneratorAgent:
    """Agent 3: Philosophical response generation (stub)."""

    def __init__(self, registry: AgentRegistry):
        """
        Args:
            registry: registry of expert agents per discipline
        """
        self.registry = registry

    def generate(self, question: str, class_label: str,
                 retrieved_passages: List[str], philosopher: str = 'filosofia') -> str:
        """
        Generate a response in the philosopher's style.

        TODO: Implement with LLM (week 3)
        """
        expert_agent = self.registry.get(philosopher)
        persona = expert_agent.get_persona()

        # Stub: concatenation of passages
        response = f"[Response in the style of {persona.split('Sei ')[-1].split('.')[0]}]\n\n"
        response += f"Alla domanda '{question}':\n\n"

        if retrieved_passages:
            response += "Relevant passages from the corpus:\n"
            for i, passage in enumerate(retrieved_passages, 1):
                response += f"{i}. {passage}\n"

        response += "\n[Reply generated - full implementation in week 3 with LLM]"

        return response


class QuizGeneratorAgent:
    """Agent 4: Quiz and follow-up question generation (stub)."""

    def generate(self, question: str, topic: str, class_label: str) -> str:
        """
        Generate verification and follow-up questions.

        TODO: Implement with LLM (week 3)
        """
        quiz_templates = {
            'definition': [
                f"Which of these is a characteristic of {topic}?",
                f"How would you define the concept of {topic} in your own words?"
            ],
            'comparison': [
                f"What are the similarities between {topic}?",
                f"What difference did you notice between {topic}?"
            ],
            'example': [
                f"Can you give another example of {topic}?",
                f"In what historical context does {topic} occur?"
            ],
            'indepth': [
                f"What are the critical aspects of {topic}?",
                f"How does the concept of {topic} evolve over time?"
            ],
            'quiz': [
                f"How well do you know {topic}? Try now!",
                f"Are you ready for a test on {topic}?"
            ]
        }

        template_list = quiz_templates.get(class_label, quiz_templates['definition'])
        quiz = template_list[0] if template_list else f"Verifica su {topic}"

        return quiz + "\n\n[Answer options - full implementation with LLM in week 3]"


class PhiloMindPipeline:
    """Integrated PhiloMind pipeline."""

    def __init__(self,
                 classifier_model_path: str = 'models-utils/bilstm_baseline_final.pt',
                 vocab_path: str = 'models-utils/vocab.pkl',
                 label2idx_path: str = 'models-utils/label2idx.json',
                 retriever_path: str = 'models-utils/tfidf_retriever.pkl',
                 config_path: str = 'disciplines/config.json',
                 device: str = 'cpu'):
        """
        Initialize the pipeline with all agents.
        """
        print("\n" + "="*60)
        print("Initializing PhiloMind Pipeline")
        print("="*60)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Agent 1: Classificatore
        print("\n[Agent 1] Loading Question Classifier...")
        self.classifier_agent = QuestionClassificationAgent(
            model_path=classifier_model_path,
            vocab_path=vocab_path,
            label2idx_path=label2idx_path,
            device=self.device
        )

        # Agent 2: Retriever
        print("[Agent 2] Loading Document Retriever...")
        self.retriever_agent = DocumentRetrievalAgent(retriever_path)

        # Agent 3: Response Generator
        print("[Agent 3] Loading Response Generator...")
        self.registry = AgentRegistry(config_path)
        self.response_agent = ResponseGeneratorAgent(self.registry)

        # Agent 4: Quiz Generator
        print("[Agent 4] Loading Quiz Generator...")
        self.quiz_agent = QuizGeneratorAgent()

        print("\n✅ Pipeline ready!")

    def process(self, question: str, top_k_passages: int = 3) -> PipelineOutput:
        """
        Process a question through the entire pipeline.

        Returns:
            PipelineOutput with all results
        """
        print(f"\n📝 Question: {question}")

        # Agent 1: Classificazione
        print("\n[1/4] Classificazione...")
        classification = self.classifier_agent.classify(question)
        print(f"  Label: {classification.predicted_label} ({classification.confidence:.2%})")
        print(f"  Top 3: {', '.join([f'{l}({c:.0%})' for l, c in classification.top_3_labels])}")

        # Extract topic from question
        topic = question.split()[:3][-1] if len(question.split()) > 2 else 'concept'

        # Agent 2: Retrieval
        print("\n[2/4] Retrieving philosophical passages...")
        retrieval = self.retriever_agent.retrieve(question, top_k=top_k_passages)
        print(f"  {len(retrieval.passages)} passages retrieved")
        for i, (text, source) in enumerate(zip(retrieval.passages, retrieval.sources), 1):
            print(f"    [{i}] ({retrieval.scores[i-1]:.3f}) {source['philosopher']} - {source['work']}")
            print(f"        {text[:80]}...")

        # Agent 3: Response
        print("\n[3/4] Generating response...")
        response = self.response_agent.generate(
            question=question,
            class_label=classification.predicted_label,
            retrieved_passages=retrieval.passages,
            philosopher=self.registry.list_disciplines()[0]
        )
        print(f"  Response generated")

        # Agent 4: Quiz
        print("\n[4/4] Generating quiz...")
        quiz = self.quiz_agent.generate(
            question=question,
            topic=topic,
            class_label=classification.predicted_label
        )
        print(f"  Quiz generated")

        return PipelineOutput(
            question=question,
            classification=classification,
            retrieval=retrieval,
            response=response,
            quiz=quiz
        )

    def batch_process(self, questions: List[str]) -> List[PipelineOutput]:
        """Process a batch of questions."""
        return [self.process(q) for q in questions]


def format_output(output: PipelineOutput) -> str:
    """Format output for display."""
    result = f"""
{'='*70}
QUESTION: {output.question}
{'='*70}

📌 CLASSIFICATION (Agent 1)
  Predicted class: {output.classification.predicted_label}
  Confidence: {output.classification.confidence:.2%}
  Alternatives: {', '.join([f'{l}({c:.0%})' for l, c in output.classification.top_3_labels])}

📚 PHILOSOPHICAL PASSAGES (Agent 2)
"""

    for i, (passage, source, score) in enumerate(
        zip(output.retrieval.passages, output.retrieval.sources, output.retrieval.scores), 1
    ):
        result += f"""
  [{i}] {source['philosopher']} - {source['work']}       (relevance: {score:.2%})
      {passage[:150]}...
"""

    result += f"""

💡 PHILOSOPHICAL RESPONSE (Agent 3)
{output.response}

❓ VERIFICATION (Agent 4)
{output.quiz}

{'='*70}
"""

    return result


if __name__ == '__main__':
    # Test the pipeline
    pipeline = PhiloMindPipeline()

    # Test questions
    test_questions = [
        "What is Cartesian dualism?",
        "How do Plato and Aristotle differ?",
        "What does 'eternal return' mean in Nietzsche?"
    ]

    print("\n" + "="*70)
    print("TEST PIPELINE")
    print("="*70)

    for question in test_questions:
        output = pipeline.process(question)
        print(format_output(output))
        print("\n")

