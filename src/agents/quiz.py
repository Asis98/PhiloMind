"""Quiz generator agent with multiple-choice options."""

import random


class QuizGenerator:
    """Generates quiz and follow-up questions with multiple choice options."""

    TEMPLATES = {
        'definition': {
            'question': "Which of the following best defines {topic}?",
            'correct': "It is the philosophical concept concerning {topic_desc}.",
            'distractors': [
                "It is a modern scientific theory unrelated to philosophy.",
                "It refers to a historical event from the 19th century.",
                "It is a literary genre popular in ancient Greece."
            ]
        },
        'comparison': {
            'question': "What is a key difference between the philosophies being compared?",
            'correct': "They differ fundamentally in their approach to {topic_desc}.",
            'distractors': [
                "They are actually the same theory under different names.",
                "The difference is purely chronological, not conceptual.",
                "One is Eastern philosophy, the other Western."
            ]
        },
        'example': {
            'question': "Which scenario best illustrates {topic}?",
            'correct': "A situation where {topic_desc} manifests in practice.",
            'distractors': [
                "A purely mathematical calculation.",
                "An unrelated everyday chore.",
                "A biological process."
            ]
        },
        'deepening': {
            'question': "What is a significant implication of {topic}?",
            'correct': "It challenges our understanding of {topic_desc}.",
            'distractors': [
                "It has no practical implications whatsoever.",
                "It only applies to ancient philosophical debates.",
                "It was disproven by modern science."
            ]
        },
        'quiz': {
            'question': "Test your knowledge: what do you know about {topic}?",
            'correct': "{topic_desc} is a key concept in philosophical discourse.",
            'distractors': [
                "It is a term from computer science.",
                "It refers to a geographical location.",
                "It is a type of musical composition."
            ]
        }
    }

    def __init__(self):
        random.seed(42)

    def generate(self, question: str, topic: str, class_label: str) -> Dict[str, str]:
        template = self.TEMPLATES.get(class_label, self.TEMPLATES['definition'])
        topic_desc = topic if len(topic) > 3 else 'this philosophical concept'
        q = template['question'].format(topic=topic)
        correct = template['correct'].format(topic_desc=topic_desc)
        distractors = [d for d in template['distractors']]
        options = distractors + [correct]
        random.shuffle(options)
        return {
            'question': q,
            'options': options,
            'correct_answer': correct,
            'explanation': f"Based on the classification '{class_label}' of the question: \"{question}\""
        }
