"""Quiz generator that produces questions independently from retrieved passages."""

import random
import re
from typing import List, Dict


PHILOSOPHERS = [
    "Plato", "Aristotle", "Socrates", "Descartes", "Kant", "Nietzsche",
    "Hegel", "Marx", "Hume", "Locke", "Rousseau", "Spinoza", "Leibniz",
    "Schopenhauer", "Heidegger", "Wittgenstein", "Husserl", "Sartre",
    "Foucault", "Arendt", "Kierkegaard", "Russell", "Rawls", "Deleuze",
    "Derrida", "Habermas", "Adorno", "Bacon", "Hobbes", "Berkeley",
    "Epicurus", "Zeno", "Plotinus", "Augustine", "Aquinas", "Anaximander"
]

_LABEL_FORMAT = {
    'definition': 'best defines',
    'comparison': 'best describes the distinction involving',
    'example': 'is the best example of',
    'deepening': 'best explains the significance of',
    'quiz': 'is true about',
}

_PHILOSOPHER_LOWER = {p.lower() for p in PHILOSOPHERS}


def _find_philosopher(text: str) -> str:
    text_lower = text.lower()
    for p in PHILOSOPHERS:
        if p.lower() in text_lower:
            return p
    return ""


def _extract_key_terms(text: str) -> List[str]:
    terms = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    stop = {
        'the', 'this', 'that', 'with', 'from', 'which', 'what',
        'when', 'where', 'how', 'why', 'have', 'has', 'had',
        'been', 'being', 'into', 'upon', 'than', 'thus', 'very',
        'such', 'some', 'more', 'most', 'many', 'also', 'only',
        'just', 'even', 'though', 'while', 'since', 'after',
        'before', 'between', 'under', 'over', 'without', 'within',
    }
    return list(dict.fromkeys(t for t in terms if t.lower() not in stop and len(t) > 3))


class QuizGenerator:
    """Generates quiz questions independently of retrieval results.

    Uses the question text, topic, and classification label to produce
    meaningful quizzes even when no passages are available.
    """

    def __init__(self):
        random.seed(42)

    def generate(self, question: str, topic: str, class_label: str,
                 passages: List[Dict] = None) -> Dict[str, str]:
        """Generate a quiz from the question alone (passages ignored)."""
        return self._generate_independent(question, topic, class_label)

    def _generate_independent(self, question: str, topic: str,
                              class_label: str) -> Dict[str, str]:
        """Generate a meaningful quiz using only question, topic and classification."""
        philosopher = _find_philosopher(question)
        topic_desc = topic if len(topic) > 3 else _extract_topic_fallback(question)

        is_philosopher_topic = topic_desc.lower() in _PHILOSOPHER_LOWER

        label_format = _LABEL_FORMAT.get(class_label, 'is true about')

        has_and = ' and ' in topic_desc.lower()
        mention_philosopher = philosopher and not is_philosopher_topic and not has_and

        if mention_philosopher:
            question_text = f"Which of the following {label_format} {topic_desc}, according to {philosopher}?"
        else:
            question_text = f"Which of the following {label_format} {topic_desc}?"

        correct = _build_correct_answer(topic_desc, class_label, philosopher, question)
        distractors = _build_distractors(topic_desc, class_label, philosopher)

        distractors = distractors[:3]
        options = distractors + [correct]
        random.shuffle(options)

        return {
            'question': question_text,
            'options': options,
            'correct_answer': correct,
            'explanation': (
                f"Based on the classification '{class_label}' "
                f"of your question: \"{question}\""
            ),
        }


def _extract_topic_fallback(question: str) -> str:
    words = [w for w in re.sub(r'[^\w\s]', '', question).split()
             if w[0].isupper() or len(w) > 5]
    return words[0] if words else 'this philosophical concept'


def _capitalize(word: str) -> str:
    return word[0].upper() + word[1:] if word else word


def _build_correct_answer(topic: str, class_label: str,
                          philosopher: str, question: str) -> str:
    tc = _capitalize(topic)
    is_philosopher = topic.lower() in _PHILOSOPHER_LOWER
    has_and = ' and ' in topic.lower()

    if class_label == 'definition':
        if is_philosopher:
            return (
                f"The philosophy of {topic} centers on fundamental questions "
                f"about reality, knowledge, and human existence."
            )
        return (
            f"{tc} is a philosophical concept concerning fundamental "
            f"questions about reality and knowledge."
        )

    if class_label == 'comparison':
        if has_and:
            verb = 'represent' if ' and ' in topic.lower() else 'represents'
            return (
                f"{tc} {verb} fundamentally different approaches "
                f"in the history of philosophy."
            )
        if is_philosopher:
            other = next((p for p in PHILOSOPHERS if p.lower() != topic.lower()), 'other thinkers')
            return (
                f"{topic}\u2019s philosophical approach differs fundamentally "
                f"from {other}\u2019s on several key questions."
            )
        return (
            f"{tc} is interpreted differently across philosophical "
            f"traditions, revealing fundamental distinctions in thought."
        )

    if class_label == 'example':
        if is_philosopher:
            return (
                f"{topic}\u2019s ideas are best understood through specific "
                f"examples and thought experiments found in his works."
            )
        return (
            f"{tc} can be illustrated through various philosophical "
            f"thought experiments and historical cases."
        )

    if class_label == 'deepening':
        if is_philosopher:
            return (
                f"{topic}\u2019s philosophy raises profound questions about "
                f"knowledge, reality, and the human condition."
            )
        return (
            f"{tc} raises fundamental questions about knowledge, "
            f"reality, and human existence."
        )

    if is_philosopher:
        return (
            f"{topic} is a central figure in the philosophical tradition, "
            f"known for influential ideas about reality and knowledge."
        )
    return (
        f"{tc} is a key concept in the philosophical tradition."
    )


def _build_distractors(topic: str, class_label: str,
                       philosopher: str) -> List[str]:
    is_philosopher = topic.lower() in _PHILOSOPHER_LOWER
    has_and = ' and ' in topic.lower()

    if class_label == 'definition':
        if is_philosopher:
            return [
                f"{topic} was primarily a scientist, not a philosopher.",
                f"{topic}\u2019s work focused only on political theory.",
                f"{topic} wrote exclusively about mathematics and logic.",
                f"{topic}\u2019s ideas have no relevance to modern philosophy.",
            ]
        return [
            f"{topic} is a scientific theory developed in the 19th century.",
            f"{topic} refers to a specific historical event in ancient Greece.",
            f"{topic} is a modern political ideology with no philosophical roots.",
            f"{topic} was coined as a literary term in the Renaissance.",
        ]

    if class_label == 'comparison':
        if has_and:
            return [
                f"{topic} held identical philosophical views on all major questions.",
                f"Neither {topic} made any original contributions to philosophy.",
                f"The ideas of {topic} were entirely borrowed from medieval thinkers.",
                f"{topic} never engaged in philosophical debate.",
            ]
        if is_philosopher:
            return [
                f"{topic}\u2019s views are identical to all other philosophers.",
                f"{topic} never engaged with other philosophical traditions.",
                f"The ideas of {topic} were entirely borrowed from earlier thinkers.",
                f"{topic} rejected all forms of philosophical comparison.",
            ]
        return [
            f"All philosophers agree on the meaning and implications of {topic}.",
            f"{topic} was not discussed by any major philosopher in history.",
            f"The concept of {topic} is purely scientific with no philosophical value.",
            f"{topic} has a single, universally accepted definition.",
        ]

    if class_label == 'example':
        if is_philosopher:
            return [
                f"{topic}\u2019s philosophy cannot be illustrated with concrete examples.",
                f"The only record of {topic}\u2019s ideas comes from a single fragment.",
                f"{topic} never used examples to explain philosophical concepts.",
                f"No complete works of {topic} survive to the present day.",
            ]
        return [
            f"{topic} cannot be illustrated with concrete examples at all.",
            f"The only known example of {topic} is found in a single ancient text.",
            f"{topic} is a purely abstract concept with no real-world connections.",
            f"No philosopher has ever provided an example of {topic}.",
        ]

    if class_label == 'deepening':
        if is_philosopher:
            return [
                f"{topic}\u2019s philosophy is no longer studied in academia.",
                f"{topic} only wrote about a single, narrow topic.",
                f"The significance of {topic}\u2019s work has been completely overturned.",
                f"{topic} contributed nothing to metaphysical or epistemological questions.",
            ]
        return [
            f"{topic} is not considered relevant in contemporary philosophy.",
            f"The significance of {topic} was definitively resolved in ancient times.",
            f"{topic} is mainly studied in economics and sociology, not philosophy.",
            f"Modern philosophy has proven that {topic} is a meaningless concept.",
        ]

    if is_philosopher:
        return [
            f"{topic} was not a philosopher but a literary figure.",
            f"{topic} never wrote about {topic.lower() if topic.lower() != topic.lower() else 'philosophy'}.",
            f"{topic}\u2019s ideas were disproven by modern science.",
            f"{topic} is primarily known for contributions to mathematics.",
        ]
    return [
        f"{philosopher or 'Any major philosopher'} never wrote about {topic}.",
        f"{topic} was invented in the 20th century with no historical basis.",
        f"{topic} is a term from modern psychology, not philosophy.",
        f"The concept of {topic} has been abandoned by contemporary philosophers.",
    ]
