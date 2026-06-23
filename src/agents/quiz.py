"""Quiz generator agent that creates follow-up questions."""


class QuizGenerator:
    """Generates quiz and follow-up questions based on classification."""

    TEMPLATES = {
        'definizione': [
            "Which of the following is a key characteristic of {topic}?",
            "How would you define the concept of {topic} in your own words?"
        ],
        'confronto': [
            "What are the similarities between {topic}?",
            "What differences did you notice regarding {topic}?"
        ],
        'esempio': [
            "Can you provide another example of {topic}?",
            "In what historical context does {topic} appear?"
        ],
        'approfondimento': [
            "What are the critical aspects of {topic}?",
            "How does the concept of {topic} evolve over time?"
        ],
        'quiz': [
            "How well do you know {topic}? Test yourself!",
            "Are you ready for a test on {topic}?"
        ]
    }

    def generate(self, question: str, topic: str, class_label: str) -> str:
        templates = self.TEMPLATES.get(class_label, self.TEMPLATES['definizione'])
        quiz_template = templates[0] if templates else "Test on {topic}"
        quiz = quiz_template.format(topic=topic)
        return quiz + "\n\n[Answer options require LLM integration]"
