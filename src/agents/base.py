"""Base agent interface for discipline-specific expert agents."""


class ExpertAgent:
    """An expert agent representing a specific philosophical discipline."""

    def __init__(self, discipline, persona, corpus_path):
        self.discipline = discipline
        self.persona = persona
        self.corpus_path = corpus_path

    def get_persona(self):
        return self.persona
