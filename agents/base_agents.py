class ExpertAgent:
    def __init__(self, discipline, persona, corpus_path):
        self.discipline = discipline
        self.persona = persona
        self.corpus_path = corpus_path
        self.chunks = []

    def retrieve(self, query, top_k=3):
        raise NotImplementedError

    def get_persona(self):
        return self.persona