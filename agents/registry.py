# registry.py
import json
from agents.base_agents import ExpertAgent

class AgentRegistry:
    def __init__(self, config_path):
        self.config_path = config_path
        self.agents = {}
        self._load_all()

    def _load_all(self):
        config = json.load(open(self.config_path))
        for name, settings in config.items():
            self.agents[name] = ExpertAgent(
                discipline=name,
                persona=settings['persona'],
                corpus_path=f"disciplines/{name}/corpus_chunks.csv"
            )

    def list_disciplines(self):
        return list(self.agents.keys())

    def get(self, discipline):
        return self.agents.get(discipline, list(self.agents.values())[0])

    def add_discipline(self, name, persona):
        # logica completa in settimana 4
        pass