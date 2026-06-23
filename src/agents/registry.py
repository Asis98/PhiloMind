"""Registry for managing discipline-specific expert agents."""

import json
from .base import ExpertAgent


class AgentRegistry:
    """Registry that loads and manages expert agents by discipline."""

    def __init__(self, config_path):
        self.config_path = config_path
        self.agents = {}
        self._load_all()

    def _load_all(self):
        with open(self.config_path) as f:
            config = json.load(f)
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
