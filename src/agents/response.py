"""Response generator agent that composes philosophical answers."""

from typing import List
from .registry import AgentRegistry


class ResponseGenerator:
    """Generates philosophical responses in the style of relevant philosophers."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def generate(self, question: str, class_label: str,
                 retrieved_passages: List[str],
                 philosopher: str = 'filosofia') -> str:
        expert = self.registry.get(philosopher)
        persona = expert.get_persona()
        style = persona.split('Sei ')[-1].split('.')[0] if 'Sei ' in persona else 'Socrates'
        response = f"[Response in the style of {style}]\n\n"
        response += f"On the question '{question}':\n\n"
        if retrieved_passages:
            response += "Relevant passages from the corpus:\n"
            for i, passage in enumerate(retrieved_passages, 1):
                response += f"{i}. {passage}\n"
        response += "\n[Complete response requires LLM integration]"
        return response
