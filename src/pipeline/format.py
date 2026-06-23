"""Output formatting for pipeline results."""

import json
from .dataclasses import PipelineOutput


def format_output(output: PipelineOutput) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"QUESTION: {output.question}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  CLASSIFICATION")
    lines.append(f"  Predicted class: {output.classification.predicted_label}")
    lines.append(f"  Confidence: {output.classification.confidence:.2%}")
    alt = ', '.join([f'{l}({c:.0%})' for l, c in output.classification.top_3_labels])
    lines.append(f"  Alternatives: {alt}")
    lines.append("")
    lines.append("  RETRIEVED PASSAGES")
    for i, p in enumerate(output.retrieval.passages, 1):
        source_info = []
        if p.source.get('philosopher', 'Unknown') != 'Unknown':
            source_info.append(p.source['philosopher'])
        if p.source.get('work', 'Unknown') != 'Unknown':
            source_info.append(p.source['work'])
        if p.source.get('subject', ''):
            source_info.append(f"[{p.source['subject']}]")
        tag = f" ({', '.join(source_info)})" if source_info else ""
        s_type = f" [TEACHER]" if p.source_type == 'teacher' else ""
        lines.append(f"  [{i}]{tag} (score: {p.score:.2%}){s_type}")
        lines.append(f"       {p.text[:200]}...")
    lines.append("")
    lines.append("  GENERATED RESPONSE")
    for line in output.response.split('\n'):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("  QUIZ / FOLLOW-UP")
    if isinstance(output.quiz, dict):
        lines.append(f"  Q: {output.quiz['question']}")
        for i, opt in enumerate(output.quiz.get('options', []), 1):
            lines.append(f"     {i}. {opt}")
    else:
        lines.append(f"  {output.quiz}")
    lines.append("")
    lines.append("=" * 70)
    return '\n'.join(lines)


def format_json_output(output: PipelineOutput) -> str:
    return json.dumps(output.to_dict(), indent=2, ensure_ascii=False)
