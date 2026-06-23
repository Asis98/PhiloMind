"""Output formatting for pipeline results."""

from .dataclasses import PipelineOutput


def format_output(output: PipelineOutput) -> str:
    """Format pipeline output for display."""
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
        lines.append(f"  [{i}] {p.source['philosopher']} - {p.source['work']} (score: {p.score:.2%})")
        lines.append(f"       {p.text[:150]}...")
    lines.append("")
    lines.append("  GENERATED RESPONSE")
    lines.append(f"  {output.response[:200]}")
    lines.append("")
    lines.append("  QUIZ / FOLLOW-UP")
    lines.append(f"  {output.quiz[:200]}")
    lines.append("")
    lines.append("=" * 70)
    return '\n'.join(lines)
