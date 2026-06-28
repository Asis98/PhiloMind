"""Modern Gradio frontend for PhiloMind with card-based pipeline results."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import gradio as gr
from src.pipeline.core import PhiloMindPipeline
from src.ingestion.indexer import MaterialIndexer

pipeline = None
indexer = MaterialIndexer()
query_history = []

# ---- Custom CSS ----
CUSTOM_CSS = """
.philo-card {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.philo-card h3 {
    margin: 0 0 0.5rem 0;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--body-text-color-subdued, #6b7280);
}
.badge {
    display: inline-block;
    padding: 0.2em 0.7em;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-definition { background: #dbeafe; color: #1e40af; }
.badge-comparison { background: #fce7f3; color: #9d174d; }
.badge-example    { background: #d1fae5; color: #065f46; }
.badge-deepening  { background: #fef3c7; color: #92400e; }
.badge-quiz       { background: #e0e7ff; color: #3730a3; }
.badge-hybrid     { background: #ede9fe; color: #5b21b6; }
.badge-teacher    { background: #ffedd5; color: #9a3412; }

.passage-box {
    border-left: 3px solid;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    background: var(--background-fill-secondary, #f9fafb);
}
.passage-score {
    font-weight: 700;
    font-size: 0.9rem;
}
.source-tag {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued, #6b7280);
}
"""


def get_pipeline(use_teacher=False):
    global pipeline
    if pipeline is None:
        base = Path(__file__).resolve().parent.parent.parent
        teacher_path = str(base / 'models' / 'retrieval' / 'teacher_tfidf.pkl')
        pipeline = PhiloMindPipeline(
            retriever_path=str(base / 'models' / 'retrieval' / 'tfidf.pkl'),
            config_path=str(base / 'config' / 'disciplines.json'),
            teacher_retriever_path=teacher_path if Path(teacher_path).exists() else None
        )
    return pipeline


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

def upload_file(file, subject):
    if file is None:
        return "No file selected.", None, None
    if not subject or not subject.strip():
        return "Subject field is required.", None, None
    ext = Path(file.name).suffix.lower()
    if ext not in ('.txt', '.md'):
        return f"Unsupported format: {ext}. Use .txt or .md.", None, None
    dest = Path(__file__).resolve().parent.parent.parent / 'data' / 'materials' / Path(file.name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(file.name, 'rb') as src, open(dest, 'wb') as dst:
        dst.write(src.read())
    try:
        indexer.load_retriever()
    except FileNotFoundError:
        pass
    try:
        result = indexer.index_material(str(dest), subject.strip())
        msg = f"Uploaded and indexed ({result['chunks']} chunks for '{result['subject']}')."
        return msg, result, get_material_status()
    except Exception as e:
        return f"Error: {str(e)}", None, get_material_status()


def get_material_status():
    status = indexer.get_material_status()
    lines = [f"Total files: {status['total_files']} | Total chunks: {status['total_chunks']}"]
    lines.append(f"Subjects: {', '.join(status['subjects']) if status['subjects'] else 'None'}")
    lines.append("")
    for f in status['files']:
        lines.append(f"- {f['source_file']} | {f['subject']} | {f['chunks']} chunks | {f.get('last_indexed', '')[:10]}")
    return '\n'.join(lines) if status['files'] else "No materials uploaded yet."


def search_materials(query):
    if not query or not query.strip():
        return get_material_status()
    results = indexer.search_materials(query.strip())
    if not results:
        return f"No results found for '{query}'."
    seen = set()
    lines = []
    for r in results:
        key = r['source_file']
        if key not in seen:
            seen.add(key)
            lines.append(f"- {r['source_file']} | Subject: {r['subject']}")
    return '\n'.join(lines)


def delete_material(file_name):
    if not file_name or not file_name.strip():
        return "Please enter a file name to delete.", get_material_status()
    success = indexer.delete_material(file_name.strip())
    if success:
        return f"Deleted '{file_name}' from index.", get_material_status()
    return f"File '{file_name}' not found.", get_material_status()


# ---------------------------------------------------------------------------
# Query helpers — card-based HTML rendering
# ---------------------------------------------------------------------------

def _badge(label):
    css_class = f"badge badge-{label}"
    return f'<span class="{css_class}">{label}</span>'


def _classification_card(class_result):
    label = class_result['predicted_label']
    conf = class_result['confidence']
    top3 = class_result['top_3_labels']
    bar_pct = max(conf * 100, 2)
    alt_tags = ' '.join(
        f'<span style="font-size:0.75rem;color:#6b7280;margin-right:0.5rem;">{item["label"]} ({item["score"]:.0%})</span>'
        for item in top3
    )
    return f"""<div class="philo-card">
    <h3>Classification</h3>
    <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        {_badge(label)}
        <div style="flex:1;min-width:120px;">
            <div style="height:8px;background:#e5e7eb;border-radius:999px;overflow:hidden;">
                <div style="height:100%;width:{bar_pct}%;background:#6366f1;border-radius:999px;transition:width 0.4s;"></div>
            </div>
        </div>
        <span style="font-weight:600;font-size:0.9rem;">{conf:.1%}</span>
    </div>
    <div style="margin-top:0.4rem;">{alt_tags}</div>
</div>"""


def _response_card(response_text):
    lines = response_text.strip().split('\n')
    body = '<br>'.join(f'  {line}' for line in lines)
    return f"""<div class="philo-card">
    <h3>Response</h3>
    <div style="font-size:0.95rem;line-height:1.6;white-space:pre-wrap;">{response_text}</div>
</div>"""


def _passages_card(passages):
    if not passages:
        return '<div class="philo-card"><h3>Retrieved Passages</h3><p style="color:#9ca3af;">No passages retrieved.</p></div>'

    items = []
    for i, p in enumerate(passages):
        text = p['text'][:300]
        score = p['score']
        src = p['source']
        st = p['source_type']
        phil = src.get('philosopher', '')
        work = src.get('work', '')
        source_str = f"{phil} — <em>{work}</em>" if phil and work else phil or work or 'Unknown'

        type_tag = ''
        if st == 'teacher':
            type_tag = '<span class="badge badge-teacher" style="margin-left:0.5rem;">TEACHER</span>'
        elif st == 'hybrid':
            pass  # default style

        border_color = '#6366f1'
        bar_color = '#6366f1'
        if st == 'teacher':
            border_color = '#f97316'
            bar_color = '#f97316'

        pct = max(score * 100, 0)
        items.append(f"""<div class="passage-box" style="border-color:{border_color};">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="source-tag">{source_str}{type_tag}</span>
        <span class="passage-score" style="color:{border_color};">{pct:.0f}%</span>
    </div>
    <div style="height:4px;background:#e5e7eb;border-radius:999px;margin:0.4rem 0;overflow:hidden;">
        <div style="height:100%;width:{pct}%;background:{bar_color};border-radius:999px;"></div>
    </div>
    <div style="font-size:0.88rem;color:var(--body-text-color,#374151);margin-top:0.2rem;">{text}...</div>
</div>""")

    body = '\n'.join(items)
    return f"""<div class="philo-card">
    <h3>Retrieved Passages ({len(passages)})</h3>
    {body}
</div>"""


def _quiz_card(quiz_dict):
    if not quiz_dict or not isinstance(quiz_dict, dict):
        return ''
    q = quiz_dict.get('question', '')
    options = quiz_dict.get('options', [])
    opts_html = '<ol style="margin:0.5rem 0 0 1.2rem;">'
    for opt in options:
        opts_html += f'<li style="margin-bottom:0.25rem;">{opt}</li>'
    opts_html += '</ol>'
    return f"""<div class="philo-card">
    <h3>Quiz</h3>
    <div style="font-weight:600;margin-bottom:0.25rem;">{q}</div>
    {opts_html}
</div>"""


def run_query(question, top_k, use_teacher):
    if not question or not question.strip():
        return [None, None, None, None, None, None]

    global query_history
    query_history.append(question.strip())
    if len(query_history) > 20:
        query_history = query_history[-20:]
    history_str = '\n'.join(f"{i+1}. {q}" for i, q in enumerate(reversed(query_history)))

    try:
        pl = get_pipeline(use_teacher=use_teacher)
        output = pl.process(question.strip(), top_k=int(top_k) if top_k else 3)

        class_html = _classification_card(output.classification.to_dict())
        response_html = _response_card(output.response)
        passages_html = _passages_card([p.to_dict() for p in output.retrieval.passages])
        quiz_html = _quiz_card(output.quiz) if output.quiz else ''
        divider = '<hr style="border:none;border-top:1px solid var(--border-color-primary,#e5e7eb);margin:0.25rem 0;">'
        all_html = f'{class_html}{divider}{response_html}{divider}{passages_html}{divider}{quiz_html}'

        return all_html, history_str
    except Exception as e:
        import traceback
        err_html = f'<div class="philo-card" style="border-color:#ef4444;"><h3 style="color:#ef4444;">Error</h3><pre>{traceback.format_exc()}</pre></div>'
        return err_html, history_str


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="PhiloMind") as demo:
    gr.Markdown(
        """# PhiloMind
        *Philosophical Question Answering with Hybrid Retrieval*""",
    )

    with gr.Tabs():
        # ---- Query Tab ----
        with gr.TabItem("Query"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Philosophical question",
                        placeholder="e.g., What is Cartesian dualism?",
                        lines=3,
                    )
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Passages (top-k)"
                        )
                        use_teacher = gr.Checkbox(
                            label="Boost teacher materials", value=True
                        )
                    query_btn = gr.Button("Ask PhiloMind", variant="primary", size="lg")

                    history_output = gr.Textbox(
                        label="Recent Queries", lines=6, max_lines=8,
                        interactive=False,
                    )

                with gr.Column(scale=5):
                    results_placeholder = gr.HTML(
                        value='<div class="philo-card" style="text-align:center;padding:3rem;"><p style="color:#9ca3af;">Your results will appear here</p></div>',
                    )

            query_btn.click(
                fn=run_query,
                inputs=[query_input, top_k_slider, use_teacher],
                outputs=[results_placeholder, history_output],
            )

        # ---- Upload Tab ----
        with gr.TabItem("Upload Material"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Select file (.txt, .md)")
                    subject_input = gr.Textbox(
                        label="Subject (required)",
                        placeholder="e.g., Ethics, Epistemology, Metaphysics...",
                    )
                    upload_btn = gr.Button("Upload & Index", variant="primary")
                with gr.Column():
                    upload_output = gr.Textbox(label="Upload result", lines=3)
            with gr.Row():
                status_output = gr.Textbox(label="Index Status", lines=8)
            upload_btn.click(
                fn=upload_file,
                inputs=[file_input, subject_input],
                outputs=[upload_output, gr.State(), status_output],
            )

        # ---- Materials Tab ----
        with gr.TabItem("Manage Materials"):
            with gr.Row():
                with gr.Column():
                    refresh_btn = gr.Button("Refresh Status", variant="secondary")
                    search_input = gr.Textbox(
                        label="Search materials",
                        placeholder="Search by keyword or subject...",
                    )
                    search_btn = gr.Button("Search", variant="secondary")
                with gr.Column():
                    delete_input = gr.Textbox(
                        label="Delete file",
                        placeholder="Enter file name to delete...",
                    )
                    delete_btn = gr.Button("Delete", variant="stop")
            materials_output = gr.Textbox(label="Material Status", lines=15)
            refresh_btn.click(fn=get_material_status, inputs=[], outputs=[materials_output])
            search_btn.click(fn=search_materials, inputs=[search_input], outputs=[materials_output])
            delete_btn.click(
                fn=delete_material,
                inputs=[delete_input],
                outputs=[gr.Textbox(), materials_output],
            )
            demo.load(fn=get_material_status, inputs=[], outputs=[materials_output])

        # ---- Help Tab ----
        with gr.TabItem("Help & Guide"):
            gr.Markdown("""
            ## How to Use PhiloMind

            ### 1. Upload Materials
            Upload `.txt` or `.md` files with philosophical content, tag them by subject.

            ### 2. Query the System
            Type a philosophical question. The pipeline:
            1. **Classifies** the question type (definition, comparison, example, deepening, quiz)
            2. **Retrieves** relevant passages using a hybrid of TF-IDF + dense embeddings (MiniLM) + cross-encoder reranking
            3. **Generates** a response citing the retrieved sources
            4. **Creates** a quiz question to test understanding

            ### 3. Understanding Results
            - **Classification** — question type with confidence score
            - **Response** — AI-generated answer grounded in the retrieved passages
            - **Retrieved Passages** — source texts with relevance scores
            - **Quiz** — multiple-choice question for self-assessment

            ### Example Questions
            - "What is Plato's theory of Forms?"
            - "How do Aristotle and Plato differ?"
            - "What is the categorical imperative?"
            - "Explain Nietzsche's concept of the Übermensch"
            - "What does Heidegger mean by Dasein?"
            - "Compare Kant's categorical imperative with Mill's utilitarianism"
            """)

    gr.Markdown("---")
    gr.Markdown("PhiloMind v2.0 | Hybrid Retrieval: TF-IDF + MiniLM + Cross-Encoder | Classification: BiLSTM / DistilBERT")


if __name__ == '__main__':
    demo.launch(
        server_name="127.0.0.1", server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
