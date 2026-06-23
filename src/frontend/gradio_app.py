"""Gradio-based frontend for PhiloMind."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import gradio as gr
from src.pipeline.core import PhiloMindPipeline
from src.pipeline.format import format_output
from src.ingestion.indexer import MaterialIndexer

pipeline = None
indexer = MaterialIndexer()


def get_pipeline():
    global pipeline
    if pipeline is None:
        base = Path(__file__).resolve().parent.parent.parent
        pipeline = PhiloMindPipeline(
            retriever_path=str(base / 'models' / 'retrieval' / 'tfidf.pkl'),
            config_path=str(base / 'config' / 'disciplines.json')
        )
    return pipeline


def upload_file(file, subject):
    if file is None:
        return "No file selected.", None
    if not subject or not subject.strip():
        return "Subject field is required.", None
    ext = Path(file.name).suffix.lower()
    if ext not in ('.txt', '.md'):
        return f"Unsupported format: {ext}. Use .txt or .md.", None
    dest = Path(__file__).resolve().parent.parent.parent / 'data' / 'materials' / Path(file.name).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(file.name, 'rb') as src, open(dest, 'wb') as dst:
        dst.write(src.read())
    try:
        indexer.load_retriever()
    except FileNotFoundError:
        pass
    result = indexer.index_material(str(dest), subject.strip())
    return f"Uploaded and indexed ({result['chunks']} chunks for '{subject}').", result


def list_uploaded():
    materials = indexer.list_materials()
    if not materials:
        return "No materials uploaded."
    return '\n'.join([f"- {m['source_file']} | Subject: {m['subject']} | Chunks: {m['chunks']}" for m in materials])


def run_query(question, top_k):
    if not question or not question.strip():
        return "Please enter a question."
    pl = get_pipeline()
    output = pl.process(question.strip(), top_k=int(top_k) if top_k else 3)
    return format_output(output)


with gr.Blocks(title="PhiloMind - Teacher Frontend") as demo:
    gr.Markdown("# PhiloMind - Teacher Frontend")
    gr.Markdown("Upload teaching materials and query the philosophical system.")

    with gr.Tab("Upload Material"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Select file (.txt, .md)")
                subject_input = gr.Textbox(label="Subject (required)", placeholder="e.g., Philosophy, Ethics, History...")
                upload_btn = gr.Button("Upload & Index", variant="primary")
            with gr.Column():
                upload_output = gr.Textbox(label="Upload result", lines=4)
        upload_btn.click(fn=upload_file, inputs=[file_input, subject_input], outputs=[upload_output])

    with gr.Tab("Uploaded Materials"):
        refresh_btn = gr.Button("Refresh List")
        materials_output = gr.Textbox(label="Indexed materials", lines=15)
        refresh_btn.click(fn=list_uploaded, inputs=[], outputs=[materials_output])
        demo.load(fn=list_uploaded, inputs=[], outputs=[materials_output])

    with gr.Tab("Query Pipeline"):
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Philosophical question", placeholder="e.g., What is Cartesian dualism?", lines=3)
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Passages to retrieve (top-k)")
                query_btn = gr.Button("Submit Query", variant="primary")
            with gr.Column():
                query_output = gr.Textbox(label="Pipeline Result", lines=25)
        query_btn.click(fn=run_query, inputs=[query_input, top_k_slider], outputs=[query_output])

    gr.Markdown("---")
    gr.Markdown("PhiloMind v1.0 | University Project")


if __name__ == '__main__':
    demo.launch(server_name="127.0.0.1", server_port=7860)
