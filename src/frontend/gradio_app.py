"""Gradio-based frontend for PhiloMind with query history, search, and status tracking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import gradio as gr
from src.pipeline.core import PhiloMindPipeline
from src.pipeline.format import format_output
from src.ingestion.indexer import MaterialIndexer

pipeline = None
indexer = MaterialIndexer()
query_history = []


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


def list_uploaded():
    materials = indexer.list_materials()
    if not materials:
        return "No materials uploaded."
    return '\n'.join([f"- {m['source_file']} | Subject: {m['subject']} | Chunks: {m['chunks']}" for m in materials])


def run_query(question, top_k, use_teacher):
    if not question or not question.strip():
        return "Please enter a question.", None
    global query_history
    query_history.append(question.strip())
    if len(query_history) > 20:
        query_history = query_history[-20:]
    try:
        pl = get_pipeline(use_teacher=use_teacher)
        output = pl.process(question.strip(), top_k=int(top_k) if top_k else 3)
        result = format_output(output)
        return result, '\n'.join(f"{i+1}. {q}" for i, q in enumerate(reversed(query_history)))
    except Exception as e:
        return f"Error: {str(e)}", '\n'.join(f"{i+1}. {q}" for i, q in enumerate(reversed(query_history)))


with gr.Blocks(title="PhiloMind - Teacher Frontend") as demo:
    gr.Markdown("# PhiloMind - Philosophy Teaching Assistant")
    gr.Markdown("Upload teaching materials and query the philosophical system. "
                "Teacher materials are prioritized in search results.")

    with gr.Tab("Upload Material"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Select file (.txt, .md)")
                subject_input = gr.Textbox(label="Subject (required)",
                                           placeholder="e.g., Ethics, Epistemology, Metaphysics...")
                upload_btn = gr.Button("Upload & Index", variant="primary")
            with gr.Column():
                upload_output = gr.Textbox(label="Upload result", lines=3)
        with gr.Row():
            status_output = gr.Textbox(label="Index Status", lines=8)
        upload_btn.click(fn=upload_file, inputs=[file_input, subject_input],
                         outputs=[upload_output, gr.State(), status_output])

    with gr.Tab("Manage Materials"):
        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("Refresh Status")
                search_input = gr.Textbox(label="Search materials", placeholder="Search by keyword or subject...")
                search_btn = gr.Button("Search", variant="secondary")
            with gr.Column():
                delete_input = gr.Textbox(label="Delete file", placeholder="Enter file name to delete...")
                delete_btn = gr.Button("Delete", variant="stop")
        materials_output = gr.Textbox(label="Material Status", lines=15)
        refresh_btn.click(fn=get_material_status, inputs=[], outputs=[materials_output])
        search_btn.click(fn=search_materials, inputs=[search_input], outputs=[materials_output])
        delete_btn.click(fn=delete_material, inputs=[delete_input],
                         outputs=[gr.Textbox(), materials_output])
        demo.load(fn=get_material_status, inputs=[], outputs=[materials_output])

    with gr.Tab("Query Pipeline"):
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Philosophical question",
                                         placeholder="e.g., What is Cartesian dualism?", lines=3)
                with gr.Row():
                    top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1,
                                             label="Passages to retrieve (top-k)")
                    use_teacher = gr.Checkbox(label="Boost teacher materials", value=True)
                query_btn = gr.Button("Submit Query", variant="primary")
                gr.Markdown("### Recent Queries")
                history_output = gr.Textbox(label="Query History", lines=6, interactive=False)
            with gr.Column():
                query_output = gr.Textbox(label="Pipeline Result", lines=25)
        query_btn.click(fn=run_query,
                        inputs=[query_input, top_k_slider, use_teacher],
                        outputs=[query_output, history_output])

    with gr.Tab("Help & Guide"):
        gr.Markdown("""
        ## How to Use PhiloMind

        ### 1. Upload Materials
        - Go to **Upload Material** tab
        - Select a `.txt` or `.md` file with philosophical content
        - Enter a subject (e.g., Ethics, Metaphysics, Epistemology)
        - Click **Upload & Index** - the system will chunk and index your material

        ### 2. Manage Materials
        - View indexing status: total files, chunks, and subjects
        - Search by keyword to find specific materials
        - Delete files that are no longer needed

        ### 3. Query the System
        - Type a philosophical question in English
        - Adjust how many passages to retrieve (top-k)
        - Enable **Boost teacher materials** to prioritize your uploaded content
        - View query history for quick reference

        ### 4. Understanding Results
        - **Classification**: What type of question (definition, comparison, example, etc.)
        - **Retrieved Passages**: Relevant texts from the corpus + your materials
        - **Response**: AI-generated philosophical answer with citations
        - **Quiz**: Multiple-choice question to test understanding

        ### Example Questions
        - "What is Plato's theory of Forms?"
        - "How do Aristotle and Plato differ?"
        - "What is the categorical imperative?"
        - "Explain Nietzsche's concept of the Übermensch"
        """)

    gr.Markdown("---")
    gr.Markdown("PhiloMind v2.0 | University Project | Week 4 Complete")


if __name__ == '__main__':
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
