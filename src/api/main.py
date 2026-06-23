"""FastAPI-based REST API for PhiloMind."""

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.pipeline.core import PhiloMindPipeline
from src.pipeline.dataclasses import PipelineOutput
from src.pipeline.format import format_output
from src.ingestion.indexer import MaterialIndexer

app = FastAPI(title="PhiloMind API", version="1.0.0")
pipeline = None
indexer = MaterialIndexer()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class PipelineResponse(BaseModel):
    question: str
    classification: dict
    passages: List[dict]
    response: str
    quiz: str


def get_pipeline():
    global pipeline
    if pipeline is None:
        base = Path(__file__).resolve().parent.parent.parent
        pipeline = PhiloMindPipeline(
            retriever_path=str(base / 'models' / 'retrieval' / 'tfidf.pkl'),
            config_path=str(base / 'config' / 'disciplines.json')
        )
    return pipeline


@app.on_event("startup")
async def startup():
    get_pipeline()
    try:
        indexer.load_retriever()
    except Exception:
        pass


@app.get("/")
async def root():
    return {"app": "PhiloMind API", "version": "1.0.0", "status": "running"}


@app.post("/materials/upload")
async def upload_material(file: UploadFile = File(...), subject: str = Form(...)):
    if not subject.strip():
        raise HTTPException(status_code=400, detail="The 'subject' field is required")
    ext = Path(file.filename).suffix.lower()
    if ext not in ('.txt', '.md'):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Use .txt or .md")
    materials_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'materials'
    materials_dir.mkdir(parents=True, exist_ok=True)
    file_path = materials_dir / file.filename
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)
    try:
        indexer.load_retriever()
    except FileNotFoundError:
        pass
    result = indexer.index_material(str(file_path), subject.strip())
    return {"status": "ok", "file": file.filename, "subject": subject, "chunks": result['chunks']}


@app.get("/materials")
async def list_materials():
    materials = indexer.list_materials()
    return {"materials": materials, "count": len(materials)}


@app.post("/query")
async def query_pipeline(req: QueryRequest) -> PipelineResponse:
    pl = get_pipeline()
    output = pl.process(req.question, top_k=req.top_k)
    return PipelineResponse(
        question=output.question,
        classification={
            "predicted_label": output.classification.predicted_label,
            "confidence": output.classification.confidence,
            "top_3": [{"label": l, "score": c} for l, c in output.classification.top_3_labels]
        },
        passages=[
            {"text": p.text, "source": p.source, "score": p.score}
            for p in output.retrieval.passages
        ],
        response=output.response,
        quiz=output.quiz
    )


@app.get("/query/{question:path}")
async def query_get(question: str, top_k: int = 3):
    pl = get_pipeline()
    output = pl.process(question, top_k=top_k)
    return format_output(output)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
