"""FastAPI-based REST API for PhiloMind."""

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.pipeline.core import PhiloMindPipeline
from src.pipeline.dataclasses import PipelineOutput
from src.pipeline.format import format_output
from src.ingestion.indexer import MaterialIndexer

app = FastAPI(title="PhiloMind API", version="2.0.0")
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
    quiz: dict


class MaterialResponse(BaseModel):
    status: str
    file: str
    subject: str
    chunks: int


class MaterialsListResponse(BaseModel):
    materials: list
    count: int


def get_pipeline():
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


@app.on_event("startup")
async def startup():
    get_pipeline()
    try:
        indexer.load_retriever()
    except Exception:
        pass


@app.get("/")
async def root():
    return {"app": "PhiloMind API", "version": "2.0.0", "status": "running"}


@app.post("/materials/upload", response_model=MaterialResponse)
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


@app.get("/materials", response_model=MaterialsListResponse)
async def list_materials():
    materials = indexer.list_materials()
    return {"materials": materials, "count": len(materials)}


@app.get("/materials/status")
async def material_status():
    status = indexer.get_material_status()
    return status


@app.delete("/materials/{file_name}")
async def delete_material(file_name: str):
    success = indexer.delete_material(file_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found")
    return {"status": "deleted", "file": file_name}


@app.post("/query", response_model=PipelineResponse)
async def query_pipeline(req: QueryRequest) -> PipelineResponse:
    pl = get_pipeline()
    output = pl.process(req.question, top_k=req.top_k)
    return PipelineResponse(
        question=output.question,
        classification=output.classification.to_dict(),
        passages=[p.to_dict() for p in output.retrieval.passages],
        response=output.response,
        quiz=output.quiz
    )


@app.get("/query/{question:path}")
async def query_get(question: str, top_k: int = 3):
    pl = get_pipeline()
    output = pl.process(question, top_k=top_k)
    return output.to_dict()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
