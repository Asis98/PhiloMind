"""Indexes teacher materials into the TF-IDF retriever."""

import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.retrieval.tfidf import TFIDFRetriever
from .parser import TextParser, Chunker


class MaterialIndexer:
    """Indexes uploaded materials with subject metadata."""

    def __init__(self, retriever_path: str = None, materials_dir: str = None):
        base = Path(__file__).resolve().parent.parent.parent
        self.retriever_path = retriever_path or str(base / 'models' / 'retrieval' / 'tfidf.pkl')
        self.materials_dir = Path(materials_dir or base / 'data' / 'materials')
        self.materials_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.materials_dir / 'index.csv'
        self.retriever = None

    def load_retriever(self):
        self.retriever = TFIDFRetriever()
        if Path(self.retriever_path).exists():
            self.retriever.load(self.retriever_path)

    def index_material(self, file_path: str, subject: str) -> Dict:
        text = TextParser.parse(file_path)
        chunks = Chunker.chunk_text(text)
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                'text': chunk, 'subject': subject,
                'source_file': Path(file_path).name, 'chunk_id': i
            })
        df = pd.DataFrame(records)
        existing = pd.DataFrame()
        if self.index_path.exists():
            existing = pd.read_csv(self.index_path)
        pd.concat([existing, df], ignore_index=True).to_csv(self.index_path, index=False)
        self.retriever.corpus_df = pd.concat([self.retriever.corpus_df, df], ignore_index=True)
        self.retriever.fit(self.retriever.corpus_df['text'].values)
        self.retriever.save(self.retriever_path)
        return {'file': file_path, 'subject': subject, 'chunks': len(chunks)}

    def list_materials(self) -> List[Dict]:
        if not self.index_path.exists():
            return []
        df = pd.read_csv(self.index_path)
        summary = df.groupby(['source_file', 'subject']).agg(
            chunks=('chunk_id', 'count')
        ).reset_index()
        return summary.to_dict('records')
