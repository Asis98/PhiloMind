"""Indexes teacher materials into the TF-IDF retriever with status tracking."""

import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from src.retrieval.tfidf import TFIDFRetriever
from .parser import TextParser, Chunker


class MaterialIndexer:
    """Indexes uploaded materials with subject metadata and status tracking."""

    def __init__(self, retriever_path: str = None, materials_dir: str = None):
        base = Path(__file__).resolve().parent.parent.parent
        self.retriever_path = retriever_path or str(base / 'models' / 'retrieval' / 'tfidf.pkl')
        self.teacher_retriever_path = str(base / 'models' / 'retrieval' / 'teacher_tfidf.pkl')
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
                'source_file': Path(file_path).name, 'chunk_id': i,
                'indexed_at': datetime.now().isoformat()
            })
        df = pd.DataFrame(records)
        existing = pd.DataFrame()
        if self.index_path.exists():
            existing = pd.read_csv(self.index_path)
        pd.concat([existing, df], ignore_index=True).to_csv(self.index_path, index=False)
        teacher_retriever = TFIDFRetriever()
        all_data = pd.read_csv(self.index_path)
        teacher_retriever.corpus_df = all_data
        teacher_retriever.fit(all_data['text'].values)
        Path(self.teacher_retriever_path).parent.mkdir(parents=True, exist_ok=True)
        teacher_retriever.save(self.teacher_retriever_path)
        if self.retriever and Path(self.retriever_path).exists():
            self.retriever.load(self.retriever_path)
            self.retriever.corpus_df = pd.concat([self.retriever.corpus_df, df], ignore_index=True)
            self.retriever.fit(self.retriever.corpus_df['text'].values)
            self.retriever.save(self.retriever_path)
        return {'file': Path(file_path).name, 'subject': subject, 'chunks': len(chunks)}

    def list_materials(self) -> List[Dict]:
        if not self.index_path.exists():
            return []
        df = pd.read_csv(self.index_path)
        summary = df.groupby(['source_file', 'subject']).agg(
            chunks=('chunk_id', 'count'),
            last_indexed=('indexed_at', 'max')
        ).reset_index()
        return summary.to_dict('records')

    def search_materials(self, query: str) -> List[Dict]:
        if not self.index_path.exists():
            return []
        df = pd.read_csv(self.index_path)
        q = query.lower()
        mask = df['text'].str.lower().str.contains(q, na=False) | df['subject'].str.lower().str.contains(q, na=False)
        results = df[mask].head(20)
        return results.to_dict('records')

    def get_material_status(self) -> Dict:
        if not self.index_path.exists():
            return {'total_files': 0, 'total_chunks': 0, 'subjects': [], 'files': []}
        df = pd.read_csv(self.index_path)
        subjects = df['subject'].unique().tolist()
        files = df.groupby('source_file').agg(
            chunks=('chunk_id', 'count'),
            subject=('subject', 'first'),
            last_indexed=('indexed_at', 'max')
        ).reset_index().to_dict('records')
        return {
            'total_files': df['source_file'].nunique(),
            'total_chunks': len(df),
            'subjects': subjects,
            'files': files
        }

    def delete_material(self, source_file: str) -> bool:
        if not self.index_path.exists():
            return False
        df = pd.read_csv(self.index_path)
        df = df[df['source_file'] != source_file]
        df.to_csv(self.index_path, index=False)
        teacher_retriever = TFIDFRetriever()
        all_data = pd.read_csv(self.index_path)
        if len(all_data) > 0:
            teacher_retriever.corpus_df = all_data
            teacher_retriever.fit(all_data['text'].values)
            teacher_retriever.save(self.teacher_retriever_path)
        else:
            Path(self.teacher_retriever_path).unlink(missing_ok=True)
        return True
