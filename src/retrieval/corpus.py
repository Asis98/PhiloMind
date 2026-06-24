"""Corpus preparation with semantic-aware chunking."""

import pandas as pd
from pathlib import Path
import re


PHILOSOPHER_NAMES = [
    "Plato", "Aristotle", "Socrates", "Descartes", "Kant", "Nietzsche",
    "Hegel", "Marx", "Hume", "Locke", "Rousseau", "Spinoza", "Leibniz",
    "Schopenhauer", "Heidegger", "Wittgenstein", "Husserl", "Sartre",
    "Foucault", "Arendt", "Kierkegaard", "Russell", "Rawls", "Deleuze",
    "Derrida", "Habermas", "Adorno", "Bacon", "Hobbes", "Berkeley",
    "Epicurus", "Zeno", "Plotinus", "Proclus", "Augustine", "Aquinas"
]


class CorpusPreparer:
    """Prepares and chunks the philosophical corpus with semantic awareness."""

    @staticmethod
    def load_kaggle_corpus(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False)
        if 'sentence_str' in df.columns:
            df['text'] = df['sentence_str']
        elif 'sentence' in df.columns:
            df['text'] = df['sentence']
        elif 'text' not in df.columns:
            text_cols = [c for c in df.columns if df[c].dtype == 'object']
            if text_cols:
                df['text'] = df[text_cols[0]]
        if 'author' in df.columns:
            df['philosopher'] = df['author']
        elif 'philosopher' not in df.columns:
            df['philosopher'] = 'Unknown'
        if 'title' in df.columns and 'work' not in df.columns:
            df['work'] = df['title']
        elif 'work' not in df.columns:
            df['work'] = 'Unknown'
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 20]
        return df.reset_index(drop=True)

    @staticmethod
    def create_chunks(df: pd.DataFrame, chunk_size: int = 120, min_chunk: int = 30) -> pd.DataFrame:
        chunks = []
        for idx, row in df.iterrows():
            text = row['text']
            philosopher = row.get('philosopher', 'Unknown')
            work = row.get('work', 'Unknown')
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_length = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                word_count = len(sentence.split())
                if current_length + word_count > chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) >= min_chunk:
                        chunks.append({
                            'text': chunk_text,
                            'philosopher': philosopher,
                            'work': work,
                            'original_idx': idx
                        })
                    current_chunk = [sentence]
                    current_length = word_count
                else:
                    current_chunk.append(sentence)
                    current_length += word_count
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= min_chunk:
                    chunks.append({
                        'text': chunk_text,
                        'philosopher': philosopher,
                        'work': work,
                        'original_idx': idx
                    })
        result = pd.DataFrame(chunks)
        result = result[result['text'].str.len() > 80]
        result = result.reset_index(drop=True)
        return result

    @staticmethod
    def create_augmented_corpus(raw_df: pd.DataFrame, base_chunks: pd.DataFrame) -> pd.DataFrame:
        extra_rows = []
        for _, row in raw_df.iterrows():
            text = str(row.get('text', ''))
            philosopher = str(row.get('philosopher', 'Unknown'))
            work = str(row.get('work', 'Unknown'))
            for name in PHILOSOPHER_NAMES:
                if name.lower() in text.lower():
                    if any(p.lower() == name.lower() for p in [philosopher]):
                        continue
                    sentence_count = len(re.split(r'(?<=[.!?])\s+', text))
                    if sentence_count <= 3 and len(text.split()) > 20:
                        extra_rows.append({
                            'text': text,
                            'philosopher': name,
                            'work': work,
                            'original_idx': row.name
                        })
        if extra_rows:
            extra_df = pd.DataFrame(extra_rows)
            combined = pd.concat([base_chunks, extra_df], ignore_index=True)
            return combined.drop_duplicates(subset=['text']).reset_index(drop=True)
        return base_chunks

    @staticmethod
    def build_corpus_index(raw_path: str, output_dir: Path, sample_size: int = 5000):
        print(f"Loading corpus from {raw_path}...")
        raw = CorpusPreparer.load_kaggle_corpus(raw_path)
        if len(raw) > sample_size:
            print(f"Sampling {sample_size} from {len(raw)} documents...")
            raw = raw.sample(n=sample_size, random_state=42)
        print(f"Creating chunks ({len(raw)} documents)...")
        chunks = CorpusPreparer.create_chunks(raw, chunk_size=150, min_chunk=50)
        print(f"Augmenting with philosopher mentions...")
        chunks = CorpusPreparer.create_augmented_corpus(raw, chunks)
        import numpy as np
        noise_keywords = ['defense', 'military', 'weapon', 'chemical', 'biological']
        noise_mask = chunks['text'].str.lower().str.contains('|'.join(noise_keywords), na=False)
        chunks = chunks[~noise_mask].reset_index(drop=True)
        print(f"Final corpus: {len(chunks)} chunks")
        chunks.to_csv(output_dir / 'corpus_chunks.csv', index=False)
        return chunks
