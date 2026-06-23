"""Corpus preparation utilities for the philosophical text corpus."""

import pandas as pd
from pathlib import Path


class CorpusPreparer:
    """Prepares and chunks the philosophical corpus."""

    @staticmethod
    def load_kaggle_corpus(csv_path: str) -> pd.DataFrame:
        """Load and normalize the Kaggle philosophy corpus."""
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
        df = df[df['text'].str.len() > 10]
        return df.reset_index(drop=True)

    @staticmethod
    def create_chunks(df: pd.DataFrame, chunk_size: int = 200) -> pd.DataFrame:
        """Split texts into semantic chunks of approximately chunk_size words."""
        chunks = []
        for idx, row in df.iterrows():
            sentences = row['text'].split('.')
            current_chunk, current_length = [], 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                word_count = len(sentence.split())
                if current_length + word_count > chunk_size and current_chunk:
                    chunks.append({
                        'text': '. '.join(current_chunk) + '.',
                        'philosopher': row.get('philosopher', 'Unknown'),
                        'work': row.get('work', 'Unknown'),
                        'original_idx': idx
                    })
                    current_chunk, current_length = [sentence], word_count
                else:
                    current_chunk.append(sentence)
                    current_length += word_count
            if current_chunk:
                chunks.append({
                    'text': '. '.join(current_chunk) + '.',
                    'philosopher': row.get('philosopher', 'Unknown'),
                    'work': row.get('work', 'Unknown'),
                    'original_idx': idx
                })
        return pd.DataFrame(chunks)
