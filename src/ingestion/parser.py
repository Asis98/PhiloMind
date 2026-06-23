"""Text and markdown parser for teacher materials."""

import re
from pathlib import Path
from typing import List


class TextParser:
    """Parses .txt and .md files into plain text."""

    @staticmethod
    def parse_txt(path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    @staticmethod
    def parse_md(path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'>\s*', '', text, flags=re.MULTILINE)
        return text

    @staticmethod
    def parse(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == '.md':
            return TextParser.parse_md(path)
        elif ext == '.txt':
            return TextParser.parse_txt(path)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use .txt or .md.")


class Chunker:
    """Splits text into semantic chunks for indexing."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks, current_chunk, current_length = [], [], 0
        for sentence in sentences:
            word_count = len(sentence.split())
            if current_length + word_count > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_words, overlap_count = [], 0
                for s in reversed(current_chunk):
                    swc = len(s.split())
                    if overlap_count + swc <= overlap:
                        overlap_words.insert(0, s)
                        overlap_count += swc
                    else:
                        break
                current_chunk, current_length = overlap_words, overlap_count
            current_chunk.append(sentence)
            current_length += word_count
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
