"""Run PhiloMind application components."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="PhiloMind - Run application")
    parser.add_argument("mode", nargs="?", default="help",
                        choices=["api", "frontend", "gradio", "train", "test", "test-retrieval", "help"],
                        help="Component to run")
    args = parser.parse_args()

    if args.mode == "help":
        print("Usage: python run.py [mode]")
        print("  api             Start FastAPI server on :8000")
        print("  frontend        Start Gradio UI on :7860")
        print("  gradio          Same as frontend")
        print("  train           Train BiLSTM classifier")
        print("  test            Test pipeline with sample question")
        print("  test-retrieval  Compare TF-IDF vs Dense vs Hybrid")
        return

    python = sys.executable

    if args.mode == "api":
        subprocess.run([python, "-m", "uvicorn", "src.api.main:app",
                       "--host", "0.0.0.0", "--port", "8000", "--reload"])
    elif args.mode in ("frontend", "gradio"):
        subprocess.run([python, "-m", "src.frontend.gradio_app"])
    elif args.mode == "train":
        subprocess.run([python, "-m", "src.classification.bilstm"])
    elif args.mode == "test":
        subprocess.run([python, "-c", """
from src.pipeline.core import PhiloMindPipeline
from src.pipeline.format import format_output
pl = PhiloMindPipeline()
print(format_output(pl.process("What is Cartesian dualism?")))
"""])
    elif args.mode == "test-retrieval":
        subprocess.run([python, "-c", """
import pandas as pd
from pathlib import Path
from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.embeddings import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

df = pd.read_csv("models/retrieval/corpus_chunks.csv")
tfidf = TFIDFRetriever()
tfidf.load("models/retrieval/tfidf.pkl")

dense = DenseRetriever()
dense.load("models/retrieval/dense_embeddings.npy")
dense.corpus_df = df

hybrid = HybridRetriever(tfidf_retriever=tfidf, dense_retriever=dense, alpha=0.3)

for q in [
    "What is Cartesian dualism?",
    "Compare Aristotle and Plato",
    "What is the categorical imperative?"
]:
    print("\\n" + "=" * 60)
    print(f"  Query: {q}")
    print("=" * 60)
    print("  TF-IDF:", [(hybrid.tfidf.get_source(i).get('philosopher'), '') for i, t, s in tfidf.retrieve(q, top_k=2)])
    print("  Dense: ", [(hybrid.tfidf.get_source(i).get('philosopher'), '') for i, t, s in dense.retrieve(q, top_k=2)])
    print("  Hybrid:", [(hybrid.get_source(i).get('philosopher'), f"score:{s:.1%}") for i, t, s, st in hybrid.retrieve(q, top_k=2)])
"""]),


if __name__ == "__main__":
    main()
