"""Run PhiloMind application components."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="PhiloMind - Run application")
    parser.add_argument("mode", nargs="?", default="help",
                        choices=["api", "frontend", "gradio", "train", "test", "help"],
                        help="Component to run")
    args = parser.parse_args()

    if args.mode == "help":
        print("Usage: python run.py [mode]")
        print("  api       Start FastAPI server on :8000")
        print("  frontend  Start Gradio UI on :7860")
        print("  gradio    Same as frontend")
        print("  train     Train BiLSTM classifier")
        print("  test      Test pipeline with sample question")
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


if __name__ == "__main__":
    main()
