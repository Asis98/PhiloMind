"""PhiloMind setup script."""
import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("  PhiloMind Setup")
    print("=" * 60)
    print(f"\n  Python: {sys.version}")

    print("\n  Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=False
    )

    print("\n  Creating directories...")
    for d in ["data/materials", "models/bilstm", "models/retrieval", "models/distilbert", "tests"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("\n  Setup complete.")
    print("\n  Next steps:")
    print("    Train BiLSTM:      python -m src.classification.bilstm")
    print("    Build retriever:   python -m src.retrieval.tfidf")
    print("    Start API:         python -m src.api.main")
    print("    Start frontend:    python -m src.frontend.gradio_app")


if __name__ == "__main__":
    main()
