#!/usr/bin/env python3
"""
PhiloMind Setup Script - Settimana 2
Installa dipendenze e prepara l'ambiente.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Esegui un comando e mostra il risultato."""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode != 0:
        print(f"⚠️  Comando fallito: {cmd}")
        return False

    return True

def setup_environment():
    """Setup completo per Settimana 2."""

    print("""
╔═══════════════════════════════════════════════════════════╗
║                   PhiloMind Setup                         ║
║                   Settimana 2 - Setup                     ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Step 1: Check Python version
    print(f"🐍 Python version: {sys.version}")

    # Step 2: Install dependencies
    print("\n" + "="*60)
    print("STEP 1: Installing Dependencies")
    print("="*60)

    # Core dependencies
    core_deps = [
        "torch",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
    ]

    for dep in core_deps:
        print(f"\n📦 Installing {dep}...")
        cmd = f"{sys.executable} -m pip install {dep} -q"
        subprocess.run(cmd, shell=True)

    # Step 3: Create directories
    print("\n" + "="*60)
    print("STEP 2: Creating directories")
    print("="*60)

    dirs = [
        'data/raw',
        'data/labels',
        'data/scripts',
        'models',
        'notebooks',
        'agents',
        'app',
        'reports'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}/")

    # Step 4: Verify corpus
    print("\n" + "="*60)
    print("STEP 3: Checking dataset files")
    print("="*60)

    files_to_check = {
        'data/raw/philosophy_data.csv': 'Corpus filosofico',
        'data/labels/questions_labeled.csv': 'Domande etichettate',
        'disciplines/config.json': 'Configurazione discipline'
    }

    for file_path, desc in files_to_check.items():
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024*1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB) - {desc}")
        else:
            print(f"⚠️  {file_path} non trovato - {desc}")

    # Step 5: Run data augmentation
    print("\n" + "="*60)
    print("STEP 4: Data Augmentation")
    print("="*60)

    if Path('data/labels/questions_labeled.csv').exists():
        print("▶️  Esegui: python data/scripts/data_augmentation.py")
        print("\n🔄 Questo genererà:")
        print("   - data/labels/questions_augmented.csv (424 domande)")
        print("   - data/labels/questions_train.csv (337 domande)")
        print("   - data/labels/questions_test.csv (87 domande)")

    # Step 6: Summary
    print("\n" + "="*60)
    print("✅ SETUP COMPLETATO")
    print("="*60)

    print("""
Prossimi step:

1️⃣  Data Preparation:
    python data/scripts/data_augmentation.py

2️⃣  Train BiLSTM Baseline:
    python models/bilstm_classifier.py
    (Questo allenerà Agent 1)

3️⃣  Build TF-IDF Retriever:
    python models/tfidf_retriever.py
    (Questo allenerà Agent 2)

4️⃣  Test Pipeline:
    python pipeline.py
    (Questo testa l'intera pipeline Agent 1+2+3+4)

5️⃣  Evaluation (Notebook):
    jupyter notebook notebooks/WEEK2_EVALUATION.md

Per maggiori dettagli, vedi:
  - ROADMAP_WEEK2.md (Timeline + Deliverable)
  - notebooks/WEEK2_EVALUATION.md (Codice di valutazione)
  - requirements.txt (Tutte le dipendenze)
""")

if __name__ == '__main__':
    setup_environment()

