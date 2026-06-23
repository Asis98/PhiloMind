#!/bin/bash
# PhiloMind - Quick Start Script
echo "========================================"
echo "  PhiloMind - Philosophy Teaching Assistant"
echo "  Quick Start"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found."
    exit 1
fi
echo "[OK] Python found"

# Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip3 install -r requirements.txt

# Check models
echo ""
echo "[2/3] Checking models..."
[ -f models/bilstm/final.pt ] && echo "[OK] BiLSTM model found" || echo "[WARN] BiLSTM model not found"
[ -f models/retrieval/tfidf.pkl ] && echo "[OK] TF-IDF retriever found" || echo "[WARN] TF-IDF retriever not found"

# Start frontend
echo ""
echo "[3/3] Starting PhiloMind frontend..."
echo ""
echo "  Open your browser to: http://127.0.0.1:7860"
echo ""
echo "  Press Ctrl+C to stop"
echo "========================================"
python3 run.py frontend
