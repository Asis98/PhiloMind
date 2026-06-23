@echo off
REM PhiloMind - Quick Start Script
echo ========================================
echo   PhiloMind - Philosophy Teaching Assistant
echo   Quick Start
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    pause
    exit /b 1
)
echo [OK] Python found

REM Install dependencies
echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Some dependencies may have failed
) else (
    echo [OK] Dependencies installed
)

REM Check if model exists
echo.
echo [2/3] Checking models...
if exist models\bilstm\final.pt (
    echo [OK] BiLSTM model found
) else (
    echo [WARN] BiLSTM model not found. Train with: python run.py train
)
if exist models\retrieval\tfidf.pkl (
    echo [OK] TF-IDF retriever found
) else (
    echo [WARN] TF-IDF retriever not found. Run: python -m src.retrieval.tfidf
)

REM Start frontend
echo.
echo [3/3] Starting PhiloMind frontend...
echo.
echo   Open your browser to: http://127.0.0.1:7860
echo.
echo   Press Ctrl+C to stop
echo ========================================
python run.py frontend
