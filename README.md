# PhiloMind

**Philosophical Question Answering System** — A modular NLP pipeline for classifying philosophical questions, retrieving relevant passages, and generating responses.

## Architecture

```
src/
├── classification/     # Question classifiers (BiLSTM, DistilBERT)
│   ├── base.py         # Abstract classifier interface
│   ├── bilstm.py       # BiLSTM model + training
│   └── distilbert.py   # DistilBERT fine-tuning
├── retrieval/          # Passage retrieval
│   ├── tfidf.py        # TF-IDF retriever
│   └── corpus.py       # Corpus preparation & chunking
├── agents/             # Response & quiz generation
│   ├── base.py         # Expert agent base class
│   ├── registry.py     # Agent registry by discipline
│   ├── response.py     # Response generator
│   └── quiz.py         # Quiz generator
├── pipeline/           # Orchestration
│   ├── core.py         # PhiloMindPipeline
│   ├── dataclasses.py  # Data transfer objects
│   └── format.py       # Output formatting
├── ingestion/          # Teacher material upload
│   ├── parser.py       # .txt/.md parser
│   └── indexer.py      # TF-IDF material indexer
├── api/                # FastAPI REST API
│   └── main.py         # Endpoints
├── frontend/           # Gradio UI
│   └── gradio_app.py   # Teacher frontend
└── data/               # Data utilities
    └── augmentation.py # Data augmentation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train BiLSTM classifier
python -m src.classification.bilstm

# Build TF-IDF retriever
python -m src.retrieval.tfidf

# Run the pipeline test
python -c "
from src.pipeline.core import PhiloMindPipeline
from src.pipeline.format import format_output
pl = PhiloMindPipeline()
print(format_output(pl.process('What is Cartesian dualism?')))
"

# Start API server
python -m src.api.main

# Start Gradio frontend
python -m src.frontend.gradio_app
```

## Components

### Agent 1 — Question Classification
- **BiLSTM**: Baseline bidirectional LSTM with embedding layer (5 classes)
- **DistilBERT**: Pre-trained transformer fine-tuned on philosophical questions
- Shared interface via `BaseClassifier` abstract class

### Agent 2 — Passage Retrieval
- TF-IDF vectorization with cosine similarity
- Corpus of ~360K philosophical sentences (Kaggle Philosophy Data)
- Supports dynamic indexing of teacher-uploaded materials

### Agent 3 — Response Generator (Stub)
- Generates responses in the style of relevant philosophers
- Placeholder for LLM integration

### Agent 4 — Quiz Generator (Stub)
- Creates follow-up questions based on classification
- Class-specific templates (definition, comparison, example, deep dive, quiz)

## Data

- `data/raw/philosophy_data.csv` — Kaggle philosophy corpus (English)
- `data/labels/questions_train.csv` — 337 labeled questions for training
- `data/labels/questions_test.csv` — 87 labeled questions for evaluation
- `data/labels/questions_labeled.csv` — 400 original human-labeled questions

## Models

- `models/bilstm/final.pt` — Trained BiLSTM weights
- `models/bilstm/vocab.pkl` — Vocabulary
- `models/bilstm/label2idx.json` — Label mapping
- `models/retrieval/tfidf.pkl` — TF-IDF retriever
- `models/retrieval/corpus.csv` — Chunked corpus

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/materials/upload` | POST | Upload .txt/.md material |
| `/materials` | GET | List uploaded materials |
| `/query` | POST | Run full pipeline |

## Legacy / Evolution

The project evolved through weekly iterations. Original materials are preserved for reference:

- `models/bilstm_classifier.py` — Original BiLSTM training script (Week 2 baseline)
- `models/tfidf_retriever.py` — Original TF-IDF retriever
- `legacy/pipeline_original.py` — Original monolithic pipeline
- `notebooks/` — Jupyter notebooks for data exploration and evaluation

The refactored code in `src/` supersedes these with a cleaner modular architecture.

## License

University project — for educational purposes.
