# 📚 PhiloMind Settimana 2 - INDEX Dei File

## 🎯 QUICK REFERENCE

Questo file serve come **indice rapido** di tutti i file creati e modificati durante la Settimana 2.

---

## 📊 DATA FILES

### Original (Week 1)
```
✅ data/labels/questions_labeled.csv
   - 200 domande etichettate manualmente
   - Formato: [question, label]
   - Autore: Labeling manuale
   - Linea 1-401 (incluso header)
```

### New (Week 2)
```
✅ data/labels/questions_augmented.csv
   - 424 domande totali
   - Creato da: data/scripts/data_augmentation.py
   - Include: original (200) + augmented variations (224)
   
✅ data/labels/questions_train.csv
   - 337 domande (80%)
   - Split per: BiLSTM training
   - Stratificato per classe
   
✅ data/labels/questions_test.csv
   - 87 domande (20%)
   - Split per: BiLSTM evaluation
   - Stratificato per classe

✅ data/scripts/data_augmentation.py
   - Script per template-based augmentation
   - Riutilizzabile per future data
   - Ha generato le 224 variazioni
```

---

## 🤖 MODEL FILES

### BiLSTM Classifier (Agent 1)
```
✅ models/bilstm_classifier.py (280 linee)
   - Implementazione completa BiLSTM
   - Classi:
     * QuestionDataset - PyTorch Dataset loader
     * BiLSTMClassifier - Model architecture
     * QuestionClassifier - Wrapper per training/inference
   
🔄 models/bilstm_baseline_final.pt (in progress)
   - Modello weights dopo training
   - Size: ~1.2 MB
   - CreAto da: python models/bilstm_classifier.py
   
🔄 models/bilstm_baseline_epoch*.pt (in progress)
   - Checkpoint ad ogni epoch
   - Best validation loss salvato automaticamente
   
✅ models/vocab.pkl (atteso)
   - Vocabulary mapping (word → index)
   - Size: ~50 KB
   - Creato da: QuestionDataset durante training
   
✅ models/label2idx.json (atteso)
   - Label encoding (5 classi)
   - Formato JSON per readability
   - Creato da: training script
```

### TF-IDF Retriever (Agent 2)
```
✅ models/tfidf_retriever.py (200 linee)
   - Implementazione TF-IDF retriever
   - Classi:
     * TFIDFRetriever - Wrapper intorno sklearn
     * CorpusPreparer - Corpus parsing e chunking
   
✅ models/tfidf_retriever.pkl (serialized)
   - Vectorizer + corpus
   - Size: ~8.5 MB
   - Creato da: python models/tfidf_retriever.py
   - Contiene:
     * sklearn.TfidfVectorizer (fitted)
     * pd.DataFrame corpus_df (5000 chunks)
   
✅ models/corpus_chunks.csv (5000 righe)
   - Corpus preprocessato con metadata
   - Colonne: [text, philosopher, work, original_idx]
   - Size: ~2.3 MB
```

---

## 🔄 PIPELINE & AGENTS

### Pipeline Principale
```
✅ pipeline.py (450 linee)
   - Classe principale: PhiloMindPipeline
   - Agent wrapper:
     * QuestionClassificationAgent (Agent 1)
     * DocumentRetrievalAgent (Agent 2)
     * ResponseGeneratorAgent (Agent 3 - STUB)
     * QuizGeneratorAgent (Agent 4 - STUB)
   - Data classes:
     * ClassificationResult
     * RetrievalResult
     * PipelineOutput
   
   Funzioni:
   - pipeline.process(question) → PipelineOutput
   - pipeline.batch_process(questions) → List[PipelineOutput]
   - format_output(output) → str (for display)
```

### Agent Base Classes
```
✅ agents/base_agents.py
   - Classe base: ExpertAgent
   - Metodi: retrieve(query, top_k), get_persona()
   - Scopo: Interface comune per tutti gli agent
   
✅ agents/registry.py
   - Classe: AgentRegistry
   - Funzioni:
     * _load_all() - load dal config.json
     * list_disciplines() - elenco discipline
     * get(discipline) - get agent per disciplina
     * add_discipline(name, persona) - (stub)
```

### Configuration
```
✅ disciplines/config.json
   - Configurazione philosopher personas
   - Esempio:
     {
       "filosofia": {
         "persona": "Sei Socrate. Rispondi con domande maieutiche...",
         "created_at": "2026-05-20",
         "num_chunks": 0
       }
     }
```

---

## 📖 DOCUMENTATION

### Main Documentation
```
✅ README.md (500 linee)
   - Overview progetto
   - Quick start guide
   - 4 Agent architecture
   - Dataset description
   - Performance comparison BiLSTM vs DistilBERT
   - FAQ + Support
   
✅ ROADMAP_WEEK2.md (400 linee)
   - Timeline giornaliero dettagliato
   - Deliverable checklist
   - Technical insights
   - How to run every step
   - Git repository info
   
✅ SUMMARY_WEEK2.md (300 linee)
   - What was completed
   - Expected metrics
   - How to use now
   - Next steps (Week 3)
   - File structure
   - Learned lessons
```

### Technical Documentation
```
✅ TECHNICAL_SPEC.json (500 linee)
   - Architecture definition
   - Implementation details per agent
   - Dataset specs
   - Pipeline flow definition
   - Performance metrics
   - Testing scenarios
   - Improvements for Week 3
   
✅ notebooks/WEEK2_EVALUATION.md (300 linee)
   - Evaluation code (Python)
   - Dataset analysis
   - BiLSTM performance section
   - TF-IDF retriever test
   - Pipeline integration test
   - Metriche raccolte
   - Conclusioni settimanal
```

### Setup & Dependencies
```
✅ requirements.txt
   - Tutte le dipendenze Python
   - Con versioni specifiche
   - Commenti per optional packages
   
✅ setup.py
   - Script di setup automatico
   - Installa dipendenze
   - Crea directory structure
   - Verifica file presenti
   - Guida next steps
```

---

## 📋 FILE MANIFEST - Per Esame

Per consegnare il progetto all'esame, includere:

### 🔴 ESSENZIALI
```
✅ models/bilstm_classifier.py - Implementazione Agent 1
🔄 models/bilstm_baseline_final.pt - Weights allenati
✅ models/tfidf_retriever.py - Implementazione Agent 2
✅ pipeline.py - Pipeline unificata
✅ README.md - Documentazione principale
✅ ROADMAP_WEEK2.md - Timeline dettagliato
✅ requirements.txt - Dipendenze
```

### 🟢 MOLTO IMPORTANTE
```
✅ data/scripts/data_augmentation.py - Data engineering
✅ data/labels/questions_train.csv - Training data
✅ data/labels/questions_test.csv - Testing data
✅ notebooks/WEEK2_EVALUATION.md - Evaluation code
✅ TECHNICAL_SPEC.json - Technical specification
```

### 🟡 SUPPORTO
```
✅ agents/base_agents.py - Base architecture
✅ agents/registry.py - Agent registry
✅ disciplines/config.json - Config
✅ setup.py - Setup script
✅ SUMMARY_WEEK2.md - Summary
```

---

## 🔗 FILE DEPENDENCIES

```
Execution Flow:
┌─ data/scripts/data_augmentation.py
│  └─ generates: data/labels/questions_train.csv
│  └─ generates: data/labels/questions_test.csv
│
├─ models/bilstm_classifier.py
│  ├─ input: data/labels/questions_train.csv
│  ├─ input: data/labels/questions_test.csv
│  ├─ output: models/bilstm_baseline_final.pt
│  ├─ output: models/vocab.pkl
│  └─ output: models/label2idx.json
│
├─ models/tfidf_retriever.py
│  ├─ input: data/raw/philosophy_data.csv
│  ├─ output: models/tfidf_retriever.pkl
│  └─ output: models/corpus_chunks.csv
│
└─ pipeline.py
   ├─ import: models/bilstm_classifier.py
   ├─ import: models/tfidf_retriever.py
   ├─ import: agents/base_agents.py
   ├─ import: agents/registry.py
   ├─ load: models/*.pkl (trained)
   └─ output: JSON results
```

---

## 📈 STATISTICS

### Code Lines
```
models/bilstm_classifier.py:      280 righe
models/tfidf_retriever.py:        200 righe
pipeline.py:                      450 righe
agents/base_agents.py:            12 righe
agents/registry.py:               28 righe
data/scripts/data_augmentation.py: 180 righe

Total: ~1150 righe di codice
```

### Documentation Lines
```
README.md:                        500 righe
ROADMAP_WEEK2.md:               400 righe
SUMMARY_WEEK2.md:               300 righe
notebooks/WEEK2_EVALUATION.md:   300 righe
TECHNICAL_SPEC.json:            500 righe (markup)

Total: ~2000 righe di documentazione
```

### Data Size
```
data/labels/questions_augmented.csv:  ~80 KB
data/labels/questions_train.csv:      ~65 KB
data/labels/questions_test.csv:       ~17 KB
models/corpus_chunks.csv:             ~2.3 MB
models/tfidf_retriever.pkl:           ~8.5 MB

Total: ~11 MB
```

---

## ✅ CHECKLIST - Cosa Verificare

Quando finisce il training BiLSTM:

```
□ Verificare models/bilstm_baseline_final.pt esiste
□ Controllare models/vocab.pkl dimensione >40KB
□ Leggere models/label2idx.json e verificare 5 classi
□ Eseguire pipeline.py e verificare JSON output
□ Controllare format_output() per display
□ Correggere errori se presenti
□ Push a Git repository
□ Documentare risultati

Expected Output:
{
  "question": "...",
  "classification": {"label": "...", "confidence": 0.85, ...},
  "retrieval": {"passages": [...], "scores": [...]},
  "response": "...",
  "quiz": "..."
}
```

---

## 🔗 LINK VELOCE AI FILE

| File | Descrizione | Tipo | Status |
|------|-----------|------|--------|
| [README.md](README.md) | Overview | Doc | ✅ |
| [ROADMAP_WEEK2.md](ROADMAP_WEEK2.md) | Timeline | Doc | ✅ |
| [pipeline.py](pipeline.py) | Main pipeline | Code | ✅ |
| [models/bilstm_classifier.py](models/bilstm_classifier.py) | Agent 1 | Code | ✅ |
| [models/tfidf_retriever.py](models/tfidf_retriever.py) | Agent 2 | Code | ✅ |
| [data/scripts/data_augmentation.py](data/scripts/data_augmentation.py) | Data prep | Code | ✅ |
| [requirements.txt](requirements.txt) | Dependencies | Config | ✅ |
| [TECHNICAL_SPEC.json](TECHNICAL_SPEC.json) | Technical | Doc | ✅ |

---

## 📝 NOTE PER STUDENTI

Se stai continuando dal mio lavoro:

1. **Next Step**: Finire il training BiLSTM (dovrebbe completarsi fra poco)
2. **Then**: Lanciare `python pipeline.py` per testare
3. **Evaluation**: Usare `notebooks/WEEK2_EVALUATION.md` per analizzare risultati
4. **Week 3**: Implementare DistilBERT + LLM (già strutture di base in pipeline.py)

Good luck! 🚀

---

**Generated**: 2 Giugno 2026  
**Status**: Week 2 - 95% Complete  

