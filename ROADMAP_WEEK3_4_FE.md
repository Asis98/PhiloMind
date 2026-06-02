# PhiloMind - Roadmap Settimane 3 e 4 (con FE Docente)

Data: 2026-06-02

## Obiettivo aggiuntivo richiesto
Aggiungere una parte Frontend dove un docente puo:
1. caricare materiale didattico,
2. specificare la materia/disciplina,
3. usare l'applicativo PhiloMind con il materiale caricato.

## Scelta tecnologia FE
- FE scelto: **Gradio** (Python-first, rapido per demo universitaria).
- Struttura suggerita: `app/gradio_app.py` con tab Upload/Materiali/Query.

---

## Settimana 3 - Focus: DistilBERT + FE docente MVP

### 1) Agent 1: DistilBERT fine-tuned
- Implementare `models/distilbert_finetune.py`.
- Addestrare su `data/labels/questions_train.csv` e validare su `data/labels/questions_test.csv`.
- Salvare modello e metriche in `models/` e `reports/`.
- Confrontare risultati con baseline BiLSTM (accuracy, F1, confusion matrix).

### 2) Backend minimo per FE
- Creare API in `app/api/` con endpoint minimi:
  - `POST /materials/upload`
  - `GET /materials`
  - `POST /materials/index`
  - `POST /query`
- Collegare `POST /query` alla pipeline esistente.

### 3) FE Docente MVP
- Creare UI in `app/gradio_app.py` (Gradio):
  - pagina upload materiale,
  - selezione materia,
  - lista materiali caricati,
  - pagina query all'app.
- Mostrare output completo:
  - classe domanda,
  - passaggi recuperati,
  - risposta filosofica,
  - quiz/follow-up.

### 4) Ingestion base materiale docente
- Parser minimo per `.txt` e `.md`.
- Chunking del contenuto.
- Indicizzazione nel retriever con metadato `subject`.

### Deliverable settimana 3
- DistilBERT funzionante e confrontato con BiLSTM.
- FE docente MVP funzionante (upload + materia + query).
- API base documentata.

---

## Settimana 4 - Focus: hardening + demo finale

### 1) Retrieval migliorato per materia
- Aggiungere filtro retrieval per `subject`.
- Priorita ai materiali docente, fallback al corpus generale.
- Migliorare ranking TF-IDF e logging fonti.

### 2) Agent 3 e Agent 4
- Raffinare output risposta filosofica con citazione fonti.
- Raffinare quiz/follow-up in base a classe domanda e materia.
- Stabilizzare formato JSON di output per FE.

### 3) FE docente v2
- Migliorare UX:
  - stato indicizzazione (`uploaded/indexed/error`),
  - filtri per materia,
  - storico query recenti.
- Gestione errori upload e messaggi utente in componenti Gradio.

### 4) Pacchetto finale esame
- README finale con architettura completa.
- Script avvio rapido backend + frontend.
- Report confronto BiLSTM vs DistilBERT.
- Demo end-to-end: upload docente -> query -> risposta + quiz.

### Deliverable settimana 4
- Sistema stabile e dimostrabile in locale.
- Demo completa pronta per presentazione.

---

## Priorita assolute (se tempo ridotto)
1. FE docente (Gradio): upload + materia + query.
2. DistilBERT fine-tuned con confronto rispetto a BiLSTM.
3. Retrieval filtrato per materia.
4. Rifiniture avanzate (solo se resta tempo).

