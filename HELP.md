# Guida a PhiloMind

## Caricare materiale didattico per una nuova materia

Puoi aggiungere una materia (es. fisica, storia dell'arte, biologia) al sistema. Ecco i passi.

### 1. Prepara i testi di partenza

Il sistema ha bisogno di due cose:

**A — Un corpus di testi** (es. libri, dispense, Wikipedia) da cui fare retrieval.
Crea un file CSV con almeno le colonne `author`, `title`, `text` (una riga per documento/frase).
Esempio per `data/raw/fisica_data.csv`:

```
author,title,text
Newton,Principia,Il moto dei corpi celesti segue le leggi della gravitazione universale.
Einstein,Relatività,"La velocità della luce è costante in tutti i sistemi di riferimento inerziali."
```

**B — Domande etichettate** (opzionale, per la classificazione).
Se vuoi che il sistema classifichi le domande sulla nuova materia (es. "Cos'è la relatività?" → label `definition`), aggiungile a `data/labels/questions_labeled.csv`. Le label esistenti sono:
- `definition` — domande "cos'è X?"
- `comparison` — confronti
- `example` — chiede esempi
- `deepening` — approfondimenti
- `quiz` — richieste di quiz

Se non aggiungi domande, il classificatore funziona comunque (le label sono sul *tipo* di domanda, non sulla materia).

### 2. Modifica i file di configurazione

**A — `config/disciplines.json`** — aggiungi la nuova materia:
```json
{
  "philosophy": { ... },
  "physics": {
    "persona": "You are Isaac Newton. Rispondi con precisione basandoti su evidenze empiriche.",
    "created_at": "2026-06-28",
    "num_chunks": 0
  }
}
```
Il campo `persona` viene usato come prompt per l'LLM nella generazione delle risposte.

**B — `src/retrieval/tfidf.py`** — estendi le liste di entità per migliorare il retrieval:
- `PHILOSOPHERS` → aggiungi i nomi dei protagonisti della nuova materia (Newton, Einstein, Galileo, ecc.)
- `CONCEPTS` → aggiungi i concetti chiave (relatività, quanto, entropia, ecc.)
- `CONCEPT_AUTHOR_MAP` → mappa concetti ad autori

**C — `src/data/augmentation.py`** (opzionale) — estendi le liste `CONCEPTS` e `PHILOSOPHERS` per generare domande di augmentation sulla nuova materia.

**D — `src/agents/quiz.py`** (opzionale) — estendi la lista `PHILOSOPHERS` per generare quiz migliori sulla nuova materia.

### 3. Ricostruisci i modelli

Esegui:
```bash
python scripts/build_models.py
```

Questo comando fa tre cose:
1. **Ricostruisce il corpus** — carica il CSV raw, lo chunk in segmenti da ~120 parole, salva in `models/retrieval/corpus_chunks.csv`
2. **Riaddestra TF-IDF + Dense** — vettorizza i chunk, calcola embeddings con SentenceTransformer, salva in `models/retrieval/`
3. **Riaddestra il classificatore BiLSTM** — carica le domande etichettate, fa lo split deduplicato, augmenta, allena con early stopping, salva in `models/bilstm/`

Tempo stimato: ~1-2 minuti.

### 4. Quando puoi fare domande

Subito dopo aver eseguito `build_models.py`. Il sistema è pronto quando vedi:

```
============================================================
  All models built successfully!
============================================================
```

A questo punto puoi fare domande come:
- "What is relativity?" → classificata come `definition`, cerca chunk su relatività
- "Compare Newton and Einstein" → classificata come `comparison`
- "Test me on quantum mechanics" → classificata come `quiz`, genera MCQ

### 5. Metodo alternativo: upload via API

Se preferisci non toccare i file, puoi usare l'API:
```bash
curl -X POST http://localhost:8000/materials/upload \
  -F "file=@dispensa_fisica.txt" \
  -F "subject=physics"
```
Poi ricostruisci i modelli con `python scripts/build_models.py`.

---

## Architettura del sistema (per capire cosa succede)

```
Domanda utente
  │
  ├─→ Classificatore (BiLSTM) → label (definition/comparison/example/deepening/quiz)
  │
  ├─→ Topic Extractor → estrae il concetto/filosofo dalla domanda
  │
  ├─→ Hybrid Retrieval (TF-IDF + Dense + Cross-Encoder) → chunk rilevanti
  │     │
  │     └─→ Filtra per subject se specificato
  │
  ├─→ QuizGenerator (parallelo al retrieval) → MCQ dalla label + topic
  │
  └─→ ResponseGenerator → risposta testuale formato con citazioni
```

Il classificatore distingue il **tipo** di domanda, non la materia. Questo significa che funziona per qualsiasi materia senza riaddestramento — finché le domande seguono gli stessi pattern linguistici.

---

## Cosa NON cambia (e perché)

| Componente | Cambia? | Motivo |
|---|---|---|
| Classificatore BiLSTM | No, se non aggiungi nuove label | Le 5 label sono tipi di domanda, non dipendono dalla materia |
| QuizGenerator | Opzionale | Le template funzionano per qualsiasi concetto; migliorarle è un'estetica |
| Pipeline core | No | Già agnostico rispetto alla materia |

| Componente | Cambia? | Come |
|---|---|---|
| Corpus chunks | Sì | Aggiungi i testi della nuova materia |
| TF-IDF vettori | Sì | Rebuild con `build_models.py` |
| Embedding densi | Sì | Rebuild con `build_models.py` |
| `disciplines.json` | Sì | Aggiungi entry per la nuova materia |
| Liste entità TF-IDF | Consigliato | Migliora il retrieval della nuova materia |
| Liste augmentation | Consigliato | Migliora la generazione di dati sintetici |
