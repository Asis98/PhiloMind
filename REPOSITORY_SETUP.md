# Setup repository GitHub per `PhiloMind`

## Stato attuale verificato

Nel workspace attuale:
- `git` **non è installato** o non è nel `PATH`
- `gh` (GitHub CLI) **non è installato** o non è nel `PATH`
- è stato creato un file `.gitignore` adatto al progetto

## Cosa verrà versionato

Da committare normalmente:
- codice Python (`agents/`, `models/*.py`, `pipeline.py`, `setup.py`)
- configurazioni (`requirements.txt`, `disciplines/config.json`)
- dataset piccoli/label (`data/labels/*.csv`)
- documentazione (`README.md`, roadmap, report)

Escluso da Git tramite `.gitignore`:
- `.venv/`, `.idea/`, `logs/`
- dataset grandi (`data/raw/philosophy_data.csv`, `data/raw/sentences.csv`)
- modelli addestrati e artefatti (`models/*.pt`, `models/*.pkl`, `models/corpus_chunks.csv`)

## Opzione A — via sito GitHub + Git locale

1. Installa Git for Windows
2. Crea su GitHub un nuovo repository vuoto chiamato `PhiloMind`
3. Esegui questi comandi in PowerShell dentro la cartella del progetto

```powershell
cd C:\repository\PhiloMind
git init
git branch -M main
git add .
git commit -m "Initial commit: week 2 baseline"
git remote add origin https://github.com/<TUO_USERNAME>/PhiloMind.git
git push -u origin main
```

## Opzione B — via GitHub CLI

Dopo aver installato sia Git che GitHub CLI:

```powershell
cd C:\repository\PhiloMind
git init
git branch -M main
git add .
git commit -m "Initial commit: week 2 baseline"
gh auth login
gh repo create PhiloMind --public --source . --remote origin --push
```

Se vuoi repo privata:

```powershell
gh repo create PhiloMind --private --source . --remote origin --push
```

## Configurazione identità Git

Se è il primo uso di Git su questa macchina:

```powershell
git config --global user.name "Il Tuo Nome"
git config --global user.email "tua-email@example.com"
```

## Controlli utili

```powershell
git status
git remote -v
git log --oneline -n 5
```

## Se il push fallisce per file troppo grandi

Probabile causa: hai aggiunto file prima del `.gitignore` o ci sono artefatti grossi già staged.

Pulizia tipica:

```powershell
git rm --cached data/raw/philosophy_data.csv
git rm --cached data/raw/sentences.csv
git rm --cached models/*.pt
git rm --cached models/*.pkl
git rm --cached models/corpus_chunks.csv
git add .gitignore
git commit -m "Remove large generated artifacts from tracking"
```

## Consiglio pratico per questo progetto

Per una prima repository pulita, committa:
- codice
- documentazione
- dataset etichettato piccolo
- configurazioni

Evita di committare:
- corpus raw da centinaia di MB
- checkpoint e modelli addestrati
- log locali

## Struttura minima consigliata nel primo commit

```text
PhiloMind/
  agents/
  data/labels/
  disciplines/
  models/
  notebooks/
  pipeline.py
  README.md
  requirements.txt
  .gitignore
```

