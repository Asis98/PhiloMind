# TODO Settimana 3 - Checklist operativa

## A. Classificatore DistilBERT
- [ ] Creare `models/distilbert_finetune.py`
- [ ] Addestrare su `questions_train.csv`
- [ ] Valutare su `questions_test.csv`
- [ ] Salvare metriche in `reports/`
- [ ] Confronto BiLSTM vs DistilBERT

## B. API base
- [ ] Creare `app/api/main.py`
- [ ] Endpoint upload/list/index/query
- [ ] Test endpoint locali

## C. FE docente MVP
- [ ] Creare `app/gradio_app.py`
- [ ] Aggiungere `gradio` a `requirements.txt`
- [ ] Form upload file
- [ ] Campo materia obbligatorio
- [ ] Lista materiali caricati
- [ ] Pagina query con output pipeline

## D. Ingestion materiale
- [ ] Parser `.txt`
- [ ] Parser `.md`
- [ ] Chunking contenuto
- [ ] Indicizzazione con metadato materia

## E. Test e demo
- [ ] Test end-to-end con almeno 1 materiale docente
- [ ] Verifica output: classe + retrieval + risposta + quiz
- [ ] Aggiornare documentazione

