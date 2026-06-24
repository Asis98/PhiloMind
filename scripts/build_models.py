"""Build/rebuild all models: corpus index, TF-IDF retriever, and train DistilBERT."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.retrieval.tfidf import TFIDFRetriever
from src.retrieval.corpus import CorpusPreparer


def rebuild_retriever():
    print("=" * 60)
    print("  Building TF-IDF Corpus Index")
    print("=" * 60)
    
    base = Path(__file__).resolve().parent.parent
    raw_path = base / 'data' / 'raw' / 'philosophy_data.csv'
    models_dir = base / 'models'
    
    if not raw_path.exists():
        print(f"Raw corpus not found at {raw_path}")
        print("Creating sample corpus instead...")
        chunk_size = 500
        example_texts = [
            "Plato was a philosopher in Classical Greece and the founder of the Academy in Athens. He is widely considered the most pivotal figure in the development of philosophy. His theory of Forms proposes that the physical world is not the real world; instead, ultimate reality exists beyond our physical world. The Forms are abstract, perfect, unchanging concepts or ideals that transcend time and space. The Form of the Good, according to Plato, is the highest form and the source of all other forms. In the Republic, Plato uses the allegory of the cave to illustrate his metaphysical theory.",
            "Aristotle was a Greek philosopher and polymath during the Classical period in Ancient Greece. He was the founder of the Lyceum and the Peripatetic school of philosophy. Aristotle's ethics, particularly his concept of virtue ethics, emphasizes the role of habit and character. He argued that eudaimonia (happiness) is the highest human good and that virtue is the mean between two extremes. Unlike Plato, Aristotle believed that form and matter are inseparable, and that knowledge comes from sensory experience.",
            "René Descartes was a French philosopher, mathematician, and scientist. He is often called the father of modern philosophy. His most famous statement is 'Cogito, ergo sum' (I think, therefore I am). Descartes developed a method of radical doubt, questioning everything that could possibly be doubted. He argued for a dualism between mind and body, claiming that they are distinct substances. His work laid the foundation for rationalism in continental philosophy.",
            "Immanuel Kant was a German philosopher who is a central figure in modern philosophy. In his Critique of Pure Reason, he argued that space and time are a priori forms of sensibility. Kant proposed the categorical imperative as the supreme principle of morality: 'Act only according to that maxim whereby you can at the same time will that it should become a universal law.' He distinguished between phenomena (things as they appear) and noumena (things in themselves).",
            "Friedrich Nietzsche was a German philosopher known for his critiques of traditional European morality and religion. He introduced the concept of the Übermensch (Overman) as a goal for humanity to strive toward. His idea of the 'will to power' suggests that the fundamental drive of humans is to exercise and expand their power. The 'eternal return' is a thought experiment that asks whether one would be willing to live one's life over and over again for eternity. Nietzsche's work has profoundly influenced existentialism and postmodern thought.",
        ]
        corpus_chunks = pd.DataFrame({
            'text': example_texts,
            'philosopher': ['Plato', 'Aristotle', 'Descartes', 'Kant', 'Nietzsche'],
            'work': ['Republic', 'Nicomachean Ethics', 'Meditations on First Philosophy', 'Critique of Pure Reason', 'Thus Spoke Zarathustra']
        })
        corpus_chunks.to_csv(models_dir / 'retrieval' / 'corpus_chunks.csv', index=False)
        print(f"Created {len(corpus_chunks)} sample chunks")
    else:
        print(f"Using corpus from {raw_path}")
        df = pd.read_csv(raw_path, low_memory=False)
        print(f"Total documents: {len(df)}")
        corpus_chunks = CorpusPreparer.build_corpus_index(str(raw_path), models_dir / 'retrieval', sample_size=15000)
    
    retriever = TFIDFRetriever(corpus_df=corpus_chunks, max_features=10000)
    retriever.fit(corpus_chunks['text'].values)
    
    save_path = models_dir / 'retrieval' / 'tfidf.pkl'
    retriever.save(str(save_path))
    print(f"Retriever saved to {save_path}")
    print(f"Vocabulary size: {len(retriever.vectorizer.get_feature_names_out())}")
    print(f"Corpus chunks: {len(corpus_chunks)}")
    
    test_queries = [
        "What is Plato's theory of Forms?",
        "Explain Cartesian dualism",
        "What is the categorical imperative?",
        "Compare Aristotle and Plato",
        "What does Nietzsche mean by the will to power?"
    ]
    
    print("\n" + "=" * 60)
    print("  Test Retrieval")
    print("=" * 60)
    for q in test_queries:
        results = retriever.retrieve(q, top_k=2)
        print(f"\nQuery: {q}")
        for idx, text, score in results:
            source = retriever.get_source(idx)
            print(f"  [{source['philosopher']} - {source['work']}] (score: {score:.2%})")
            print(f"  {text[:120]}...")
    
    print("\n[OK] Retriever built successfully!")
    return retriever


def train_classifier():
    print("\n" + "=" * 60)
    print("  Training BiLSTM Classifier")
    print("=" * 60)
    
    base = Path(__file__).resolve().parent.parent
    train_path = base / 'data' / 'labels' / 'questions_train.csv'
    test_path = base / 'data' / 'labels' / 'questions_test.csv'
    models_dir = base / 'models' / 'bilstm'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if not train_path.exists():
        all_dfs = []
        label_path = base / 'data' / 'labels' / 'questions_labeled.csv'
        if label_path.exists():
            df = pd.read_csv(label_path)
            print(f"Using labeled data ({len(df)} samples)")
            all_dfs.append(df)
        aug_path = base / 'data' / 'labels' / 'questions_augmented.csv'
        if aug_path.exists():
            aug_df = pd.read_csv(aug_path)
            print(f"Using augmented data ({len(aug_df)} samples)")
            all_dfs.append(aug_df)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['question'])
            from sklearn.model_selection import train_test_split
            train, test = train_test_split(combined, test_size=0.2, random_state=42, stratify=combined['label'])
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)
            print(f"Created train ({len(train)}) / test ({len(test)}) split")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Labels: {train_df['label'].unique()}")
    print(train_df['label'].value_counts())
    
    from src.classification.bilstm import BiLSTMClassifier, build_vocab, BiLSTMTrainer
    from src.classification.bilstm import QuestionDataset
    from torch.utils.data import DataLoader
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vocab = build_vocab(train_df, min_freq=1)
    label2idx = {l: i for i, l in enumerate(sorted(train_df['label'].unique()))}
    
    with open(models_dir / 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(models_dir / 'label2idx.json', 'w') as f:
        json.dump(label2idx, f, indent=2)
    
    train_dataset = QuestionDataset(train_df, vocab, label2idx, max_length=50)
    test_dataset = QuestionDataset(test_df, vocab, label2idx, max_length=50)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    classifier = BiLSTMClassifier(
        vocab_size=len(vocab), embedding_dim=100, hidden_dim=64,
        n_classes=len(label2idx), n_layers=2, dropout=0.3, device=device
    )
    trainer = BiLSTMTrainer(model=classifier.model, device=device)
    history = trainer.train(train_loader, test_loader, epochs=30, patience=5,
                            save_path=str(models_dir / 'final.pt'))
    
    classifier.model.load_state_dict(torch.load(models_dir / 'final.pt', map_location=device))
    
    from sklearn.metrics import classification_report
    all_preds = []
    all_labels = []
    classifier.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            lengths = batch['length'].to(device)
            labels = batch['label'].to(device)
            logits = classifier.model(tokens, lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    idx2label = {v: k for k, v in label2idx.items()}
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[idx2label[i] for i in range(len(idx2label))]))
    
    print(f"\n[OK] BiLSTM trained and saved to {models_dir / 'final.pt'}")


def main():
    retriever = rebuild_retriever()
    train_classifier()
    print("\n" + "=" * 60)
    print("  All models built successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
