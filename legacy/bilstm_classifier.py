"""
BiLSTM Baseline Classifier for philosophical question classification.
RNN model to demonstrate attention (via BiLSTM) and transfer learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from pathlib import Path

BASE_DIR = Path("models-utils")
BASE_DIR.mkdir(parents=True, exist_ok=True)

class QuestionDataset(Dataset):
    """Dataset for philosophical questions."""

    def __init__(self, df, vocab, label2idx, max_length=50):
        self.questions = df['question'].values
        self.labels = df['label'].values
        self.vocab = vocab
        self.label2idx = label2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        label = self.labels[idx]

        # Basic tokenization (split)
        tokens = question.lower().split()[:self.max_length]

        # Convert to indices
        token_indices = []
        for token in tokens:
            # Remove punctuation
            clean_token = token.strip(',.!?;:\'"')
            if clean_token in self.vocab:
                token_indices.append(self.vocab[clean_token])
            else:
                token_indices.append(self.vocab.get('<UNK>', 1))  # unknown token

        # Padding
        if len(token_indices) < self.max_length:
            token_indices += [self.vocab.get('<PAD>', 0)] * (self.max_length - len(token_indices))

        seq_len = max(1, min(len([t for t in tokens if t]), self.max_length))

        return {
            'tokens': torch.LongTensor(token_indices),
            'length': seq_len,
            'label': torch.LongTensor([self.label2idx[label]])[0]
        }


class BiLSTMClassifier(nn.Module):
    """BiLSTM Classifier with embedding layer."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,  # BiLSTM
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Output layer after forward+backward concatenation
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, lengths):
        # tokens: (batch_size, max_length)

        # Embedding
        embedded = self.dropout(self.embedding(tokens))  # (batch_size, max_length, embedding_dim)

        # Packing padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # BiLSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Take only the last hidden state from both directions
        # hidden: (n_layers*2, batch_size, hidden_dim)

        # Concatenate forward and backward of the last layer
        hidden_fwd = hidden[-2, :, :]  # last layer, forward
        hidden_bwd = hidden[-1, :, :]  # last layer, backward
        hidden_concat = torch.cat((hidden_fwd, hidden_bwd), dim=-1)  # (batch_size, 2*hidden_dim)

        # Classify
        output = self.fc(hidden_concat)
        return output


class QuestionClassifier:
    """Wrapper for training and inference."""

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, n_classes=5,
                 n_layers=2, dropout=0.3, device='cpu'):
        self.device = torch.device(device)
        self.vocab_size = vocab_size

        self.model = BiLSTMClassifier(
            vocab_size,
            embedding_dim,
            hidden_dim,
            n_classes,
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-3)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            tokens = batch['tokens'].to(self.device)
            lengths = batch['length'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(tokens, lengths)
            loss = self.criterion(logits, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                tokens = batch['tokens'].to(self.device)
                lengths = batch['length'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(tokens, lengths)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'loss': avg_loss,
            'accuracy': acc,
            'f1': f1,
            'confusion_matrix': cm
        }

    def train(self, train_loader, val_loader, epochs=30, patience=5):
        """Full training with early stopping."""
        patience_counter = 0

        print(f"\n[START] TRAINING BiLSTM (device={self.device}, epochs={epochs})")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                print(f"  [OK] Improvement! Saving the model...")
                self.save_model(BASE_DIR / f"bilstm_baseline_epoch{epoch + 1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n[STOP] Early stopping dopo {epoch+1} epochs")
                    break

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def predict(self, tokens, lengths):
        """Predict on batch."""
        self.model.eval()
        with torch.no_grad():
            tokens = tokens.to(self.device)
            lengths = lengths.to(self.device)
            logits = self.model(tokens, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy(), probs.cpu().numpy()

    def save_model(self, path):
        """Save the model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"  Model saved to {path}")

    def load_model(self, path):
        """Load the model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"  Model loaded from {path}")


def build_vocab(train_df, min_freq=2):
    """Build vocabulary from training set."""
    counter = Counter()

    for text in train_df['question']:
        tokens = text.lower().split()
        for token in tokens:
            clean_token = token.strip(',.!?;:\'"')
            if clean_token:
                counter[clean_token] += 1

    # Filter by minimum frequency
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2

    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    print(f"[INFO] Vocabulary size: {len(vocab)}")
    return vocab


if __name__ == '__main__':
    # Paths
    train_path = '../data/labels/questions_train.csv'
    test_path = '../data/labels/questions_test.csv'

    # load data
    print("[INFO] Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Label mapping
    label2idx = {
        'definition': 0,
        'comparison': 1,
        'example': 2,
        'indepth': 3,
        'quiz': 4
    }
    idx2label = {v: k for k, v in label2idx.items()}

    # Build vocab
    vocab = build_vocab(train_df, min_freq=1)

    # Dataset e DataLoader
    print("\n[INFO] Dataset e DataLoader...")
    train_dataset = QuestionDataset(train_df, vocab, label2idx, max_length=50)
    test_dataset = QuestionDataset(test_df, vocab, label2idx, max_length=50)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Device: {device}")

    classifier = QuestionClassifier(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=64,
        n_classes=5,
        n_layers=2,
        dropout=0.3,
        device=device
    )

    # Train
    history = classifier.train(train_loader, test_loader, epochs=30, patience=5)

    # Final evaluation
    print("\n" + "="*60)
    print("[INFO] FINAL EVALUATION")
    print("="*60)

    final_metrics = classifier.evaluate(test_loader)

    print(f"\nTest Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {final_metrics['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(final_metrics['confusion_matrix'])

    print("\nClassification Report:")
    # Ricrea predizioni per report full
    classifier.model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(classifier.device)
            lengths = batch['length'].to(classifier.device)
            labels = batch['label'].to(classifier.device)

            logits = classifier.model(tokens, lengths)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=list(idx2label.values())))

    # Save everything
    save_dir = BASE_DIR

    classifier.save_model(save_dir / "bilstm_baseline_final.pt")

    with open(save_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(save_dir / "label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)

    print(f"\n[OK] Model and vocabulary saved in {save_dir}/")


