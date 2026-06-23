"""BiLSTM classifier for philosophical questions.
Implements a bidirectional LSTM with embedding layer as a baseline model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import pickle
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, List

from .base import BaseClassifier


class QuestionDataset(Dataset):
    """Dataset for philosophical questions with tokenization."""

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
        tokens = question.lower().split()[:self.max_length]
        token_indices = []
        for token in tokens:
            clean_token = token.strip(',.!?;:\'"')
            if clean_token in self.vocab:
                token_indices.append(self.vocab[clean_token])
            else:
                token_indices.append(self.vocab.get('<UNK>', 1))
        if len(token_indices) < self.max_length:
            token_indices += [self.vocab.get('<PAD>', 0)] * (self.max_length - len(token_indices))
        seq_len = max(1, min(len([t for t in tokens if t]), self.max_length))
        return {
            'tokens': torch.LongTensor(token_indices),
            'length': seq_len,
            'label': torch.LongTensor([self.label2idx[label]])[0]
        }


class BiLSTM(nn.Module):
    """BiLSTM with embedding and fully-connected classification head."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, lengths):
        embedded = self.dropout(self.embedding(tokens))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_fwd, hidden_bwd), dim=-1)
        return self.fc(hidden_concat)


class BiLSTMClassifier(BaseClassifier):
    """BiLSTM classifier wrapper with training, evaluation and prediction."""

    def __init__(self, vocab_size=1, embedding_dim=100, hidden_dim=64, n_classes=5,
                 n_layers=2, dropout=0.3, device='cpu'):
        self.device = torch.device(device)
        self.model = BiLSTM(vocab_size, embedding_dim, hidden_dim, n_classes,
                            n_layers=n_layers, dropout=dropout).to(self.device)
        self.idx2label = None

    def predict(self, question: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        self.model.eval()
        tokens = question.lower().split()[:50]
        token_indices = []
        for token in tokens:
            clean_token = token.strip(',.!?;:\'"')
            if clean_token in self.vocab:
                token_indices.append(self.vocab[clean_token])
            else:
                token_indices.append(self.vocab.get('<UNK>', 1))
        if len(token_indices) < 50:
            token_indices += [self.vocab.get('<PAD>', 0)] * (50 - len(token_indices))
        tokens_tensor = torch.LongTensor([token_indices]).to(self.device)
        lengths = torch.LongTensor([min(len(tokens), 50)]).to(self.device)
        with torch.no_grad():
            logits = self.model(tokens_tensor, lengths)
            probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = self.idx2label[pred_idx]
        confidence = float(probs[0, pred_idx])
        top_3_indices = np.argsort(probs[0].cpu().numpy())[::-1][:3]
        top_3 = [(self.idx2label[idx], float(probs[0, idx])) for idx in top_3_indices]
        return pred_label, confidence, top_3

    def load(self, model_path, vocab_path, label2idx_path):
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label2idx_path, 'r') as f:
            label2idx = json.load(f)
        self.idx2label = {v: k for k, v in label2idx.items()}
        self.model = BiLSTM(len(self.vocab), 100, 64, len(label2idx), n_layers=2, dropout=0.3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))


class BiLSTMTrainer:
    """Handles BiLSTM training loop with early stopping."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.optimizer = optim.Adam(model.parameters(), lr=2e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            tokens = batch['tokens'].to(self.device)
            lengths = batch['length'].to(self.device)
            labels = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(tokens, lengths)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
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
        return {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }

    def train(self, train_loader, val_loader, epochs=30, patience=5, save_path=None):
        patience_counter = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f} - Val F1: {val_metrics['f1']:.4f}")
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                if save_path:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}


def build_vocab(train_df, min_freq=2):
    """Build vocabulary from training data."""
    counter = Counter()
    for text in train_df['question']:
        for token in text.lower().split():
            clean_token = token.strip(',.!?;:\'"')
            if clean_token:
                counter[clean_token] += 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab
