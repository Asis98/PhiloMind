"""DistilBERT fine-tuning for question classification.
Uses a pre-trained transformer model with a classification head.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from typing import Tuple, List

from .base import BaseClassifier

LABEL_MAP = {
    'definition': 0, 'comparison': 1, 'example': 2,
    'deepening': 3, 'quiz': 4
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = 5


class PhilosophyDataset(Dataset):
    """Dataset for philosophical questions with DistilBERT tokenizer."""

    def __init__(self, df, tokenizer, max_length=128):
        self.questions = df['question'].values
        self.labels = df['label'].map(LABEL_MAP).values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.questions[idx]),
            truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class DistilBERTClassifier(BaseClassifier):
    """DistilBERT classifier wrapper with fine-tuning and prediction."""

    def __init__(self, model_name='distilbert-base-uncased', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_LABELS
        ).to(self.device)

    def predict(self, question: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        self.model.eval()
        encoding = self.tokenizer(
            question, truncation=True, padding='max_length',
            max_length=128, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = IDX_TO_LABEL[pred_idx]
        confidence = float(probs[0, pred_idx])
        top_3_indices = torch.argsort(logits[0], descending=True)[:3].cpu().numpy()
        top_3 = [(IDX_TO_LABEL[idx], float(probs[0, idx])) for idx in top_3_indices]
        return pred_label, confidence, top_3

    def save(self, path):
        model_path = str(Path(path).with_suffix(''))
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load(self, path):
        model_path = str(Path(path).with_suffix(''))
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)


class DistilBERTFineTuner:
    """Handles DistilBERT fine-tuning with early stopping."""

    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses = []
        self.val_losses = []

    def prepare_data(self, train_path, test_path, batch_size=16, max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_dataset = PhilosophyDataset(train_df, self.tokenizer, max_length)
        test_dataset = PhilosophyDataset(test_df, self.tokenizer, max_length)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def train_epoch(self, model, optimizer, scheduler=None):
        model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self, model, loader):
        model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / len(loader)
        return {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'predictions': all_preds,
            'true_labels': all_labels
        }

    def train(self, model, epochs=10, learning_rate=2e-5, patience=3, save_path=None):
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, optimizer, scheduler)
            val_metrics = self.evaluate(model, self.test_loader)
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f} - Val F1: {val_metrics['f1']:.4f}")
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                if save_path:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(Path(save_path).with_suffix('')))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}
