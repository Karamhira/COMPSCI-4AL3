import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import os

# ==========================
# CONFIG
# ==========================
CONFIG = {
    "model_name": "microsoft/deberta-v3-large",
    "max_length": 256,
    "batch_size": 8,  
    "lr": 2e-5,
    "epochs": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "accumulation_steps": 4,
    "num_folds": 3,
    "seed": 42
}

# ==========================
# REPRODUCIBILITY
# ==========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# ==========================
# DATASET
# ==========================
class TextDataset(Dataset):
    def __init__(self, texts, labels1, labels2, tokenizer, max_length):
        self.texts = texts
        self.labels1 = labels1
        self.labels2 = labels2
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            "input_ids": enc['input_ids'].squeeze(0),
            "attention_mask": enc['attention_mask'].squeeze(0),
            "label1": torch.tensor(self.labels1[idx], dtype=torch.long),
            "label2": torch.tensor(self.labels2[idx], dtype=torch.long)
        }

# ==========================
# MODEL
# ==========================
class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, num_classes1, num_classes2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier1 = nn.Linear(hidden_size, num_classes1)
        self.classifier2 = nn.Linear(hidden_size, num_classes2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = outputs.last_hidden_state[:, 0]  # CLS token
        x = self.dropout(x)
        logits1 = self.classifier1(x)
        logits2 = self.classifier2(x)
        return logits1, logits2

# ==========================
# TRAIN & EVAL
# ==========================
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels1 = batch['label1'].to(device)
        labels2 = batch['label2'].to(device)

        with torch.amp.autocast():
            logits1, logits2 = model(input_ids, attention_mask)
            loss1 = nn.CrossEntropyLoss()(logits1, labels1)
            loss2 = nn.CrossEntropyLoss()(logits2, labels2)
            loss = (loss1 + loss2) / accumulation_steps

        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds1, preds2, truths1, truths2 = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels1 = batch['label1'].to(device)
            labels2 = batch['label2'].to(device)

            logits1, logits2 = model(input_ids, attention_mask)
            preds1.extend(torch.argmax(logits1, dim=1).cpu().numpy())
            preds2.extend(torch.argmax(logits2, dim=1).cpu().numpy())
            truths1.extend(labels1.cpu().numpy())
            truths2.extend(labels2.cpu().numpy())

    acc1 = accuracy_score(truths1, preds1)
    acc2 = accuracy_score(truths2, preds2)
    return acc1, acc2

# ==========================
# MAIN TRAIN LOOP WITH K-FOLD
# ==========================
def main():
    df = pd.read_csv("your_dataset.csv")  # columns: text, category_level_1, category_level_2
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    df['label1'] = le1.fit_transform(df['category_level_1'])
    df['label2'] = le2.fit_transform(df['category_level_2'])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    skf = StratifiedKFold(n_splits=CONFIG['num_folds'], shuffle=True, random_state=CONFIG['seed'])
    fold = 0
    all_fold_acc1, all_fold_acc2 = [], []

    for train_idx, val_idx in skf.split(df['text'], df['label1']):
        fold += 1
        print(f"\n=== Fold {fold} ===")
        train_texts, val_texts = df['text'].iloc[train_idx], df['text'].iloc[val_idx]
        train_labels1, val_labels1 = df['label1'].iloc[train_idx], df['label1'].iloc[val_idx]
        train_labels2, val_labels2 = df['label2'].iloc[train_idx], df['label2'].iloc[val_idx]

        train_dataset = TextDataset(train_texts.tolist(), train_labels1.tolist(), train_labels2.tolist(),
                                    tokenizer, CONFIG['max_length'])
        val_dataset = TextDataset(val_texts.tolist(), val_labels1.tolist(), val_labels2.tolist(),
                                  tokenizer, CONFIG['max_length'])
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

        model = MultiTaskClassifier(CONFIG['model_name'], num_classes1=len(le1.classes_),
                                    num_classes2=len(le2.classes_)).to(CONFIG['device'])
        optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
        total_steps = len(train_loader) * CONFIG['epochs']
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps),
                                                    num_training_steps=total_steps)
        scaler = torch.amp.GradScaler(device="cuda")

        best_acc1 = 0
        for epoch in range(CONFIG['epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                         CONFIG['device'], scaler, CONFIG['accumulation_steps'])
            val_acc1, val_acc2 = evaluate(model, val_loader, CONFIG['device'])
            print(f"Train Loss: {train_loss:.4f} | VAL ACC L1: {val_acc1:.4f} | VAL ACC L2: {val_acc2:.4f}")

            # Save best fold model
            if val_acc1 > best_acc1:
                best_acc1 = val_acc1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'le1': le1,
                    'le2': le2,
                    'tokenizer': tokenizer
                }, f"best_multitask_model_fold{fold}.pth")
        all_fold_acc1.append(best_acc1)
        all_fold_acc2.append(val_acc2)

    print(f"\nAverage L1 ACC: {np.mean(all_fold_acc1):.4f} | Average L2 ACC: {np.mean(all_fold_acc2):.4f}")

if __name__ == "__main__":
    main()
