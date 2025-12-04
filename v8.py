import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ----------------------------
#  Config / hyperparameters
# ----------------------------
PRETRAINED_MODEL = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 8
LR_BACKBONE = 1e-5
LR_CLASSIFIER = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "dataset/MN-DS-news-classification.csv"
GRAD_CLIP = 1.0  # Gradient clipping

# ----------------------------
#  Seed
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
#  Dataset + DataLoader
# ----------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels_lvl1, labels_lvl2, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels_lvl1 = labels_lvl1
        self.labels_lvl2 = labels_lvl2
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_lvl1": torch.tensor(self.labels_lvl1[idx], dtype=torch.long),
            "label_lvl2": torch.tensor(self.labels_lvl2[idx], dtype=torch.long)
        }

# ----------------------------
#  Model
# ----------------------------
class HierarchicalRoberta(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, pretrained_model=PRETRAINED_MODEL, dropout_prob=0.3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier_lvl1 = nn.Linear(hidden_size, num_labels_lvl1)
        self.classifier_lvl2 = nn.Linear(hidden_size, num_labels_lvl2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits1 = self.classifier_lvl1(pooled)
        logits2 = self.classifier_lvl2(pooled)
        return logits1, logits2

# ----------------------------
#  Load + preprocess data
# ----------------------------
df = pd.read_csv(CSV_PATH)
df["text_full"] = df["title"].fillna("") + " " + df["content"].fillna("")

le1 = LabelEncoder()
df["label_lvl1"] = le1.fit_transform(df["category_level_1"])
le2 = LabelEncoder()
df["label_lvl2"] = le2.fit_transform(df["category_level_2"])

X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    df["text_full"], df["label_lvl1"], df["label_lvl2"],
    test_size=0.2,
    random_state=SEED,
    stratify=df["label_lvl1"]
)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
train_ds = NewsDataset(X_train.tolist(), y1_train.tolist(), y2_train.tolist(), tokenizer)
val_ds = NewsDataset(X_val.tolist(), y1_val.tolist(), y2_val.tolist(), tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
#  Class weights for imbalance
# ----------------------------
counts_lvl1 = np.bincount(y1_train)
counts_lvl2 = np.bincount(y2_train)
weight_lvl1 = torch.tensor(1.0 / (counts_lvl1 + 1e-6), dtype=torch.float).to(DEVICE)
weight_lvl2 = torch.tensor(1.0 / (counts_lvl2 + 1e-6), dtype=torch.float).to(DEVICE)

criterion_lvl1 = nn.CrossEntropyLoss(weight=weight_lvl1)
criterion_lvl2 = nn.CrossEntropyLoss(weight=weight_lvl2)

# ----------------------------
#  Initialize model + optimizer + scheduler
# ----------------------------
model = HierarchicalRoberta(
    num_labels_lvl1=len(le1.classes_),
    num_labels_lvl2=len(le2.classes_)
).to(DEVICE)

# Separate LRs for backbone vs classifiers
optimizer = AdamW([
    {"params": model.roberta.parameters(), "lr": LR_BACKBONE},
    {"params": model.classifier_lvl1.parameters(), "lr": LR_CLASSIFIER},
    {"params": model.classifier_lvl2.parameters(), "lr": LR_CLASSIFIER},
], weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# ----------------------------
#  Training + evaluation loop
# ----------------------------
best_f1w2 = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels1 = batch["label_lvl1"].to(DEVICE)
        labels2 = batch["label_lvl2"].to(DEVICE)

        logits1, logits2 = model(input_ids, attention_mask)
        loss1 = criterion_lvl1(logits1, labels1)
        loss2 = criterion_lvl2(logits2, labels2)
        loss = loss1 + loss2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} — avg training loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    all_p1, all_l1, all_p2, all_l2 = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels1 = batch["label_lvl1"].to(DEVICE)
            labels2 = batch["label_lvl2"].to(DEVICE)

            logits1, logits2 = model(input_ids, attention_mask)
            preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
            preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
            all_p1.extend(preds1)
            all_l1.extend(labels1.cpu().numpy())
            all_p2.extend(preds2)
            all_l2.extend(labels2.cpu().numpy())

    acc1 = accuracy_score(all_l1, all_p1)
    f1w1 = f1_score(all_l1, all_p1, average='weighted')
    f1m1 = f1_score(all_l1, all_p1, average='macro')
    acc2 = accuracy_score(all_l2, all_p2)
    f1w2 = f1_score(all_l2, all_p2, average='weighted')
    f1m2 = f1_score(all_l2, all_p2, average='macro')
    print(f"Validation — Level1: acc {acc1:.4f}, F1w {f1w1:.4f}, F1m {f1m1:.4f}")
    print(f"Validation — Level2: acc {acc2:.4f}, F1w {f1w2:.4f}, F1m {f1m2:.4f}")

    # Save best model based on Level2 weighted F1
    if f1w2 > best_f1w2:
        best_f1w2 = f1w2
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved best model at epoch {epoch+1} with Level2 weighted F1: {best_f1w2:.4f}")
