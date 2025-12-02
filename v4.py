# ============================
# Full advanced training script
# ============================
import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
import pandas as pd
import numpy as np
import gc
import random
from tqdm.auto import tqdm

# --------------------
# Config
# --------------------
MODEL_NAME = "bert-base-uncased"           # or "distilbert-base-uncased" for faster runs (MN-DS baseline)
MAX_LEN = 384
BATCH_SIZE = 8
GRAD_ACCUM = 4                             # effective batch size 32
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
FREEZE_LAYERS_UNTIL = 5                    # -1 to disable freezing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 3                               # early stopping patience (val F1)
SWA_START = int(0.75 * (EPOCHS))           # when to start SWA (fraction of epochs)
SAVE_PATH = "model_advanced.pth"

SEED = 42
def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_all()

# --------------------
# Load data + tokenizer
# --------------------
df = pd.read_csv("./dataset/MN-DS-news-classification.csv")
df["text"] = df["title"].fillna("") + " " + df["content"].fillna("")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

le1 = LabelEncoder()
df["label_lvl1"] = le1.fit_transform(df["category_level_1"])
le2 = LabelEncoder()
df["label_lvl2"] = le2.fit_transform(df["category_level_2"])

X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    df["text"], df["label_lvl1"], df["label_lvl2"],
    test_size=0.2, random_state=42, stratify=df["label_lvl1"]
)

# --------------------
# Dataset + dynamic collate
# --------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels1, labels2):
        self.texts = texts
        self.labels1 = labels1
        self.labels2 = labels2
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return {"text": str(self.texts.iloc[idx]), "labels1": int(self.labels1.iloc[idx]), "labels2": int(self.labels2.iloc[idx])}

def collate_fn(batch):
    texts = [b["text"] for b in batch]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    labels1 = torch.tensor([b["labels1"] for b in batch], dtype=torch.long)
    labels2 = torch.tensor([b["labels2"] for b in batch], dtype=torch.long)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels1": labels1, "labels2": labels2}

train_ds = NewsDataset(X_train.reset_index(drop=True), y1_train.reset_index(drop=True), y2_train.reset_index(drop=True))
val_ds   = NewsDataset(X_val.reset_index(drop=True),   y1_val.reset_index(drop=True),   y2_val.reset_index(drop=True))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --------------------
# Model with multi-sample dropout + optional R-Drop support
# --------------------
class AdvancedHierarchicalModel(nn.Module):
    def __init__(self, model_name, num_labels1, num_labels2, ms_dropout_n=5, dropout_p=0.3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.ms_dropout_n = ms_dropout_n
        self.classifier1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, num_labels1)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, num_labels2)
        )
        # small helper dropout modules for multi-sample dropout
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(ms_dropout_n)])

        # optionally freeze lower layers
        if FREEZE_LAYERS_UNTIL >= 0:
            try:
                # Works for BERT-like "encoder.layer.X" naming
                for name, param in self.backbone.named_parameters():
                    if "encoder.layer" in name:
                        layer_num = int(name.split(".")[2])
                        if layer_num <= FREEZE_LAYERS_UNTIL:
                            param.requires_grad = False
            except Exception:
                pass

    def forward(self, input_ids, attention_mask, return_all=False):
        # return_all True -> return list of logits for R-Drop / multi-sample purposes
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # use pooled_output if available or [CLS] token
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]

        logits1_samples = []
        logits2_samples = []
        # multi-sample dropout: run several dropout masks and average logits
        for d in self.dropouts:
            f = d(feat)
            logits1_samples.append(self.classifier1(f))
            logits2_samples.append(self.classifier2(f))

        # stack sample logits: (n_samples, batch, classes)
        l1 = torch.stack(logits1_samples, dim=0)
        l2 = torch.stack(logits2_samples, dim=0)

        # average across dropout samples
        logits1 = l1.mean(dim=0)
        logits2 = l2.mean(dim=0)

        if return_all:
            return logits1, logits2, l1, l2
        return logits1, logits2

# --------------------
# Adversarial FGM helper
# --------------------
class FGM:
    def __init__(self, model, emb_name='embeddings', epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        # add perturbation to embeddings
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if param.grad is None:
                    continue
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / (norm + 1e-12)
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# --------------------
# Losses: CE + R-Drop KL
# --------------------
ce_loss = nn.CrossEntropyLoss()
def rdrop_kl_loss(p_logits, q_logits, mask=None):
    # p_logits, q_logits shape: (batch, classes)
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    kl = F.kl_div(p, q.exp(), reduction='batchmean') + F.kl_div(q, p.exp(), reduction='batchmean')
    return 0.5 * kl

# --------------------
# Build model, optimizer, scheduler, AMP, SWA
# --------------------
model = AdvancedHierarchicalModel(MODEL_NAME, num_labels1=len(le1.classes_), num_labels2=len(le2.classes_)).to(DEVICE)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps = int(math.ceil(len(train_loader) / 1) * EPOCHS / 1)  # we will step scheduler per optimizer.step()
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

scaler = GradScaler()
fgm = FGM(model, emb_name="embeddings", epsilon=0.8)   # epsilon tuned small

use_rdrop = True   # apply R-Drop consistency regularization
rdrop_alpha = 5.0  # weight for R-Drop KL loss

use_swa = True
swa_model = AveragedModel(model) if use_swa else None
swa_start_step = int(SWA_START * len(train_loader)) if use_swa else None

# --------------------
# Train / eval helper
# --------------------
def evaluate(model, loader):
    model.eval()
    preds1, labs1, preds2, labs2 = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels1 = batch['labels1'].to(DEVICE)
            labels2 = batch['labels2'].to(DEVICE)
            logits1, logits2 = model(input_ids, attention_mask)
            preds1.extend(torch.argmax(logits1, dim=1).cpu().numpy())
            preds2.extend(torch.argmax(logits2, dim=1).cpu().numpy())
            labs1.extend(labels1.cpu().numpy())
            labs2.extend(labels2.cpu().numpy())
    f1_1 = f1_score(labs1, preds1, average='weighted')
    f1_2 = f1_score(labs2, preds2, average='weighted')
    return f1_1, f1_2

# --------------------
# Training loop with all bells
# --------------------
best_val = -1.0
best_state = None
no_improve = 0
global_step = 0

print("Start training with advanced techniques...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels1 = batch['labels1'].to(DEVICE)
        labels2 = batch['labels2'].to(DEVICE)

        with autocast():
            if use_rdrop:
                # two forward passes with dropout active (return_all to get per-sample logits)
                logits1_a, logits2_a, _, _ = model(input_ids, attention_mask, return_all=True)
                logits1_b, logits2_b, _, _ = model(input_ids, attention_mask, return_all=True)

                # average (we already have averaged logits in model's default output if not return_all)
                # here compute CE on each and R-Drop KL between the two
                ce_a = ce_loss(logits1_a.mean(dim=0) if logits1_a.dim()==3 else logits1_a, labels1) + \
                       ce_loss(logits2_a.mean(dim=0) if logits2_a.dim()==3 else logits2_a, labels2)
                ce_b = ce_loss(logits1_b.mean(dim=0) if logits1_b.dim()==3 else logits1_b, labels1) + \
                       ce_loss(logits2_b.mean(dim=0) if logits2_b.dim()==3 else logits2_b, labels2)
                ce = 0.5 * (ce_a + ce_b)
                # R-Drop KL on the averaged logits per head
                kl1 = rdrop_kl_loss(logits1_a.mean(dim=0), logits1_b.mean(dim=0))
                kl2 = rdrop_kl_loss(logits2_a.mean(dim=0), logits2_b.mean(dim=0))
                loss = ce + rdrop_alpha * 0.5 * (kl1 + kl2)
            else:
                logits1, logits2 = model(input_ids, attention_mask)
                loss = ce_loss(logits1, labels1) + ce_loss(logits2, labels2)

            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        # FGM adversarial step (embedding perturbation)
        fgm.attack()
        with autocast():
            # forward again for adversarial loss
            logits1_adv, logits2_adv = model(input_ids, attention_mask)
            loss_adv = ce_loss(logits1_adv, labels1) + ce_loss(logits2_adv, labels2)
            loss_adv = (loss_adv / GRAD_ACCUM)
        scaler.scale(loss_adv).backward()
        fgm.restore()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # SWA update
            if use_swa and global_step >= swa_start_step:
                swa_model.update_parameters(model)

        running_loss += loss.item() * GRAD_ACCUM
        if (step + 1) % 50 == 0:
            pbar.set_postfix({'avg_loss': running_loss / (step + 1)})

    # end epoch: evaluate on val
    val_f1_1, val_f1_2 = evaluate(model, val_loader)
    val_f1 = 0.5 * (val_f1_1 + val_f1_2)   # combined metric for early stopping
    print(f"\nEpoch {epoch+1} validation F1s -> lvl1: {val_f1_1:.4f}, lvl2: {val_f1_2:.4f}, combined: {val_f1:.4f}")

    if val_f1 > best_val:
        best_val = val_f1
        best_state = copy.deepcopy(model.state_dict())
        torch.save({'model_state_dict': best_state, 'le_lvl1': le1, 'le_lvl2': le2, 'tokenizer': tokenizer}, SAVE_PATH)
        no_improve = 0
        print(f"Saved best model (combined val F1={best_val:.4f}) to {SAVE_PATH}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, early stopping.")
            break

    gc.collect()

# Finalize SWA (if used)
if use_swa:
    print("Applying SWA params and updating BN (if any)...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
    # save swa model
    swa_state = swa_model.module.state_dict() if hasattr(swa_model, "module") else swa_model.state_dict()
    torch.save({'model_state_dict': swa_state, 'le_lvl1': le1, 'le_lvl2': le2, 'tokenizer': tokenizer}, "model_swa.pth")
    print("Saved model_swa.pth")

print("Training finished.")
