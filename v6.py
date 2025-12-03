#!/usr/bin/env python3
"""
Grandmaster ModernBERT (Local) — Pure PyTorch, no HuggingFace.

- Local tokenizer + vocab
- Learned positional embeddings
- TransformerEncoder (PyTorch)
- Two-head hierarchical classifier
- Mixed precision training (torch.amp)
- Safe max-pooling (computed in float32)
- Weighted sampling / focal loss optional
- Save/load local models & vocab
"""

import os
import json
import math
import random
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import amp
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# -----------------------
# CONFIG
# -----------------------
CONFIG = {
    "data_csv": "./dataset/MN-DS-news-classification.csv",
    "text_columns": ["title", "content"],
    "lvl1_col": "category_level_1",
    "lvl2_col": "category_level_2",
    "output_dir": "./modernbert_local_models",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # tokenizer / vocab
    "vocab_size": 50000,
    "min_freq": 2,
    "max_len": 256,

    # model
    "emb_dim": 256,
    "n_layers": 4,
    "n_heads": 8,
    "ff_dim": 512,
    "dropout": 0.1,

    # training
    "batch_size": 16,
    "accumulation_steps": 2,
    "num_epochs": 6,
    "lr": 2e-4,
    "weight_decay": 1e-6,
    "warmup_steps_ratio": 0.05,
    "use_focal_loss": False,
    "focal_gamma": 2.0,
    "seeds": [42, 43, 44],
    "num_workers": 4,

    # saving
    "save_dir": "./modernbert_local_models",
    "vocab_path": "./modernbert_local_models/vocab.json"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# -----------------------
# Utilities: tokenizer & vocab
# -----------------------
def simple_tokenize(text: str) -> List[str]:
    # lightweight whitespace + punctuation split
    text = str(text).lower()
    tokens = []
    token = []
    for ch in text:
        if ch.isalnum():
            token.append(ch)
        else:
            if token:
                tokens.append(''.join(token))
                token = []
            if ch.strip():
                tokens.append(ch)
    if token:
        tokens.append(''.join(token))
    return [t for t in tokens if t.strip()]

def build_vocab(texts: List[str], cfg: Dict) -> Dict[str,int]:
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= cfg["min_freq"]]
    items.sort(key=lambda x: x[1], reverse=True)
    topk = items[:cfg["vocab_size"]]
    vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
    idx = len(vocab)
    for tok, _ in topk:
        if tok in vocab: continue
        vocab[tok] = idx; idx += 1
    return vocab

def text_to_ids(text: str, vocab: Dict[str,int], max_len: int) -> List[int]:
    toks = simple_tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in toks][:max_len-2]
    # add cls/sep
    ids = [vocab["<CLS>"]] + ids + [vocab["<SEP>"]]
    if len(ids) < max_len:
        # padding handled later in collate
        return ids
    return ids[:max_len]

# -----------------------
# Dataset + collate
# -----------------------
class NewsLocalDataset(Dataset):
    def __init__(self, texts: List[str], lvl1: List[int], lvl2: List[int], vocab: Dict[str,int], max_len: int):
        self.texts = texts
        self.lvl1 = lvl1
        self.lvl2 = lvl2
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        ids = text_to_ids(t, self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "lvl1": torch.tensor(self.lvl1[idx], dtype=torch.long),
            "lvl2": torch.tensor(self.lvl2[idx], dtype=torch.long),
            "length": len(ids)
        }

def collate_batch(batch, pad_id=0, max_len=None):
    lengths = [b["length"] for b in batch]
    max_l = max(lengths) if max_len is None else min(max(lengths), max_len)
    padded = []
    for b in batch:
        ids = b["input_ids"][:max_l].tolist()
        if len(ids) < max_l:
            ids = ids + [pad_id] * (max_l - len(ids))
        padded.append(ids)
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()
    lvl1 = torch.stack([b["lvl1"] for b in batch])
    lvl2 = torch.stack([b["lvl2"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "lvl1": lvl1, "lvl2": lvl2}

# -----------------------
# Model: local "ModernBERT" style (learned pos emb + TransformerEncoder)
# -----------------------
class ModernBERTLocal(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int, n_layers:int, n_heads:int, ff_dim:int, dropout:float, max_len:int, num_lvl1:int, num_lvl2:int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.layernorm = nn.LayerNorm(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        hidden = emb_dim
        # pooling projection if needed
        self.proj = nn.Linear(hidden, hidden)
        self.head1 = nn.Linear(hidden, num_lvl1)
        self.head2 = nn.Linear(hidden, num_lvl2)

        # init
        nn.init.xavier_uniform_(self.head1.weight)
        nn.init.xavier_uniform_(self.head2.weight)
        if self.head1.bias is not None:
            nn.init.zeros_(self.head1.bias)
        if self.head2.bias is not None:
            nn.init.zeros_(self.head2.bias)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, L)
        B, L = input_ids.shape
        positions = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.layernorm(x)

        # prepare transformer mask: True for positions to attend (0 = pad)
        if attention_mask is None:
            attn_mask = (input_ids != 0).long()
        else:
            attn_mask = attention_mask

        # Transformer wants src_key_padding_mask with True where padding -> mask out
        src_key_padding_mask = (attn_mask == 0)  # (B, L) bool

        # encoder output (B, L, emb)
        enc_out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # pooling: mean pool + safe max pool (Fix A)
        mask = attn_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (enc_out * mask).sum(1)        # (B, emb)
        denom = mask.sum(1).clamp(min=1e-9)
        mean_pool = summed / denom              # (B, emb)

        # safe max in float32
        enc_fp32 = enc_out.float()
        mask_fp32 = mask.float()
        masked = enc_fp32.masked_fill(mask_fp32 == 0, -1e9)
        max_pool, _ = masked.max(1)             # (B, emb)
        max_pool = max_pool.to(enc_out.dtype)

        # choose pooling strategy (mean by default)
        pooled = mean_pool
        # pooled = torch.cat([mean_pool, max_pool], dim=1)  # if you want concat, update proj/head dims

        feat = torch.tanh(self.proj(self.dropout(pooled)))
        logits1 = self.head1(feat)
        logits2 = self.head2(feat)
        return logits1, logits2

# -----------------------
# Losses
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# -----------------------
# Sampler builder
# -----------------------
def build_sampler(df_train: pd.DataFrame, label_col: str, device: str):
    counts = df_train[label_col].value_counts()
    class_weights = 1.0 / counts
    sample_weights = df_train[label_col].map(class_weights).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # class weight vector
    max_label = int(df_train[label_col].max())
    class_weight_tensor = torch.ones(max_label + 1, dtype=torch.float, device=device)
    for lbl, w in class_weights.items():
        class_weight_tensor[int(lbl)] = float(w)
    return sampler, class_weight_tensor

# -----------------------
# Eval
# -----------------------
def evaluate(model: ModernBERTLocal, dataloader: DataLoader, device: str):
    model.eval()
    preds1, preds2, trues1, trues2 = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            l1 = batch["lvl1"].to(device)
            l2 = batch["lvl2"].to(device)
            logits1, logits2 = model(ids, mask)
            p1 = torch.argmax(logits1, dim=1)
            p2 = torch.argmax(logits2, dim=1)
            preds1.extend(p1.cpu().tolist())
            preds2.extend(p2.cpu().tolist())
            trues1.extend(l1.cpu().tolist())
            trues2.extend(l2.cpu().tolist())
    return {
        "lvl1_acc": accuracy_score(trues1, preds1),
        "lvl2_acc": accuracy_score(trues2, preds2),
        "lvl1_macro_f1": f1_score(trues1, preds1, average="macro"),
        "lvl2_macro_f1": f1_score(trues2, preds2, average="macro")
    }

# -----------------------
# Train one seed
# -----------------------
def train_one_seed(seed: int, cfg: Dict, df: pd.DataFrame, le1: LabelEncoder, le2: LabelEncoder, vocab: Dict[str,int]):
    # reproducibility
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = cfg["device"]

    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=seed)

    train_ds = NewsLocalDataset(train_df["text"].tolist(), train_df["lvl1_id"].tolist(), train_df["lvl2_id"].tolist(), vocab, cfg["max_len"])
    val_ds = NewsLocalDataset(val_df["text"].tolist(), val_df["lvl1_id"].tolist(), val_df["lvl2_id"].tolist(), vocab, cfg["max_len"])

    sampler, class_weight_tensor = build_sampler(train_df, "lvl2_id", device)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                              collate_fn=lambda b: collate_batch(b, pad_id=vocab["<PAD>"], max_len=cfg["max_len"]),
                              num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=lambda b: collate_batch(b, pad_id=vocab["<PAD>"], max_len=cfg["max_len"]),
                             num_workers=cfg["num_workers"])

    num_lvl1 = len(le1.classes_); num_lvl2 = len(le2.classes_)
    model = ModernBERTLocal(len(vocab), cfg["emb_dim"], cfg["n_layers"], cfg["n_heads"], cfg["ff_dim"], cfg["dropout"], cfg["max_len"], num_lvl1, num_lvl2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = math.ceil(len(train_loader) * cfg["num_epochs"] / cfg["accumulation_steps"])
    warmup_steps = max(1, int(cfg["warmup_steps_ratio"] * total_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], total_steps=max(1,total_steps), pct_start=warmup_steps/max(1,total_steps), anneal_strategy="cos", final_div_factor=100.0)

    # losses
    if cfg["use_focal_loss"]:
        loss_lvl2 = FocalLoss(cfg["focal_gamma"], weight=class_weight_tensor.to(device))
    else:
        loss_lvl2 = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
    # lvl1 weights (balance by freq)
    lvl1_counts = train_df["lvl1_id"].value_counts()
    lvl1_weights = torch.tensor([1.0 / lvl1_counts.get(i, 1.0) for i in range(num_lvl1)], dtype=torch.float, device=device)
    loss_lvl1 = nn.CrossEntropyLoss(weight=lvl1_weights)

    # GradScaler (safe default)
    scaler = GradScaler()

    best_val = -1.0
    model_dir = os.path.join(cfg["save_dir"], f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"

    for epoch in range(cfg["num_epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Seed {seed} Epoch {epoch+1}")
        for step, batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y1 = batch["lvl1"].to(device)
            y2 = batch["lvl2"].to(device)

            with autocast(device_type=autocast_device):
                logits1, logits2 = model(ids, mask)
                l1 = loss_lvl1(logits1, y1)
                l2 = loss_lvl2(logits2, y2)
                loss = 0.5 * l1 + 1.0 * l2

            scaler.scale(loss / cfg["accumulation_steps"]).backward()
            running_loss += loss.item()

            if (step + 1) % cfg["accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                try:
                    scheduler.step()
                except Exception:
                    pass

            if (step + 1) % 200 == 0:
                pbar.set_postfix({"loss": running_loss / (step + 1)})

        # validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"[Seed {seed}] Epoch {epoch+1} Validation: lvl1_acc={val_metrics['lvl1_acc']:.4f} lvl2_acc={val_metrics['lvl2_acc']:.4f} lvl2_macro_f1={val_metrics['lvl2_macro_f1']:.4f}")

        if val_metrics["lvl2_macro_f1"] > best_val:
            best_val = val_metrics["lvl2_macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "vocab": vocab
            }, os.path.join(model_dir, "best_model.pt"))
            with open(os.path.join(model_dir, "meta.json"), "w") as f:
                json.dump({"seed": seed, "num_lvl1": num_lvl1, "num_lvl2": num_lvl2}, f)

    # save vocab for this seed (stable across seeds)
    with open(os.path.join(model_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    return model_dir

# -----------------------
# Load & ensemble
# -----------------------
def load_local_model(model_dir: str, cfg: Dict):
    with open(os.path.join(model_dir, "vocab.json"), "r") as f:
        vocab = json.load(f)
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    model = ModernBERTLocal(len(vocab), cfg["emb_dim"], cfg["n_layers"], cfg["n_heads"], cfg["ff_dim"], cfg["dropout"], cfg["max_len"], meta["num_lvl1"], meta["num_lvl2"])
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=cfg["device"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg["device"])
    model.eval()
    return model, vocab

def ensemble_predict(texts: List[str], model_dirs: List[str], cfg: Dict, le1: LabelEncoder, le2: LabelEncoder):
    device = cfg["device"]
    models = []
    vocabs = []
    for d in model_dirs:
        m, v = load_local_model(d, cfg)
        models.append(m); vocabs.append(v)
    vocab = vocabs[0]
    all_ids = [text_to_ids(t, vocab, cfg["max_len"]) for t in texts]
    pad = vocab.get("<PAD>", 0)
    maxlen = max(len(x) for x in all_ids)
    padded = [x + [pad]*(maxlen - len(x)) for x in all_ids]
    input_ids = torch.tensor(padded, dtype=torch.long).to(device)

    logits1_list, logits2_list = [], []
    with torch.no_grad():
        for m in models:
            l1, l2 = m(input_ids, (input_ids!=pad).long().to(device))
            logits1_list.append(l1.cpu().numpy())
            logits2_list.append(l2.cpu().numpy())

    avg1 = np.mean(np.stack(logits1_list, axis=0), axis=0)
    avg2 = np.mean(np.stack(logits2_list, axis=0), axis=0)
    preds1 = np.argmax(avg1, axis=1)
    preds2 = np.argmax(avg2, axis=1)
    return le1.inverse_transform(preds1), le2.inverse_transform(preds2)

# -----------------------
# Run pipeline
# -----------------------
def run_full_pipeline(cfg: Dict):
    df = pd.read_csv(cfg["data_csv"])
    df["text"] = df[cfg["text_columns"]].fillna("").agg(" ".join, axis=1)
    df = df.dropna(subset=["text", cfg["lvl1_col"], cfg["lvl2_col"]]).reset_index(drop=True)

    le1 = LabelEncoder(); le2 = LabelEncoder()
    df["lvl1_id"] = le1.fit_transform(df[cfg["lvl1_col"]])
    df["lvl2_id"] = le2.fit_transform(df[cfg["lvl2_col"]])

    print("[INFO] Loaded dataset:", len(df), "rows")
    print("[INFO] Level-1 classes:", len(le1.classes_), "Level-2 classes:", len(le2.classes_))

    print("[INFO] Building vocab...")
    vocab = build_vocab(df["text"].tolist(), cfg)
    with open(cfg["vocab_path"], "w") as f:
        json.dump(vocab, f)
    print("[INFO] Vocab size:", len(vocab))

    model_dirs = []
    for seed in cfg["seeds"]:
        print(f"\n===== TRAINING SEED {seed} =====")
        md = train_one_seed(seed, cfg, df, le1, le2, vocab)
        model_dirs.append(md)

    # quick ensemble sanity check
    _, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=cfg["seeds"][0])
    sample_texts = val_df["text"].tolist()[:256]
    lvl1_preds, lvl2_preds = ensemble_predict(sample_texts, model_dirs, cfg, le1, le2)
    print("[INFO] Ensemble sample outputs:")
    for i in range(min(10, len(lvl2_preds))):
        print(f" L1: {lvl1_preds[i]}  L2: {lvl2_preds[i]}")

    print("[INFO] All done. Models saved to:", cfg["save_dir"])

# -----------------------
# Entry
# -----------------------
if __name__ == "__main__":
    run_full_pipeline(CONFIG)
