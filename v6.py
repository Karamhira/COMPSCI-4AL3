#!/usr/bin/env python3
"""
Grandmaster (Local ModernBERT replacement) — Pure PyTorch, NO HuggingFace.

Key features:
- Local whitespace tokenizer + vocab (no HF)
- Trainable Embedding + BiLSTM encoder
- Two-head classifier for hierarchical labels
- Weighted sampling, optional focal loss
- Mixed precision + grad accumulation
- Multi-seed training + simple ensemble
- Save/Load tokenizer (vocab) + models
"""

import os
import random
import math
import json
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler

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
    "output_dir": "./grandmaster_local_models",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "seeds": [42, 43, 44],
    "max_len": 200,
    "vocab_size": 60000,          # top-k words kept
    "min_freq": 2,                # min freq for token to be included
    "embedding_dim": 300,
    "encoder_hidden": 256,
    "encoder_layers": 1,
    "dropout": 0.2,
    "batch_size": 16,
    "accumulation_steps": 2,
    "num_epochs": 6,
    "lr": 2e-4,
    "weight_decay": 1e-6,
    "warmup_proportion": 0.05,
    "freeze_epochs": 1,
    "use_focal_loss": False,
    "focal_gamma": 2.0,
    "num_workers": 4,
    # Optional: path to pretrained embedding .npy and vocab mapping .json (local)
    "pretrained_embedding_path": None,
    "pretrained_embedding_vocab": None,
    # Tokenizer/saving
    "vocab_path": "./grandmaster_local_models/vocab.json",
    "save_dir": "./grandmaster_local_models",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def simple_tokenize(text: str) -> List[str]:
    # basic whitespace + punctuation separation (lightweight)
    # you can replace with a better local tokenizer if desired
    text = str(text).lower()
    # keep punctuation as separate tokens
    chars = []
    token = []
    for ch in text:
        if ch.isalnum():
            token.append(ch)
        else:
            if token:
                chars.append(''.join(token))
                token = []
            if ch.strip():
                chars.append(ch)
    if token:
        chars.append(''.join(token))
    # filter spaces tokens
    tokens = [t for t in chars if t.strip()]
    return tokens

# -----------------------
# Build vocab from dataset
# -----------------------
def build_vocab(texts: List[str], cfg: Dict):
    counter = Counter()
    for t in texts:
        tokens = simple_tokenize(t)
        counter.update(tokens)
    # filter by min_freq and top-k
    tokens_and_freq = [(tok, freq) for tok, freq in counter.items() if freq >= cfg["min_freq"]]
    tokens_and_freq.sort(key=lambda x: x[1], reverse=True)
    topk = tokens_and_freq[:cfg["vocab_size"]]
    # special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for tok, _ in topk:
        vocab[tok] = idx
        idx += 1
    return vocab

def text_to_ids(text: str, vocab: Dict[str,int], max_len: int):
    tokens = simple_tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens][:max_len]
    # padding handled in collate
    return ids

# -----------------------
# Dataset
# -----------------------
class NewsLocalDataset(Dataset):
    def __init__(self, texts: List[str], lvl1_ids: List[int], lvl2_ids: List[int], vocab: Dict[str,int], max_len: int):
        self.texts = texts
        self.lvl1 = lvl1_ids
        self.lvl2 = lvl2_ids
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = text_to_ids(text, self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "lvl1": torch.tensor(self.lvl1[idx], dtype=torch.long),
            "lvl2": torch.tensor(self.lvl2[idx], dtype=torch.long),
            "length": len(input_ids)
        }

def collate_fn(batch, pad_id=0, max_len=None):
    # batch: list of dicts
    lengths = [b["length"] for b in batch]
    max_l = max(lengths) if max_len is None else min(max(lengths), max_len)
    padded = []
    for b in batch:
        ids = b["input_ids"][:max_l].tolist()
        ids = ids + [pad_id] * (max_l - len(ids))
        padded.append(ids)
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()
    lvl1 = torch.stack([b["lvl1"] for b in batch])
    lvl2 = torch.stack([b["lvl2"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "lvl1": lvl1, "lvl2": lvl2}

# -----------------------
# Model (Embedding + BiLSTM + pooling + heads)
# -----------------------
class HierLocalModel(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int, enc_hidden:int, num_lvl1:int, num_lvl2:int, emb_weights:torch.Tensor=None, dropout=0.2, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if emb_weights is not None:
            try:
                self.embedding.weight.data.copy_(emb_weights)
            except Exception:
                pass
        self.encoder = nn.LSTM(input_size=emb_dim, hidden_size=enc_hidden, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        hidden_size = enc_hidden * 2  # bidirectional
        # pooling projections
        self.proj = nn.Linear(hidden_size, hidden_size)
        # heads
        self.head1 = nn.Linear(hidden_size, num_lvl1)
        self.head2 = nn.Linear(hidden_size, num_lvl2)
        # init
        nn.init.xavier_uniform_(self.head1.weight)
        nn.init.xavier_uniform_(self.head2.weight)
        if self.head1.bias is not None:
            nn.init.zeros_(self.head1.bias)
        if self.head2.bias is not None:
            nn.init.zeros_(self.head2.bias)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, L)
        emb = self.embedding(input_ids)               # (B, L, E)
        outputs, _ = self.encoder(emb)               # (B, L, 2H)
        # attention_mask optional; do mean + max pooling
        mask = (input_ids != 0).unsqueeze(-1).float()  # (B,L,1)
        summed = (outputs * mask).sum(1)              # (B, 2H)
        denom = mask.sum(1).clamp(min=1e-9)
        mean_pool = summed / denom                    # (B,2H)
        max_pool, _ = (outputs.masked_fill(mask==0, -1e9)).max(1)  # (B,2H)
        pooled = torch.cat([mean_pool, max_pool], dim=1) if False else mean_pool  # choose mean or concat
        # project
        feat = torch.tanh(self.proj(self.dropout(pooled)))  # (B, 2H)
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
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)  # (B,)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# -----------------------
# Sampler & weights
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
# Evaluate
# -----------------------
def evaluate(model: HierLocalModel, dataloader: DataLoader, device: str):
    model.eval()
    preds1, preds2, trues1, trues2 = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            l1 = batch["lvl1"].to(device)
            l2 = batch["lvl2"].to(device)
            logits1, logits2 = model(ids)
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
def train_one_seed(seed: int, cfg: Dict, df: pd.DataFrame, le1: LabelEncoder, le2: LabelEncoder, vocab: Dict[str,int], emb_weights: torch.Tensor=None):
    set_seed(seed)
    device = cfg["device"]

    # split
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=seed)

    # datasets
    tokenizer = None  # we use local vocab
    train_ds = NewsLocalDataset(train_df["text"].tolist(), train_df["lvl1_id"].tolist(), train_df["lvl2_id"].tolist(), vocab, cfg["max_len"])
    val_ds = NewsLocalDataset(val_df["text"].tolist(), val_df["lvl1_id"].tolist(), val_df["lvl2_id"].tolist(), vocab, cfg["max_len"])

    sampler, class_weight_tensor = build_sampler(train_df, "lvl2_id", device)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, collate_fn=lambda b: collate_fn(b, pad_id=0, max_len=cfg["max_len"]), num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id=0, max_len=cfg["max_len"]), num_workers=cfg["num_workers"])

    num_lvl1 = len(le1.classes_)
    num_lvl2 = len(le2.classes_)
    model = HierLocalModel(len(vocab), cfg["embedding_dim"], cfg["encoder_hidden"], num_lvl1, num_lvl2, emb_weights, dropout=cfg["dropout"], num_layers=cfg["encoder_layers"]).to(device)

    # optionally freeze embeddings/encoder for first epoch(s)
    if cfg["freeze_epochs"] > 0:
        for name, p in model.named_parameters():
            if "embedding" in name:
                p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = math.ceil(len(train_loader) * cfg["num_epochs"] / cfg["accumulation_steps"])
    warmup_steps = int(cfg["warmup_proportion"] * total_steps)
    # simple scheduler: cosine with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))

    # loss
    if cfg["use_focal_loss"]:
        loss_lvl2 = FocalLoss(cfg["focal_gamma"], weight=class_weight_tensor.to(device))
    else:
        loss_lvl2 = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
    # level1 class weights
    lvl1_counts = train_df["lvl1_id"].value_counts()
    lvl1_weights = torch.tensor([1.0 / lvl1_counts.get(i, 1.0) for i in range(len(le1.classes_))], dtype=torch.float, device=device)
    loss_lvl1 = nn.CrossEntropyLoss(weight=lvl1_weights)

    scaler = GradScaler()
    best_val_f1 = -1.0
    model_dir = os.path.join(cfg["save_dir"], f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(cfg["num_epochs"]):
        model.train()
        if epoch == cfg["freeze_epochs"]:
            # unfreeze embeddings
            for name, p in model.named_parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))

        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Seed {seed} Epoch {epoch+1}")
        for step, batch in pbar:
            ids = batch["input_ids"].to(device)
            y1 = batch["lvl1"].to(device)
            y2 = batch["lvl2"].to(device)

            autocast_device = "cuda" if device.startswith("cuda") else "cpu"
            with amp.autocast(device_type=autocast_device):
                logits1, logits2 = model(ids)
                l1 = loss_lvl1(logits1, y1)
                l2 = loss_lvl2(logits2, y2)
                loss = 0.5 * l1 + 1.0 * l2

            scaler.scale(loss / cfg["accumulation_steps"]).backward()
            running_loss += loss.item()

            if (step + 1) % cfg["accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # scheduler step per optimizer step
                try:
                    scheduler.step()
                except Exception:
                    pass

            if (step + 1) % 200 == 0:
                pbar.set_postfix({"loss": running_loss / (step + 1)})

        # validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"[Seed {seed}] Epoch {epoch+1} Validation: lvl1_acc={val_metrics['lvl1_acc']:.4f} lvl2_acc={val_metrics['lvl2_acc']:.4f} lvl2_macro_f1={val_metrics['lvl2_macro_f1']:.4f}")

        # save best
        if val_metrics["lvl2_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["lvl2_macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "vocab": vocab
            }, os.path.join(model_dir, "best_model.pt"))
            # save simple metadata
            with open(os.path.join(model_dir, "meta.json"), "w") as f:
                json.dump({"seed": seed, "num_lvl1": num_lvl1, "num_lvl2": num_lvl2}, f)

    # save vocab
    with open(os.path.join(model_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    return model_dir

# -----------------------
# Ensemble inference
# -----------------------
def load_local_model(model_dir: str, cfg: Dict):
    # load vocab and model
    with open(os.path.join(model_dir, "vocab.json"), "r") as f:
        vocab = json.load(f)
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    # create model
    model = HierLocalModel(len(vocab), cfg["embedding_dim"], cfg["encoder_hidden"], meta["num_lvl1"], meta["num_lvl2"], emb_weights=None, dropout=cfg["dropout"], num_layers=cfg["encoder_layers"])
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
        models.append(m)
        vocabs.append(v)

    # For simplicity, use first vocab to tokenize (we saved same vocab each seed)
    vocab = vocabs[0]
    # tokenization
    all_ids = [text_to_ids(t, vocab, cfg["max_len"]) for t in texts]
    # pad
    pad_id = vocab.get("<PAD>", 0)
    maxlen = max(len(x) for x in all_ids)
    padded = [x + [pad_id]*(maxlen - len(x)) for x in all_ids]
    input_ids = torch.tensor(padded, dtype=torch.long).to(device)

    logits1_list = []
    logits2_list = []
    with torch.no_grad():
        for model in models:
            l1, l2 = model(input_ids)
            logits1_list.append(l1.cpu().numpy())
            logits2_list.append(l2.cpu().numpy())

    avg1 = np.mean(np.stack(logits1_list, axis=0), axis=0)
    avg2 = np.mean(np.stack(logits2_list, axis=0), axis=0)
    preds1 = np.argmax(avg1, axis=1)
    preds2 = np.argmax(avg2, axis=1)
    return le1.inverse_transform(preds1), le2.inverse_transform(preds2)

# -----------------------
# Run full pipeline
# -----------------------
def run_full_pipeline(cfg: Dict):
    # load data
    df = pd.read_csv(cfg["data_csv"])
    df["text"] = df[cfg["text_columns"]].fillna("").agg(" ".join, axis=1)
    df = df.dropna(subset=["text", cfg["lvl1_col"], cfg["lvl2_col"]]).reset_index(drop=True)

    le1 = LabelEncoder(); le2 = LabelEncoder()
    df["lvl1_id"] = le1.fit_transform(df[cfg["lvl1_col"]])
    df["lvl2_id"] = le2.fit_transform(df[cfg["lvl2_col"]])

    print("[INFO] Loaded dataset:", len(df), "rows")
    print("[INFO] Level-1 classes:", len(le1.classes_), "Level-2 classes:", len(le2.classes_))

    # build vocabulary from full dataset (stable across seeds)
    print("[INFO] Building vocabulary...")
    vocab = build_vocab(df["text"].tolist(), cfg)
    with open(cfg["vocab_path"], "w") as f:
        json.dump(vocab, f)
    print("[INFO] Vocab size:", len(vocab))

    # optional load pretrained embeddings if provided (user-supplied)
    emb_weights = None
    if cfg.get("pretrained_embedding_path") and cfg.get("pretrained_embedding_vocab"):
        emb_weights = np.load(cfg["pretrained_embedding_path"])
        # user must ensure emb_weights shape == (vocab_size, emb_dim)
        emb_weights = torch.tensor(emb_weights, dtype=torch.float)

    model_dirs = []
    for seed in cfg["seeds"]:
        print(f"\n========== TRAINING SEED {seed} ==========")
        md = train_one_seed(seed, cfg, df, le1, le2, vocab, emb_weights)
        model_dirs.append(md)

    # quick ensemble check
    _, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=cfg["seeds"][0])
    sample_texts = val_df["text"].tolist()[:256]
    lvl1_preds, lvl2_preds = ensemble_predict(sample_texts, model_dirs, cfg, le1, le2)
    print("[INFO] Ensemble example (first 10):")
    for i in range(min(10, len(lvl2_preds))):
        print(f" L1: {lvl1_preds[i]}  L2: {lvl2_preds[i]}")

    print("[INFO] All done. Models saved to:", cfg["save_dir"])

# -----------------------
# Entry
# -----------------------
if __name__ == "__main__":
    run_full_pipeline(CONFIG)
