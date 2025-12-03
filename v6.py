#!/usr/bin/env python3
"""
Grandmaster Final pipeline — pretrained encoder + two-head hierarchical finetune.

Drop-in single file:
- Uses HF pretrained encoder (configurable)
- Shared encoder + two classification heads
- Class-balanced sampling, focal loss option
- Mixed precision (torch.amp), GradScaler (correct API)
- Layer freezing warmup then unfreeze
- OneCycleLR scheduler with warmup
- Gradient accumulation + clipping
- SWA optional for final epochs
- Multi-seed ensembling
"""

import os, math, random, time, json
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch import amp
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)

# -----------------------
# CONFIG — tune these
# -----------------------
CONFIG = {
    # Data
    "data_csv": "./dataset/MN-DS-news-classification.csv",
    "text_columns": ["title", "content"],
    "lvl1_col": "category_level_1",
    "lvl2_col": "category_level_2",

    # Model (set to a pretrained model you can download or a local path)
    # Recommended: "microsoft/deberta-v3-large" or "microsoft/deberta-v3-small" if memory is low.
    # For best results use a large model.
    "model_name_or_path": "microsoft/deberta-v3-large",

    # Tokenization / encoding
    "max_len": 256,
    "use_fast_tokenizer": True,

    # Training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seeds": [42, 43, 44],
    "num_epochs": 12,
    "batch_size": 16,            # set 32 if GPU mem allows
    "accumulation_steps": 1,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "freeze_epochs": 1,         # freeze base encoder for this many epochs (warmup)
    "layerwise_lr_decay": 0.8,  # optional discriminative LR multiplier across layers
    "use_focal_loss": False,
    "focal_gamma": 2.0,
    "use_swa": True,
    "swa_start_epoch": 9,
    "swa_lr": 1e-5,

    # Data loader
    "num_workers": 4,

    # Save
    "output_dir": "./grandmaster_final_models",
    "save_best_only": True,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# -----------------------
# Utility dataset
# -----------------------
class NewsHFDataset(Dataset):
    def __init__(self, texts: List[str], lvl1: List[int], lvl2: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.lvl1 = lvl1
        self.lvl2 = lvl2
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lvl1": torch.tensor(self.lvl1[idx], dtype=torch.long),
            "lvl2": torch.tensor(self.lvl2[idx], dtype=torch.long),
        }
        return item

# -----------------------
# Model: shared encoder + two heads
# -----------------------
class HierModel(nn.Module):
    def __init__(self, model_name_or_path: str, num_lvl1: int, num_lvl2: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
        hidden = config.hidden_size

        # pooling projection and heads
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(hidden, hidden)
        self.head1 = nn.Linear(hidden, num_lvl1)
        self.head2 = nn.Linear(hidden, num_lvl2)

        # init heads
        nn.init.xavier_uniform_(self.head1.weight)
        nn.init.xavier_uniform_(self.head2.weight)
        if self.head1.bias is not None:
            nn.init.zeros_(self.head1.bias)
        if self.head2.bias is not None:
            nn.init.zeros_(self.head2.bias)

    def forward(self, input_ids, attention_mask):
        # encoder returns last_hidden_state and possibly pooler_output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # mean pooling over non-pad tokens
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
            summed = (last_hidden * mask).sum(1)
            denom = mask.sum(1).clamp(min=1e-9)
            pooled = summed / denom

        feat = torch.tanh(self.proj(self.dropout(pooled)))
        logits1 = self.head1(feat)
        logits2 = self.head2(feat)
        return logits1, logits2

# -----------------------
# Loss helpers
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
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
# Sampler builder
# -----------------------
def build_weighted_sampler(df_train: pd.DataFrame, label_col: str):
    counts = df_train[label_col].value_counts().to_dict()
    # weight = 1/count
    weight_map = {k: 1.0 / v for k, v in counts.items()}
    sample_weights = df_train[label_col].map(weight_map).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # class weight tensor for loss
    max_label = int(df_train[label_col].max())
    class_weight = torch.ones(max_label + 1, dtype=torch.float)
    for lbl, cnt in counts.items():
        class_weight[int(lbl)] = float(1.0 / cnt)
    return sampler, class_weight

# -----------------------
# Utility: set seed
# -----------------------
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------
# Training / evaluation
# -----------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_p1, all_p2, all_t1, all_t2 = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            t1 = batch["lvl1"].to(device)
            t2 = batch["lvl2"].to(device)
            l1, l2 = model(ids, mask)
            p1 = torch.argmax(l1, dim=1)
            p2 = torch.argmax(l2, dim=1)
            all_p1.extend(p1.cpu().numpy().tolist())
            all_p2.extend(p2.cpu().numpy().tolist())
            all_t1.extend(t1.cpu().numpy().tolist())
            all_t2.extend(t2.cpu().numpy().tolist())
    return {
        "lvl1_acc": accuracy_score(all_t1, all_p1),
        "lvl2_acc": accuracy_score(all_t2, all_p2),
        "lvl1_macro_f1": f1_score(all_t1, all_p1, average="macro"),
        "lvl2_macro_f1": f1_score(all_t2, all_p2, average="macro"),
    }

def train_one_seed(seed:int, cfg:Dict, df:pd.DataFrame, le1:LabelEncoder, le2:LabelEncoder, tokenizer):
    set_seed(seed)
    device = cfg["device"]

    # split (stratify by lvl2 to keep second-level distribution)
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=seed)
    train_texts = train_df["text"].tolist(); val_texts = val_df["text"].tolist()
    train_y1 = train_df["lvl1_id"].tolist(); val_y1 = val_df["lvl1_id"].tolist()
    train_y2 = train_df["lvl2_id"].tolist(); val_y2 = val_df["lvl2_id"].tolist()

    train_ds = NewsHFDataset(train_texts, train_y1, train_y2, tokenizer, cfg["max_len"])
    val_ds = NewsHFDataset(val_texts, val_y1, val_y2, tokenizer, cfg["max_len"])

    sampler, class_weight = build_weighted_sampler(train_df, "lvl2_id")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    num_lvl1 = len(le1.classes_); num_lvl2 = len(le2.classes_)
    model = HierModel(cfg["model_name_or_path"], num_lvl1=num_lvl1, num_lvl2=num_lvl2).to(device)

    # optionally freeze encoder parameters for warmup
    if cfg["freeze_epochs"] > 0:
        for name, p in model.encoder.named_parameters():
            p.requires_grad = False

    # prepare optimizer with weight decay on all except biases & LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg["lr"], eps=1e-8)

    # scheduler (linear warmup -> linear decay)
    total_steps = int(len(train_loader) * cfg["num_epochs"] / max(1, cfg["accumulation_steps"]))
    warmup_steps = max(1, int(cfg["warmup_ratio"] * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # losses
    device_tensor = device
    if cfg["use_focal_loss"]:
        loss_lvl2 = FocalLoss(gamma=cfg["focal_gamma"], weight=class_weight.to(device))
    else:
        loss_lvl2 = nn.CrossEntropyLoss(weight=class_weight.to(device))
    # lvl1 class weights (balance by freq)
    lvl1_counts = train_df["lvl1_id"].value_counts().to_dict()
    lvl1_weights = torch.tensor([1.0 / lvl1_counts.get(i, 1.0) for i in range(num_lvl1)], dtype=torch.float).to(device)
    loss_lvl1 = nn.CrossEntropyLoss(weight=lvl1_weights)

    scaler = GradScaler()

    # optionally SWA
    swa_model = None
    if cfg["use_swa"]:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg["swa_lr"])

    best_val_metric = -1.0
    model_dir = os.path.join(cfg["output_dir"], f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    global_step = 0

    for epoch in range(cfg["num_epochs"]):
        model.train()
        if epoch == cfg["freeze_epochs"]:
            # unfreeze encoder for full finetune
            for name, p in model.encoder.named_parameters():
                p.requires_grad = True
            # re-create optimizer to include encoder params
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Seed {seed} Epoch {epoch+1}/{cfg['num_epochs']}")
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y1 = batch["lvl1"].to(device)
            y2 = batch["lvl2"].to(device)

            with autocast(device_type=autocast_device):
                logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
                l1 = loss_lvl1(logits1, y1)
                l2 = loss_lvl2(logits2, y2)
                loss = 0.5 * l1 + 1.0 * l2

            # accumulate grads
            scaler.scale(loss / cfg["accumulation_steps"]).backward()
            running_loss += loss.item()

            if (step + 1) % cfg["accumulation_steps"] == 0:
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # SWA
                if cfg["use_swa"] and epoch >= cfg["swa_start_epoch"]:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

            if (step + 1) % 200 == 0:
                pbar.set_postfix({"loss": running_loss / (step + 1)})

        # end epoch evaluation
        val_metrics = evaluate(model, val_loader, device)
        print(f"[Seed {seed}] Epoch {epoch+1} VAL lvl1_acc={val_metrics['lvl1_acc']:.4f} lvl2_acc={val_metrics['lvl2_acc']:.4f} lvl2_macro_f1={val_metrics['lvl2_macro_f1']:.4f}")

        # checkpoint best (by lvl2 macro f1)
        if val_metrics["lvl2_macro_f1"] > best_val_metric:
            best_val_metric = val_metrics["lvl2_macro_f1"]
            if cfg["save_best_only"]:
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                }
                torch.save(save_dict, os.path.join(model_dir, "best.pt"))
                # also save tokenizer indices info externally by caller

    # Finalize SWA if used
    if cfg["use_swa"] and swa_model is not None:
        # update bn (requires a dataloader with training data and model in eval)
        try:
            from torch.optim.swa_utils import update_bn
            update_bn(train_loader, swa_model)
            model = swa_model.module
            # save swa model
            torch.save({"model_state_dict": swa_model.module.state_dict(), "cfg": cfg}, os.path.join(model_dir, "best_swa.pt"))
        except Exception:
            pass

    return model_dir

# -----------------------
# Ensemble inference helpers
# -----------------------
def load_model_for_inference(model_dir: str, cfg:Dict, device:str, num_lvl1:int, num_lvl2:int):
    # load model weights
    ckpt = torch.load(os.path.join(model_dir, "best.pt"), map_location=device)
    model = HierModel(cfg["model_name_or_path"], num_lvl1=num_lvl1, num_lvl2=num_lvl2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device); model.eval()
    return model

def ensemble_predict(texts: List[str], model_dirs: List[str], cfg:Dict, tokenizer, le1:LabelEncoder, le2:LabelEncoder):
    device = cfg["device"]
    models = []
    for d in model_dirs:
        # load num classes from le1/le2 (the model class uses args to create heads)
        models.append(load_model_for_inference(d, cfg, device, len(le1.classes_), len(le2.classes_)))

    enc = tokenizer(texts, truncation=True, padding=True, max_length=cfg["max_len"], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    logits1_collect = []
    logits2_collect = []
    with torch.no_grad():
        for m in models:
            l1, l2 = m(input_ids=input_ids, attention_mask=attention_mask)
            logits1_collect.append(l1.cpu().numpy())
            logits2_collect.append(l2.cpu().numpy())

    avg1 = np.mean(np.stack(logits1_collect, axis=0), axis=0)
    avg2 = np.mean(np.stack(logits2_collect, axis=0), axis=0)
    p1 = np.argmax(avg1, axis=1); p2 = np.argmax(avg2, axis=1)
    return le1.inverse_transform(p1), le2.inverse_transform(p2)

# -----------------------
# Run full pipeline
# -----------------------
def run_full_pipeline(cfg:Dict):
    # load data
    df = pd.read_csv(cfg["data_csv"])
    df["text"] = df[cfg["text_columns"]].fillna("").agg(" ".join, axis=1)
    df = df.dropna(subset=["text", cfg["lvl1_col"], cfg["lvl2_col"]]).reset_index(drop=True)

    le1 = LabelEncoder(); le2 = LabelEncoder()
    df["lvl1_id"] = le1.fit_transform(df[cfg["lvl1_col"]])
    df["lvl2_id"] = le2.fit_transform(df[cfg["lvl2_col"]])

    print("[INFO] Loaded dataset:", len(df), "rows")
    print("[INFO] Level-1 classes:", len(le1.classes_), "Level-2 classes:", len(le2.classes_))

    # tokenizer
    print("[INFO] Loading tokenizer:", cfg["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=cfg["use_fast_tokenizer"])

    # train multiple seeds and ensemble
    model_dirs = []
    for seed in cfg["seeds"]:
        print(f"\n=== TRAINING SEED {seed} ===")
        md = train_one_seed(seed, cfg, df, le1, le2, tokenizer)
        model_dirs.append(md)

    # quick ensemble sanity check on holdout 256 samples
    _, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=cfg["seeds"][0])
    sample_texts = val_df["text"].tolist()[:256]
    lvl1_preds, lvl2_preds = ensemble_predict(sample_texts, model_dirs, cfg, tokenizer, le1, le2)
    print("[INFO] Ensemble sample predictions (first 10):")
    for i in range(min(10, len(lvl2_preds))):
        print("L1:", lvl1_preds[i], "L2:", lvl2_preds[i])

    print("[INFO] Done. Models saved to:", cfg["output_dir"])

if __name__ == "__main__":
    run_full_pipeline(CONFIG)
