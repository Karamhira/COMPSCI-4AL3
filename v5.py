
import os
import random
import math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -----------------------
# CONFIG
# -----------------------
CONFIG = {
    "data_csv": "./dataset/MN-DS-news-classification.csv",
    # Columns in CSV:
    "text_columns": ["title", "content"],   # will be concatenated
    "lvl1_col": "category_level_1",
    "lvl2_col": "category_level_2",
    "model_name": "microsoft/deberta-v3-base",
    "t5_aug_model": "t5-small",   # for paraphrasing (smaller for speed)
    "max_len": 256,
    "batch_size": 8,
    "accumulation_steps": 4,  # effective batch size = batch_size * accumulation_steps
    "num_epochs": 4,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_proportion": 0.1,
    "freeze_layers": 8,  # number of bottom encoder layers to freeze for epoch 0
    "augment_min_samples": 150,  # only augment classes with <= this samples
    "augment_factor": 2,  # how many augmented copies per original for target small classes
    "use_t5_aug": True,
    "focal_loss_gamma": 2.0,
    "use_focal_loss": False,  # optional, sometimes helps
    "seeds": [42, 43, 44],  # ensemble seeds
    "output_dir": "./grandmaster_models",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------
# Data load + preprocess
# -----------------------
def load_and_prepare_df(cfg) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    df = pd.read_csv(cfg["data_csv"])
    # create text column
    df["text"] = df[cfg["text_columns"]].fillna("").agg(" ".join, axis=1)
    # drop empties
    df = df.dropna(subset=["text", cfg["lvl1_col"], cfg["lvl2_col"]]).reset_index(drop=True)

    # encode level labels
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    df["lvl1_id"] = le1.fit_transform(df[cfg["lvl1_col"]])
    df["lvl2_id"] = le2.fit_transform(df[cfg["lvl2_col"]])

    return df, le1, le2

# -----------------------
# Optional T5-based paraphrase augmentor (slow but useful)
# -----------------------
class T5Paraphraser:
    def __init__(self, model_name="t5-small", device="cpu"):
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def paraphrase(self, text: str, num_return_sequences: int = 1, max_length=256) -> List[str]:
        # A simple prompt-style paraphrase
        input_text = f"paraphrase: {text} </s>"
        encoding = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        outputs = self.model.generate(
            **encoding,
            max_length=max_length,
            num_beams=4,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            early_stopping=True
        )
        decs = [self.tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in outputs]
        return decs

# -----------------------
# Dataset
# -----------------------
class NewsHierDataset(Dataset):
    def __init__(self, texts: List[str], lvl1_ids: List[int], lvl2_ids: List[int], tokenizer, max_len=256):
        self.texts = texts
        self.lvl1_ids = lvl1_ids
        self.lvl2_ids = lvl2_ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lvl1": torch.tensor(self.lvl1_ids[idx], dtype=torch.long),
            "lvl2": torch.tensor(self.lvl2_ids[idx], dtype=torch.long),
        }
        return item

# -----------------------
# Model (shared encoder + two heads)
# -----------------------
class HierarchicalClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_lvl1: int, num_lvl2: int, dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=config)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier_lvl1 = nn.Linear(hidden_size, num_lvl1)
        self.classifier_lvl2 = nn.Linear(hidden_size, num_lvl2)

        # init heads
        nn.init.xavier_uniform_(self.classifier_lvl1.weight)
        nn.init.xavier_uniform_(self.classifier_lvl2.weight)
        if self.classifier_lvl1.bias is not None:
            nn.init.zeros_(self.classifier_lvl1.bias)
        if self.classifier_lvl2.bias is not None:
            nn.init.zeros_(self.classifier_lvl2.bias)

    def forward(self, input_ids, attention_mask):
        # encoder returns last_hidden_state and optionally pooled output depending on model; for DeBERTa we use pooler if present or mean pooling
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # try to use pooled output if exists else mean pool
        pooled = None
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # mean pool over tokens weighted by attention
            last_hidden = outputs.last_hidden_state  # (B, L, H)
            attn = attention_mask.unsqueeze(-1).expand_as(last_hidden)  # (B, L, H)
            sum_emb = (last_hidden * attn).sum(1)
            denom = attn.sum(1).clamp(min=1e-9)
            pooled = sum_emb / denom

        pooled = self.dropout(pooled)
        logits1 = self.classifier_lvl1(pooled)
        logits2 = self.classifier_lvl2(pooled)
        return logits1, logits2

# -----------------------
# Losses: CrossEntropy with class weights + optional FocalLoss
# -----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)  # negative CE (per example)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# -----------------------
# Training & evaluation functions
# -----------------------
def build_sampler(df_train: pd.DataFrame, label_col: str):
    counts = df_train[label_col].value_counts()
    class_weights = 1.0 / counts
    sample_weights = df_train[label_col].map(class_weights).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # return sample weights and class weights mapping also
    class_weight_tensor = torch.tensor([class_weights.get(i, 0.0) for i in sorted(counts.index)], dtype=torch.float)
    return sampler, class_weight_tensor.to(CONFIG["device"])

def freeze_bottom_layers(model: HierarchicalClassifier, freeze_layers: int):
    """
    Freeze bottom `freeze_layers` transformer layers (encoder.model.encoder.layer[0:freeze_layers]).
    Works for DeBERTa-like architectures which expose encoder.layer
    """
    # try to find encoder layers attribute for many HF models
    # handle DeBERTa: model.encoder.encoder.layer
    layers = None
    enc = model.encoder
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
        layers = enc.encoder.layer
    elif hasattr(enc, "layer"):
        layers = enc.layer
    else:
        # unknown architecture — do not freeze
        return
    # freeze bottom layers
    for i, layer in enumerate(layers):
        if i < freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False

def unfreeze_all(model: HierarchicalClassifier):
    for p in model.parameters():
        p.requires_grad = True

def evaluate(model: HierarchicalClassifier, dataloader: DataLoader, device: str, le1: LabelEncoder, le2: LabelEncoder):
    model.eval()
    preds1, preds2, true1, true2 = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            l1 = batch["lvl1"].to(device)
            l2 = batch["lvl2"].to(device)
            logits1, logits2 = model(ids, mask)
            p1 = torch.argmax(logits1, dim=1)
            p2 = torch.argmax(logits2, dim=1)
            preds1.extend(p1.cpu().numpy())
            preds2.extend(p2.cpu().numpy())
            true1.extend(l1.cpu().numpy())
            true2.extend(l2.cpu().numpy())
    acc1 = accuracy_score(true1, preds1)
    acc2 = accuracy_score(true2, preds2)
    f1_1 = f1_score(true1, preds1, average="macro")
    f1_2 = f1_score(true2, preds2, average="macro")
    return {"lvl1_acc": acc1, "lvl2_acc": acc2, "lvl1_macro_f1": f1_1, "lvl2_macro_f1": f1_2,
            "preds2": preds2, "true2": true2}

# -----------------------
# Main training routine for one seed
# -----------------------
def train_one_seed(seed: int, cfg: Dict, df: pd.DataFrame, le1: LabelEncoder, le2: LabelEncoder):
    set_seed(seed)
    device = cfg["device"]

    # split
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=seed)

    # augmentation for rare classes (optional)
    if cfg["use_t5_aug"]:
        # create paraphraser
        print("[INFO] Building T5 paraphraser (may be slow)...")
        t5para = T5Paraphraser(model_name=cfg["t5_aug_model"], device=device)
        # find rare classes
        counts = train_df["lvl2_id"].value_counts()
        rare_classes = counts[counts <= cfg["augment_min_samples"]].index.tolist()
        if len(rare_classes) > 0:
            new_rows = []
            for cls in tqdm(rare_classes, desc="Paraphrasing rare classes"):
                subset = train_df[train_df["lvl2_id"] == cls]
                for _, row in subset.iterrows():
                    for _ in range(cfg["augment_factor"]):
                        try:
                            paras = t5para.paraphrase(row["text"], num_return_sequences=1)
                            new_text = paras[0] if len(paras) > 0 else row["text"]
                            new_rows.append({
                                "text": new_text,
                                "lvl1_id": row["lvl1_id"],
                                "lvl2_id": row["lvl2_id"]
                            })
                        except Exception as e:
                            # on error, skip aug
                            continue
            if len(new_rows) > 0:
                aug_df = pd.DataFrame(new_rows)
                # append to train
                train_df = pd.concat([train_df, aug_df], ignore_index=True)
                print(f"[INFO] Added {len(new_rows)} augmented rows. New train size: {len(train_df)}")

    # tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_dataset = NewsHierDataset(train_df["text"].tolist(), train_df["lvl1_id"].tolist(), train_df["lvl2_id"].tolist(), tokenizer, max_len=cfg["max_len"])
    val_dataset = NewsHierDataset(val_df["text"].tolist(), val_df["lvl1_id"].tolist(), val_df["lvl2_id"].tolist(), tokenizer, max_len=cfg["max_len"])

    # sampler for lvl2
    sampler, class_weight_tensor = build_sampler(train_df, "lvl2_id")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # model initialization
    num_lvl1 = len(le1.classes_)
    num_lvl2 = len(le2.classes_)
    model = HierarchicalClassifier(cfg["model_name"], num_lvl1=num_lvl1, num_lvl2=num_lvl2).to(device)

    # freeze bottom layers for warmup epoch
    freeze_bottom_layers(model, cfg["freeze_layers"])

    # optimizer + scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
    total_steps = math.ceil(len(train_loader) / 1.0) * cfg["num_epochs"]  # approximate - accumulation handled separately
    warmup_steps = int(cfg["warmup_proportion"] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # losses
    if cfg["use_focal_loss"]:
        loss_lvl2 = FocalLoss(gamma=cfg["focal_loss_gamma"], weight=class_weight_tensor)
    else:
        loss_lvl2 = nn.CrossEntropyLoss(weight=class_weight_tensor)
    # level-1 class weight (optional: balance by lvl1 freq)
    lvl1_counts = train_df["lvl1_id"].value_counts()
    lvl1_class_weights = 1.0 / lvl1_counts
    lvl1_weight_tensor = torch.tensor([lvl1_class_weights.get(i, 0.0) for i in sorted(lvl1_counts.index)], dtype=torch.float).to(device)
    loss_lvl1 = nn.CrossEntropyLoss(weight=lvl1_weight_tensor)

    scaler = GradScaler()
    best_val_f1 = -1.0
    model_dir = os.path.join(cfg["output_dir"], f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    # training loop
    global_step = 0
    for epoch in range(cfg["num_epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        # If epoch > 0, unfreeze all layers (we froze only for epoch 0)
        if epoch == 1:
            print("[INFO] Unfreezing all layers.")
            unfreeze_all(model)
            # re-create optimizer because some params now require grad
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Seed {seed} Epoch {epoch+1}")
        for step, batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y1 = batch["lvl1"].to(device)
            y2 = batch["lvl2"].to(device)

            with autocast():
                logits1, logits2 = model(ids, mask)
                loss1 = loss_lvl1(logits1, y1)
                loss2 = loss_lvl2(logits2, y2)
                # multi-task weighting: emphasize level2 a bit more if you want
                loss = 0.5 * loss1 + 1.0 * loss2

            scaler.scale(loss / cfg["accumulation_steps"]).backward()
            running_loss += loss.item()

            if (step + 1) % cfg["accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if (step + 1) % 200 == 0:
                pbar.set_postfix({"loss": running_loss / (step + 1)})

        # end epoch: evaluate
        val_metrics = evaluate(model, val_loader, device, le1, le2)
        print(f"[Seed {seed}] Epoch {epoch+1} Validation: lvl1_acc={val_metrics['lvl1_acc']:.4f} lvl2_acc={val_metrics['lvl2_acc']:.4f} lvl2_macro_f1={val_metrics['lvl2_macro_f1']:.4f}")

        # checkpoint best by level2 macro f1
        if val_metrics["lvl2_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["lvl2_macro_f1"]
            save_path = os.path.join(model_dir, "best_model.pt")
            print(f"[Seed {seed}] New best lvl2_macro_f1={best_val_f1:.4f} — saving model to {save_path}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cfg": cfg,
                "le1_classes": le1.classes_.tolist(),
                "le2_classes": le2.classes_.tolist()
            }, save_path)

    # save tokenizer and final config
    tokenizer.save_pretrained(model_dir)
    return model_dir  # return directory where best model is saved

# -----------------------
# Ensemble inference
# -----------------------
def load_model_for_inference(model_dir: str, cfg: Dict, device: str, num_lvl1: int, num_lvl2: int):
    # load tokenizer from model_dir (saved earlier) if present
    tokenizer = AutoTokenizer.from_pretrained(model_dir) if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "tokenizer_config.json")) else AutoTokenizer.from_pretrained(cfg["model_name"])
    model = HierarchicalClassifier(cfg["model_name"], num_lvl1=num_lvl1, num_lvl2=num_lvl2)
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return tokenizer, model

def ensemble_predict(texts: List[str], model_dirs: List[str], cfg: Dict, le1: LabelEncoder, le2: LabelEncoder):
    device = cfg["device"]
    # load all models + tokenizers
    models = []
    tokenizers = []
    for d in model_dirs:
        tokenizer, model = load_model_for_inference(d, cfg, device, num_lvl1=len(le1.classes_), num_lvl2=len(le2.classes_))
        tokenizers.append(tokenizer)
        models.append(model)

    # tokenize with first tokenizer (all tokenizers same config/size normally)
    tok = tokenizers[0]
    enc = tok(texts, truncation=True, padding=True, max_length=cfg["max_len"], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # collect logits
    logits1_list = []
    logits2_list = []
    with torch.no_grad():
        for model in models:
            l1, l2 = model(input_ids, attention_mask)
            logits1_list.append(l1.cpu().numpy())
            logits2_list.append(l2.cpu().numpy())

    # average
    avg_logits1 = np.mean(np.stack(logits1_list, axis=0), axis=0)
    avg_logits2 = np.mean(np.stack(logits2_list, axis=0), axis=0)

    preds1 = np.argmax(avg_logits1, axis=1)
    preds2 = np.argmax(avg_logits2, axis=1)

    lvl1_preds = le1.inverse_transform(preds1)
    lvl2_preds = le2.inverse_transform(preds2)
    return lvl1_preds, lvl2_preds

# -----------------------
# Run full training (multi-seed) and ensemble
# -----------------------
def run_full_pipeline(cfg: Dict):
    df, le1, le2 = load_and_prepare_df(cfg)
    print("[INFO] Loaded dataset: total rows =", len(df))
    print("[INFO] Level-1 classes:", len(le1.classes_), "Level-2 classes:", len(le2.classes_))

    model_dirs = []
    for seed in cfg["seeds"]:
        print(f"\n========== TRAINING SEED {seed} ==========")
        model_dir = train_one_seed(seed, cfg, df, le1, le2)
        model_dirs.append(model_dir)

    # Sanity check: ensemble on validation (we'll do a quick val sample)
    # Build a small val set
    _, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=cfg["seeds"][0])
    sample_texts = val_df["text"].tolist()[:512]  # up to 512 to keep memory moderate
    lvl1_preds, lvl2_preds = ensemble_predict(sample_texts, model_dirs, cfg, le1, le2)
    print("[INFO] Ensemble produced", len(lvl2_preds), "predictions. Example (first 10):")
    for i in range(min(10, len(lvl2_preds))):
        print(f"Pred L1: {lvl1_preds[i]}  Pred L2: {lvl2_preds[i]}")

    print("\nALL DONE. Models saved to:", cfg["output_dir"])
    print("To run inference, use ensemble_predict(texts, model_dirs, CONFIG, le1, le2)")

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    run_full_pipeline(CONFIG)
