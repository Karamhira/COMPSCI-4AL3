import os
import random
import math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler

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
from sklearn.metrics import accuracy_score, f1_score

# -----------------------
# CONFIG
# -----------------------
CONFIG = {
    "data_csv": "./dataset/MN-DS-news-classification.csv",
    "text_columns": ["title", "content"],
    "lvl1_col": "category_level_1",
    "lvl2_col": "category_level_2",
    "model_name": "microsoft/deberta-v3-base",
    # paraphraser model (flan/t5 variants may be faster/better)
    "t5_aug_model": "ramsrigouthamg/t5_paraphraser",
    "max_len": 256,
    "batch_size": 8,
    "accumulation_steps": 4,
    "num_epochs": 4,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_proportion": 0.1,
    "freeze_layers": 8,
    "augment_min_samples": 150,     # classes with <= this will be considered 'rare'
    "augment_factor": 2,            # times to augment each rare sample (effective multiplier)
    "use_t5_aug": True,
    "paraphrase_batch_size": 32,    # batch size for paraphraser
    "paraphrase_sampling": True,    # use sampling (fast) vs beam search (higher quality slower)
    "paraphrase_top_k": 50,
    "paraphrase_top_p": 0.95,
    "paraphrase_temp": 0.8,
    "focal_loss_gamma": 2.0,
    "use_focal_loss": False,
    "seeds": [42, 43, 44],
    "output_dir": "./grandmaster_models",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "aug_cache_path": "./grandmaster_models/augmented_train.csv"  # cache location
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("[CONFIG] Using device:", CONFIG["device"])

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------
# Data load + preprocess
# -----------------------
def load_and_prepare_df(cfg) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    df = pd.read_csv(cfg["data_csv"])
    df["text"] = df[cfg["text_columns"]].fillna("").agg(" ".join, axis=1)
    df = df.dropna(subset=["text", cfg["lvl1_col"], cfg["lvl2_col"]]).reset_index(drop=True)

    le1 = LabelEncoder()
    le2 = LabelEncoder()
    df["lvl1_id"] = le1.fit_transform(df[cfg["lvl1_col"]])
    df["lvl2_id"] = le2.fit_transform(df[cfg["lvl2_col"]])
    return df, le1, le2

# -----------------------
# Fast GPU-batched paraphraser (with caching)
# -----------------------
def build_paraphraser(cfg):
    """
    Loads paraphraser model & tokenizer using fast tokenizer +
    forcing safetensors to avoid torch.load vulnerability checks.
    """
    from transformers import T5TokenizerFast, T5ForConditionalGeneration

    device = cfg["device"]
    print("[INFO] Loading paraphraser:", cfg["t5_aug_model"], "on", device)

    # Fast tokenizer → no protobuf issues
    tokenizer = T5TokenizerFast.from_pretrained(
        cfg["t5_aug_model"],
        use_fast=True,
        legacy=False
    )

    # Force safetensors → avoids torch 2.6 requirement
    model = T5ForConditionalGeneration.from_pretrained(
        cfg["t5_aug_model"],
        use_safetensors=True,
        torch_dtype="auto"
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device



def paraphrase_batch(model, tokenizer, device, sentences: List[str], cfg) -> List[str]:
    """
    Paraphrase a batch of sentences. Uses sampling (fast) by default.
    Falls back to returning original sentences on any failure.
    """
    prompts = [f"paraphrase: {s}" for s in sentences]
    enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=cfg["max_len"])
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_length": cfg["max_len"],
        "num_return_sequences": 1,
        "early_stopping": True,
    }

    if cfg["paraphrase_sampling"]:
        gen_kwargs.update({
            "do_sample": True,
            "top_k": cfg["paraphrase_top_k"],
            "top_p": cfg["paraphrase_top_p"],
            "temperature": cfg["paraphrase_temp"],
            "num_beams": 1
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": 4
        })

    try:
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # outputs will be (batch_size * num_return_sequences,)
        # we requested num_return_sequences=1 => length matches batch
        return decoded
    except Exception as e:
        print("[WARN] paraphrase_batch failed:", e)
        return sentences  # fallback: return originals

def augment_rare_classes_once(df_train: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Checks cache; if absent, paraphrases rare classes in batches and saves cached augmented dataset.
    Returns augmented train dataframe.
    """
    cache_path = cfg["aug_cache_path"]
    if cfg["use_t5_aug"] and os.path.exists(cache_path):
        print("[INFO] Found augmented cache at", cache_path, "- loading it.")
        return pd.read_csv(cache_path)

    if not cfg["use_t5_aug"]:
        print("[INFO] use_t5_aug set to False: skipping augmentation.")
        return df_train

    counts = df_train["lvl2_id"].value_counts()
    rare_classes = counts[counts <= cfg["augment_min_samples"]].index.tolist()
    if len(rare_classes) == 0:
        print("[INFO] No rare classes found; skipping augmentation.")
        return df_train

    model, tokenizer, device = build_paraphraser(cfg)

    # build subset of rare samples to augment
    rare_df = df_train[df_train["lvl2_id"].isin(rare_classes)].reset_index(drop=True)
    texts = rare_df["text"].tolist()
    lvl1_ids = rare_df["lvl1_id"].tolist()
    lvl2_ids = rare_df["lvl2_id"].tolist()

    new_rows = []
    batch_size = cfg.get("paraphrase_batch_size", 32)
    print(f"[INFO] Paraphrasing {len(texts)} rare samples in batches of {batch_size} ...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        paras = paraphrase_batch(model, tokenizer, device, batch_texts, cfg)
        # we may want to create multiple augmented copies per original (augment_factor)
        for j, p in enumerate(paras):
            orig_idx = i + j
            for _ in range(cfg["augment_factor"]):
                new_rows.append({
                    "text": p,
                    "lvl1_id": int(lvl1_ids[orig_idx]),
                    "lvl2_id": int(lvl2_ids[orig_idx])
                })

    if len(new_rows) > 0:
        aug_df = pd.concat([df_train, pd.DataFrame(new_rows)], ignore_index=True)
        # keep original label columns if present (for compatibility)
        # attempt to preserve lvl label names if available
        # Save cache (without extra big metadata)
        aug_df.to_csv(cache_path, index=False)
        print(f"[INFO] Augmentation complete — added {len(new_rows)} rows. Saved cache to {cache_path}")
        return aug_df

    print("[INFO] No augmented rows created.")
    return df_train

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
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lvl1": torch.tensor(self.lvl1_ids[idx], dtype=torch.long),
            "lvl2": torch.tensor(self.lvl2_ids[idx], dtype=torch.long),
        }

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

        nn.init.xavier_uniform_(self.classifier_lvl1.weight)
        nn.init.xavier_uniform_(self.classifier_lvl2.weight)
        if self.classifier_lvl1.bias is not None:
            nn.init.zeros_(self.classifier_lvl1.bias)
        if self.classifier_lvl2.bias is not None:
            nn.init.zeros_(self.classifier_lvl2.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state
            attn = attention_mask.unsqueeze(-1).expand_as(last_hidden)
            pooled = (last_hidden * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        logits1 = self.classifier_lvl1(pooled)
        logits2 = self.classifier_lvl2(pooled)
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
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# -----------------------
# Sampler + utilities
# -----------------------
def build_sampler(df_train: pd.DataFrame, label_col: str, device: str):
    counts = df_train[label_col].value_counts()
    class_weights = 1.0 / counts
    sample_weights = df_train[label_col].map(class_weights).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # build class weight vector ordered by label index (0..C-1)
    max_label = int(df_train[label_col].max())
    class_weight_tensor = torch.ones(max_label + 1, dtype=torch.float)
    for lbl, w in class_weights.items():
        class_weight_tensor[int(lbl)] = float(w)
    return sampler, class_weight_tensor.to(device)

def freeze_bottom_layers(model: HierarchicalClassifier, freeze_layers: int):
    layers = getattr(getattr(model.encoder, "encoder", None), "layer", None)
    if layers is None:
        return
    for i, layer in enumerate(layers):
        if i < freeze_layers:
            for p in layer.parameters():
                p.requires_grad = False

def unfreeze_all(model: HierarchicalClassifier):
    for p in model.parameters():
        p.requires_grad = True

# -----------------------
# Evaluation
# -----------------------
def evaluate(model: HierarchicalClassifier, dataloader: DataLoader, device: str):
    model.eval()
    preds1, preds2, true1, true2 = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            l1 = batch["lvl1"].to(device)
            l2 = batch["lvl2"].to(device)
            logits1, logits2 = model(ids, mask)
            preds1.extend(torch.argmax(logits1, dim=1).cpu().numpy())
            preds2.extend(torch.argmax(logits2, dim=1).cpu().numpy())
            true1.extend(l1.cpu().numpy())
            true2.extend(l2.cpu().numpy())
    return {
        "lvl1_acc": accuracy_score(true1, preds1),
        "lvl2_acc": accuracy_score(true2, preds2),
        "lvl1_macro_f1": f1_score(true1, preds1, average="macro"),
        "lvl2_macro_f1": f1_score(true2, preds2, average="macro")
    }

# -----------------------
# Train single seed
# -----------------------
def train_one_seed(seed: int, cfg: Dict, df: pd.DataFrame, le1: LabelEncoder, le2: LabelEncoder):
    set_seed(seed)
    device = cfg["device"]

    # split
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["lvl2_id"], random_state=seed)

    # tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_dataset = NewsHierDataset(train_df["text"].tolist(), train_df["lvl1_id"].tolist(), train_df["lvl2_id"].tolist(), tokenizer, max_len=cfg["max_len"])
    val_dataset = NewsHierDataset(val_df["text"].tolist(), val_df["lvl1_id"].tolist(), val_df["lvl2_id"].tolist(), tokenizer, max_len=cfg["max_len"])

    # sampler + loaders
    sampler, class_weight_tensor = build_sampler(train_df, "lvl2_id", device)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # model init
    num_lvl1, num_lvl2 = len(le1.classes_), len(le2.classes_)
    model = HierarchicalClassifier(cfg["model_name"], num_lvl1, num_lvl2).to(device)
    freeze_bottom_layers(model, cfg["freeze_layers"])

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
    total_steps = math.ceil(len(train_loader) * cfg["num_epochs"] / cfg["accumulation_steps"])
    warmup_steps = int(cfg["warmup_proportion"] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # losses
    if cfg["use_focal_loss"]:
        loss_lvl2 = FocalLoss(cfg["focal_loss_gamma"], class_weight_tensor)
    else:
        loss_lvl2 = nn.CrossEntropyLoss(weight=class_weight_tensor)
    lvl1_counts = train_df["lvl1_id"].value_counts()
    lvl1_weights = {i: 1.0/lvl1_counts.get(i, 1.0) for i in range(len(le1.classes_))}
    lvl1_weight_tensor = torch.tensor([lvl1_weights[i] for i in range(len(le1.classes_))], dtype=torch.float).to(device)
    loss_lvl1 = nn.CrossEntropyLoss(weight=lvl1_weight_tensor)

    scaler = GradScaler()
    best_val_f1 = -1.0
    model_dir = os.path.join(cfg["output_dir"], f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)

    # training loop
    for epoch in range(cfg["num_epochs"]):
        model.train()
        if epoch == 1:
            print("[INFO] Unfreezing all layers for full fine-tuning.")
            unfreeze_all(model)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Seed {seed} Epoch {epoch+1}")
        for step, batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y1 = batch["lvl1"].to(device)
            y2 = batch["lvl2"].to(device)

            # proper autocast device selection
            autocast_device = "cuda" if device.startswith("cuda") else "cpu"
            with amp.autocast(device_type=autocast_device):
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
                scheduler.step()

            if (step + 1) % 200 == 0:
                pbar.set_postfix({"loss": running_loss / (step + 1)})

        # validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"[Seed {seed}] Epoch {epoch+1} Validation: lvl2_macro_f1={val_metrics['lvl2_macro_f1']:.4f} lvl2_acc={val_metrics['lvl2_acc']:.4f}")

        if val_metrics["lvl2_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["lvl2_macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cfg": cfg
            }, os.path.join(model_dir, "best_model.pt"))
            # also save tokenizer
            tokenizer.save_pretrained(model_dir)

    return model_dir

# -----------------------
# Ensemble inference
# -----------------------
def load_model_for_inference(model_dir: str, cfg: Dict, device: str, num_lvl1: int, num_lvl2: int):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = HierarchicalClassifier(cfg["model_name"], num_lvl1, num_lvl2).to(device)
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return tokenizer, model

def ensemble_predict(texts: List[str], model_dirs: List[str], cfg: Dict, le1: LabelEncoder, le2: LabelEncoder):
    device = cfg["device"]
    tokenizers, models = [], []
    for d in model_dirs:
        tok, mdl = load_model_for_inference(d, cfg, device, len(le1.classes_), len(le2.classes_))
        tokenizers.append(tok); models.append(mdl)

    enc = tokenizers[0](texts, truncation=True, padding=True, max_length=cfg["max_len"], return_tensors="pt")
    input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    logits1_list, logits2_list = [], []
    with torch.no_grad():
        for model in models:
            l1, l2 = model(input_ids, attention_mask)
            logits1_list.append(l1.cpu().numpy())
            logits2_list.append(l2.cpu().numpy())

    avg_logits1 = np.mean(np.stack(logits1_list, axis=0), axis=0)
    avg_logits2 = np.mean(np.stack(logits2_list, axis=0), axis=0)
    preds1 = np.argmax(avg_logits1, axis=1)
    preds2 = np.argmax(avg_logits2, axis=1)
    return le1.inverse_transform(preds1), le2.inverse_transform(preds2)

# -----------------------
# Run full pipeline (with augmentation once)
# -----------------------
def run_full_pipeline(cfg: Dict):
    df, le1, le2 = load_and_prepare_df(cfg)
    print("[INFO] Loaded dataset:", len(df), "rows")
    print("[INFO] Level-1 classes:", len(le1.classes_), "Level-2 classes:", len(le2.classes_))

    # Perform augmentation ONCE (cached)
    if cfg["use_t5_aug"]:
        print("[INFO] Starting augmentation (cached). This will run only if cache absent.")
        df_aug = augment_rare_classes_once(df, cfg)
    else:
        df_aug = df

    # Train per-seed
    model_dirs = []
    for seed in cfg["seeds"]:
        print(f"\n========== TRAINING SEED {seed} ==========")
        model_dir = train_one_seed(seed, cfg, df_aug, le1, le2)
        model_dirs.append(model_dir)

    # Quick ensemble check on validation subset
    _, val_df = train_test_split(df_aug, test_size=0.15, stratify=df_aug["lvl2_id"], random_state=cfg["seeds"][0])
    sample_texts = val_df["text"].tolist()[:512]
    lvl1_preds, lvl2_preds = ensemble_predict(sample_texts, model_dirs, cfg, le1, le2)
    print("[INFO] Ensemble example predictions (first 10):")
    for i in range(min(10, len(lvl2_preds))):
        print(f" L1: {lvl1_preds[i]}  L2: {lvl2_preds[i]}")

    print("[INFO] All done. Models saved to", cfg["output_dir"])

# -----------------------
# Entry
# -----------------------
if __name__ == "__main__":
    run_full_pipeline(CONFIG)
