# updated_hier_roberta_multidrop_fp16_fix.py
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Optional Lookahead optimizer (try import, otherwise fall back)
try:
    from torch_optimizer import Lookahead
    HAVE_LOOKAHEAD = True
except Exception:
    HAVE_LOOKAHEAD = False

# ----------------------------
# Config / hyperparameters
# ----------------------------
PRETRAINED_MODEL = "roberta-large"
MAX_LEN = 384
BATCH_SIZE = 8
EPOCHS = 16         # increased default (you can reduce)
LR = 2e-5
BACKBONE_LR = 5e-6
CLASSIFIER_LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06   # smaller warmup for longer training
SEED = 42
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./dataset/MN-DS-news-classification.csv"

# Safer dropout ensemble (expanded)
DROPOUTS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# --------- Gradient accumulation setting ----------
ACC_STEPS = 4  # effective batch size = BATCH_SIZE * ACC_STEPS

# Hierarchical loss weights (light)
L1_WEIGHT = 1.2
L2_WEIGHT = 1.0

# R-Drop weight
R_DROP_WEIGHT = 0.5

# ----------------------------
# Seeds
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# Utilities (mean_pool kept for reference)
# ----------------------------
def mean_pool(hidden_states, mask):
    mask = mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

# ----------------------------
# Dataset
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
# Attention pooling (lightweight) - SAFE for fp16 now
# ----------------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.att = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, mask):
        """
        hidden_states: (batch, seq_len, hidden)
        mask: (batch, seq_len) -- 1 for real tokens, 0 for padding
        Works safely under autocast/float16 by using a dtype-appropriate fill value.
        """
        # ensure mask is on same device
        mask = mask.to(device=hidden_states.device)

        scores = self.att(hidden_states).squeeze(-1)           # (batch, seq_len)

        # boolean mask
        mask_bool = mask.to(dtype=torch.bool)

        # choose a safe large negative fill value compatible with scores dtype
        # e.g., float16 min ~ -65504, so using torch.finfo(scores.dtype).min is safe
        if scores.is_floating_point():
            fill_value = torch.finfo(scores.dtype).min
        else:
            # fallback to a large negative float that will be cast
            fill_value = -1e4

        # masked_fill where mask == 0 (i.e., ~mask_bool)
        scores = scores.masked_fill(~mask_bool, fill_value)

        weights = torch.softmax(scores, dim=1)                # (batch, seq_len)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # (batch, hidden)
        return pooled

# ----------------------------
# Model with hierarchical gating matrix + R-Drop support
# ----------------------------
class HierarchicalRobertaMultiDropout(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, hierarchy_prior=None,
                 pretrained_model=PRETRAINED_MODEL, dropouts=DROPOUTS, alpha=0.5):
        """
        hierarchy_prior: numpy array shape (num_lvl1, num_lvl2) expressing P(lvl2 | lvl1) or 0/1 mapping.
        We'll register it as a buffer and normalize rows for a prior.
        """
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size

        # attention pooling
        self.att_pool = AttentionPooling(hidden_size)

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        self.classifier_lvl1 = nn.Linear(hidden_size, num_labels_lvl1)
        self.classifier_lvl2 = nn.Linear(hidden_size, num_labels_lvl2)

        # gating: dataset-derived prior matrix
        if hierarchy_prior is None:
            # fallback: learn a linear projection if no prior provided
            self.register_buffer("hierarchy_prior", torch.zeros(num_labels_lvl1, num_labels_lvl2))
        else:
            # make sure normalized by row
            prior = np.array(hierarchy_prior, dtype=np.float32)
            # avoid zero rows; add tiny epsilon
            row_sums = prior.sum(axis=1, keepdims=True) + 1e-9
            normalized = prior / row_sums
            self.register_buffer("hierarchy_prior", torch.tensor(normalized, dtype=torch.float32))

        self.alpha = alpha  # gating weight

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.att_pool(outputs.last_hidden_state, attention_mask)  # (batch, hidden)

        logits1_all, logits2_all = [], []

        for dropout in self.dropouts:
            dropped = dropout(pooled)
            logits1 = self.classifier_lvl1(dropped)  # (batch, num_lvl1)
            logits2 = self.classifier_lvl2(dropped)  # (batch, num_lvl2)

            # compute soft gating using logits1 -> softmax -> prior multiplication
            gate = torch.softmax(logits1, dim=1)   # (batch, num_lvl1)
            # hierarchy_prior: (num_lvl1, num_lvl2) -> gate @ prior => (batch, num_lvl2)
            if self.hierarchy_prior.numel() > 0:
                prior = gate @ self.hierarchy_prior  # (batch, num_lvl2)
                logits2 = logits2 + self.alpha * prior

            logits1_all.append(logits1)
            logits2_all.append(logits2)

        logits1 = torch.stack(logits1_all, dim=0).mean(dim=0)
        logits2 = torch.stack(logits2_all, dim=0).mean(dim=0)
        return logits1, logits2

# ----------------------------
# Load + preprocess data
# ----------------------------
df = pd.read_csv(CSV_PATH)
df["text_full"] = df["title"].fillna("") + " " + df["content"].fillna("")

le1 = LabelEncoder()
df["label_lvl1"] = le1.fit_transform(df["category_level_1"])
le2 = LabelEncoder()
df["label_lvl2"] = le2.fit_transform(df["category_level_2"])

# Build hierarchy prior matrix P(lvl2 | lvl1)
num_lvl1 = len(le1.classes_)
num_lvl2 = len(le2.classes_)
hierarchy = np.zeros((num_lvl1, num_lvl2), dtype=np.float32)
for _, row in df.iterrows():
    i = int(row["label_lvl1"])
    j = int(row["label_lvl2"])
    hierarchy[i, j] = 1.0
# normalize rows later inside model init (we pass raw matrix)

# split (stratify by level1 to keep balanced)
X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    df["text_full"], df["label_lvl1"], df["label_lvl2"],
    test_size=0.2, random_state=SEED, stratify=df["label_lvl1"]
)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
train_ds = NewsDataset(X_train.tolist(), y1_train.tolist(), y2_train.tolist(), tokenizer)
val_ds = NewsDataset(X_val.tolist(), y1_val.tolist(), y2_val.tolist(), tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# Class weights & loss (with label smoothing)
# ----------------------------
counts_lvl1 = np.bincount(y1_train)
counts_lvl2 = np.bincount(y2_train)
weight_lvl1 = torch.tensor(1.0 / (counts_lvl1 + 1e-6), dtype=torch.float).to(DEVICE)
weight_lvl2 = torch.tensor(1.0 / (counts_lvl2 + 1e-6), dtype=torch.float).to(DEVICE)

criterion_lvl1 = nn.CrossEntropyLoss(weight=weight_lvl1, label_smoothing=0.1)
criterion_lvl2 = nn.CrossEntropyLoss(weight=weight_lvl2, label_smoothing=0.1)
kl_div_loss = nn.KLDivLoss(reduction="batchmean")  # for R-Drop (expects log-probs vs probs)

# ----------------------------
# Model, optimizer, scheduler, scaler
# ----------------------------
model = HierarchicalRobertaMultiDropout(
    num_labels_lvl1=num_lvl1,
    num_labels_lvl2=num_lvl2,
    hierarchy_prior=hierarchy,
    pretrained_model=PRETRAINED_MODEL,
    dropouts=DROPOUTS,
    alpha=0.5
).to(DEVICE)

# ---------- Layer-wise LR decay helper ----------
def get_llrd_params(model, base_lr=BACKBONE_LR, head_lr=CLASSIFIER_LR, decay=0.8):
    """
    Create parameter groups where lower layers get smaller lr; embeddings smallest.
    decay: multiplicative factor per layer from top->bottom (0.8 typical)
    """
    opt_params = []

    # embeddings
    try:
        embeddings_params = list(model.roberta.embeddings.parameters())
        opt_params.append({"params": embeddings_params, "lr": base_lr * (decay ** 24)})
    except Exception:
        # fallback
        pass

    # encoder layers
    num_layers = model.roberta.config.num_hidden_layers
    for i in range(num_layers):
        layer = model.roberta.encoder.layer[i]
        # compute lr multiplier: deeper layers (higher i) get higher lr
        # we want layer 0 (bottom) smallest, layer num_layers-1 largest (closer to base_lr)
        layer_decay = decay ** (num_layers - 1 - i)
        lr = base_lr * (layer_decay + 0.0)
        opt_params.append({"params": layer.parameters(), "lr": lr})

    # pooler or final layernorms if any
    try:
        pooler_params = list(model.roberta.pooler.parameters())
        if pooler_params:
            opt_params.append({"params": pooler_params, "lr": base_lr})
    except Exception:
        pass

    # heads
    opt_params.append({"params": model.classifier_lvl1.parameters(), "lr": head_lr})
    opt_params.append({"params": model.classifier_lvl2.parameters(), "lr": head_lr})

    return opt_params

optimizer = AdamW(get_llrd_params(model, base_lr=BACKBONE_LR, head_lr=CLASSIFIER_LR), weight_decay=WEIGHT_DECAY)

# Optional Lookahead wrapper
if HAVE_LOOKAHEAD:
    optimizer = Lookahead(optimizer, k=5, alpha=0.5)

# total optimizer steps must account for gradient accumulation
steps_per_epoch = (len(train_loader) + ACC_STEPS - 1) // ACC_STEPS
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

scaler = GradScaler()

# ----------------------------
# Training + evaluation (with R-Drop)
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    accumulated_steps = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch+1}/{EPOCHS}")
    optimizer.zero_grad()

    for step, batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels1 = batch["label_lvl1"].to(DEVICE)
        labels2 = batch["label_lvl2"].to(DEVICE)

        # Two forward passes for R-Drop (stochasticity via dropout) inside autocast
        with autocast(device_type='cuda' if DEVICE.type == 'cuda' else 'cpu'):
            logits1_a, logits2_a = model(input_ids, attention_mask)
            logits1_b, logits2_b = model(input_ids, attention_mask)

            # classification losses (use logits_a for main)
            loss1 = criterion_lvl1(logits1_a, labels1)
            loss2 = criterion_lvl2(logits2_a, labels2)
            cls_loss = L1_WEIGHT * loss1 + L2_WEIGHT * loss2

            # R-Drop KL between a and b for both levels
            # KL expects input as log-probs and target probs (we average symmetric)
            logp1_a = torch.log_softmax(logits1_a, dim=1)
            p1_b = torch.softmax(logits1_b, dim=1)
            kl1 = kl_div_loss(logp1_a, p1_b)

            logp1_b = torch.log_softmax(logits1_b, dim=1)
            p1_a = torch.softmax(logits1_a, dim=1)
            kl1 += kl_div_loss(logp1_b, p1_a)
            kl1 = 0.5 * kl1

            logp2_a = torch.log_softmax(logits2_a, dim=1)
            p2_b = torch.softmax(logits2_b, dim=1)
            kl2 = kl_div_loss(logp2_a, p2_b)

            logp2_b = torch.log_softmax(logits2_b, dim=1)
            p2_a = torch.softmax(logits2_a, dim=1)
            kl2 += kl_div_loss(logp2_b, p2_a)
            kl2 = 0.5 * kl2

            rdrop_loss = (kl1 + kl2)

            raw_loss = cls_loss + R_DROP_WEIGHT * rdrop_loss
            loss = raw_loss / ACC_STEPS

        scaler.scale(loss).backward()
        accumulated_steps += 1
        running_loss += raw_loss.item()

        # update on accumulation
        if accumulated_steps % ACC_STEPS == 0:
            # unscale, clip, step, update scaler, zero_grad, then scheduler.step()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)    # optimizer.step() under GradScaler
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()          # <-- CORRECT ORDER: scheduler.step() AFTER optimizer.step()

        pbar.set_postfix({"loss": f"{running_loss / (step + 1):.4f}"})

    # leftover gradients
    if accumulated_steps % ACC_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} — avg training loss: {avg_loss:.4f}")

    # ----------------------------
    # Validation
    # ----------------------------
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
    print("** Validation results **")
    print(f"Level1 — acc: {acc1:.4f}, F1-weighted: {f1w1:.4f}, F1-macro: {f1m1:.4f}")
    print(f"Level2 — acc: {acc2:.4f}, F1-weighted: {f1w2:.4f}, F1-macro: {f1m2:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"hier_roberta_multidrop_epoch{epoch+1}.pt")

print("Training complete.")
