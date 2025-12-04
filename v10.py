import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ---------------------------------
# Config
# ---------------------------------
PRETRAINED_MODEL = "roberta-large"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 8
LR = 2e-5
BACKBONE_LR = 5e-6
CLASSIFIER_LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./dataset/MN-DS-news-classification.csv"
DROPOUTS = [0.1, 0.15, 0.2, 0.25, 0.3]
ACC_STEPS = 4
L1_WEIGHT = 1.2
L2_WEIGHT = 1.0

# ---------------------------------
# Seeds
# ---------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------
# Mixout for regularization
# ---------------------------------
class MixLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mixout_prob=0.5):
        super().__init__(in_features, out_features, bias)
        self.mixout_prob = mixout_prob
        self.register_buffer("target_params", self.weight.data.clone())

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(self.weight) < self.mixout_prob).float()
            weight = mask * self.target_params + (1 - mask) * self.weight
        else:
            weight = self.weight
        return nn.functional.linear(x, weight, self.bias)

# ---------------------------------
# Multi-Head Attention Pooling
# ---------------------------------
class MultiHeadAttentionPool(nn.Module):
    def __init__(self, hidden_size, heads=4):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(hidden_size, heads)

    def forward(self, hidden_states, mask):
        scores = self.W(hidden_states)  # (B, L, H)
        # FP16-safe mask
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, torch.finfo(scores.dtype).min / 2)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("blh,bld->bhd", weights, hidden_states)
        pooled = pooled.mean(dim=1)  # average heads
        return pooled

# ---------------------------------
# Dataset
# ---------------------------------
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
            text, padding='max_length', truncation=True,
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_lvl1": torch.tensor(self.labels_lvl1[idx], dtype=torch.long),
            "label_lvl2": torch.tensor(self.labels_lvl2[idx], dtype=torch.long),
        }

# ---------------------------------
# Model
# ---------------------------------
class HierarchicalRobertaMultiDropout(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2,
                 pretrained_model=PRETRAINED_MODEL, dropouts=DROPOUTS):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size

        self.pool = MultiHeadAttentionPool(hidden_size, heads=4)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])

        self.classifier_lvl1 = MixLinear(hidden_size, num_labels_lvl1, mixout_prob=0.5)
        self.classifier_lvl2 = MixLinear(hidden_size, num_labels_lvl2, mixout_prob=0.5)
        self.proj_lvl1_to_lvl2 = MixLinear(num_labels_lvl1, num_labels_lvl2, mixout_prob=0.7)
        self.alpha = 0.5

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(out.last_hidden_state, attention_mask)

        logits1_all, logits2_all = [], []

        for drop in self.dropouts:
            h = drop(pooled)
            logits1 = self.classifier_lvl1(h)
            logits2 = self.classifier_lvl2(h)

            gate = torch.sigmoid(self.proj_lvl1_to_lvl2(logits1))
            logits2 = logits2 + self.alpha * gate

            logits1_all.append(logits1)
            logits2_all.append(logits2)

        return torch.stack(logits1_all).mean(0), torch.stack(logits2_all).mean(0)

# ---------------------------------
# Load data
# ---------------------------------
df = pd.read_csv(CSV_PATH)
df["text_full"] = df["title"].fillna("") + " " + df["content"].fillna("")

le1 = LabelEncoder()
df["label_lvl1"] = le1.fit_transform(df["category_level_1"])
le2 = LabelEncoder()
df["label_lvl2"] = le2.fit_transform(df["category_level_2"])

X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    df["text_full"], df["label_lvl1"], df["label_lvl2"],
    test_size=0.2, random_state=SEED, stratify=df["label_lvl1"]
)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
train_ds = NewsDataset(X_train.tolist(), y1_train.tolist(), y2_train.tolist(), tokenizer)
val_ds = NewsDataset(X_val.tolist(), y1_val.tolist(), y2_val.tolist(), tokenizer)

# Weighted sampler for Level-2 balance
counts_lvl2 = np.bincount(y2_train)
weights = 1.0 / (counts_lvl2 + 1e-9)
sample_weights = torch.tensor([weights[y] for y in y2_train], dtype=torch.float)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------
# Loss
# ---------------------------------
weight_lvl1 = torch.tensor(1/(np.bincount(y1_train)+1e-6), dtype=torch.float).to(DEVICE)
weight_lvl2 = torch.tensor(1/(np.bincount(y2_train)+1e-6), dtype=torch.float).to(DEVICE)
criterion_lvl1 = nn.CrossEntropyLoss(weight=weight_lvl1)
criterion_lvl2 = nn.CrossEntropyLoss(weight=weight_lvl2)

# ---------------------------------
# Model + optimizer + scheduler + scaler
# ---------------------------------
model = HierarchicalRobertaMultiDropout(len(le1.classes_), len(le2.classes_)).to(DEVICE)

optimizer = AdamW([
    {'params': model.roberta.parameters(), 'lr': BACKBONE_LR},
    {'params': model.classifier_lvl1.parameters(), 'lr': CLASSIFIER_LR},
    {'params': model.classifier_lvl2.parameters(), 'lr': CLASSIFIER_LR},
    {'params': model.proj_lvl1_to_lvl2.parameters(), 'lr': CLASSIFIER_LR},
], weight_decay=WEIGHT_DECAY)

steps_per_epoch = (len(train_loader) + ACC_STEPS - 1) // ACC_STEPS
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

scaler = GradScaler(device="cuda")  # FIX: ensures .scale() works

# ---------------------------------
# Training loop
# ---------------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    acc_steps = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y1 = batch["label_lvl1"].to(DEVICE)
        y2 = batch["label_lvl2"].to(DEVICE)

        with autocast(device_type="cuda"):
            log1, log2 = model(input_ids, mask)
            loss = L1_WEIGHT * criterion_lvl1(log1, y1) + \
                   L2_WEIGHT * criterion_lvl2(log2, y2)
            scaler.scale(loss / ACC_STEPS).backward()

        acc_steps += 1
        running_loss += loss.item()

        if acc_steps % ACC_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        pbar.set_postfix({"loss": f"{running_loss/(acc_steps):.4f}"})

    # Validation
    model.eval()
    A1, P1, A2, P2 = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y1 = batch["label_lvl1"].to(DEVICE)
            y2 = batch["label_lvl2"].to(DEVICE)

            log1, log2 = model(ids, mask)
            P1.extend(torch.argmax(log1,1).cpu().numpy())
            A1.extend(y1.cpu().numpy())
            P2.extend(torch.argmax(log2,1).cpu().numpy())
            A2.extend(y2.cpu().numpy())

    print("VAL:")
    print("L1:", accuracy_score(A1,P1), f1_score(A1,P1,average="macro"))
    print("L2:", accuracy_score(A2,P2), f1_score(A2,P2,average="macro"))

    torch.save(model.state_dict(), f"hier_v12_epoch{epoch+1}.pt")
