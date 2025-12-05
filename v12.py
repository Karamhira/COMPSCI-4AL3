import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ----------------------------
#  Config / hyperparameters
# ----------------------------
PRETRAINED_MODEL = "roberta-large"  # Larger model for better performance
MAX_LEN = 384  # Increased to capture more context
BATCH_SIZE = 8  # Reduced for roberta-large (larger model needs more memory)
EPOCHS = 15  # More epochs for better convergence
LR = 2e-5  # Slightly higher LR for faster convergence to 90%
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05  # Shorter warmup for faster learning
GRAD_CLIP = 1.0  # Gradient clipping
LOSS_WEIGHT_LVL1 = 0.65  # Favor Level 1 more to hit 90% target
LOSS_WEIGHT_LVL2 = 0.35  # Reduce Level 2 weight
LABEL_SMOOTHING = 0.05  # Reduce label smoothing (was too aggressive)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "MN-DS-news-classification.csv"  # Fixed path

# GPU verification
if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
else:
    print("⚠️  CUDA not available, using CPU (training will be slower)")

# set seed
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
    def __init__(self, num_labels_lvl1, num_labels_lvl2, pretrained_model=PRETRAINED_MODEL, dropout_prob=0.25):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size
        
        # Better pooling: mean pooling with attention mask
        self.pooling_dropout = nn.Dropout(dropout_prob)
        
        # Intermediate layers for better feature extraction
        intermediate_size = int(hidden_size * 0.75)  # Slightly larger intermediate
        
        # Level 1 pathway with residual connection
        self.lvl1_intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.classifier_lvl1 = nn.Sequential(
            nn.Dropout(dropout_prob * 0.5),
            nn.Linear(hidden_size, num_labels_lvl1)
        )
        
        # Level 2 pathway with residual connection
        self.lvl2_intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.classifier_lvl2 = nn.Sequential(
            nn.Dropout(dropout_prob * 0.5),
            nn.Linear(hidden_size, num_labels_lvl2)
        )

    def mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Combine CLS token and mean pooling for better representation
        cls_token = last_hidden[:, 0, :]  # CLS token
        mean_pooled = self.mean_pooling(last_hidden, attention_mask)
        # Weighted combination: 40% CLS, 60% mean (CLS is good for classification, mean captures full context)
        pooled = 0.4 * cls_token + 0.6 * mean_pooled
        pooled = self.pooling_dropout(pooled)
        
        # Separate pathways for each level with residual connections
        features_lvl1 = self.lvl1_intermediate(pooled) + pooled  # Residual connection
        features_lvl2 = self.lvl2_intermediate(pooled) + pooled  # Residual connection
        
        logits1 = self.classifier_lvl1(features_lvl1)
        logits2 = self.classifier_lvl2(features_lvl2)
        return logits1, logits2

# ----------------------------
#  Load + preprocess data
# ----------------------------
if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    # combine title + content for full text
    df["text_full"] = df["title"].fillna("") + " " + df["content"].fillna("")

    # encode labels
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

    # Optimized DataLoader with pin_memory for faster GPU transfer
    # Using num_workers=0 on Windows to avoid multiprocessing issues
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 on Windows to avoid multiprocessing issues
        pin_memory=True if DEVICE.type == "cuda" else False  # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    # ----------------------------
    #  Class weights for imbalance (improved calculation)
    # ----------------------------
    counts_lvl1 = np.bincount(y1_train)
    counts_lvl2 = np.bincount(y2_train)

    # Use inverse frequency with smoothing
    total_lvl1 = counts_lvl1.sum()
    total_lvl2 = counts_lvl2.sum()
    weight_lvl1 = torch.tensor(total_lvl1 / (len(counts_lvl1) * (counts_lvl1 + 1e-6)), dtype=torch.float).to(DEVICE)
    weight_lvl2 = torch.tensor(total_lvl2 / (len(counts_lvl2) * (counts_lvl2 + 1e-6)), dtype=torch.float).to(DEVICE)

    # Normalize weights
    weight_lvl1 = weight_lvl1 / weight_lvl1.sum() * len(weight_lvl1)
    weight_lvl2 = weight_lvl2 / weight_lvl2.sum() * len(weight_lvl2)

    # Use label smoothing for better generalization
    criterion_lvl1 = nn.CrossEntropyLoss(weight=weight_lvl1, label_smoothing=LABEL_SMOOTHING)
    criterion_lvl2 = nn.CrossEntropyLoss(weight=weight_lvl2, label_smoothing=LABEL_SMOOTHING)

    # ----------------------------
    #  Initialize model, optimizer, scheduler
    # ----------------------------
    model = HierarchicalRoberta(
        num_labels_lvl1=len(le1.classes_),
        num_labels_lvl2=len(le2.classes_)
    ).to(DEVICE)

    # Compile model for faster training (PyTorch 2.0+)
    # Note: torch.compile requires Triton which doesn't work well on Windows
    # Disabled for Windows compatibility - model will still train fine without it
    # if hasattr(torch, 'compile') and DEVICE.type == "cuda":
    #     try:
    #         model = torch.compile(model, mode='default')
    #         print("✅ Model compiled for faster training")
    #     except Exception as e:
    #         print(f"⚠️  Model compilation failed: {e}, continuing without compilation")

    # Different learning rates for backbone and classifier
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "roberta" in n],
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "roberta" not in n],
            "lr": LR * 2,  # Higher LR for classifier layers
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": LR,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    # Use cosine annealing for better convergence (smoother decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=0.5
    )

    # Mixed precision training for GPU speedup (optimized settings)
    if DEVICE.type == "cuda":
        scaler = GradScaler('cuda', init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5)
        print("✅ Mixed precision training enabled (AMP)")
    else:
        scaler = None

    # ----------------------------
    #  Training + eval loop
    # ----------------------------
    best_lvl1_f1 = 0.0
    best_lvl2_f1 = 0.0
    patience = 4  # More patience to allow model to converge
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels1 = batch["label_lvl1"].to(DEVICE)
            labels2 = batch["label_lvl2"].to(DEVICE)

            # Mixed precision training for GPU
            if scaler is not None:
                with autocast('cuda'):
                    logits1, logits2 = model(input_ids, attention_mask)
                    loss1 = criterion_lvl1(logits1, labels1)
                    loss2 = criterion_lvl2(logits2, labels2)
                    loss = LOSS_WEIGHT_LVL1 * loss1 + LOSS_WEIGHT_LVL2 * loss2

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()  # Fixed: accumulate loss for GPU path
            else:
                logits1, logits2 = model(input_ids, attention_mask)
                loss1 = criterion_lvl1(logits1, labels1)
                loss2 = criterion_lvl2(logits2, labels2)
                loss = LOSS_WEIGHT_LVL1 * loss1 + LOSS_WEIGHT_LVL2 * loss2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                running_loss += loss.item()
            
            scheduler.step()  # Fixed: scheduler should step for both GPU and CPU paths

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} — avg training loss: {avg_loss:.4f}")

        # evaluation (also use AMP for faster inference)
        model.eval()
        all_p1, all_l1, all_p2, all_l2 = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels1 = batch["label_lvl1"].to(DEVICE)
                labels2 = batch["label_lvl2"].to(DEVICE)

                # Use AMP for faster validation inference
                if scaler is not None:
                    with autocast('cuda'):
                        logits1, logits2 = model(input_ids, attention_mask)
                else:
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
        print(f"Level1 — acc: {acc1:.4f}, F1‑weighted: {f1w1:.4f}, F1‑macro: {f1m1:.4f}")
        print(f"Level2 — acc: {acc2:.4f}, F1‑weighted: {f1w2:.4f}, F1‑macro: {f1m2:.4f}")
        
        # Track best performance
        current_lvl1_f1 = (f1w1 + f1m1) / 2
        current_lvl2_f1 = (f1w2 + f1m2) / 2
        
        if current_lvl1_f1 > best_lvl1_f1 or current_lvl2_f1 > best_lvl2_f1:
            best_lvl1_f1 = max(best_lvl1_f1, current_lvl1_f1)
            best_lvl2_f1 = max(best_lvl2_f1, current_lvl2_f1)
            patience_counter = 0
            print("✅ New best performance!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏸️  Early stopping triggered (no improvement for {patience} epochs)")
                break
        
        print("-" * 60)

