import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# SIMPLIFIED Config (Focus on stability)
# ----------------------------
PRETRAINED_MODEL = "roberta-large"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 10  # Start with fewer epochs
LR = 2e-5  # Standard LR
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./dataset/MN-DS-news-classification.csv"

# ----------------------------
# Seeds
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# SIMPLE Dataset (No augmentation for now)
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
# SIMPLE Model (Independent classifiers)
# ----------------------------
class SimpleHierarchicalRoberta(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, pretrained_model=PRETRAINED_MODEL):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size
        
        # Simple mean pooling
        self.pooling = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size)
        )
        
        # Independent classifiers
        self.classifier_lvl1 = nn.Linear(hidden_size, num_labels_lvl1)
        self.classifier_lvl2 = nn.Linear(hidden_size, num_labels_lvl2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Apply pooling layer
        pooled = self.pooling(pooled)
        
        # Separate predictions
        logits1 = self.classifier_lvl1(pooled)
        logits2 = self.classifier_lvl2(pooled)
        
        return logits1, logits2

# ----------------------------
# Data Loading
# ----------------------------
print("Loading and preprocessing data...")
df = pd.read_csv(CSV_PATH)
df["text_full"] = df["title"].fillna("") + " " + df["content"].fillna("")

le1 = LabelEncoder()
df["label_lvl1"] = le1.fit_transform(df["category_level_1"])
le2 = LabelEncoder()
df["label_lvl2"] = le2.fit_transform(df["category_level_2"])

print(f"Level1 classes: {len(le1.classes_)}")
print(f"Level2 classes: {len(le2.classes_)}")

X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
    df["text_full"], df["label_lvl1"], df["label_lvl2"],
    test_size=0.2, random_state=SEED, stratify=df["label_lvl1"]
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# ----------------------------
# Create datasets
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
train_ds = NewsDataset(X_train.tolist(), y1_train.tolist(), y2_train.tolist(), tokenizer)
val_ds = NewsDataset(X_val.tolist(), y1_val.tolist(), y2_val.tolist(), tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ----------------------------
# Class weights (IMPORTANT for Level2)
# ----------------------------
print("Computing class weights for Level2...")
counts_lvl2 = np.bincount(y2_train)

# Calculate balanced class weights
weight_lvl2 = torch.tensor(1.0 / (counts_lvl2 + 1e-6), dtype=torch.float32)
weight_lvl2 = weight_lvl2 / weight_lvl2.sum() * len(weight_lvl2)  # Normalize
weight_lvl2 = weight_lvl2.to(DEVICE)

print(f"Level2 class weights range: {weight_lvl2.min():.2f} to {weight_lvl2.max():.2f}")

# Loss functions with class weights for Level2
criterion_lvl1 = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_lvl2 = nn.CrossEntropyLoss(weight=weight_lvl2, label_smoothing=0.1)

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
print("Initializing simple model...")
model = SimpleHierarchicalRoberta(
    num_labels_lvl1=len(le1.classes_),
    num_labels_lvl2=len(le2.classes_)
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Simple optimizer (all parameters same LR)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Linear schedule
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = GradScaler()

# ----------------------------
# Training
# ----------------------------
print("\n" + "="*60)
print("Starting SIMPLE training (debug mode)")
print("="*60)

# Store predictions for analysis
all_predictions = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct_lvl1 = 0
    total_correct_lvl2 = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{EPOCHS}")
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels1 = batch["label_lvl1"].to(DEVICE)
        labels2 = batch["label_lvl2"].to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast(device_type=DEVICE.type):
            logits1, logits2 = model(input_ids, attention_mask)
            
            loss1 = criterion_lvl1(logits1, labels1)
            loss2 = criterion_lvl2(logits2, labels2)
            
            # Equal weighting
            loss = loss1 + loss2
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Training accuracy
        preds1 = torch.argmax(logits1, dim=1)
        preds2 = torch.argmax(logits2, dim=1)
        total_correct_lvl1 += (preds1 == labels1).sum().item()
        total_correct_lvl2 += (preds2 == labels2).sum().item()
        total_samples += labels1.size(0)
        
        total_loss += loss.item()
        
        train_acc1 = total_correct_lvl1 / total_samples
        train_acc2 = total_correct_lvl2 / total_samples
        
        pbar.set_postfix({
            "loss": f"{total_loss/(pbar.n+1):.4f}",
            "train_L1": f"{train_acc1:.4f}",
            "train_L2": f"{train_acc2:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f}, Train L1: {train_acc1:.4f}, Train L2: {train_acc2:.4f}")
    
    # ----------------------------
    # Validation
    # ----------------------------
    model.eval()
    all_p1, all_l1, all_p2, all_l2 = [], [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            logits1, logits2 = model(input_ids, attention_mask)
            preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
            preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
            
            all_p1.extend(preds1)
            all_l1.extend(batch["label_lvl1"].numpy())
            all_p2.extend(preds2)
            all_l2.extend(batch["label_lvl2"].numpy())
    
    # Store for analysis
    if epoch == 0:
        all_predictions.append({
            'epoch': epoch + 1,
            'preds_lvl2': all_p2[:100],  # Store first 100 predictions
            'true_lvl2': all_l2[:100]
        })
    
    # Compute metrics
    acc1 = accuracy_score(all_l1, all_p1)
    f1w1 = f1_score(all_l1, all_p1, average='weighted')
    acc2 = accuracy_score(all_l2, all_p2)
    f1w2 = f1_score(all_l2, all_p2, average='weighted')
    
    print("\n" + "="*50)
    print(f"Epoch {epoch+1} Validation Results")
    print("="*50)
    print(f"Level1 — Acc: {acc1:.4f} | F1-w: {f1w1:.4f}")
    print(f"Level2 — Acc: {acc2:.4f} | F1-w: {f1w2:.4f}")
    print("="*50 + "\n")
    
    # Early stopping check
    if epoch == 0:
        expected_lvl2 = 1 / len(le2.classes_)  # Random chance
        print(f"Expected random Level2 accuracy: {expected_lvl2:.4f} (1/{len(le2.classes_)})")
        print(f"Actual Level2 accuracy: {acc2:.4f}")
        
        if acc2 < expected_lvl2 * 2:  # Less than 2x random chance
            print("\n⚠️ WARNING: Level2 accuracy is too low!")
            print("Possible issues:")
            print("1. Labels might be shuffled")
            print("2. Class imbalance is severe")
            print("3. Model isn't learning Level2 features")
            
            # Analyze predictions
            print("\nFirst 10 Level2 predictions:")
            for i in range(min(10, len(all_p2))):
                print(f"  True: {le2.classes_[all_l2[i]]} ({all_l2[i]}) | Pred: {le2.classes_[all_p2[i]]} ({all_p2[i]})")
    
    # Save model
    torch.save(model.state_dict(), f"simple_model_epoch{epoch+1}.pt")

# ----------------------------
# Final Analysis
# ----------------------------
print("\n" + "="*70)
print("TRAINING DIAGNOSTICS")
print("="*70)

# Check class distribution
print("\nLevel2 Class Distribution (Top 20):")
sorted_indices = np.argsort(counts_lvl2)[::-1]
for i, idx in enumerate(sorted_indices[:20]):
    print(f"  {le2.classes_[idx]:30s}: {counts_lvl2[idx]:4d} samples ({counts_lvl2[idx]/len(y2_train):.2%})")

# Check if model is predicting only majority class
print(f"\nMajority class baseline for Level2: {counts_lvl2.max()/len(y2_train):.4f}")

# Show confusion patterns
print("\nMost common Level2 predictions:")
if len(all_p2) > 0:
    pred_counts = np.bincount(all_p2, minlength=len(le2.classes_))
    top_pred_indices = np.argsort(pred_counts)[-5:][::-1]
    for idx in top_pred_indices:
        if pred_counts[idx] > 0:
            print(f"  {le2.classes_[idx]:30s}: {pred_counts[idx]:3d} predictions")

print("\n" + "="*70)
print("NEXT STEPS BASED ON RESULTS")
print("="*70)

if acc2 < 0.3:
    print("⚠️ Level2 accuracy < 30% - Serious problem!")
    print("Try these fixes:")
    print("1. CHECK LABELS: Make sure Level2 labels are correct")
    print("2. SIMPLER MODEL: Use roberta-base instead of roberta-large")
    print("3. BATCH SIZE: Increase to 16 or 32")
    print("4. LEARNING RATE: Try 3e-5 or 5e-5")
    print("5. EPOCHS: Train for 20+ epochs")
elif acc2 < 0.5:
    print("⚠️ Level2 accuracy 30-50% - Needs improvement")
    print("Try:")
    print("1. Add class weights (already done)")
    print("2. Train longer (20+ epochs)")
    print("3. Add data augmentation")
    print("4. Try focal loss instead of weighted CE")
elif acc2 < 0.7:
    print("✅ Level2 accuracy 50-70% - Good progress")
    print("To reach 80%:")
    print("1. Train for 20-30 epochs")
    print("2. Add hierarchical connection (carefully)")
    print("3. Try model ensemble")
    print("4. Use deberta-v3-large")
else:
    print("🎉 Level2 accuracy > 70% - Excellent!")
    print("Continue training to reach 80%")

# Save final model
final_state = {
    'model_state_dict': model.state_dict(),
    'le1_classes': le1.classes_,
    'le2_classes': le2.classes_,
    'class_weights_lvl2': weight_lvl2.cpu().numpy(),
    'metrics': {'acc1': acc1, 'acc2': acc2, 'f1w1': f1w1, 'f1w2': f1w2}
}
torch.save(final_state, "simple_final_model.pt")
print(f"\nModel saved as 'simple_final_model.pt'")