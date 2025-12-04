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
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ----------------------------
# OPTIMIZED Config for 90%/80% targets
# ----------------------------
PRETRAINED_MODEL = "roberta-large"  # Consider "microsoft/deberta-v3-large" for potential 1-2% boost
MAX_LEN = 512
BATCH_SIZE = 8  # Keep this, but increase effective batch via accumulation
EPOCHS = 12  # Increased from 8
LR = 2e-5
BACKBONE_LR = 1e-5  # Increased from 5e-6
CLASSIFIER_LR = 3e-5  # Increased from 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06  # Reduced warmup for faster convergence
SEED = 42
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./dataset/MN-DS-news-classification.csv"

# More aggressive dropout ensemble
DROPOUTS = [0.15, 0.2, 0.25, 0.3, 0.35]  # More regularization

# Gradient accumulation
ACC_STEPS = 4

# Adjusted loss weights - focus MORE on Level2
L1_WEIGHT = 1.0
L2_WEIGHT = 1.5  # Increased from 1.0 to focus on Level2

# Early stopping
PATIENCE = 4
MIN_DELTA = 0.001

# ----------------------------
# Seeds
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# Enhanced Focal Loss for Level2 (better for imbalanced)
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# ----------------------------
# Dataset with simple augmentation
# ----------------------------
class EnhancedNewsDataset(Dataset):
    def __init__(self, texts, labels_lvl1, labels_lvl2, tokenizer, max_len=MAX_LEN, augment=False):
        self.texts = texts
        self.labels_lvl1 = labels_lvl1
        self.labels_lvl2 = labels_lvl2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Simple augmentation for training only
        if self.augment and random.random() < 0.3:
            # Randomly swap two words
            words = text.split()
            if len(words) > 5:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
                text = ' '.join(words)
        
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
# Enhanced Model with better pooling and classification
# ----------------------------
class EnhancedHierarchicalRoberta(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, pretrained_model=PRETRAINED_MODEL, dropouts=DROPOUTS):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size

        # Enhanced attention pooling
        self.att_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Also use CLS token for robustness
        self.pool_proj = nn.Linear(hidden_size * 2, hidden_size)  # Combine attention + CLS
        
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        
        # Enhanced Level1 classifier
        self.classifier_lvl1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels_lvl1)
        )
        
        # Enhanced Level2 classifier
        self.classifier_lvl2 = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels_lvl2)
        )
        
        # Improved hierarchical connection
        self.proj_lvl1_to_lvl2 = nn.Sequential(
            nn.Linear(num_labels_lvl1, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels_lvl2)
        )
        self.alpha = nn.Parameter(torch.tensor(0.3))  # Learnable gate weight
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Attention pooling
        scores = self.att_pool(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        att_pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        
        # CLS token
        cls_pooled = hidden_states[:, 0, :]
        
        # Combine both pooling methods
        combined = torch.cat([att_pooled, cls_pooled], dim=-1)
        pooled = self.pool_proj(combined)
        
        logits1_all, logits2_all = [], []

        for dropout in self.dropouts:
            dropped = dropout(pooled)
            
            # Level1 predictions
            logits1 = self.classifier_lvl1(dropped)
            
            # Level2 with improved gate
            logits2_base = self.classifier_lvl2(dropped)
            
            # Use soft probabilities from Level1
            lvl1_probs = F.softmax(logits1, dim=-1)
            gate_signal = torch.sigmoid(self.proj_lvl1_to_lvl2(lvl1_probs))
            
            # Additive gate (more stable than multiplication)
            logits2 = logits2_base + self.alpha * gate_signal
            
            logits1_all.append(logits1)
            logits2_all.append(logits2)

        logits1 = torch.stack(logits1_all, dim=0).mean(dim=0)
        logits2 = torch.stack(logits2_all, dim=0).mean(dim=0)
        return logits1, logits2

# ----------------------------
# Load + preprocess data
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
# Create datasets with augmentation
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
train_ds = EnhancedNewsDataset(X_train.tolist(), y1_train.tolist(), y2_train.tolist(), tokenizer, augment=True)
val_ds = EnhancedNewsDataset(X_val.tolist(), y1_val.tolist(), y2_val.tolist(), tokenizer, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ----------------------------
# Improved Class weights & loss
# ----------------------------
print("Computing enhanced class weights...")
counts_lvl1 = np.bincount(y1_train)
counts_lvl2 = np.bincount(y2_train)

# More balanced weights
weight_lvl1 = torch.tensor(len(counts_lvl1) / (counts_lvl1 + 1), dtype=torch.float).to(DEVICE)
weight_lvl2 = torch.tensor(len(counts_lvl2) / (counts_lvl2 + 1), dtype=torch.float).to(DEVICE)

# Normalize weights
weight_lvl1 = weight_lvl1 / weight_lvl1.mean()
weight_lvl2 = weight_lvl2 / weight_lvl2.mean()

print(f"Level1 weight range: {weight_lvl1.min():.2f} - {weight_lvl1.max():.2f}")
print(f"Level2 weight range: {weight_lvl2.min():.2f} - {weight_lvl2.max():.2f}")

# Focal loss for Level2 (better for imbalance), regular for Level1
criterion_lvl1 = nn.CrossEntropyLoss(weight=weight_lvl1, label_smoothing=0.15)
criterion_lvl2 = FocalLoss(alpha=0.25, gamma=2.0)

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
print("Initializing enhanced model...")
model = EnhancedHierarchicalRoberta(
    num_labels_lvl1=len(le1.classes_),
    num_labels_lvl2=len(le2.classes_)
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer with layer-wise decay
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in param_optimizer 
                  if 'roberta' in n and not any(nd in n for nd in no_decay)],
        'lr': BACKBONE_LR,
        'weight_decay': WEIGHT_DECAY
    },
    {
        'params': [p for n, p in param_optimizer 
                  if 'roberta' in n and any(nd in n for nd in no_decay)],
        'lr': BACKBONE_LR,
        'weight_decay': 0.0
    },
    {
        'params': [p for n, p in param_optimizer 
                  if 'roberta' not in n and not any(nd in n for nd in no_decay)],
        'lr': CLASSIFIER_LR,
        'weight_decay': WEIGHT_DECAY
    },
    {
        'params': [p for n, p in param_optimizer 
                  if 'roberta' not in n and any(nd in n for nd in no_decay)],
        'lr': CLASSIFIER_LR,
        'weight_decay': 0.0
    },
]

optimizer = AdamW(optimizer_grouped_parameters)

# Cosine annealing scheduler (better than linear)
from torch.optim.lr_scheduler import CosineAnnealingLR
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

# Warmup then cosine
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = GradScaler()

# ----------------------------
# Training with early stopping
# ----------------------------
best_acc2 = 0
patience_counter = 0
best_epoch = 0
best_model_state = None

print("\n" + "="*60)
print("Starting ENHANCED training for 90%/80% targets")
print("="*60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    
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
            
            # Dynamic weighting - emphasize Level2 more as training progresses
            epoch_factor = min(1.0, (epoch + 1) / 4)  # Ramp up over first 4 epochs
            current_l2_weight = L2_WEIGHT * epoch_factor
            
            loss = L1_WEIGHT * loss1 + current_l2_weight * loss2
            loss = loss / ACC_STEPS  # For gradient accumulation
        
        scaler.scale(loss).backward()
        
        total_loss += loss.item() * ACC_STEPS
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        
        # Gradient accumulation
        if (pbar.n + 1) % ACC_STEPS == 0 or (pbar.n + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        pbar.set_postfix({
            "loss": f"{total_loss/(pbar.n+1):.4f}",
            "loss1": f"{total_loss1/(pbar.n+1):.4f}",
            "loss2": f"{total_loss2/(pbar.n+1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_loss1 = total_loss1 / len(train_loader)
    avg_loss2 = total_loss2 / len(train_loader)
    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} (L1: {avg_loss1:.4f}, L2: {avg_loss2:.4f})")
    
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
    
    # Early stopping and model saving
    if acc2 > best_acc2 + MIN_DELTA:
        best_acc2 = acc2
        best_acc1 = acc1
        best_epoch = epoch + 1
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc2': best_acc2,
            'best_acc1': best_acc1,
            'acc2': acc2,
            'acc1': acc1,
        }
        torch.save(checkpoint, "best_model_checkpoint.pt")
        print(f"✓ New best model saved! (Level1: {acc1:.4f}, Level2: {acc2:.4f})")
    else:
        patience_counter += 1
        print(f"✗ No improvement for {patience_counter}/{PATIENCE} epochs")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save checkpoint every 3 epochs
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pt")

# Load best model
if best_model_state is not None:
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(best_model_state)

# Final evaluation
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

acc1 = accuracy_score(all_l1, all_p1)
f1w1 = f1_score(all_l1, all_p1, average='weighted')
acc2 = accuracy_score(all_l2, all_p2)
f1w2 = f1_score(all_l2, all_p2, average='weighted')

print("\n" + "="*70)
print("FINAL RESULTS (Best Model)")
print("="*70)
print(f"Level1 — Accuracy: {acc1:.4f} | F1-weighted: {f1w1:.4f}")
print(f"Level2 — Accuracy: {acc2:.4f} | F1-weighted: {f1w2:.4f}")
print("="*70)

# Performance assessment
if acc1 >= 0.90 and acc2 >= 0.80:
    print("\n🎯 TARGETS ACHIEVED! Excellent work!")
elif acc1 >= 0.89 and acc2 >= 0.78:
    print("\n✅ Very close! Just a bit more tuning needed.")
    print("   Try these quick fixes:")
    print("   1. Train for 2-3 more epochs")
    print("   2. Reduce learning rate by 50% for final epochs")
    print("   3. Add test-time augmentation")
elif acc1 >= 0.87 and acc2 >= 0.75:
    print("\n⚠️ Good but not at targets. Next steps:")
    print("   1. Switch to 'microsoft/deberta-v3-large'")
    print("   2. Increase epochs to 15-20")
    print("   3. Use 5-fold cross validation")
else:
    print("\n❌ Need more improvement. Consider:")
    print("   1. Different model architecture")
    print("   2. More data augmentation")
    print("   3. Hyperparameter tuning")

# Save final model
final_state = {
    'model_state_dict': model.state_dict(),
    'le1_classes': le1.classes_,
    'le2_classes': le2.classes_,
    'tokenizer_config': tokenizer.init_kwargs,
    'metrics': {'acc1': acc1, 'acc2': acc2, 'f1w1': f1w1, 'f1w2': f1w2}
}
torch.save(final_state, "enhanced_final_model.pt")
print(f"\nEnhanced model saved as 'enhanced_final_model.pt'")

# Quick improvements you can try immediately:
print("\n" + "="*70)
print("QUICK WINS FOR BETTER PERFORMANCE:")
print("="*70)
print("1. MODEL SWITCH (run this now):")
print("   PRETRAINED_MODEL = 'microsoft/deberta-v3-large'")
print("   # Often gives 1-3% boost over RoBERTa")

print("\n2. LEARNING RATE SCHEDULE:")
print("   scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)")
print("   # Better convergence than linear")

print("\n3. TEST TIME AUGMENTATION (simple):")
print("   def predict_with_tta(text):")
print("       # Make predictions on 3 slight variations")
print("       # Average the results")
print("       # Can add 0.5-1.5% accuracy")

print("\n4. ENSEMBLE (if you have time):")
print("   # Train 3 models with different seeds")
print("   # Average their predictions")
print("   # Usually adds 1-3% accuracy")
print("="*70)