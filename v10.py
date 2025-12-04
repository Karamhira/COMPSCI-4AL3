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
# Enhanced Config / hyperparameters
# ----------------------------
PRETRAINED_MODEL = "roberta-large"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 12  # Increased for better convergence
LR = 2e-5
BACKBONE_LR = 1e-5  # Increased for faster backbone learning
CLASSIFIER_LR = 2e-4  # Increased for classifier
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06  # Reduced warmup
SEED = 42
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./dataset/MN-DS-news-classification.csv"

# More aggressive dropout ensemble
DROPOUTS = [0.2, 0.25, 0.3, 0.35, 0.4]

# Gradient accumulation
ACC_STEPS = 4

# Adjusted loss weights
L1_WEIGHT = 1.0
L2_WEIGHT = 1.5  # Increased to focus more on level2

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
# Focal Loss for imbalanced classes
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ----------------------------
# Enhanced Dataset with augmentation
# ----------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels_lvl1, labels_lvl2, tokenizer, max_len=MAX_LEN, augment=False):
        self.texts = texts
        self.labels_lvl1 = labels_lvl1
        self.labels_lvl2 = labels_lvl2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def simple_augment(self, text):
        """Simple text augmentation for training"""
        words = text.split()
        if len(words) > 10 and self.augment and random.random() < 0.3:
            # Random swap
            if random.random() < 0.3:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            # Random delete
            if random.random() < 0.2 and len(words) > 5:
                del_idx = random.randint(0, len(words)-1)
                words.pop(del_idx)
        return ' '.join(words)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment:
            text = self.simple_augment(text)
        
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
# Enhanced Attention pooling with residual
# ----------------------------
class EnhancedAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, mask):
        # Get attention scores
        scores = self.att(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        
        # Weighted sum
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        
        # Also take [CLS] token and combine
        cls_token = hidden_states[:, 0, :]
        pooled = self.norm(pooled + self.proj(cls_token))
        
        return pooled

# ----------------------------
# Enhanced Model with deeper classifiers
# ----------------------------
class EnhancedHierarchicalRoberta(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, pretrained_model=PRETRAINED_MODEL, dropouts=DROPOUTS):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size
        
        # Enhanced pooling
        self.pooling = EnhancedAttentionPooling(hidden_size)
        
        # Multi-dropout ensemble
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        
        # Enhanced Level1 classifier with residual
        self.lvl1_pre = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )
        self.classifier_lvl1 = nn.Linear(512, num_labels_lvl1)
        
        # Enhanced Level2 classifier
        self.lvl2_pre = nn.Sequential(
            nn.Linear(hidden_size, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )
        self.classifier_lvl2 = nn.Linear(512, num_labels_lvl2)
        
        # Enhanced gate mechanism
        self.gate_projection = nn.Sequential(
            nn.Linear(num_labels_lvl1, 256),
            nn.GELU(),
            nn.Linear(256, num_labels_lvl2)
        )
        
        # Learnable gate weight
        self.gate_weight = nn.Parameter(torch.tensor(0.7))
        
        # Layer for combining features
        self.feature_fusion = nn.Linear(hidden_size + num_labels_lvl1, hidden_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooling(outputs.last_hidden_state, attention_mask)
        
        logits1_all, logits2_all = [], []
        
        for dropout in self.dropouts:
            dropped = dropout(pooled)
            
            # Level1 predictions
            lvl1_features = self.lvl1_pre(dropped)
            logits1 = self.classifier_lvl1(lvl1_features)
            
            # Level2 predictions with enhanced features
            lvl2_features = self.lvl2_pre(dropped)
            
            # Enhanced gate: combine features from Level1 predictions
            lvl1_probs = F.softmax(logits1, dim=-1)
            gate_signal = torch.sigmoid(self.gate_projection(lvl1_probs))
            
            # Combine Level1 information with base features
            combined_features = torch.cat([dropped, lvl1_probs], dim=-1)
            fused_features = self.feature_fusion(combined_features)
            
            # Final Level2 logits
            logits2_base = self.classifier_lvl2(lvl2_features)
            logits2_gated = logits2_base * (1 + self.gate_weight * gate_signal)
            
            logits1_all.append(logits1)
            logits2_all.append(logits2_gated)
        
        # Ensemble average
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
train_ds = NewsDataset(X_train.tolist(), y1_train.tolist(), y2_train.tolist(), tokenizer, augment=True)
val_ds = NewsDataset(X_val.tolist(), y1_val.tolist(), y2_val.tolist(), tokenizer, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ----------------------------
# Class weights & loss functions
# ----------------------------
print("Computing class weights...")
counts_lvl1 = np.bincount(y1_train)
counts_lvl2 = np.bincount(y2_train)

# Smooth class weights
weight_lvl1 = torch.tensor(len(counts_lvl1) / (counts_lvl1 + 1e-6), dtype=torch.float).to(DEVICE)
weight_lvl2 = torch.tensor(len(counts_lvl2) / (counts_lvl2 + 1e-6), dtype=torch.float).to(DEVICE)

weight_lvl1 = weight_lvl1 / weight_lvl1.mean()
weight_lvl2 = weight_lvl2 / weight_lvl2.mean()

print(f"Level1 weight range: {weight_lvl1.min():.2f} - {weight_lvl1.max():.2f}")
print(f"Level2 weight range: {weight_lvl2.min():.2f} - {weight_lvl2.max():.2f}")

# Enhanced loss functions with focal loss for level2
criterion_lvl1 = nn.CrossEntropyLoss(weight=weight_lvl1, label_smoothing=0.15)
criterion_lvl2 = FocalLoss(alpha=0.75, gamma=2.0)  # Focal loss for imbalanced level2

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
print("Initializing model...")
model = EnhancedHierarchicalRoberta(
    num_labels_lvl1=len(le1.classes_),
    num_labels_lvl2=len(le2.classes_)
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Separate optimizer groups with different learning rates
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
        "lr": BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY
    },
    {
        "params": [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay)],
        "lr": BACKBONE_LR,
        "weight_decay": 0.0
    },
    {
        "params": [p for n, p in model.named_parameters() if "roberta" not in n and not any(nd in n for nd in no_decay)],
        "lr": CLASSIFIER_LR,
        "weight_decay": WEIGHT_DECAY
    },
    {
        "params": [p for n, p in model.named_parameters() if "roberta" not in n and any(nd in n for nd in no_decay)],
        "lr": CLASSIFIER_LR,
        "weight_decay": 0.0
    },
]

optimizer = AdamW(optimizer_grouped_parameters)

# Cosine annealing scheduler
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = GradScaler()

# ----------------------------
# Training with early stopping
# ----------------------------
best_f1_lvl2 = 0
patience_counter = 0
best_model_state = None

print("\n" + "="*50)
print("Starting training...")
print("="*50)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels1 = batch["label_lvl1"].to(DEVICE, non_blocking=True)
        labels2 = batch["label_lvl2"].to(DEVICE, non_blocking=True)
        
        with autocast(device_type='cuda' if DEVICE.type == 'cuda' else 'cpu'):
            logits1, logits2 = model(input_ids, attention_mask)
            
            # Combined loss with adaptive weights
            loss1 = criterion_lvl1(logits1, labels1)
            loss2 = criterion_lvl2(logits2, labels2)
            
            # Dynamic weight adjustment based on epoch
            epoch_factor = min(1.0, (epoch + 1) / 3)
            current_l2_weight = L2_WEIGHT * epoch_factor
            
            raw_loss = L1_WEIGHT * loss1 + current_l2_weight * loss2
            loss = raw_loss / ACC_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % ACC_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        running_loss += raw_loss.item()
        pbar.set_postfix({
            "loss": f"{running_loss / (step + 1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} — avg training loss: {avg_loss:.4f}")
    
    # ----------------------------
    # Enhanced Validation
    # ----------------------------
    model.eval()
    all_p1, all_l1, all_p2, all_l2 = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
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
    f1m1 = f1_score(all_l1, all_p1, average='macro')
    acc2 = accuracy_score(all_l2, all_p2)
    f1w2 = f1_score(all_l2, all_p2, average='weighted')
    f1m2 = f1_score(all_l2, all_p2, average='macro')
    
    print("\n" + "="*50)
    print("** Validation results **")
    print(f"Level1 — acc: {acc1:.4f}, F1-weighted: {f1w1:.4f}, F1-macro: {f1m1:.4f}")
    print(f"Level2 — acc: {acc2:.4f}, F1-weighted: {f1w2:.4f}, F1-macro: {f1m2:.4f}")
    print("="*50 + "\n")
    
    # Early stopping check
    current_f1 = (f1w1 + f1w2) / 2  # Average of both levels
    
    if current_f1 - best_f1_lvl2 > MIN_DELTA:
        best_f1_lvl2 = current_f1
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1_lvl2,
            'acc1': acc1,
            'acc2': acc2,
        }, "best_model_checkpoint.pt")
        print(f"✓ New best model saved! (Avg F1: {current_f1:.4f})")
        
        # Also save the best model separately
        torch.save(best_model_state, f"best_model_epoch{epoch+1}.pt")
    else:
        patience_counter += 1
        print(f"✗ No improvement for {patience_counter}/{PATIENCE} epochs")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save regular checkpoint
    torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pt")

# Load best model for final evaluation
if best_model_state is not None:
    print("\nLoading best model for final evaluation...")
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

# Final metrics
acc1 = accuracy_score(all_l1, all_p1)
f1w1 = f1_score(all_l1, all_p1, average='weighted')
acc2 = accuracy_score(all_l2, all_p2)
f1w2 = f1_score(all_l2, all_p2, average='weighted')

print("\n" + "="*60)
print("FINAL RESULTS (Best Model)")
print("="*60)
print(f"Level1 — Accuracy: {acc1:.4f} | F1-weighted: {f1w1:.4f}")
print(f"Level2 — Accuracy: {acc2:.4f} | F1-weighted: {f1w2:.4f}")
print("="*60)

if acc1 >= 0.90 and acc2 >= 0.80:
    print("🎯 TARGETS ACHIEVED!")
elif acc1 >= 0.88 and acc2 >= 0.75:
    print("✅ Close to targets - consider training a bit longer")
else:
    print("⚠️ Targets not reached - consider adjusting hyperparameters")

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'le1_classes': le1.classes_,
    'le2_classes': le2.classes_,
    'tokenizer_config': tokenizer.init_kwargs,
}, "final_hierarchical_model.pt")
print("\nModel saved as 'final_hierarchical_model.pt'")