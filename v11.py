import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Optimized Advanced Config
# ----------------------------
PRETRAINED_MODEL = "roberta-large"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 20
LR = 1.5e-5
BACKBONE_LR = 8e-6
CLASSIFIER_LR = 3e-4
WEIGHT_DECAY = 0.02
WARMUP_RATIO = 0.08
SEED = 42
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./dataset/MN-DS-news-classification.csv"

# Disable MixUp for text (it doesn't work with token indices)
USE_MIXUP = False  # Can't use MixUp with token indices
USE_CUTMIX = False  # Can't use CutMix with text
USE_LABEL_SMOOTHING = True
SMOOTHING = 0.15
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

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
# Advanced Loss Functions
# ----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ----------------------------
# Dataset with Advanced Augmentation
# ----------------------------
class AdvancedNewsDataset(Dataset):
    def __init__(self, texts, labels_lvl1, labels_lvl2, tokenizer, max_len=MAX_LEN, augment=False):
        self.texts = texts
        self.labels_lvl1 = labels_lvl1
        self.labels_lvl2 = labels_lvl2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def apply_augmentation(self, text):
        """Advanced text augmentation"""
        words = text.split()
        if len(words) < 5:
            return text
            
        if self.augment:
            # Random swap
            if random.random() < 0.3 and len(words) > 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            
            # Random deletion
            if random.random() < 0.1 and len(words) > 3:
                idx = random.randint(0, len(words)-1)
                words.pop(idx)
            
            # Random insertion (duplicate)
            if random.random() < 0.1:
                idx = random.randint(0, len(words)-1)
                words.insert(idx, words[idx])
        
        return ' '.join(words)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment:
            text = self.apply_augmentation(text)
        
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
# Advanced Pooling Methods (Simplified)
# ----------------------------
class AttentionPooling(nn.Module):
    """Simplified attention pooling"""
    def __init__(self, hidden_size):
        super().__init__()
        self.att = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, mask):
        scores = self.att(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e4)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return self.norm(pooled)

class MeanPooling(nn.Module):
    """Mean pooling"""
    def forward(self, hidden_states, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

# ----------------------------
# Optimized Advanced Model
# ----------------------------
class OptimizedHierarchicalRoberta(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, pretrained_model=PRETRAINED_MODEL):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size
        
        # Dual pooling
        self.att_pooling = AttentionPooling(hidden_size)
        self.mean_pooling = MeanPooling()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_size)
        )
        
        # Level1 classifier
        self.classifier_lvl1 = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels_lvl1)
        )
        
        # Level2 classifier with Level1 guidance
        self.classifier_lvl2 = nn.Sequential(
            nn.Linear(hidden_size + num_labels_lvl1, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels_lvl2)
        )
        
        # Learnable gate for hierarchical connection
        self.gate_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Dual pooling
        att_pooled = self.att_pooling(hidden_states, attention_mask)
        mean_pooled = self.mean_pooling(hidden_states, attention_mask)
        
        # Fuse
        pooled = torch.cat([att_pooled, mean_pooled], dim=-1)
        fused = self.fusion(pooled)
        
        # Level1 predictions
        logits1 = self.classifier_lvl1(fused)
        
        # Level2 with hierarchical information
        lvl1_probs = F.softmax(logits1, dim=-1)
        lvl2_input = torch.cat([fused, lvl1_probs], dim=-1)
        logits2 = self.classifier_lvl2(lvl2_input)
        
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
train_ds = AdvancedNewsDataset(X_train.tolist(), y1_train.tolist(), y2_train.tolist(), tokenizer, augment=True)
val_ds = AdvancedNewsDataset(X_val.tolist(), y1_val.tolist(), y2_val.tolist(), tokenizer, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ----------------------------
# Class weights & loss functions
# ----------------------------
print("Computing class weights...")
counts_lvl1 = np.bincount(y1_train)
counts_lvl2 = np.bincount(y2_train)

# Focal loss for Level2
if USE_FOCAL_LOSS:
    criterion_lvl2 = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
else:
    criterion_lvl2 = nn.CrossEntropyLoss(label_smoothing=SMOOTHING if USE_LABEL_SMOOTHING else 0.0)

criterion_lvl1 = nn.CrossEntropyLoss(label_smoothing=SMOOTHING if USE_LABEL_SMOOTHING else 0.0)

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
print("Initializing optimized model...")
model = OptimizedHierarchicalRoberta(
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

# Linear schedule with warmup
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = GradScaler()

# ----------------------------
# Training with Gradient Accumulation
# ----------------------------
ACC_STEPS = 4  # Effective batch size = 8 * 4 = 32
best_acc2 = 0
patience_counter = 0
best_model_state = None
training_history = []

print("\n" + "="*60)
print("Starting OPTIMIZED training...")
print("="*60)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels1 = batch["label_lvl1"].to(DEVICE)
        labels2 = batch["label_lvl2"].to(DEVICE)
        
        with autocast(device_type=DEVICE.type):
            logits1, logits2 = model(input_ids, attention_mask)
            
            loss1 = criterion_lvl1(logits1, labels1)
            loss2 = criterion_lvl2(logits2, labels2)
            
            # Emphasize Level2 more
            loss = loss1 + loss2 * 1.3
            
            # Scale loss for gradient accumulation
            loss = loss / ACC_STEPS
        
        scaler.scale(loss).backward()
        
        total_loss += loss.item() * ACC_STEPS
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        
        # Gradient accumulation step
        if (step + 1) % ACC_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        pbar.set_postfix({
            "loss": f"{total_loss/(step+1):.4f}",
            "loss1": f"{total_loss1/(step+1):.4f}",
            "loss2": f"{total_loss2/(step+1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_loss1 = total_loss1 / len(train_loader)
    avg_loss2 = total_loss2 / len(train_loader)
    print(f"Epoch {epoch+1} — Total Loss: {avg_loss:.4f}, L1: {avg_loss1:.4f}, L2: {avg_loss2:.4f}")
    
    # ----------------------------
    # Validation
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
    acc2 = accuracy_score(all_l2, all_p2)
    f1w2 = f1_score(all_l2, all_p2, average='weighted')
    
    print("\n" + "="*50)
    print(f"Epoch {epoch+1} Validation Results")
    print("="*50)
    print(f"Level1 — Acc: {acc1:.4f} | F1-w: {f1w1:.4f}")
    print(f"Level2 — Acc: {acc2:.4f} | F1-w: {f1w2:.4f}")
    print("="*50 + "\n")
    
    # Save training history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'train_loss1': avg_loss1,
        'train_loss2': avg_loss2,
        'val_acc1': acc1,
        'val_acc2': acc2,
        'val_f1w1': f1w1,
        'val_f1w2': f1w2
    })
    
    # Early stopping and model saving
    if acc2 > best_acc2 + 0.001:  # 0.1% improvement
        best_acc2 = acc2
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc2': best_acc2,
            'acc1': acc1,
            'acc2': acc2,
            'le1_classes': le1.classes_,
            'le2_classes': le2.classes_,
            'training_history': training_history
        }
        torch.save(checkpoint, "best_model_checkpoint.pt")
        print(f"✓ New best model saved! (Level2 Acc: {acc2:.4f})")
    else:
        patience_counter += 1
        print(f"✗ No improvement for {patience_counter}/5 epochs")
        
        if patience_counter >= 5:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save periodic checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pt")

# Load best model
if best_model_state is not None:
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(best_model_state)

# Final detailed evaluation
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

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Level1 — Accuracy: {acc1:.4f} | F1-weighted: {f1w1:.4f}")
print(f"Level2 — Accuracy: {acc2:.4f} | F1-weighted: {f1w2:.4f}")
print("="*70)

# Show training progress
print("\nTraining Progress:")
for hist in training_history[-5:]:  # Last 5 epochs
    print(f"Epoch {hist['epoch']}: L1={hist['val_acc1']:.4f}, L2={hist['val_acc2']:.4f}")

# Analyze performance
if acc1 >= 0.90 and acc2 >= 0.80:
    print("\n🎯 TARGETS ACHIEVED! Excellent!")
elif acc1 >= 0.88 and acc2 >= 0.78:
    print("\n✅ Close to targets! Consider training a bit longer.")
    print("   Try increasing epochs to 25-30 for further improvement.")
elif acc1 >= 0.85 and acc2 >= 0.75:
    print("\n⚠️ Good progress but not at targets yet.")
    print("   Consider trying these improvements:")
    print("   1. Increase epochs to 25-30")
    print("   2. Try 'microsoft/deberta-v3-large' model")
    print("   3. Add more data augmentation")
else:
    print("\n❌ Needs improvement. Try:")
    print("   1. Switch to 'microsoft/deberta-v3-large'")
    print("   2. Increase batch size to 16 if memory allows")
    print("   3. Use 5-fold cross validation")
    print("   4. Try ensemble of multiple models")

# Save final model
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'le1_classes': le1.classes_,
    'le2_classes': le2.classes_,
    'tokenizer_config': tokenizer.init_kwargs,
    'metrics': {
        'acc1': acc1,
        'f1w1': f1w1,
        'acc2': acc2,
        'f1w2': f1w2
    },
    'training_history': training_history
}
torch.save(final_checkpoint, "optimized_final_model.pt")
print("\nOptimized model saved as 'optimized_final_model.pt'")

# Additional advanced techniques you can try:
print("\n" + "="*70)
print("ADVANCED TECHNIQUES TO REACH TARGETS:")
print("="*70)
print("1. MODEL ENSEMBLING:")
print("   - Train 3-5 models with different seeds")
print("   - Average their predictions")
print("   - Can gain 1-3% accuracy")

print("\n2. PSEUDO-LABELING:")
print("   - Use model's high-confidence predictions")
print("   - Add them back to training data")
print("   - Retrain with enlarged dataset")

print("\n3. MODEL SWITCHING:")
print("   - Try 'microsoft/deberta-v3-large'")
print("   - Often outperforms RoBERTa on classification")

print("\n4. K-FOLD CROSS VALIDATION:")
print("   - Train on 5 different splits")
print("   - Ensemble all 5 models")

print("\n5. TEST TIME AUGMENTATION:")
print("   - Make predictions on multiple augmentations")
print("   - Average the predictions")
print("="*70)