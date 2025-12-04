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
# Advanced Config
# ----------------------------
PRETRAINED_MODEL = "roberta-large"  # Consider "microsoft/deberta-v3-large" for better performance
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 20  # More epochs for gradual improvement
LR = 1.5e-5
BACKBONE_LR = 8e-6
CLASSIFIER_LR = 3e-4
WEIGHT_DECAY = 0.02  # Slightly stronger regularization
WARMUP_RATIO = 0.08
SEED = 42
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./dataset/MN-DS-news-classification.csv"

# Advanced features
USE_MIXUP = True  # MixUp augmentation
MIXUP_ALPHA = 0.2
USE_CUTMIX = True  # CutMix augmentation
CUTMIX_ALPHA = 1.0
USE_STOCHASTIC_DEPTH = True  # Stochastic depth regularization
DROP_PATH_RATE = 0.1
USE_LABEL_SMOOTHING = True
SMOOTHING = 0.15
USE_FOCAL_LOSS = True  # For Level2 imbalance
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

class SymmetricCrossEntropy(nn.Module):
    """Symmetric CE for better calibration"""
    def __init__(self, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets)
        
        # Reverse KL divergence
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        rce = -torch.sum(probs * log_probs, dim=1).mean()
        
        return self.alpha * ce + self.beta * rce

# ----------------------------
# Advanced Augmentation Functions
# ----------------------------
def mixup_data(x, y1, y2, alpha=1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    
    return mixed_x, y1_a, y1_b, y2_a, y2_b, lam

def mixup_criterion(criterion, pred1, pred2, y_a, y_b, lam):
    """MixUp loss calculation"""
    return lam * criterion(pred1, y_a) + (1 - lam) * criterion(pred1, y_b), \
           lam * criterion(pred2, y_a) + (1 - lam) * criterion(pred2, y_b)

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
# Advanced Pooling Methods
# ----------------------------
class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for richer representations"""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, mask):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (self.head_dim ** 0.5)
        
        # Mask padding
        mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
        scores = scores.masked_fill(mask == 0, -1e4)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        out = out.contiguous().view(batch_size, seq_len, hidden_size)
        
        # Weighted sum
        weights = attn_weights.mean(dim=1)  # Average over heads
        pooled = torch.bmm(weights, hidden_states).squeeze(1)
        
        pooled = self.out_proj(pooled)
        pooled = self.norm(pooled)
        
        return pooled

class GeMTextPooling(nn.Module):
    """Generalized Mean Pooling for text"""
    def __init__(self, hidden_size, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, hidden_states, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        hidden_states = hidden_states.clamp(min=self.eps)
        
        # Apply mask and compute GeM
        gem = (hidden_states ** self.p * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1)
        
        gem = gem / counts
        gem = gem ** (1.0 / self.p)
        
        return gem

# ----------------------------
# Advanced Model Architecture
# ----------------------------
class AdvancedHierarchicalRoberta(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, pretrained_model=PRETRAINED_MODEL):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.roberta.config.hidden_size
        
        # Advanced pooling (ensemble of multiple pooling methods)
        self.pooling1 = MultiHeadAttentionPooling(hidden_size, num_heads=4)
        self.pooling2 = GeMTextPooling(hidden_size, p=3.0)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3)
        )
        
        # Level1 classifier with residual connections
        self.lvl1_features = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2)
        )
        self.classifier_lvl1 = nn.Linear(256, num_labels_lvl1)
        
        # Level2 classifier with hierarchical information
        self.lvl2_features = nn.Sequential(
            nn.Linear(hidden_size + num_labels_lvl1, 768),  # Include Level1 info
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2)
        )
        self.classifier_lvl2 = nn.Linear(256, num_labels_lvl2)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.fusion, self.lvl1_features, self.lvl2_features]:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                    
        # Initialize classifier heads
        nn.init.xavier_uniform_(self.classifier_lvl1.weight)
        nn.init.constant_(self.classifier_lvl1.bias, 0)
        nn.init.xavier_uniform_(self.classifier_lvl2.weight)
        nn.init.constant_(self.classifier_lvl2.bias, 0)
        
    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Ensemble pooling
        pooled1 = self.pooling1(hidden_states, attention_mask)
        pooled2 = self.pooling2(hidden_states, attention_mask)
        
        # Concatenate and fuse
        pooled = torch.cat([pooled1, pooled2], dim=-1)
        fused = self.fusion(pooled)
        
        # Level1 features and predictions
        lvl1_feat = self.lvl1_features(fused)
        logits1 = self.classifier_lvl1(lvl1_feat)
        
        # Level2 with Level1 information (soft probabilities)
        lvl1_probs = F.softmax(logits1 / self.temperature, dim=-1)
        lvl2_input = torch.cat([fused, lvl1_probs], dim=-1)
        
        lvl2_feat = self.lvl2_features(lvl2_input)
        logits2 = self.classifier_lvl2(lvl2_feat)
        
        if return_features:
            return logits1, logits2, lvl1_feat, lvl2_feat
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

# ----------------------------
# Class weights & loss functions
# ----------------------------
print("Computing class weights...")
counts_lvl1 = np.bincount(y1_train)
counts_lvl2 = np.bincount(y2_train)

# Focal loss for Level2, standard for Level1
if USE_FOCAL_LOSS:
    criterion_lvl2 = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
else:
    criterion_lvl2 = nn.CrossEntropyLoss(label_smoothing=SMOOTHING if USE_LABEL_SMOOTHING else 0.0)

criterion_lvl1 = nn.CrossEntropyLoss(label_smoothing=SMOOTHING if USE_LABEL_SMOOTHING else 0.0)

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
print("Initializing advanced model...")
model = AdvancedHierarchicalRoberta(
    num_labels_lvl1=len(le1.classes_),
    num_labels_lvl2=len(le2.classes_)
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Layer-wise learning rate decay
def get_optimizer(model, lr, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'roberta' in n and not any(nd in n for nd in no_decay)],
            'lr': BACKBONE_LR,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'roberta' in n and any(nd in n for nd in no_decay)],
            'lr': BACKBONE_LR,
            'weight_decay': 0.0
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'roberta' not in n and not any(nd in n for nd in no_decay)],
            'lr': CLASSIFIER_LR,
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'roberta' not in n and any(nd in n for nd in no_decay)],
            'lr': CLASSIFIER_LR,
            'weight_decay': 0.0
        },
    ]
    return AdamW(optimizer_grouped_parameters)

optimizer = get_optimizer(model, LR, WEIGHT_DECAY)

# Cosine annealing with warm restarts
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=len(train_loader) * 3,  # Restart every 3 epochs
    T_mult=1,
    eta_min=1e-7
)

scaler = GradScaler()

# ----------------------------
# Training with Advanced Features
# ----------------------------
best_acc2 = 0
patience_counter = 0
best_model_state = None

print("\n" + "="*60)
print("Starting ADVANCED training...")
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
        
        # Apply MixUp if enabled
        if USE_MIXUP and np.random.random() < 0.5:
            input_ids, labels1_a, labels1_b, labels2_a, labels2_b, lam = mixup_data(
                input_ids, labels1, labels2, MIXUP_ALPHA
            )
        
        with autocast(device_type=DEVICE.type):
            logits1, logits2 = model(input_ids, attention_mask)
            
            if USE_MIXUP and np.random.random() < 0.5:
                loss1, loss2 = mixup_criterion(
                    criterion_lvl1, logits1, logits2, 
                    labels1_a, labels1_b, lam
                )
                _, loss2 = mixup_criterion(
                    criterion_lvl2, logits1, logits2,
                    labels2_a, labels2_b, lam
                )
            else:
                loss1 = criterion_lvl1(logits1, labels1)
                loss2 = criterion_lvl2(logits2, labels2)
            
            # Combined loss
            loss = loss1 + loss2 * 1.2  # Slightly emphasize Level2
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        
        pbar.set_postfix({
            "loss": f"{total_loss/(pbar.n+1):.4f}",
            "loss1": f"{total_loss1/(pbar.n+1):.4f}",
            "loss2": f"{total_loss2/(pbar.n+1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} — Loss: {avg_loss:.4f} (L1: {total_loss1/len(train_loader):.4f}, L2: {total_loss2/len(train_loader):.4f})")
    
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
    f1m1 = f1_score(all_l1, all_p1, average='macro')
    acc2 = accuracy_score(all_l2, all_p2)
    f1w2 = f1_score(all_l2, all_p2, average='weighted')
    f1m2 = f1_score(all_l2, all_p2, average='macro')
    
    print("\n" + "="*60)
    print(f"Epoch {epoch+1} Validation Results")
    print("="*60)
    print(f"Level1 — Acc: {acc1:.4f} | F1-w: {f1w1:.4f} | F1-m: {f1m1:.4f}")
    print(f"Level2 — Acc: {acc2:.4f} | F1-w: {f1w2:.4f} | F1-m: {f1m2:.4f}")
    print("="*60 + "\n")
    
    # Early stopping and model saving
    if acc2 > best_acc2 + MIN_DELTA:
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
        }
        torch.save(checkpoint, "best_model_checkpoint.pt")
        print(f"✓ New best model saved! (Level2 Acc: {acc2:.4f})")
    else:
        patience_counter += 1
        print(f"✗ No improvement for {patience_counter}/{PATIENCE} epochs")
        
        if patience_counter >= PATIENCE:
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
all_logits1, all_logits2 = [], []

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
        all_logits1.append(logits1.cpu())
        all_logits2.append(logits2.cpu())

# Ensemble with temperature scaling
all_logits1 = torch.cat(all_logits1, dim=0)
all_logits2 = torch.cat(all_logits2, dim=0)

# Temperature scaling for better calibration
with torch.no_grad():
    temperature = model.temperature.item()
    scaled_logits1 = all_logits1 / temperature
    scaled_logits2 = all_logits2 / temperature
    final_preds1 = torch.argmax(scaled_logits1, dim=1).numpy()
    final_preds2 = torch.argmax(scaled_logits2, dim=1).numpy()

# Final metrics
acc1 = accuracy_score(all_l1, final_preds1)
f1w1 = f1_score(all_l1, final_preds1, average='weighted')
acc2 = accuracy_score(all_l2, final_preds2)
f1w2 = f1_score(all_l2, final_preds2, average='weighted')

print("\n" + "="*70)
print("FINAL RESULTS WITH TEMPERATURE SCALING")
print("="*70)
print(f"Level1 — Accuracy: {acc1:.4f} | F1-weighted: {f1w1:.4f}")
print(f"Level2 — Accuracy: {acc2:.4f} | F1-weighted: {f1w2:.4f}")
print(f"Temperature: {temperature:.3f}")
print("="*70)

# Show problematic classes
print("\nTop 10 most confused Level2 classes:")
unique_labels = np.unique(all_l2)
conf_matrix = np.zeros((len(unique_labels), len(unique_labels)))
for true, pred in zip(all_l2, final_preds2):
    conf_matrix[true, pred] += 1

# Find most confused pairs
confusions = []
for i in range(len(unique_labels)):
    for j in range(len(unique_labels)):
        if i != j and conf_matrix[i, j] > 5:  # At least 5 misclassifications
            confusions.append((i, j, conf_matrix[i, j]))

confusions.sort(key=lambda x: x[2], reverse=True)
for i, j, count in confusions[:10]:
    print(f"{le2.classes_[i]} → {le2.classes_[j]}: {int(count)} samples")

# Save final model
final_checkpoint = {
    'model_state_dict': model.state_dict(),
    'le1_classes': le1.classes_,
    'le2_classes': le2.classes_,
    'tokenizer_config': tokenizer.init_kwargs,
    'temperature': temperature,
    'metrics': {
        'acc1': acc1,
        'f1w1': f1w1,
        'acc2': acc2,
        'f1w2': f1w2
    }
}
torch.save(final_checkpoint, "advanced_final_model.pt")
print("\nAdvanced model saved as 'advanced_final_model.pt'")