import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os, gc
from tqdm import tqdm

# -------------------------
# Dataset class
# -------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------------
# Load dataset and encode labels
# -------------------------
df = pd.read_csv("./dataset/MN-DS-news-classification.csv")
df['text'] = df['title'].fillna('') + " " + df['content'].fillna('')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
le_lvl1 = LabelEncoder()
le_lvl2 = LabelEncoder()
df['label_lvl1'] = le_lvl1.fit_transform(df['category_level_1'])
df['label_lvl2'] = le_lvl2.fit_transform(df['category_level_2'])

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train1, y_test1, y_train2, y_test2 = train_test_split(
    df['text'], df['label_lvl1'], df['label_lvl2'],
    test_size=0.2, random_state=42, stratify=df['label_lvl1']
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Level 1 datasets
# -------------------------
train_dataset_lvl1 = NewsDataset(X_train.tolist(), y_train1.tolist(), tokenizer)
test_dataset_lvl1 = NewsDataset(X_test.tolist(), y_test1.tolist(), tokenizer)
train_loader_lvl1 = DataLoader(train_dataset_lvl1, batch_size=8, shuffle=True)
test_loader_lvl1 = DataLoader(test_dataset_lvl1, batch_size=8)

# -------------------------
# Level 1 model
# -------------------------
model_lvl1 = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=len(le_lvl1.classes_)
).to(device)
optimizer_lvl1 = AdamW(model_lvl1.parameters(), lr=2e-5)
epochs_lvl1 = 15
scaler = torch.amp.GradScaler()
use_swa = True

if use_swa:
    swa_model = AveragedModel(model_lvl1)
    swa_start = 5
    swa_scheduler = SWALR(optimizer_lvl1, swa_lr=1e-5)

# -------------------------
# Level 1 training
# -------------------------
for epoch in range(epochs_lvl1):
    model_lvl1.train()
    total_loss = 0
    for batch in tqdm(train_loader_lvl1, desc=f"Level 1 Epoch {epoch+1}/{epochs_lvl1}"):
        optimizer_lvl1.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.amp.autocast(device_type='cuda'):
            outputs = model_lvl1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer_lvl1)
        scaler.update()
        total_loss += loss.item()
    
    print(f"Level 1 Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader_lvl1):.4f}")

    if use_swa and epoch >= swa_start:
        swa_model.update_parameters(model_lvl1)
        swa_scheduler.step()

    # -------------------------
    # Validation
    # -------------------------
    model_lvl1.eval()
    all_preds_lvl1, all_labels_lvl1 = [], []
    with torch.no_grad():
        for batch in test_loader_lvl1:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model_lvl1(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds_lvl1.extend(preds.cpu().numpy())
            all_labels_lvl1.extend(labels.cpu().numpy())
    acc_lvl1 = accuracy_score(all_labels_lvl1, all_preds_lvl1)
    f1_lvl1 = f1_score(all_labels_lvl1, all_preds_lvl1, average='weighted')
    print(f"Level 1 Validation -> Accuracy: {acc_lvl1:.4f}, Weighted F1: {f1_lvl1:.4f}")

# -------------------------
# Apply SWA batch norm (without loading into HF model)
# -------------------------
if use_swa:
    update_bn(train_loader_lvl1, swa_model)
    print("SWA batch norm updated. Skipping direct loading into Hugging Face model.")

# -------------------------
# Save Level 1 model
# -------------------------
os.makedirs("models/level1", exist_ok=True)
model_lvl1.save_pretrained("models/level1")
tokenizer.save_pretrained("models/level1")

# -------------------------
# Level 2 models
# -------------------------
lvl2_results = []
for idx, lvl1_label in enumerate(le_lvl1.classes_):
    indices = [i for i, y in enumerate(y_train1) if y == idx]
    X_sub = [X_train.iloc[i] for i in indices]
    y_sub = [y_train2.iloc[i] for i in indices]
    unique_labels = sorted(set(y_sub))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_sub_local = [label_map[y] for y in y_sub]

    train_dataset_lvl2 = NewsDataset(X_sub, y_sub_local, tokenizer)
    train_loader_lvl2 = DataLoader(train_dataset_lvl2, batch_size=8, shuffle=True)

    model_lvl2 = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(unique_labels)
    ).to(device)
    optimizer_lvl2 = AdamW(model_lvl2.parameters(), lr=2e-5)
    epochs_lvl2 = 15

    # Level 2 training
    for epoch in range(epochs_lvl2):
        model_lvl2.train()
        total_loss = 0
        for batch in train_loader_lvl2:
            optimizer_lvl2.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model_lvl2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            loss.backward()
            optimizer_lvl2.step()
            total_loss += loss.item()
        print(f"Level 2 [{lvl1_label}] Epoch {epoch+1}/{epochs_lvl2} Avg Loss: {total_loss/len(train_loader_lvl2):.4f}")

    # Save Level 2 model
    save_path = f"models/level2_{idx}_{lvl1_label.replace(' ', '_')}"
    os.makedirs(save_path, exist_ok=True)
    model_lvl2.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # -------------------------
    # Level 2 validation
    # -------------------------
    test_indices = [i for i, y in enumerate(y_test1) if y == idx]
    X_sub_test = [X_test.iloc[i] for i in test_indices]
    y_sub_test = [y_test2.iloc[i] for i in test_indices]
    y_sub_test_local = [label_map[y] for y in y_sub_test if y in label_map]

    test_dataset_lvl2 = NewsDataset(X_sub_test, y_sub_test_local, tokenizer)
    test_loader_lvl2 = DataLoader(test_dataset_lvl2, batch_size=8)

    all_preds, all_labels = [], []
    model_lvl2.eval()
    with torch.no_grad():
        for batch in test_loader_lvl2:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model_lvl2(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average='weighted')
    print(f"Level 2 [{lvl1_label}] -> Accuracy: {acc:.4f}, Weighted F1: {f1_w:.4f}")
    lvl2_results.append({"level1_label": lvl1_label, "accuracy": acc, "f1_weighted": f1_w})

    del model_lvl2
    torch.cuda.empty_cache()
    gc.collect()

# -------------------------
# Summary stats
# -------------------------
df_lvl2_results = pd.DataFrame(lvl2_results)
print(df_lvl2_results.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True))

# -------------------------
# Prediction function
# -------------------------
def predict(text, tokenizer, model_lvl1, le_lvl1, le_lvl2):
    model_lvl1.eval()
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        outputs_lvl1 = model_lvl1(input_ids=input_ids, attention_mask=attention_mask)
        pred_lvl1_idx = torch.argmax(outputs_lvl1.logits, dim=1).item()
    lvl1_label = le_lvl1.inverse_transform([pred_lvl1_idx])[0]

    lvl2_path = f"models/level2_{pred_lvl1_idx}_{lvl1_label.replace(' ', '_')}"
    model_lvl2 = DistilBertForSequenceClassification.from_pretrained(lvl2_path).to(device)
    model_lvl2.eval()
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        outputs_lvl2 = model_lvl2(input_ids=input_ids, attention_mask=attention_mask)
        pred_lvl2_idx = torch.argmax(outputs_lvl2.logits, dim=1).item()
    lvl2_label = le_lvl2.inverse_transform([pred_lvl2_idx])[0]

    return lvl1_label, lvl2_label
