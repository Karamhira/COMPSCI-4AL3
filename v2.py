import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import gc
import os

# Dataset class
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

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Label encoding
le_lvl1 = LabelEncoder()
df['label_lvl1'] = le_lvl1.fit_transform(df['category_level_1'])

le_lvl2 = LabelEncoder()
df['label_lvl2'] = le_lvl2.fit_transform(df['category_level_2'])

# Train/test split
X_train, X_test, y_train1, y_test1, y_train2, y_test2 = train_test_split(
    df['text'], df['label_lvl1'], df['label_lvl2'],
    test_size=0.2, random_state=42, stratify=df['label_lvl1']
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Level 1 datasets and loaders
train_dataset_lvl1 = NewsDataset(X_train.tolist(), y_train1.tolist(), tokenizer)
test_dataset_lvl1 = NewsDataset(X_test.tolist(), y_test1.tolist(), tokenizer)
train_loader_lvl1 = DataLoader(train_dataset_lvl1, batch_size=4, shuffle=True)
test_loader_lvl1 = DataLoader(test_dataset_lvl1, batch_size=4)

# Level 1 model
model_lvl1 = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=len(le_lvl1.classes_)
)
model_lvl1.to(device)
optimizer_lvl1 = AdamW(model_lvl1.parameters(), lr=2e-5)

# Level 1 training loop
epochs_lvl1 = 2
for epoch in range(epochs_lvl1):
    model_lvl1.train()
    total_loss = 0
    for batch in train_loader_lvl1:
        optimizer_lvl1.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model_lvl1(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer_lvl1.step()
    print("Level 1 Epoch {}/{} Loss: {:.4f}".format(epoch+1, epochs_lvl1, total_loss/len(train_loader_lvl1)))

# Level 1 evaluation
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

f1_lvl1 = f1_score(all_labels_lvl1, all_preds_lvl1, average='weighted')
print("Level 1 Weighted F1:", round(f1_lvl1, 4))

# Level 2 training per Level 1 category
os.makedirs("models", exist_ok=True)
lvl2_results = []

for idx, lvl1_label in enumerate(le_lvl1.classes_):
    # Subset data for current Level 1 category
    indices = [i for i, y in enumerate(y_train1) if y == idx]
    X_sub = [X_train.iloc[i] for i in indices]
    y_sub = [y_train2.iloc[i] for i in indices]

    unique_labels = sorted(set(y_sub))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_sub_local = [label_map[y] for y in y_sub]

    train_dataset_lvl2 = NewsDataset(X_sub, y_sub_local, tokenizer)
    train_loader_lvl2 = DataLoader(train_dataset_lvl2, batch_size=4, shuffle=True)

    model_lvl2 = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(unique_labels)
    )
    model_lvl2.to(device)
    optimizer_lvl2 = AdamW(model_lvl2.parameters(), lr=2e-5)

    epochs_lvl2 = 3
    for epoch in range(epochs_lvl2):
        model_lvl2.train()
        total_loss = 0
        for batch in train_loader_lvl2:
            optimizer_lvl2.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model_lvl2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer_lvl2.step()
        print(f"Level 2 [{lvl1_label}] Epoch {epoch+1}/{epochs_lvl2}, Loss: {total_loss/len(train_loader_lvl2):.4f}")

    # Save Level 2 model
    save_path = f"models/level2_{idx}_{lvl1_label.replace(' ', '_')}"
    model_lvl2.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Evaluation on Level 2
    test_indices = [i for i, y in enumerate(y_test1) if y == idx]
    X_sub_test = [X_test.iloc[i] for i in test_indices]
    y_sub_test = [y_test2.iloc[i] for i in test_indices]
    y_sub_test_local = [label_map[y] for y in y_sub_test if y in label_map]

    test_dataset_lvl2 = NewsDataset(X_sub_test, y_sub_test_local, tokenizer)
    test_loader_lvl2 = DataLoader(test_dataset_lvl2, batch_size=4)

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
    print(f"Level 2 [{lvl1_label}] - Accuracy: {acc:.4f}, Weighted F1: {f1_w:.4f}")

    lvl2_results.append({"level1_label": lvl1_label, "accuracy": acc, "f1_weighted": f1_w})

    del model_lvl2
    torch.cuda.empty_cache()
    gc.collect()

# Level 2 summary
df_lvl2_results = pd.DataFrame(lvl2_results)
print(df_lvl2_results.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True))
