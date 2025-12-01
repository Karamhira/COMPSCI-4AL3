# -------------------------
# Imports
# -------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import gc

# -------------------------
# Dataset class
# -------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels_lvl1, labels_lvl2, tokenizer, max_len=256):
        self.texts = texts
        self.labels_lvl1 = labels_lvl1
        self.labels_lvl2 = labels_lvl2
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Tokenize text
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
            'labels_lvl1': torch.tensor(self.labels_lvl1[idx], dtype=torch.long),
            'labels_lvl2': torch.tensor(self.labels_lvl2[idx], dtype=torch.long)
        }

# -------------------------
# Model class
# -------------------------
class HierarchicalBERT(nn.Module):
    def __init__(self, num_labels_lvl1, num_labels_lvl2, dropout=0.3):
        super(HierarchicalBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_lvl1 = nn.Linear(hidden_size, num_labels_lvl1)
        self.head_lvl2 = nn.Linear(hidden_size, num_labels_lvl2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:,0]  # CLS token
        cls_emb = self.dropout(cls_emb)
        logits_lvl1 = self.head_lvl1(cls_emb)
        logits_lvl2 = self.head_lvl2(cls_emb)
        return logits_lvl1, logits_lvl2

# -------------------------
# Load data
# -------------------------
# Replace this with your dataset loading
# df = pd.read_csv("mn_ds.csv")
# Assuming df has 'text', 'category_level_1', 'category_level_2'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode labels
le_lvl1 = LabelEncoder()
df['label_lvl1'] = le_lvl1.fit_transform(df['category_level_1'])
le_lvl2 = LabelEncoder()
df['label_lvl2'] = le_lvl2.fit_transform(df['category_level_2'])

# Train/test split
X_train, X_test, y_train1, y_test1, y_train2, y_test2 = train_test_split(
    df['text'], df['label_lvl1'], df['label_lvl2'],
    test_size=0.2, random_state=42, stratify=df['label_lvl1']
)

train_dataset = NewsDataset(X_train.tolist(), y_train1.tolist(), y_train2.tolist(), tokenizer)
test_dataset = NewsDataset(X_test.tolist(), y_test1.tolist(), y_test2.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Initialize model
# -------------------------
model = HierarchicalBERT(num_labels_lvl1=len(le_lvl1.classes_),
                         num_labels_lvl2=len(le_lvl2.classes_)).to(device)

# -------------------------
# Optimizer, scheduler, loss
# -------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 5
train_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=train_steps)
criterion = nn.CrossEntropyLoss()

# -------------------------
# Training loop
# -------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_lvl1 = batch['labels_lvl1'].to(device)
        labels_lvl2 = batch['labels_lvl2'].to(device)

        logits_lvl1, logits_lvl2 = model(input_ids, attention_mask)
        loss_lvl1 = criterion(logits_lvl1, labels_lvl1)
        loss_lvl2 = criterion(logits_lvl2, labels_lvl2)
        loss = loss_lvl1 + loss_lvl2

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# -------------------------
# Evaluation
# -------------------------
model.eval()
all_preds_lvl1, all_labels_lvl1 = [], []
all_preds_lvl2, all_labels_lvl2 = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_lvl1 = batch['labels_lvl1'].to(device)
        labels_lvl2 = batch['labels_lvl2'].to(device)

        logits_lvl1, logits_lvl2 = model(input_ids, attention_mask)
        preds_lvl1 = torch.argmax(logits_lvl1, dim=1)
        preds_lvl2 = torch.argmax(logits_lvl2, dim=1)

        all_preds_lvl1.extend(preds_lvl1.cpu().numpy())
        all_labels_lvl1.extend(labels_lvl1.cpu().numpy())
        all_preds_lvl2.extend(preds_lvl2.cpu().numpy())
        all_labels_lvl2.extend(labels_lvl2.cpu().numpy())

f1_lvl1 = f1_score(all_labels_lvl1, all_preds_lvl1, average='weighted')
f1_lvl2 = f1_score(all_labels_lvl2, all_preds_lvl2, average='weighted')
print(f"Weighted F1 - Level 1: {f1_lvl1:.4f}, Level 2: {f1_lvl2:.4f}")

# -------------------------
# Save unified model
# -------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
    'le_lvl1': le_lvl1,
    'le_lvl2': le_lvl2
}, "model.pth")
print("Saved unified model to model.pth")

# -------------------------
# Inference function
# -------------------------
def predict(text, model, tokenizer, le_lvl1, le_lvl2, device):
    model.eval()
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits_lvl1, logits_lvl2 = model(input_ids, attention_mask)
        pred_lvl1 = torch.argmax(logits_lvl1, dim=1).cpu().item()
        pred_lvl2 = torch.argmax(logits_lvl2, dim=1).cpu().item()

    return le_lvl1.inverse_transform([pred_lvl1])[0], le_lvl2.inverse_transform([pred_lvl2])[0]

# Example:
# level1, level2 = predict("Some news article text here", model, tokenizer, le_lvl1, le_lvl2, device)
# print(level1, level2)
