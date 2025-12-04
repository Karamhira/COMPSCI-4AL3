import os
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import Dataset as HFDataset
from transformers import (
    RobertaModel, RobertaTokenizerFast, AutoConfig,
    Trainer, TrainingArguments, default_data_collator, EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------
# Utility: Focal Loss
# ---------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss."""
    def __init__(self, gamma=2.0, weight=None, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        probs = F.softmax(logits, dim=-1).clamp(self.eps, 1.0 - self.eps)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-12)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)

def get_layerwise_lr_decay_param_groups(model: nn.Module, base_lr: float, layer_decay: float) -> List[Dict]:
    """Layer-wise LR decay for HuggingFace RoBERTa."""
    param_groups = {}
    num_layers = len(list(model.roberta.encoder.layer))
    head_names = ["level1_head", "level2_head"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_scale = 1.0
        if any(h in name for h in head_names):
            lr_scale = 1.0
        else:
            found = False
            for i in range(num_layers):
                if f"encoder.layer.{i}." in name:
                    lr_scale = layer_decay ** (num_layers - 1 - i)
                    found = True
                    break
            if not found:
                lr_scale = layer_decay ** num_layers
        key = float(lr_scale)
        if key not in param_groups:
            param_groups[key] = {"params": [], "lr_scale": key}
        param_groups[key]["params"].append(param)

    groups_list = []
    for lr_scale, group in sorted(param_groups.items(), key=lambda x: x[0]):
        groups_list.append({"params": group["params"], "lr": base_lr * lr_scale})
    return groups_list

# ---------------------------
# Model: RoBERTa with two heads + multi-dropout
# ---------------------------
class RobertaTwoHeadMultiDropout(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels_level1: int, num_labels_level2: int,
                 n_dropouts: int = 5, dropout_prob: float = 0.3, gradient_checkpointing: bool = False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.roberta = RobertaModel.from_pretrained(model_name_or_path, config=config)
        if gradient_checkpointing:
            try:
                self.roberta.gradient_checkpointing_enable()
            except Exception:
                pass

        self.n_dropouts = n_dropouts
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(n_dropouts)])
        hidden_size = config.hidden_size

        self.level1_head = nn.Linear(hidden_size, num_labels_level1)
        self.level2_head = nn.Linear(hidden_size, num_labels_level2)

        nn.init.normal_(self.level1_head.weight, std=0.02)
        nn.init.normal_(self.level2_head.weight, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.last_hidden_state[:, 0, :]

        level1_logits_samples = []
        level2_logits_samples = []
        for d in self.dropouts:
            h = d(pooled)
            level1_logits_samples.append(self.level1_head(h))
            level2_logits_samples.append(self.level2_head(h))

        level1_logits = torch.stack(level1_logits_samples, dim=0).mean(dim=0)
        level2_logits = torch.stack(level2_logits_samples, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            l1 = labels.get("level1", None)
            l2 = labels.get("level2", None)
            # level1 CE
            if l1 is not None:
                loss1 = F.cross_entropy(level1_logits, l1, weight=labels.get("level1_weight", None).to(level1_logits.device) if labels.get("level1_weight", None) is not None else None)
            else:
                loss1 = torch.tensor(0.0, device=level1_logits.device)
            # level2 focal
            if l2 is not None:
                focal = labels.get("level2_focal", None)
                if focal is not None:
                    loss2 = focal(level2_logits, l2)
                else:
                    loss2 = F.cross_entropy(level2_logits, l2, weight=labels.get("level2_weight", None).to(level2_logits.device) if labels.get("level2_weight", None) is not None else None)
            else:
                loss2 = torch.tensor(0.0, device=level2_logits.device)
            alpha = labels.get("alpha", 0.4)
            beta = labels.get("beta", 0.6)
            loss = alpha * loss1 + beta * loss2

        return {
            "loss": loss,
            "logits_level1": level1_logits,
            "logits_level2": level2_logits,
            "hidden_state": pooled
        }

# ---------------------------
# Dataset preparation
# ---------------------------
def prepare_dataset(tokenizer, df, text_column='content', title_column='title', max_length=256):
    if title_column:
        texts = (df[title_column].fillna("") + ". " + df[text_column].fillna("")).tolist()
    else:
        texts = df[text_column].fillna("").tolist()
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    enc = {k: np.array(v) for k, v in enc.items()}
    enc['labels_level1'] = df['level1'].to_numpy()
    enc['labels_level2'] = df['level2'].to_numpy()
    return HFDataset.from_dict(enc)

def collate_fn(batch):
    collated = default_data_collator(batch)
    collated['labels'] = {
        "level1": torch.tensor([b['labels_level1'] for b in batch], dtype=torch.long),
        "level2": torch.tensor([b['labels_level2'] for b in batch], dtype=torch.long)
    }
    return collated

# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits_level1, logits_level2 = eval_pred.predictions
    labels_level1 = eval_pred.label_ids['level1']
    labels_level2 = eval_pred.label_ids['level2']
    preds1 = np.argmax(logits_level1, axis=-1)
    preds2 = np.argmax(logits_level2, axis=-1)

    metrics = {
        "level1_accuracy": accuracy_score(labels_level1, preds1),
        "level1_micro_f1": f1_score(labels_level1, preds1, average='micro', zero_division=0),
        "level1_macro_f1": f1_score(labels_level1, preds1, average='macro', zero_division=0),
        "level2_accuracy": accuracy_score(labels_level2, preds2),
        "level2_micro_f1": f1_score(labels_level2, preds2, average='micro', zero_division=0),
        "level2_macro_f1": f1_score(labels_level2, preds2, average='macro', zero_division=0)
    }
    return metrics

# ---------------------------
# Main training
# ---------------------------
def main(args):
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    df = pd.read_csv(args.data_csv)

    # MN-DS columns fix
    if "level1_category" in df.columns:
        df["level1"] = df["level1_category"]
    if "level2_category" in df.columns:
        df["level2"] = df["level2_category"]

    dataset = prepare_dataset(tokenizer, df, text_column=args.text_col, title_column=args.title_col, max_length=args.max_length)

    ds = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds = ds['train']
    val_ds = ds['test']

    # Class weights
    level1_weights = compute_class_weights(train_ds['labels_level1'], args.num_labels_level1)
    level2_weights = compute_class_weights(train_ds['labels_level2'], args.num_labels_level2)

    model = RobertaTwoHeadMultiDropout(
        model_name_or_path=args.model_name,
        num_labels_level1=args.num_labels_level1,
        num_labels_level2=args.num_labels_level2,
        n_dropouts=args.n_dropouts,
        dropout_prob=args.dropout_prob,
        gradient_checkpointing=args.gradient_checkpointing
    )

    param_groups = get_layerwise_lr_decay_param_groups(model, base_lr=args.learning_rate, layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=args.report_to if args.report_to else None,
        save_total_limit=3,
        dataloader_pin_memory=True
    )

    # Wrap model for Trainer
    class WrappedModel(nn.Module):
        def __init__(self, model, level1_weights, level2_weights, alpha=0.4, beta=0.6, focal_gamma=2.0):
            super().__init__()
            self.model = model
            self.level1_weights = level1_weights
            self.level2_weights = level2_weights
            self.alpha = alpha
            self.beta = beta
            self.focal = FocalLoss(gamma=focal_gamma, weight=self.level2_weights)

        def forward(self, **kwargs):
            labels = kwargs.get('labels', None)
            if labels is not None:
                labels_payload = {
                    "level1": labels['level1'],
                    "level2": labels['level2'],
                    "level1_weight": self.level1_weights,
                    "level2_focal": self.focal,
                    "alpha": self.alpha,
                    "beta": self.beta
                }
            else:
                labels_payload = None
            out = self.model(input_ids=kwargs.get('input_ids'), attention_mask=kwargs.get('attention_mask'),
                             labels=labels_payload)
            return out["loss"]

    wrapped = WrappedModel(model, level1_weights, level2_weights, alpha=args.alpha, beta=args.beta, focal_gamma=args.focal_gamma)

    class TwoHeadTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop('labels')
            device = next(model.parameters()).device
            out = model.model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'),
                              labels={
                                  "level1": labels["level1"].to(device),
                                  "level2": labels["level2"].to(device),
                                  "level1_weight": level1_weights.to(device),
                                  "level2_focal": FocalLoss(gamma=args.focal_gamma, weight=level2_weights.to(device)),
                                  "alpha": args.alpha,
                                  "beta": args.beta
                              })
            loss = out['loss']
            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            labels = inputs.pop("labels", None)
            out = model.model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))
            l1_logits = out['logits_level1'].detach().cpu().numpy()
            l2_logits = out['logits_level2'].detach().cpu().numpy()
            if labels is not None:
                return None, (np.stack([l1_logits, l2_logits], axis=0)), {"level1": labels['level1'].numpy(), "level2": labels['level2'].numpy()}
            else:
                return None, (np.stack([l1_logits, l2_logits], axis=0)), None

    trainer = TwoHeadTrainer(
        model=wrapped,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=val_ds)
    print("FINAL EVAL METRICS:", metrics)

# ---------------------------
# Argument Parser
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="./dataset/MN-DS-news-classification.csv")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="./roberta_mnds_model")
    parser.add_argument("--text_col", type=str, default="content")
    parser.add_argument("--title_col", type=str, default="title")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_labels_level1", type=int, default=17)
    parser.add_argument("--num_labels_level2", type=int, default=109)
    parser.add_argument("--n_dropouts", type=int, default=5)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--layer_decay", type=float, default=0.8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default="level2_macro_f1")
    args = parser.parse_args()

    main(args)
