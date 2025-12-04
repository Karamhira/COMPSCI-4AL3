import os
import math
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, ClassLabel, Dataset
from transformers import (
    RobertaModel, RobertaTokenizerFast, AutoConfig, AutoModel,
    Trainer, TrainingArguments, EvalPrediction, default_data_collator
)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

# ---------------------------
# Utilities: Losses, LLRD
# ---------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss (generalized)."""
    def __init__(self, gamma=2.0, weight=None, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (batch, C) raw logits
        targets: (batch,) integer class labels
        """
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        probs = F.softmax(logits, dim=-1).clamp(self.eps, 1.0 - self.eps)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss = ( (1 - pt) ** self.gamma ) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    # inverse frequency
    weights = 1.0 / (counts + 1e-12)
    # normalize
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)

def get_layerwise_lr_decay_param_groups(model: nn.Module, base_lr: float, layer_decay: float) -> List[Dict]:
    """
    Create param groups with layer-wise learning rate decay (LLRD).
    Works for HuggingFace transformers where encoder layers are named like:
    model.roberta.encoder.layer.{i}
    """
    # map param_name -> param
    param_groups = {}
    # identify encoder layers
    # count layers
    num_layers = len(list(model.roberta.encoder.layer))
    # top-level classifier heads -> lr multiplier 1.0
    head_names = ["level1_head", "level2_head"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # default lr scale
        lr_scale = 1.0
        # assign based on layer
        if any(h in name for h in head_names):
            lr_scale = 1.0
        else:
            # find occurrences of encoder.layer.N
            found = False
            for i in range(num_layers):
                if f"encoder.layer.{i}." in name:
                    # deeper layers (higher i) -> larger lr
                    # scale = layer_decay ** (num_layers - 1 - i)
                    # so top layers (i close to num_layers-1) have scale ~ 1.0
                    lr_scale = layer_decay ** (num_layers - 1 - i)
                    found = True
                    break
            if not found:
                # embeddings and pooler: use small lr
                lr_scale = layer_decay ** (num_layers)
        # group by lr_scale
        key = float(lr_scale)
        if key not in param_groups:
            param_groups[key] = {"params": [], "lr_scale": key}
        param_groups[key]["params"].append(param)

    # convert to list with explicit lr
    groups_list = []
    for lr_scale, group in sorted(param_groups.items(), key=lambda x: x[0]):
        groups_list.append({"params": group["params"], "lr": base_lr * lr_scale})
    return groups_list

# ---------------------------
# Model: RoBERTa with two heads + multi-sample dropout
# ---------------------------
class RobertaTwoHeadMultiDropout(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels_level1: int,
        num_labels_level2: int,
        n_dropouts: int = 5,
        dropout_prob: float = 0.3,
        use_mixout: bool = False,  # placeholder if you wish to integrate mixout later
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.roberta = RobertaModel.from_pretrained(model_name_or_path, config=config)
        if gradient_checkpointing:
            # this enables gradient checkpointing across the transformer
            try:
                self.roberta.gradient_checkpointing_enable()
            except Exception:
                pass

        self.n_dropouts = n_dropouts
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(n_dropouts)])
        hidden_size = config.hidden_size

        # two heads
        self.level1_head = nn.Linear(hidden_size, num_labels_level1)
        self.level2_head = nn.Linear(hidden_size, num_labels_level2)

        # initialize heads
        nn.init.normal_(self.level1_head.weight, std=0.02)
        nn.init.normal_(self.level2_head.weight, std=0.02)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """
        labels: dict with 'level1' and 'level2' (both LongTensor shape (batch,))
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token pooling

        # multisample dropout: average logits across multiple dropout masks
        level1_logits_samples = []
        level2_logits_samples = []
        for d in self.dropouts:
            h = d(pooled)
            level1_logits_samples.append(self.level1_head(h))
            level2_logits_samples.append(self.level2_head(h))

        # average logits
        level1_logits = torch.stack(level1_logits_samples, dim=0).mean(dim=0)
        level2_logits = torch.stack(level2_logits_samples, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            # labels is a dict: labels["level1"], labels["level2"]
            l1 = labels.get("level1", None)
            l2 = labels.get("level2", None)
            losses = []
            # level 1: weighted CE
            if l1 is not None:
                loss_fct1 = nn.CrossEntropyLoss(weight=labels.get("level1_weight", None).to(level1_logits.device) if labels.get("level1_weight", None) is not None else None)
                loss1 = loss_fct1(level1_logits, l1)
            else:
                loss1 = torch.tensor(0.0, device=level1_logits.device)
            # level 2: focal loss
            if l2 is not None:
                focal = labels.get("level2_focal", None)  # a focal loss module may be passed in via labels
                if focal is not None:
                    loss2 = focal(level2_logits, l2)
                else:
                    loss2 = nn.CrossEntropyLoss(weight=labels.get("level2_weight", None).to(level2_logits.device) if labels.get("level2_weight", None) is not None else None)(level2_logits, l2)
            else:
                loss2 = torch.tensor(0.0, device=level2_logits.device)

            # hierarchical combined loss weight (you can tune these)
            alpha = labels.get("alpha", 0.4)  # weight for level1
            beta = labels.get("beta", 0.6)    # weight for level2
            loss = alpha * loss1 + beta * loss2

        return {
            "loss": loss,
            "logits_level1": level1_logits,
            "logits_level2": level2_logits,
            "hidden_state": pooled
        }

# ---------------------------
# Data processing helpers
# ---------------------------
def prepare_dataset(tokenizer, df, text_column='text', title_column=None, max_length=256):
    # df: pandas DataFrame with columns 'text','level1','level2'
    texts = []
    if title_column:
        texts = (df[title_column].fillna("") + ". " + df[text_column].fillna("")).tolist()
    else:
        texts = df[text_column].fillna("").tolist()
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    enc = {k: np.array(v) for k, v in enc.items()}
    enc['labels_level1'] = df['level1'].to_numpy()
    enc['labels_level2'] = df['level2'].to_numpy()
    return Dataset.from_dict(enc)

# collate function to wrap label tensors and extras
def collate_fn(batch):
    # batch is list of dicts from dataset
    collated = default_data_collator(batch)
    # gather labels
    collated['labels'] = {
        "level1": torch.tensor([b['labels_level1'] for b in batch], dtype=torch.long),
        "level2": torch.tensor([b['labels_level2'] for b in batch], dtype=torch.long)
    }
    return collated

# Metrics
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits_level1, logits_level2 = eval_pred.predictions
    labels_level1 = eval_pred.label_ids['level1']
    labels_level2 = eval_pred.label_ids['level2']
    preds1 = np.argmax(logits_level1, axis=-1)
    preds2 = np.argmax(logits_level2, axis=-1)

    metrics = {}
    # Level 1
    metrics['level1_accuracy'] = accuracy_score(labels_level1, preds1)
    metrics['level1_micro_f1'] = f1_score(labels_level1, preds1, average='micro', zero_division=0)
    metrics['level1_macro_f1'] = f1_score(labels_level1, preds1, average='macro', zero_division=0)
    metrics['level1_precision_macro'] = precision_score(labels_level1, preds1, average='macro', zero_division=0)
    metrics['level1_recall_macro'] = recall_score(labels_level1, preds1, average='macro', zero_division=0)
    # Level 2
    metrics['level2_accuracy'] = accuracy_score(labels_level2, preds2)
    metrics['level2_micro_f1'] = f1_score(labels_level2, preds2, average='micro', zero_division=0)
    metrics['level2_macro_f1'] = f1_score(labels_level2, preds2, average='macro', zero_division=0)
    metrics['level2_precision_macro'] = precision_score(labels_level2, preds2, average='macro', zero_division=0)
    metrics['level2_recall_macro'] = recall_score(labels_level2, preds2, average='macro', zero_division=0)

    # combined / weighted metrics can be added as needed
    return metrics

# ---------------------------
# Main runner
# ---------------------------
def main(args):
    # ---------- tokenizer & load data ----------
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)

    import pandas as pd
    df = pd.read_csv(args.data_csv)  # expects columns: text,title (optional), level1, level2
    if args.title_col and args.title_col in df.columns:
        dataset = prepare_dataset(tokenizer, df, text_column=args.text_col, title_column=args.title_col, max_length=args.max_length)
    else:
        dataset = prepare_dataset(tokenizer, df, text_column=args.text_col, title_column=None, max_length=args.max_length)

    # simple train/dev split
    ds = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds = ds['train']
    val_ds = ds['test']

    # determine class counts -> weights
    num_l1 = args.num_labels_level1
    num_l2 = args.num_labels_level2
    level1_weights = compute_class_weights(train_ds['labels_level1'], num_l1)
    level2_weights = compute_class_weights(train_ds['labels_level2'], num_l2)

    # build model
    model = RobertaTwoHeadMultiDropout(
        model_name_or_path=args.model_name,
        num_labels_level1=num_l1,
        num_labels_level2=num_l2,
        n_dropouts=args.n_dropouts,
        dropout_prob=args.dropout_prob,
        gradient_checkpointing=args.gradient_checkpointing
    )

    # param groups (LLRD)
    param_groups = get_layerwise_lr_decay_param_groups(model, base_lr=args.learning_rate, layer_decay=args.layer_decay)

    # custom optimizer
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)

    # prepare Trainer training arguments
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

    # make a trainer-compatible model wrapper: we will provide compute_loss by delegating
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
            # move weights to device of model
            labels = kwargs.get('labels', None)
            # attach auxiliary loss objects in labels dict for model.forward
            if labels is not None:
                # labels is a dict inside collate; ensure keys exist
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

            out = self.model(input_ids=kwargs.get('input_ids', None),
                             attention_mask=kwargs.get('attention_mask', None),
                             labels=labels_payload)
            # Trainer expects: return loss and logits
            loss = out["loss"]
            logits1 = out["logits_level1"].detach().cpu().numpy()
            logits2 = out["logits_level2"].detach().cpu().numpy()
            # but Trainer will use model(**inputs) to get loss tensor and predictions come from predict
            # Here we return a simple object; Trainer handles predictions from model separately
            # For compatibility, return loss tensor and raw logits as attributes on the module output when called by Trainer
            # But huggingface Trainer expects model() returns a torch.Tensor (loss) or ModelOutput. We'll return a simple tuple
            # To be safer, return a custom object:
            return loss

    wrapped = WrappedModel(model, level1_weights, level2_weights, alpha=args.alpha, beta=args.beta, focal_gamma=args.focal_gamma)

    # Build a HF Trainer but we need to override compute_loss and prediction_step to handle dual outputs.
    class TwoHeadTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # inputs: include 'labels' as dict
            labels = inputs.pop('labels')
            # move labels to model device
            device = model.model.roberta.device if hasattr(model, "model") else next(model.parameters()).device
            labels_device = {"level1": labels["level1"].to(device), "level2": labels["level2"].to(device)}
            # call underlying model as defined above (which expects labels via kwargs)
            out = model.model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'),
                              labels={"level1": labels_device["level1"],
                                      "level2": labels_device["level2"],
                                      "level1_weight": level1_weights.to(device),
                                      "level2_focal": FocalLoss(gamma=args.focal_gamma, weight=level2_weights.to(device)),
                                      "alpha": args.alpha,
                                      "beta": args.beta})
            loss = out['loss']
            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """
            We override to return predictions for both heads in the shape expected by compute_metrics.
            """
            has_labels = "labels" in inputs and inputs["labels"] is not None
            labels = inputs.pop("labels") if has_labels else None
            # forward pass (no loss attached)
            out = model.model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))
            level1_logits = out['logits_level1']
            level2_logits = out['logits_level2']

            # move logits to CPU numpy
            level1_logits = level1_logits.detach().cpu().numpy()
            level2_logits = level2_logits.detach().cpu().numpy()

            if has_labels:
                label_level1 = labels['level1'].numpy()
                label_level2 = labels['level2'].numpy()
                # We must return tuple (loss, (preds, labels)) where preds can be anything; our compute_metrics expects (preds, label_ids)
                # To be compatible with compute_metrics (which expects EvalPrediction with predictions and label_ids),
                # Trainer will later call predict and wrap results. We'll return None for loss (Trainer handles it).
                return None, (np.stack([level1_logits, level2_logits], axis=0)), {"level1": label_level1, "level2": label_level2}
            else:
                return None, (np.stack([level1_logits, level2_logits], axis=0)), None

    # instantiate trainer
    trainer = TwoHeadTrainer(
        model=wrapped,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # scheduler will be created by trainer if needed
    )

    # run training
    trainer.train()

    # final eval
    metrics = trainer.evaluate(eval_dataset=val_ds)
    print("FINAL EVAL METRICS:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="./dataset/mn_ds_data.csv")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="./roberta_mnds_model")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--title_col", type=str, default=None)
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
    parser.add_argument("--layer_decay", type=float, default=0.8)  # layer-wise LR decay
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default=None)
    args = parser.parse_args()
    main(args)
