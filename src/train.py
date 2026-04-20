import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score
import numpy as np

def compute_class_weights(train_sent_labels):
    """Compute inverse frequency weights for sentiment classes."""
    class_counts = torch.bincount(train_sent_labels)
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum() * len(class_counts)
    return weights

def train_epoch(model, dataloader, optimizer, aspect_criterion, sentiment_criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        att_mask = batch['attention_mask'].to(device)
        aspect_labels = batch['aspect_labels'].to(device)
        sent_labels = batch['sentiment_labels'].to(device)
        optimizer.zero_grad()
        aspect_logits, sent_logits = model(input_ids, att_mask)
        loss_aspect = aspect_criterion(aspect_logits.view(-1, 3), aspect_labels.view(-1))
        loss_sent = sentiment_criterion(sent_logits, sent_labels)
        loss = loss_aspect + loss_sent
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, aspect_criterion, sentiment_criterion, device, id2label):
    model.eval()
    total_loss = 0
    all_aspect_preds, all_aspect_labels = [], []
    all_sent_preds, all_sent_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)
            sent_labels = batch['sentiment_labels'].to(device)
            aspect_logits, sent_logits = model(input_ids, att_mask)
            loss = aspect_criterion(aspect_logits.view(-1,3), aspect_labels.view(-1)) + \
                   sentiment_criterion(sent_logits, sent_labels)
            total_loss += loss.item()
            preds = torch.argmax(aspect_logits, dim=-1)
            for i in range(preds.size(0)):
                pred_seq = preds[i].cpu().numpy()
                label_seq = aspect_labels[i].cpu().numpy()
                mask = label_seq != 0
                if mask.any():
                    all_aspect_preds.append([id2label[p] for p in pred_seq[mask]])
                    all_aspect_labels.append([id2label[l] for l in label_seq[mask]])
            sent_preds = torch.argmax(sent_logits, dim=-1)
            all_sent_preds.extend(sent_preds.cpu().numpy())
            all_sent_labels.extend(sent_labels.cpu().numpy())
    aspect_f1 = f1_score(all_aspect_labels, all_aspect_preds)
    sent_acc = (np.array(all_sent_preds) == np.array(all_sent_labels)).mean()
    return total_loss / len(dataloader), aspect_f1, sent_acc

def evaluate_implicit_explicit(model, dataloader, device, id2label):
    """Compute aspect F1 separately for sentences with/without implicit aspects."""
    model.eval()
    imp_preds, imp_labels = [], []
    exp_preds, exp_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Implicit/Explicit"):
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)
            flags = batch['has_implicit'].cpu().numpy()
            aspect_logits, _ = model(input_ids, att_mask)
            preds = torch.argmax(aspect_logits, dim=-1)
            for i in range(preds.size(0)):
                pred_seq = preds[i].cpu().numpy()
                label_seq = aspect_labels[i].cpu().numpy()
                mask = label_seq != 0
                if mask.any():
                    p_str = [id2label[p] for p in pred_seq[mask]]
                    l_str = [id2label[l] for l in label_seq[mask]]
                    if flags[i] == 1:
                        imp_preds.append(p_str); imp_labels.append(l_str)
                    else:
                        exp_preds.append(p_str); exp_labels.append(l_str)
    imp_f1 = f1_score(imp_labels, imp_preds) if imp_labels else 0.0
    exp_f1 = f1_score(exp_labels, exp_preds) if exp_labels else 0.0
    return imp_f1, exp_f1