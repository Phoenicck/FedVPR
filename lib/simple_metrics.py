# -*- coding: utf-8 -*-
"""Small numpy-based metric helpers to avoid sklearn binary dependencies."""
import numpy as np


def _prepare_labels(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred])) if y_true.size or y_pred.size else np.array([], dtype=np.int64)
    return y_true, y_pred, np.asarray(labels)


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred, labels = _prepare_labels(y_true, y_pred, labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels.tolist())}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def precision_score(y_true, y_pred, average='macro', zero_division=0, labels=None):
    y_true, y_pred, labels = _prepare_labels(y_true, y_pred, labels)
    values = []
    for label in labels.tolist():
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        denom = tp + fp
        values.append(float(tp) / denom if denom > 0 else float(zero_division))
    values = np.asarray(values, dtype=np.float32)
    if average is None:
        return values
    if average == 'macro':
        return float(values.mean()) if values.size > 0 else 0.0
    raise ValueError(f'Unsupported average: {average}')


def recall_score(y_true, y_pred, average='macro', zero_division=0, labels=None):
    y_true, y_pred, labels = _prepare_labels(y_true, y_pred, labels)
    values = []
    for label in labels.tolist():
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = tp + fn
        values.append(float(tp) / denom if denom > 0 else float(zero_division))
    values = np.asarray(values, dtype=np.float32)
    if average is None:
        return values
    if average == 'macro':
        return float(values.mean()) if values.size > 0 else 0.0
    raise ValueError(f'Unsupported average: {average}')


def f1_score(y_true, y_pred, average='macro', zero_division=0, labels=None):
    precision = precision_score(y_true, y_pred, average=None, zero_division=zero_division, labels=labels)
    recall = recall_score(y_true, y_pred, average=None, zero_division=zero_division, labels=labels)
    denom = precision + recall
    values = np.where(denom > 0, 2 * precision * recall / denom, float(zero_division))
    if average is None:
        return values
    if average == 'macro':
        return float(values.mean()) if values.size > 0 else 0.0
    raise ValueError(f'Unsupported average: {average}')


def auc(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError('roc_auc_score requires both positive and negative samples.')
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[pos].sum()
    auc_value = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc_value)


def precision_recall_curve(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.ones_like(tp, dtype=np.float64), where=denom > 0)
    total_pos = max(int((y_true == 1).sum()), 1)
    recall = tp / total_pos
    thresholds = y_score
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall, thresholds
