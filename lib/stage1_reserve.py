# -*- coding: utf-8 -*-
"""Stage-1 reserve helpers for prototype/confusion-based virtual anchor initialization."""
import json
import math
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from . import simple_metrics as metrics


EPS = 1e-12


def _normalize(x, dim=1):
    return F.normalize(x, dim=dim, eps=EPS)


def compute_known_logits(features, classifier_weight, scale):
    feat = _normalize(features, dim=1)
    weight = _normalize(classifier_weight, dim=1)
    return scale * torch.mm(feat, weight.t())


def compute_virtual_logits(features, anchors, scale):
    feat = _normalize(features, dim=1)
    anc = _normalize(anchors, dim=1)
    return scale * torch.mm(feat, anc.t())


def collect_client_prototypes(model, loader, device, known_class, max_batches=0):
    model.eval()
    feature_sum = None
    sample_count = torch.zeros(known_class, dtype=torch.long)
    with torch.no_grad():
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            features = model(inputs)['feature']
            if feature_sum is None:
                feature_sum = torch.zeros(known_class, features.shape[1], device=device)
            for feat, label in zip(features, targets):
                feature_sum[label] += feat
                sample_count[label] += 1
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
    if feature_sum is None:
        feature_sum = torch.zeros(known_class, 0)
    return feature_sum.cpu(), sample_count.cpu()


def aggregate_global_prototypes(feature_sums, sample_counts):
    total_sum = torch.stack(feature_sums, 0).sum(0)
    total_count = torch.stack(sample_counts, 0).sum(0)
    prototypes = []
    for idx in range(total_sum.shape[0]):
        if int(total_count[idx].item()) <= 0:
            raise ValueError(f'Known class {idx} has zero samples during warm-up prototype aggregation.')
        proto = total_sum[idx] / total_count[idx].float()
        prototypes.append(_normalize(proto.unsqueeze(0), dim=1).squeeze(0))
    return torch.stack(prototypes, 0), total_count


def _pair_candidates(prototypes, confusion, class_names):
    candidates = []
    row_sums = confusion.sum(1)
    for i in range(prototypes.shape[0]):
        for j in range(i + 1, prototypes.shape[0]):
            sym_confusion = 0.5 * (
                float(confusion[i, j]) / max(float(row_sums[i]), 1.0) +
                float(confusion[j, i]) / max(float(row_sums[j]), 1.0)
            )
            proto_sim = 0.5 * (1.0 + float(F.cosine_similarity(
                prototypes[i].unsqueeze(0), prototypes[j].unsqueeze(0)
            ).item()))
            pair_score = 0.5 * sym_confusion + 0.5 * proto_sim
            anchor = _normalize((prototypes[i] + prototypes[j]).unsqueeze(0), dim=1).squeeze(0)
            candidates.append({
                'pair': [int(i), int(j)],
                'pair_names': [class_names[i], class_names[j]],
                'sym_confusion': sym_confusion,
                'prototype_similarity': proto_sim,
                'pair_score': pair_score,
                'anchor': anchor,
            })
    candidates.sort(key=lambda item: item['pair_score'], reverse=True)
    return candidates


def init_virtual_anchors(args, prototypes, confusion, class_names):
    candidates = _pair_candidates(prototypes, confusion, class_names)
    selected = []
    for candidate in candidates:
        if len(selected) >= args.virtue_num:
            break
        if not selected:
            selected.append(candidate)
            continue
        allow = True
        for picked in selected:
            cosine = float(F.cosine_similarity(candidate['anchor'].unsqueeze(0), picked['anchor'].unsqueeze(0)).item())
            if cosine >= args.anchor_similarity_threshold:
                allow = False
                break
        if allow:
            selected.append(candidate)
    if len(selected) < args.virtue_num:
        used = {tuple(item['pair']) for item in selected}
        for candidate in candidates:
            if tuple(candidate['pair']) in used:
                continue
            selected.append(candidate)
            if len(selected) >= args.virtue_num:
                break

    anchors = torch.stack([item['anchor'] for item in selected], 0)
    anchor_cos = torch.mm(_normalize(anchors, dim=1), _normalize(anchors, dim=1).t()).cpu().tolist()
    return {
        'initialized': True,
        'virtual_anchors': anchors.cpu(),
        'selected_anchor_pairs': [{
            'pair': item['pair'],
            'pair_names': item['pair_names'],
            'pair_score': float(item['pair_score']),
            'sym_confusion': float(item['sym_confusion']),
            'prototype_similarity': float(item['prototype_similarity']),
        } for item in selected],
        'anchor_pairwise_cosine': anchor_cos,
        'prototype_count': [],
        'known_class_names': list(class_names),
        'all_candidates': [{
            'pair': item['pair'],
            'pair_names': item['pair_names'],
            'pair_score': float(item['pair_score']),
            'sym_confusion': float(item['sym_confusion']),
            'prototype_similarity': float(item['prototype_similarity']),
        } for item in candidates],
    }


def total_int_list(sample_counts):
    if sample_counts is None:
        return []
    return [int(item) for item in sample_counts.tolist()]


def attach_counts(stage1_state, sample_counts):
    stage1_state['prototype_count'] = total_int_list(sample_counts)
    return stage1_state


def known_virtual_stats(known_logits, virtual_logits, targets, known_class):
    if virtual_logits is None or virtual_logits.numel() == 0:
        zeros = np.zeros(targets.shape[0], dtype=np.float32)
        return {
            'pred_is_virtual': np.zeros(targets.shape[0], dtype=bool),
            'pred_virtual_idx': np.full(targets.shape[0], -1, dtype=np.int64),
            'kv_margin': zeros,
            'true_virtual_margin': zeros,
        }

    full_logits = torch.cat([known_logits, virtual_logits], dim=1)
    pred = full_logits.argmax(1)
    pred_is_virtual = (pred >= known_class).detach().cpu().numpy()
    pred_virtual_idx = np.where(pred_is_virtual, pred.detach().cpu().numpy() - known_class, -1)
    true_known = known_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    kv_margin = (true_known - virtual_logits.max(1)[0]).detach().cpu().numpy()
    return {
        'pred_is_virtual': pred_is_virtual,
        'pred_virtual_idx': pred_virtual_idx,
        'kv_margin': kv_margin,
        'true_virtual_margin': kv_margin,
    }


def close_open_virtual_eval(args, model, device, closerloader, openloader, stage1_state):
    model.eval()
    anchors = stage1_state['virtual_anchors'].to(device)
    close_virtual_flags = []
    close_virtual_hist = []
    close_true_virtual_margin = []
    close_pred = []
    close_targets = []
    close_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outs = model(inputs)
            features = outs['feature']
            known_logits = compute_known_logits(features, model.main_cls.weight[:args.known_class], args.cosine_scale)
            virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)
            close_loss += criterion(known_logits, targets).item()
            stats = known_virtual_stats(known_logits, virtual_logits, targets, args.known_class)
            close_virtual_flags.append(stats['pred_is_virtual'])
            close_true_virtual_margin.append(stats['true_virtual_margin'])
            full_pred = torch.cat([known_logits, virtual_logits], dim=1).argmax(1).detach().cpu().numpy()
            close_virtual_hist.append(full_pred)
            close_pred.extend(known_logits.argmax(1).detach().cpu().numpy().tolist())
            close_targets.extend(targets.cpu().numpy().tolist())
            if args.max_eval_batches > 0 and batch_idx + 1 >= args.max_eval_batches:
                break

    open_virtual_flags = []
    open_virtual_hist = []
    open_known_virtual_margin = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(openloader):
            inputs = inputs.to(device)
            outs = model(inputs)
            features = outs['feature']
            known_logits = compute_known_logits(features, model.main_cls.weight[:args.known_class], args.cosine_scale)
            virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)
            full_logits = torch.cat([known_logits, virtual_logits], dim=1)
            pred = full_logits.argmax(1).detach().cpu().numpy()
            open_virtual_flags.append(pred >= args.known_class)
            open_virtual_hist.append(pred)
            open_known_virtual_margin.append((known_logits.max(1)[0] - virtual_logits.max(1)[0]).detach().cpu().numpy())
            if args.max_eval_batches > 0 and batch_idx + 1 >= args.max_eval_batches:
                break

    close_virtual_flags = np.concatenate(close_virtual_flags) if close_virtual_flags else np.zeros(0, dtype=bool)
    close_true_virtual_margin = np.concatenate(close_true_virtual_margin) if close_true_virtual_margin else np.zeros(0, dtype=np.float32)
    close_virtual_hist = np.concatenate(close_virtual_hist) if close_virtual_hist else np.zeros(0, dtype=np.int64)
    open_virtual_flags = np.concatenate(open_virtual_flags) if open_virtual_flags else np.zeros(0, dtype=bool)
    open_virtual_hist = np.concatenate(open_virtual_hist) if open_virtual_hist else np.zeros(0, dtype=np.int64)
    open_known_virtual_margin = np.concatenate(open_known_virtual_margin) if open_known_virtual_margin else np.zeros(0, dtype=np.float32)

    close_hist = np.bincount(close_virtual_hist[close_virtual_hist >= args.known_class] - args.known_class, minlength=args.virtue_num) if close_virtual_hist.size > 0 else np.zeros(args.virtue_num, dtype=np.int64)
    open_hist = np.bincount(open_virtual_hist[open_virtual_hist >= args.known_class] - args.known_class, minlength=args.virtue_num) if open_virtual_hist.size > 0 else np.zeros(args.virtue_num, dtype=np.int64)

    labels = np.array(close_targets, dtype=np.int64)
    preds = np.array(close_pred, dtype=np.int64)
    close_test_result = {
        'loss': close_loss / max(len(close_targets), 1),
        'acc': 100.0 * metrics.accuracy_score(labels, preds) if labels.size > 0 else 0.0,
        'f1': 100.0 * metrics.f1_score(labels, preds, average='macro', zero_division=0) if labels.size > 0 else 0.0,
        'recall': 100.0 * metrics.recall_score(labels, preds, average='macro', zero_division=0) if labels.size > 0 else 0.0,
        'precision': 100.0 * metrics.precision_score(labels, preds, average='macro', zero_division=0) if labels.size > 0 else 0.0,
    }
    osr_result = {
        'close_virtual_pred_rate': float(close_virtual_flags.mean() * 100.0) if close_virtual_flags.size > 0 else 0.0,
        'close_true_virtual_logit_margin': margin_summary(close_true_virtual_margin),
        'close_virtual_hist': close_hist.tolist(),
        'open_virtual_pred_rate': float(open_virtual_flags.mean() * 100.0) if open_virtual_flags.size > 0 else 0.0,
        'open_known_virtual_logit_margin': margin_summary(open_known_virtual_margin),
        'open_virtual_hist': open_hist.tolist(),
    }
    return osr_result, close_test_result


def margin_summary(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0, 'max': 0.0}
    return {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'p10': float(np.percentile(arr, 10)),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'max': float(arr.max()),
    }


def compute_anchor_density(features, anchors, angle_thresholds):
    if features.size == 0 or anchors is None or anchors.numel() == 0:
        return {str(int(angle)): [0 for _ in range(anchors.shape[0] if anchors is not None else 0)] for angle in angle_thresholds}
    feat = _normalize(torch.from_numpy(features), dim=1)
    anc = _normalize(anchors.detach().cpu(), dim=1)
    sim = torch.mm(feat, anc.t())
    density = {}
    for angle in angle_thresholds:
        threshold = math.cos(math.radians(float(angle)))
        density[str(int(angle))] = [int(item) for item in (sim >= threshold).sum(0).tolist()]
    return density


def save_anchor_init(args, stage1_state):
    path = osp.join(args.save_path, args.anchor_init_file)
    payload = {
        'selected_anchor_pairs': stage1_state.get('selected_anchor_pairs', []),
        'anchor_pairwise_cosine': stage1_state.get('anchor_pairwise_cosine', []),
        'prototype_count': stage1_state.get('prototype_count', []),
        'known_class_names': stage1_state.get('known_class_names', []),
        'all_candidates': stage1_state.get('all_candidates', []),
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path


def serialize_stage1_state(stage1_state):
    if stage1_state is None:
        return None
    payload = dict(stage1_state)
    if 'virtual_anchors' in payload and torch.is_tensor(payload['virtual_anchors']):
        payload['virtual_anchors'] = payload['virtual_anchors'].cpu()
    return payload
