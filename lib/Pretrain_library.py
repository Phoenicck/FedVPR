# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:03:23 2022

@author: ZML
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics


def _virtual_stats_from_prob(prob, known_class):
    if prob.shape[1] <= known_class:
        zeros = np.zeros(prob.shape[0], dtype=np.float32)
        return {
            'virtual_prob_sum': zeros,
            'virtual_prob_max': zeros,
            'pred_is_virtual': np.zeros(prob.shape[0], dtype=bool),
            'pred_virtual_idx': np.full(prob.shape[0], -1, dtype=np.int64),
            'virtual_prob_mean_per_anchor': [],
        }

    virtual_prob = prob[:, known_class:]
    pred = prob.argmax(1)
    pred_is_virtual = pred >= known_class
    pred_virtual_idx = np.where(pred_is_virtual, pred - known_class, -1)
    return {
        'virtual_prob_sum': virtual_prob.sum(1),
        'virtual_prob_max': virtual_prob.max(1),
        'pred_is_virtual': pred_is_virtual,
        'pred_virtual_idx': pred_virtual_idx,
        'virtual_prob_mean_per_anchor': virtual_prob.mean(0).tolist(),
    }


def _normalized_entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 0.0
    p = counts / total
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    return float(entropy / np.log(len(counts)))


def _margin_summary(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'p10': 0.0,
            'p50': 0.0,
            'p90': 0.0,
            'max': 0.0,
        }
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




def _feature_tensor_to_numpy(feats):
    if feats.dim() == 4:
        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
    elif feats.dim() > 2:
        feats = feats.flatten(1)
    return feats.detach().cpu().numpy().astype(np.float32)


def _build_stage1_geometry_summary(features, labels, margin_values, args):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    margin_values = np.asarray(margin_values, dtype=np.float32).reshape(-1)
    eps = float(getattr(args, 'stage1_diag_eps', 1e-12))

    if features.size == 0 or labels.size == 0:
        empty_boundary = {
            'count': 0,
            'rate': 0.0,
            'threshold': 0.0,
            'hist': [0 for _ in range(args.known_class)],
            'mean_margin': 0.0,
        }
        return {
            'feature_dim': 0,
            'class_count': 0,
            'intra_class_variance_mean': 0.0,
            'inter_class_center_distance_mean': 0.0,
            'center_norm_mean': 0.0,
            'compactness_ratio': 0.0,
            'known_true_other_margin': _margin_summary([]),
            'boundary_candidate': empty_boundary,
        }

    unique_labels = np.unique(labels)
    class_centers = []
    intra_vars = []
    center_norms = []
    for cls in unique_labels:
        cls_feats = features[labels == cls]
        if cls_feats.shape[0] == 0:
            continue
        center = cls_feats.mean(axis=0)
        class_centers.append(center)
        center_norms.append(float(np.linalg.norm(center)))
        sq_dist = np.sum((cls_feats - center) ** 2, axis=1)
        intra_vars.append(float(sq_dist.mean()) if sq_dist.size > 0 else 0.0)

    if class_centers:
        centers = np.stack(class_centers, axis=0)
        if centers.shape[0] > 1:
            diff = centers[:, None, :] - centers[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            mask = ~np.eye(centers.shape[0], dtype=bool)
            inter_vals = dists[mask]
            inter_mean = float(inter_vals.mean()) if inter_vals.size > 0 else 0.0
        else:
            inter_mean = 0.0
    else:
        inter_mean = 0.0

    intra_mean = float(np.mean(intra_vars)) if intra_vars else 0.0
    compactness = float(intra_mean / max(inter_mean, eps)) if inter_mean > 0 else 0.0

    boundary_quantile = float(np.clip(getattr(args, 'stage1_boundary_quantile', 0.2), 0.0, 1.0))
    if margin_values.size == 0:
        boundary_mask = np.zeros(labels.shape[0], dtype=bool)
        boundary_threshold = 0.0
    elif boundary_quantile <= 0.0:
        boundary_mask = np.zeros_like(margin_values, dtype=bool)
        boundary_threshold = float(margin_values.min())
    elif boundary_quantile >= 1.0:
        boundary_mask = np.ones_like(margin_values, dtype=bool)
        boundary_threshold = float(margin_values.max())
    else:
        boundary_threshold = float(np.quantile(margin_values, boundary_quantile))
        boundary_mask = margin_values <= boundary_threshold

    boundary_hist = np.bincount(labels[boundary_mask], minlength=args.known_class).tolist() if labels.size > 0 else [0 for _ in range(args.known_class)]

    return {
        'feature_dim': int(features.shape[1]),
        'class_count': int(len(unique_labels)),
        'intra_class_variance_mean': intra_mean,
        'inter_class_center_distance_mean': inter_mean,
        'center_norm_mean': float(np.mean(center_norms)) if center_norms else 0.0,
        'compactness_ratio': compactness,
        'known_true_other_margin': _margin_summary(margin_values),
        'boundary_candidate': {
            'count': int(boundary_mask.sum()),
            'rate': float(boundary_mask.mean() * 100.0) if boundary_mask.size > 0 else 0.0,
            'threshold': boundary_threshold,
            'hist': boundary_hist,
            'mean_margin': float(margin_values[boundary_mask].mean()) if boundary_mask.any() else 0.0,
        },
    }

def _get_virtual_loss_weight(args, epoch):
    warmup_epochs = max(0, int(getattr(args, 'vir_warmup_epochs', 4)))
    anneal_epochs = max(0, int(getattr(args, 'vir_anneal_epochs', 0)))
    warmup_weight = float(getattr(args, 'vir_weight_warmup', 0.5))
    main_weight = float(getattr(args, 'vir_weight_main', 0.01))

    if epoch < warmup_epochs:
        return warmup_weight
    if anneal_epochs <= 0:
        return main_weight

    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / float(anneal_epochs)))
    return main_weight + (warmup_weight - main_weight) * (1.0 - progress)


def train(args, device, epoch, net, trainloader, optimizer):
    net.train()
    train_loss = 0
    train_loss_ce = 0
    train_loss_vir = 0
    train_loss_vir_weighted = 0
    train_loss_aux = 0
    pred_list = []
    label_list = []
    output_list = []
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets, img_dirs) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.long().to(device)
        optimizer.zero_grad()
        outs = net(inputs)
        outputs = outs['outputs']
        aux_outputs = outs['aux_out']
        # Stage 1: Pre-training with Space Reservation
        # L_Stage1 = L_CE(W_known) + lambda * L_vir(W_known U W_vir)

        # 1. Split outputs into known and virtual logits
        known_logits = outputs[:, :args.known_class]
        virtual_logits = outputs[:, args.known_class:]

        # 2. Standard CrossEntropy on Known Classes
        loss_ce = criterion(known_logits, targets)

        # 3. Virtual Softmax Loss plus a small margin add-on.
        true_class_logits = known_logits.gather(1, targets.unsqueeze(1))  # [B, 1]
        vir_loss_input = torch.cat([true_class_logits, virtual_logits], dim=1)  # [B, 1 + M]
        vir_loss_targets = torch.zeros(inputs.size(0), dtype=torch.long, device=device)
        loss_vir_ce = criterion(vir_loss_input, vir_loss_targets)
        max_virtual_logits = virtual_logits.max(1, keepdim=True)[0]
        loss_vir_margin = torch.relu(args.vir_margin - (true_class_logits - max_virtual_logits)).mean()
        loss_vir = loss_vir_ce + args.vir_margin_weight * loss_vir_margin

        # Total loss with a configurable virtual-loss schedule.
        vir_weight = _get_virtual_loss_weight(args, epoch)
        weighted_loss_vir = vir_weight * loss_vir
        loss = loss_ce + weighted_loss_vir
        # Add Aux Loss
        loss_aux = criterion(aux_outputs, targets)
        loss += loss_aux
        loss.backward()
        if args.virtue_num > 0 and net.main_cls.weight.grad is not None:
            # deepcopy() drops the constructor-registered grad hook, so clamp
            # virtual anchor rows explicitly on every client update.
            net.main_cls.weight.grad[args.known_class:] = 0
        optimizer.step()
        train_loss += loss.item()
        train_loss_ce += loss_ce.item()
        train_loss_vir += loss_vir.item()
        train_loss_vir_weighted += weighted_loss_vir.item()
        train_loss_aux += loss_aux.item()
        _, predicted = outputs[:, :args.known_class].max(1)

        pred_list.extend(predicted.cpu().numpy().tolist())
        label_list.extend(targets.cpu().numpy().tolist())
        output_list.append(torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy())

    loss_avg = train_loss/(batch_idx+1)
    loss_ce_avg = train_loss_ce/(batch_idx+1)
    loss_vir_avg = train_loss_vir/(batch_idx+1)
    loss_vir_weighted_avg = train_loss_vir_weighted/(batch_idx+1)
    loss_aux_avg = train_loss_aux/(batch_idx+1)
    mean_acc = 100*metrics.accuracy_score(label_list, pred_list)
    precision = 100*metrics.precision_score(label_list, pred_list, average='macro')
    recall_macro = 100*metrics.recall_score(y_true=label_list, y_pred=pred_list, average='macro')
    f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')

    result = {'loss':loss_avg,
              'loss_ce': loss_ce_avg,
              'loss_vir': loss_vir_avg,
              'loss_vir_weighted': loss_vir_weighted_avg,
              'loss_aux': loss_aux_avg,
              'vir_weight': vir_weight,
              'acc':mean_acc,
              'f1': f1_macro,
              'recall':recall_macro,
              'precision': precision,
              }
    return result


def val(args, device, epoch, net, valloader):
    net.eval()

    val_loss = 0
    pred_list = []
    label_list = []
    criterion = nn.CrossEntropyLoss()
    known_virtual_margin = []
    known_true_virtual_logit_margin = []
    known_true_other_margin = []
    virtual_prob_sum = []
    virtual_pred_flags = []
    val_features = []
    val_feature_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, img_dirs) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            aux_outputs = outs['aux_out']
            loss = criterion(outputs, targets)
            loss += criterion(aux_outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs[:, :args.known_class].max(1)
            pred_list.extend(predicted.cpu().numpy().tolist())
            label_list.extend(targets.cpu().numpy().tolist())
            val_features.append(_feature_tensor_to_numpy(outs['discrete_feats']))
            val_feature_labels.append(targets.cpu().numpy())

            prob = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy()
            stats = _virtual_stats_from_prob(prob, args.known_class)
            virtual_prob_sum.append(stats['virtual_prob_sum'])
            virtual_pred_flags.append(stats['pred_is_virtual'])
            if outputs.shape[1] > args.known_class:
                known_max = prob[:, :args.known_class].max(1)
                virtual_max = prob[:, args.known_class:].max(1)
                known_virtual_margin.append(known_max - virtual_max)
                true_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
                virtual_max_logit = outputs[:, args.known_class:].max(1)[0].detach().cpu().numpy()
                known_true_virtual_logit_margin.append(true_logits - virtual_max_logit)
            known_only_logits = outputs[:, :args.known_class]
            true_known_logits = known_only_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            masked_known_logits = known_only_logits.clone()
            masked_known_logits.scatter_(1, targets.unsqueeze(1), float('-inf'))
            max_other_known_logits = masked_known_logits.max(1)[0]
            known_true_other_margin.append((true_known_logits - max_other_known_logits).detach().cpu().numpy())

        loss_avg = val_loss/(batch_idx+1)
        mean_acc = 100*metrics.accuracy_score(label_list, pred_list)
        precision = 100*metrics.precision_score(label_list, pred_list, average='macro')
        recall_macro = 100*metrics.recall_score(y_true=label_list, y_pred=pred_list, average='macro')
        f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')
        confusion_matrix = metrics.confusion_matrix(y_true=label_list, y_pred=pred_list)

        virtual_prob_sum = np.concatenate(virtual_prob_sum) if virtual_prob_sum else np.zeros(0, dtype=np.float32)
        virtual_pred_flags = np.concatenate(virtual_pred_flags) if virtual_pred_flags else np.zeros(0, dtype=bool)
        known_virtual_margin = np.concatenate(known_virtual_margin) if known_virtual_margin else np.zeros(0, dtype=np.float32)
        known_true_virtual_logit_margin = np.concatenate(known_true_virtual_logit_margin) if known_true_virtual_logit_margin else np.zeros(0, dtype=np.float32)
        known_true_other_margin = np.concatenate(known_true_other_margin) if known_true_other_margin else np.zeros(0, dtype=np.float32)
        val_features = np.concatenate(val_features, axis=0) if val_features else np.zeros((0, 0), dtype=np.float32)
        val_feature_labels = np.concatenate(val_feature_labels) if val_feature_labels else np.zeros(0, dtype=np.int64)
        stage1_geometry = _build_stage1_geometry_summary(val_features, val_feature_labels, known_true_other_margin, args)

        result = {'loss':loss_avg,
                      'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision': precision,
                      'confusion_matrix':confusion_matrix,
                      'known_virtual_pred_rate': float(virtual_pred_flags.mean() * 100.0) if len(virtual_pred_flags) > 0 else 0.0,
                      'known_virtual_prob_mean': float(virtual_prob_sum.mean()) if len(virtual_prob_sum) > 0 else 0.0,
                      'known_virtual_margin_mean': float(known_virtual_margin.mean()) if len(known_virtual_margin) > 0 else 0.0,
                      'known_true_virtual_logit_margin': _margin_summary(known_true_virtual_logit_margin),
                      'known_true_other_logit_margin': _margin_summary(known_true_other_margin),
                      'stage1_geometry': stage1_geometry,
                      }
    return result


def test(args, device, epoch, net, closerloader, openloader, threshold=0):
    net.eval()

    temperature = 1.
    with torch.no_grad():
        pred_list=[]
        targets_list=[]
        test_loss=0
        criterion = nn.CrossEntropyLoss()

        pred_list_temp = []
        label_list_temp = []

        for batch_idx, (inputs, targets, img_dirs) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            aux_outputs = outs['aux_out']
            loss = criterion(outputs, targets)
            loss += criterion(aux_outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs[:, :args.known_class].max(1)
            pred_list_temp.extend(predicted.cpu().numpy().tolist())
            label_list_temp.extend(targets.cpu().numpy().tolist())

        loss_avg = test_loss/(batch_idx+1)
        mean_acc = 100*metrics.accuracy_score(label_list_temp, pred_list_temp)
        precision = 100*metrics.precision_score(label_list_temp, pred_list_temp, average='macro')
        recall_macro = 100*metrics.recall_score(y_true=label_list_temp, y_pred=pred_list_temp, average='macro')
        f1_macro = 100*metrics.f1_score(y_true=label_list_temp, y_pred=pred_list_temp, average='macro')
        confusion_matrix = metrics.confusion_matrix(y_true=label_list_temp, y_pred=pred_list_temp)

        close_test_result = {'loss':loss_avg,
                      'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision':precision,
                      'confusion_matrix':confusion_matrix}

        close_prob_known_list = []
        close_virtual_prob_sum = []
        close_virtual_prob_mean_per_anchor = []
        close_raw_preds = []
        close_targets = []
        close_virtual_pred_flags = []
        close_virtual_margin = []
        close_true_virtual_logit_margin = []

        for batch_idx, (inputs, targets, img_dirs) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            prob = nn.functional.softmax(outputs/temperature,dim=-1).cpu().numpy()
            stats = _virtual_stats_from_prob(prob, args.known_class)
            close_prob_known_list.append(prob[:, :args.known_class].max(1))
            close_virtual_prob_sum.append(stats['virtual_prob_sum'])
            close_virtual_prob_mean_per_anchor.append(stats['virtual_prob_mean_per_anchor'])
            close_virtual_pred_flags.append(stats['pred_is_virtual'])
            close_raw_preds.append(prob.argmax(1))
            close_targets.append(targets.cpu().numpy())
            if prob.shape[1] > args.known_class:
                close_virtual_margin.append(prob[:, :args.known_class].max(1) - prob[:, args.known_class:].max(1))
                true_logits = outputs.gather(1, targets.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
                virtual_max_logit = outputs[:, args.known_class:].max(1)[0].detach().cpu().numpy()
                close_true_virtual_logit_margin.append(true_logits - virtual_max_logit)

        open_prob_known_list = []
        open_virtual_prob_sum = []
        open_virtual_prob_mean_per_anchor = []
        open_raw_preds = []
        open_targets = []
        open_known_virtual_logit_margin = []

        for batch_idx, (inputs, targets, img_dirs) in enumerate(openloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            prob = nn.functional.softmax(outputs/temperature,dim=-1).cpu().numpy()
            stats = _virtual_stats_from_prob(prob, args.known_class)
            open_prob_known_list.append(prob[:, :args.known_class].max(1))
            open_virtual_prob_sum.append(stats['virtual_prob_sum'])
            open_virtual_prob_mean_per_anchor.append(stats['virtual_prob_mean_per_anchor'])
            open_raw_preds.append(prob.argmax(1))
            open_targets.append(np.ones_like(targets.cpu().numpy()) * args.known_class)
            if outputs.shape[1] > args.known_class:
                known_max_logit = outputs[:, :args.known_class].max(1)[0].detach().cpu().numpy()
                virtual_max_logit = outputs[:, args.known_class:].max(1)[0].detach().cpu().numpy()
                open_known_virtual_logit_margin.append(known_max_logit - virtual_max_logit)

        close_prob_known_array = np.concatenate(close_prob_known_list) if close_prob_known_list else np.zeros(0, dtype=np.float32)
        open_prob_known_array = np.concatenate(open_prob_known_list) if open_prob_known_list else np.zeros(0, dtype=np.float32)
        prob_known_array = np.concatenate([close_prob_known_array, open_prob_known_array])
        close_virtual_prob_sum = np.concatenate(close_virtual_prob_sum) if close_virtual_prob_sum else np.zeros(0, dtype=np.float32)
        open_virtual_prob_sum = np.concatenate(open_virtual_prob_sum) if open_virtual_prob_sum else np.zeros(0, dtype=np.float32)
        close_virtual_pred_flags = np.concatenate(close_virtual_pred_flags) if close_virtual_pred_flags else np.zeros(0, dtype=bool)
        close_virtual_margin = np.concatenate(close_virtual_margin) if close_virtual_margin else np.zeros(0, dtype=np.float32)
        close_true_virtual_logit_margin = np.concatenate(close_true_virtual_logit_margin) if close_true_virtual_logit_margin else np.zeros(0, dtype=np.float32)
        open_known_virtual_logit_margin = np.concatenate(open_known_virtual_logit_margin) if open_known_virtual_logit_margin else np.zeros(0, dtype=np.float32)

        close_raw_preds = np.concatenate(close_raw_preds) if close_raw_preds else np.zeros(0, dtype=np.int64)
        open_raw_preds = np.concatenate(open_raw_preds) if open_raw_preds else np.zeros(0, dtype=np.int64)
        close_targets = np.concatenate(close_targets) if close_targets else np.zeros(0, dtype=np.int64)
        open_targets = np.concatenate(open_targets) if open_targets else np.zeros(0, dtype=np.int64)

        targets_list = np.concatenate([close_targets, open_targets])
        pred_list = np.concatenate([close_raw_preds, open_raw_preds])

        # Calculate binary labels: 0 for known, 1 for unknown
        binary_labels = (targets_list == args.known_class).astype(int)

        # Calculate AUROC and AUPR
        try:
            auroc = 100.0 * metrics.roc_auc_score(binary_labels, 1 - prob_known_array)
            precision_curve, recall_curve, _ = metrics.precision_recall_curve(binary_labels, 1 - prob_known_array)
            aupr = 100.0 * metrics.auc(recall_curve, precision_curve)
        except:
            auroc = 0.0
            aupr = 0.0

        # Calculate UNK: Average unknown recall
        unknown_mask = targets_list == args.known_class
        if np.sum(unknown_mask) > 0:
            unk_recall = 100.0 * np.mean(pred_list[unknown_mask] >= args.known_class)
        else:
            unk_recall = 0.0

        # Calculate OS*: Average per-class recall for known classes
        known_mask = targets_list < args.known_class
        try:
            known_labels = list(range(args.known_class))
            if np.sum(known_mask) > 0:
                known_targets = targets_list[known_mask]
                known_preds = pred_list[known_mask]
                os_star = 100.0 * metrics.recall_score(
                    y_true=known_targets,
                    y_pred=known_preds,
                    labels=known_labels,
                    average='macro',
                    zero_division=0
                )
            else:
                os_star = 0.0
        except:
            os_star = 0.0

        # Calculate HOS (Harmonic Open-Set)
        if (os_star + unk_recall) > 0:
            hos = 2 * os_star * unk_recall / (os_star + unk_recall)
        else:
            hos = 0.0

        # Calculate OSCR (Open Set Classification Rate)
        try:
            prob_known = np.array(prob_known_array)
            targets = np.array(targets_list)
            is_known = (targets < args.known_class)
            is_unknown = ~is_known
            correct = np.zeros_like(targets, dtype=bool)
            correct[is_known] = (pred_list[is_known] == targets[is_known])
            sorted_idx = np.argsort(-prob_known)
            correct_sorted = correct[sorted_idx]
            unknown_sorted = is_unknown[sorted_idx]
            tp_cum = np.cumsum(correct_sorted)
            fp_cum = np.cumsum(unknown_sorted)
            n_known_total = np.sum(is_known)
            n_unknown_total = np.sum(is_unknown)
            if n_known_total == 0 or n_unknown_total == 0:
                oscr = 0.0
            else:
                ccr = tp_cum / n_known_total
                fpr = fp_cum / n_unknown_total
                fpr = np.concatenate([[0.0], fpr, [1.0]])
                ccr = np.concatenate([[0.0], ccr, [ccr[-1]]])
                oscr = 100.0 * metrics.auc(fpr, ccr)
                del prob_known, targets, is_known, is_unknown, correct
                del sorted_idx, correct_sorted, unknown_sorted
                del tp_cum, fp_cum, ccr, fpr
        except Exception as e:
            print(f"Warning: OSCR calculation failed: {e}")
            oscr = 0.0

        # Map all virtual class predictions to known_class for ACC/F1/Precision/Recall
        pred_list_mapped = pred_list.copy()
        pred_list_mapped[pred_list_mapped >= args.known_class] = args.known_class

        mean_acc = 100.0 * metrics.accuracy_score(targets_list, pred_list_mapped)
        precision = 100*metrics.precision_score(targets_list, pred_list_mapped, average='macro')
        recall_macro = 100.0*metrics.recall_score(y_true=targets_list, y_pred=pred_list_mapped, average='macro')
        f1_macro = 100*metrics.f1_score(y_true=targets_list, y_pred=pred_list_mapped, average='macro')

        close_virtual_hist = np.bincount(close_raw_preds[close_raw_preds >= args.known_class] - args.known_class, minlength=args.virtue_num) if len(close_raw_preds) > 0 and args.virtue_num > 0 else np.zeros(args.virtue_num, dtype=np.int64)
        open_virtual_hist = np.bincount(open_raw_preds[open_raw_preds >= args.known_class] - args.known_class, minlength=args.virtue_num) if len(open_raw_preds) > 0 and args.virtue_num > 0 else np.zeros(args.virtue_num, dtype=np.int64)
        close_virtual_prob_mean_per_anchor = np.mean(np.array(close_virtual_prob_mean_per_anchor), axis=0).tolist() if close_virtual_prob_mean_per_anchor and close_virtual_prob_mean_per_anchor[0] else []
        open_virtual_prob_mean_per_anchor = np.mean(np.array(open_virtual_prob_mean_per_anchor), axis=0).tolist() if open_virtual_prob_mean_per_anchor and open_virtual_prob_mean_per_anchor[0] else []

        osr_result = {'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision':precision,
                      'unk': unk_recall,
                      'os_star': os_star,
                      'hos': hos,
                      'auroc': auroc,
                      'aupr': aupr,
                      'oscr': oscr,
                      'close_virtual_pred_rate': float(close_virtual_pred_flags.mean() * 100.0) if len(close_virtual_pred_flags) > 0 else 0.0,
                      'close_virtual_prob_mean': float(close_virtual_prob_sum.mean()) if len(close_virtual_prob_sum) > 0 else 0.0,
                      'close_known_virtual_margin_mean': float(close_virtual_margin.mean()) if len(close_virtual_margin) > 0 else 0.0,
                      'close_virtual_hist': close_virtual_hist.tolist(),
                      'close_virtual_entropy': _normalized_entropy(close_virtual_hist),
                      'close_virtual_prob_mean_per_anchor': close_virtual_prob_mean_per_anchor,
                      'close_true_virtual_logit_margin': _margin_summary(close_true_virtual_logit_margin),
                      'open_virtual_pred_rate': float((open_raw_preds >= args.known_class).mean() * 100.0) if len(open_raw_preds) > 0 else 0.0,
                      'open_virtual_prob_mean': float(open_virtual_prob_sum.mean()) if len(open_virtual_prob_sum) > 0 else 0.0,
                      'open_virtual_hist': open_virtual_hist.tolist(),
                      'open_virtual_entropy': _normalized_entropy(open_virtual_hist),
                      'open_virtual_prob_mean_per_anchor': open_virtual_prob_mean_per_anchor,
                      'open_known_virtual_logit_margin': _margin_summary(open_known_virtual_logit_margin),
                      }

    return osr_result, close_test_result
