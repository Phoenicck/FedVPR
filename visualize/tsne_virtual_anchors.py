#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import random
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = '/workspace/Phoenic/claude0527/FedVPR'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.fed_retinal_oct_relabel import OCT_CLASSES, OCT_PROTOCOLS, get_dataloaders, resolve_protocol  # noqa: E402
from models.ResNet_FedOSR_Pretrain import resnet18  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize FedVPR embeddings with fixed or classifier virtual anchors using t-SNE, PCA, or UMAP'
    )
    parser.add_argument('--data_root', default='/workspace/Phoenic/claude0527/FedVPR/datasets/RetinalOCT_Dataset')
    parser.add_argument('--checkpoint', default='/workspace/Phoenic/claude0527/FedVPR/results/MPretrain-DRetinalOCT-Msoftmax-BResnet18/LR0.0005-K5-U3-Seed0/best_ckpt_Pretrain_known_class_5_unknown_class_3_seed_0.pth')
    parser.add_argument('--output_dir', default='/workspace/Phoenic/claude0527/FedVPR/visualize')
    parser.add_argument('--protocol_mode', default='hard', choices=['easy', 'hard', 'random'])
    parser.add_argument('--known_class', type=int, default=5)
    parser.add_argument('--unknown_class', type=int, default=3)
    parser.add_argument('--virtue_num', type=int, default=3)
    parser.add_argument('--client_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--resize', type=int, default=144)
    parser.add_argument('--cropsize', type=int, default=128)
    parser.add_argument('--rotation', type=int, default=45)
    parser.add_argument('--dirichlet', type=float, default=0.5)
    parser.add_argument('--cosine_scale', type=float, default=16.0)
    parser.add_argument('--max_per_known_class', type=int, default=120)
    parser.add_argument('--max_per_unknown_class', type=int, default=120)
    parser.add_argument('--method', default='tsne', choices=['tsne', 'pca', 'umap'],
                        help='2D projection method')
    parser.add_argument('--fallback_to_pca', action='store_true',
                        help='fallback to PCA when the requested method is unavailable')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--tsne_seed', type=int, default=0)
    parser.add_argument('--umap_neighbors', type=int, default=20)
    parser.add_argument('--umap_min_dist', type=float, default=0.1)
    parser.add_argument('--anchor_source', default='auto', choices=['auto', 'fixed', 'classifier'],
                        help='use fixed anchors from stage1_state or virtual classifier weights')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_checkpoint(args):
    return torch.load(args.checkpoint, map_location='cpu')


def build_model(args, checkpoint_state, device):
    weight = checkpoint_state['net']['main_cls.weight']
    total_classes = int(weight.shape[0])
    num_virtual = max(0, total_classes - args.known_class)
    model = resnet18(pretrained=False, num_classes=args.known_class, num_virtual=num_virtual)
    model.load_state_dict(checkpoint_state['net'], strict=True)
    model.to(device)
    model.eval()
    return model


def _resolve_random_names(args):
    state = np.random.get_state()
    try:
        np.random.seed(args.seed)
        known_idx, unknown_idx = resolve_protocol({
            'Known_class': args.known_class,
            'unKnown_class': args.unknown_class,
            'protocol_mode': 'random',
        })
    finally:
        np.random.set_state(state)
    known_names = [OCT_CLASSES[idx] for idx in known_idx]
    unknown_names = [OCT_CLASSES[idx] for idx in unknown_idx]
    return known_names, unknown_names


def build_label_maps(args, stage1_state):
    if args.protocol_mode in OCT_PROTOCOLS:
        spec = OCT_PROTOCOLS[args.protocol_mode]
        known_names = list(spec['known'])
        unknown_names = list(spec['unknown'])
    else:
        known_names, unknown_names = _resolve_random_names(args)

    state_known = stage1_state.get('known_class_names', []) if stage1_state else []
    state_unknown = stage1_state.get('unknown_class_names', []) if stage1_state else []
    if state_known and not all(name.startswith('K') for name in state_known):
        known_names = list(state_known)
    if state_unknown and not all(name.startswith('U') for name in state_unknown):
        unknown_names = list(state_unknown)

    label_to_name = {i: name for i, name in enumerate(known_names)}
    label_to_name.update({i + len(known_names): name for i, name in enumerate(unknown_names)})
    return known_names, unknown_names, label_to_name


def build_anchor_bank(args, checkpoint_state, model, known_names):
    stage1_state = checkpoint_state.get('stage1_state', {}) or {}
    classifier_virtual = model.main_cls.weight[args.known_class:].detach().cpu().float()
    fixed_virtual = stage1_state.get('virtual_anchors')
    fixed_virtual = fixed_virtual.cpu().float() if torch.is_tensor(fixed_virtual) else None

    if args.anchor_source == 'fixed':
        source = 'fixed'
    elif args.anchor_source == 'classifier':
        source = 'classifier'
    else:
        source = 'fixed' if fixed_virtual is not None and fixed_virtual.numel() > 0 else 'classifier'

    if source == 'fixed':
        if fixed_virtual is None or fixed_virtual.numel() == 0:
            raise ValueError('Requested fixed anchor source, but checkpoint stage1_state has no virtual_anchors.')
        virtual_anchor = fixed_virtual
        pair_rows = stage1_state.get('selected_anchor_pairs', [])
        virtual_names = []
        for idx in range(virtual_anchor.shape[0]):
            if idx < len(pair_rows):
                pair_names = pair_rows[idx].get('pair_names', [])
                if pair_names and not all(name.startswith('K') for name in pair_names):
                    virtual_names.append(f"V{idx}:{'+'.join(pair_names)}")
                else:
                    pair = pair_rows[idx].get('pair', [])
                    if len(pair) == 2:
                        virtual_names.append(f"V{idx}:{known_names[pair[0]]}+{known_names[pair[1]]}")
                    else:
                        virtual_names.append(f'V{idx}')
            else:
                virtual_names.append(f'V{idx}')
    else:
        virtual_anchor = classifier_virtual
        virtual_names = [f'V{idx}' for idx in range(virtual_anchor.shape[0])]

    known_anchor = model.main_cls.weight[:args.known_class].detach().cpu().float()
    return {
        'source': source,
        'known_anchor': known_anchor,
        'virtual_anchor': virtual_anchor,
        'virtual_names': virtual_names,
        'stage1_state': stage1_state,
    }


@torch.no_grad()
def extract_embedding(model, inputs):
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    outputs = model.main_cls(x)
    return x, outputs


def compute_full_logits(args, model, features, outputs, anchor_bank, device):
    if anchor_bank['source'] == 'classifier':
        return outputs
    known_weight = model.main_cls.weight[:args.known_class]
    known_logits = args.cosine_scale * torch.mm(
        F.normalize(features, dim=1),
        F.normalize(known_weight, dim=1).t(),
    )
    virtual_anchor = anchor_bank['virtual_anchor'].to(device)
    virtual_logits = args.cosine_scale * torch.mm(
        F.normalize(features, dim=1),
        F.normalize(virtual_anchor, dim=1).t(),
    )
    return torch.cat([known_logits, virtual_logits], dim=1)


@torch.no_grad()
def collect_embeddings(args, model, anchor_bank, loader, device, max_per_class, split_name):
    feats = []
    labels = []
    split_tags = []
    pred_virtual = []
    pred_virtual_idx = []
    counts = Counter()

    for inputs, targets, _ in loader:
        label = int(targets.item())
        if counts[label] >= max_per_class:
            continue
        inputs = inputs.to(device)
        embedding, outputs = extract_embedding(model, inputs)
        full_logits = compute_full_logits(args, model, embedding, outputs, anchor_bank, device)
        pred = int(full_logits.argmax(1).item())
        is_virtual = pred >= args.known_class
        pred_idx = pred - args.known_class if is_virtual else -1
        feats.append(embedding.squeeze(0).cpu().numpy())
        labels.append(label)
        split_tags.append(split_name)
        pred_virtual.append(bool(is_virtual))
        pred_virtual_idx.append(int(pred_idx))
        counts[label] += 1

    return {
        'features': np.asarray(feats, dtype=np.float32),
        'labels': np.asarray(labels, dtype=np.int64),
        'split_tags': np.asarray(split_tags),
        'pred_virtual': np.asarray(pred_virtual, dtype=bool),
        'pred_virtual_idx': np.asarray(pred_virtual_idx, dtype=np.int64),
        'sample_counts': dict(counts),
    }


def fit_pca(embedding_matrix):
    n_samples = len(embedding_matrix)
    if n_samples <= 2:
        coords = np.zeros((n_samples, 2), dtype=np.float32)
        if n_samples == 2:
            coords[1, 0] = 1.0
        return coords, {'explained_variance_ratio': [1.0 if n_samples > 0 else 0.0, 0.0]}
    centered = embedding_matrix - embedding_matrix.mean(axis=0, keepdims=True)
    _, singular_vals, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    variance = singular_vals ** 2
    explained = variance / max(float(variance.sum()), 1e-12)
    return coords[:, :2], {'explained_variance_ratio': explained[:2].tolist()}


def _import_optional(module_name, hint):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f'Failed to import {module_name}. {hint}. Original error: {exc}') from exc


def fit_projection(method, embedding_matrix, args, perplexity=None):
    if method == 'pca':
        return fit_pca(embedding_matrix), 'pca'

    try:
        if method == 'tsne':
            sklearn_manifold = _import_optional(
                'sklearn.manifold',
                'Install a compatible scikit-learn build or rerun with --method pca / --fallback_to_pca',
            )
            TSNE = sklearn_manifold.TSNE
            n_samples = len(embedding_matrix)
            if n_samples <= 2:
                coords = np.zeros((n_samples, 2), dtype=np.float32)
                if n_samples == 2:
                    coords[1, 0] = 1.0
                return (coords, {}), 'tsne'
            perplexity = args.perplexity if perplexity is None else perplexity
            perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=args.tsne_seed,
                init='pca',
                learning_rate='auto',
                max_iter=2000,
            )
            return (reducer.fit_transform(embedding_matrix), {'perplexity': float(perplexity)}), 'tsne'

        if method == 'umap':
            umap_mod = _import_optional(
                'umap',
                'Install umap-learn or rerun with --method pca / --fallback_to_pca',
            )
            reducer = umap_mod.UMAP(
                n_components=2,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric='cosine',
                random_state=args.tsne_seed,
            )
            return (reducer.fit_transform(embedding_matrix), {
                'n_neighbors': int(args.umap_neighbors),
                'min_dist': float(args.umap_min_dist),
            }), 'umap'
    except Exception:
        if args.fallback_to_pca:
            return fit_pca(embedding_matrix), 'pca'
        raise

    raise ValueError(f'Unsupported projection method: {method}')


def save_overview_plot(coords, meta, output_path, known_names, unknown_names, method_used):
    plt.figure(figsize=(14, 11))
    cmap_known = plt.get_cmap('tab10')
    cmap_unknown = plt.get_cmap('Set2')

    for idx, name in enumerate(known_names):
        mask = (meta['kind'] == 'close') & (meta['label'] == idx)
        if mask.any():
            plt.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.65, color=cmap_known(idx), label=f'Known {name}')

    for idx, name in enumerate(unknown_names, start=len(known_names)):
        mask = (meta['kind'] == 'open') & (meta['label'] == idx)
        if mask.any():
            plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, color=cmap_unknown(idx - len(known_names)), marker='x', label=f'Unknown {name}')

    known_anchor_mask = meta['kind'] == 'known_anchor'
    if known_anchor_mask.any():
        plt.scatter(coords[known_anchor_mask, 0], coords[known_anchor_mask, 1], s=260, c='black', marker='s', label='Known anchors')
        for x, y, name in zip(coords[known_anchor_mask, 0], coords[known_anchor_mask, 1], meta['name'][known_anchor_mask]):
            plt.text(x, y, name, fontsize=9, weight='bold', ha='left', va='bottom')

    virtual_anchor_mask = meta['kind'] == 'virtual_anchor'
    if virtual_anchor_mask.any():
        plt.scatter(coords[virtual_anchor_mask, 0], coords[virtual_anchor_mask, 1], s=340, c='red', marker='*', edgecolors='black', linewidths=0.8, label='Virtual anchors')
        for x, y, name in zip(coords[virtual_anchor_mask, 0], coords[virtual_anchor_mask, 1], meta['name'][virtual_anchor_mask]):
            plt.text(x, y, name, fontsize=10, weight='bold', ha='left', va='bottom', color='darkred')

    plt.title(f'FedVPR RetinalOCT {method_used.upper()}: close/open samples with known and virtual anchors')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()


def save_virtual_focus_plot(coords, meta, output_path, method_used):
    plt.figure(figsize=(12, 10))
    colors = ['#d73027', '#1a9850', '#4575b4']
    for vid in range(3):
        mask = (meta['kind'] == 'open_virtual') & (meta['virtual_idx'] == vid)
        if mask.any():
            plt.scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.78, color=colors[vid % len(colors)], label=f'Open -> V{vid}')

    anchor_mask = meta['kind'] == 'virtual_anchor'
    if anchor_mask.any():
        plt.scatter(coords[anchor_mask, 0], coords[anchor_mask, 1], s=360, c='gold', marker='*', edgecolors='black', linewidths=1.0, label='Virtual anchors')
        for x, y, name in zip(coords[anchor_mask, 0], coords[anchor_mask, 1], meta['name'][anchor_mask]):
            plt.text(x, y, name, fontsize=11, weight='bold', ha='left', va='bottom')

    plt.title(f'FedVPR RetinalOCT {method_used.upper()}: open samples routed to virtual anchors')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_state = load_checkpoint(args)
    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() else torch.device('cpu')
    model = build_model(args, checkpoint_state, device)
    stage1_state = checkpoint_state.get('stage1_state', {}) or {}
    known_names, unknown_names, label_to_name = build_label_maps(args, stage1_state)
    anchor_bank = build_anchor_bank(args, checkpoint_state, model, known_names)

    param = {
        'Known_class': args.known_class,
        'unKnown_class': args.unknown_class,
        'Rotation': args.rotation,
        'Resize': args.resize,
        'CropSize': args.cropsize,
        'Batchsize': args.batchsize,
        'dirichlet': args.dirichlet,
        'protocol_mode': args.protocol_mode,
    }
    _, _, close_loader, open_loader, _ = get_dataloaders(args.client_num, args.data_root, args.seed, param)

    close_pack = collect_embeddings(args, model, anchor_bank, close_loader, device, args.max_per_known_class, 'close')
    open_pack = collect_embeddings(args, model, anchor_bank, open_loader, device, args.max_per_unknown_class, 'open')

    known_anchor = anchor_bank['known_anchor'].numpy().astype(np.float32)
    virtual_anchor = anchor_bank['virtual_anchor'].numpy().astype(np.float32)

    all_features = []
    meta_rows = []

    for feat, label in zip(close_pack['features'], close_pack['labels']):
        all_features.append(feat)
        meta_rows.append({'kind': 'close', 'label': int(label), 'name': label_to_name[int(label)], 'virtual_idx': -1})

    for feat, label in zip(open_pack['features'], open_pack['labels']):
        all_features.append(feat)
        meta_rows.append({'kind': 'open', 'label': int(label), 'name': label_to_name[int(label)], 'virtual_idx': -1})

    for idx, anchor in enumerate(known_anchor):
        all_features.append(anchor)
        meta_rows.append({'kind': 'known_anchor', 'label': idx, 'name': f'K{idx}:{known_names[idx]}', 'virtual_idx': -1})

    for idx, anchor in enumerate(virtual_anchor):
        all_features.append(anchor)
        meta_rows.append({'kind': 'virtual_anchor', 'label': args.known_class + idx, 'name': anchor_bank['virtual_names'][idx], 'virtual_idx': idx})

    matrix = np.asarray(all_features, dtype=np.float32)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    (coords, projection_meta), method_used = fit_projection(args.method, matrix, args)

    meta = {
        'kind': np.asarray([row['kind'] for row in meta_rows]),
        'label': np.asarray([row['label'] for row in meta_rows]),
        'name': np.asarray([row['name'] for row in meta_rows]),
        'virtual_idx': np.asarray([row['virtual_idx'] for row in meta_rows]),
    }

    output_stem = f'retinaloct_{args.protocol_mode}_{method_used}'
    overview_png = os.path.join(args.output_dir, f'{output_stem}_overview.png')
    save_overview_plot(coords, meta, overview_png, known_names, unknown_names, method_used)

    open_virtual_mask = open_pack['pred_virtual']
    open_virtual_features = open_pack['features'][open_virtual_mask]
    open_virtual_idx = open_pack['pred_virtual_idx'][open_virtual_mask]
    open_virtual_labels = open_pack['labels'][open_virtual_mask]

    focus_matrix = []
    focus_meta_rows = []
    for feat, label, vid in zip(open_virtual_features, open_virtual_labels, open_virtual_idx):
        focus_matrix.append(feat)
        focus_meta_rows.append({'kind': 'open_virtual', 'label': int(label), 'name': label_to_name[int(label)], 'virtual_idx': int(vid)})
    for idx, anchor in enumerate(virtual_anchor):
        focus_matrix.append(anchor)
        focus_meta_rows.append({'kind': 'virtual_anchor', 'label': args.known_class + idx, 'name': anchor_bank['virtual_names'][idx], 'virtual_idx': idx})

    focus_png = None
    focus_projection_meta = {}
    if focus_matrix:
        focus_matrix = np.asarray(focus_matrix, dtype=np.float32)
        focus_matrix = focus_matrix / np.clip(np.linalg.norm(focus_matrix, axis=1, keepdims=True), 1e-12, None)
        focus_perplexity = min(args.perplexity, 20.0) if args.method == 'tsne' else None
        (focus_coords, focus_projection_meta), focus_method_used = fit_projection(args.method, focus_matrix, args, perplexity=focus_perplexity)
        focus_meta = {
            'kind': np.asarray([row['kind'] for row in focus_meta_rows]),
            'virtual_idx': np.asarray([row['virtual_idx'] for row in focus_meta_rows]),
            'name': np.asarray([row['name'] for row in focus_meta_rows]),
        }
        focus_png = os.path.join(args.output_dir, f'{output_stem}_open_virtual_focus.png')
        save_virtual_focus_plot(focus_coords, focus_meta, focus_png, focus_method_used)

    virtual_hist = Counter(int(x) for x in open_virtual_idx if x >= 0)
    if virtual_anchor.shape[0] > 0:
        anchor_cos = F.normalize(torch.from_numpy(virtual_anchor), dim=1)
        anchor_cos = torch.mm(anchor_cos, anchor_cos.t()).numpy().tolist()
    else:
        anchor_cos = []

    summary = {
        'checkpoint': args.checkpoint,
        'protocol_mode': args.protocol_mode,
        'anchor_source': anchor_bank['source'],
        'projection_method_requested': args.method,
        'projection_method_used': method_used,
        'projection_meta': projection_meta,
        'focus_projection_meta': focus_projection_meta,
        'overview_png': overview_png,
        'focus_png': focus_png,
        'close_sample_counts': close_pack['sample_counts'],
        'open_sample_counts': open_pack['sample_counts'],
        'open_virtual_count': int(open_virtual_mask.sum()),
        'open_virtual_hist': {str(k): int(v) for k, v in sorted(virtual_hist.items())},
        'virtual_anchor_cosine_matrix': anchor_cos,
        'known_names': known_names,
        'unknown_names': unknown_names,
        'virtual_anchor_names': anchor_bank['virtual_names'],
    }
    summary_path = os.path.join(args.output_dir, f'{output_stem}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(parse_args())
