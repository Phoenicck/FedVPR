#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

ROOT = '/workspace/Phoenic/claude0527/FedVPR'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.fed_retinal_oct_relabel import get_dataloaders, OCT_PROTOCOLS  # noqa: E402
from models.ResNet_FedOSR_Pretrain import resnet18  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FedVPR embeddings and virtual anchors with t-SNE')
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
    parser.add_argument('--max_per_known_class', type=int, default=120)
    parser.add_argument('--max_per_unknown_class', type=int, default=120)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--tsne_seed', type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(args, device):
    model = resnet18(pretrained=False, num_classes=args.known_class, num_virtual=args.virtue_num)
    state = torch.load(args.checkpoint, map_location='cpu')['net']
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


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


@torch.no_grad()
def collect_embeddings(model, loader, device, max_per_class, split_name):
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
        pred = int(outputs.argmax(1).item())
        is_virtual = pred >= args_global.known_class
        pred_idx = pred - args_global.known_class if is_virtual else -1
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



def build_label_maps(protocol_mode):
    spec = OCT_PROTOCOLS[protocol_mode]
    known_names = spec['known']
    unknown_names = spec['unknown']
    label_to_name = {i: name for i, name in enumerate(known_names)}
    label_to_name.update({i + len(known_names): name for i, name in enumerate(unknown_names)})
    return known_names, unknown_names, label_to_name



def fit_tsne(embedding_matrix, perplexity, seed):
    n_samples = len(embedding_matrix)
    if n_samples <= 2:
        coords = np.zeros((n_samples, 2), dtype=np.float32)
        if n_samples == 2:
            coords[1, 0] = 1.0
        return coords
    perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init='pca',
        learning_rate='auto',
        max_iter=2000,
    )
    return tsne.fit_transform(embedding_matrix)



def save_overview_plot(coords, meta, output_path, known_names, unknown_names):
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

    plt.title('FedVPR RetinalOCT t-SNE: close/open samples with known and virtual anchors')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()



def save_virtual_focus_plot(coords, meta, output_path):
    plt.figure(figsize=(12, 10))
    colors = ['#d73027', '#1a9850', '#4575b4']
    for vid in range(3):
        mask = (meta['kind'] == 'open_virtual') & (meta['virtual_idx'] == vid)
        if mask.any():
            plt.scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.78, color=colors[vid % len(colors)], label=f'Open -> V{vid}')

    anchor_mask = meta['kind'] == 'virtual_anchor'
    plt.scatter(coords[anchor_mask, 0], coords[anchor_mask, 1], s=360, c='gold', marker='*', edgecolors='black', linewidths=1.0, label='Virtual anchors')
    for x, y, name in zip(coords[anchor_mask, 0], coords[anchor_mask, 1], meta['name'][anchor_mask]):
        plt.text(x, y, name, fontsize=11, weight='bold', ha='left', va='bottom')

    plt.title('FedVPR RetinalOCT t-SNE: open samples routed to virtual anchors')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()



def main(args):
    global args_global
    args_global = args
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() else torch.device('cpu')
    model = make_model(args, device)

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
    known_names, unknown_names, label_to_name = build_label_maps(args.protocol_mode)

    close_pack = collect_embeddings(model, close_loader, device, args.max_per_known_class, 'close')
    open_pack = collect_embeddings(model, open_loader, device, args.max_per_unknown_class, 'open')

    known_anchor = model.main_cls.weight[:args.known_class].detach().cpu().numpy().astype(np.float32)
    virtual_anchor = model.main_cls.weight[args.known_class:].detach().cpu().numpy().astype(np.float32)

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
        meta_rows.append({'kind': 'virtual_anchor', 'label': args.known_class + idx, 'name': f'V{idx}', 'virtual_idx': idx})

    matrix = np.asarray(all_features, dtype=np.float32)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    coords = fit_tsne(matrix, args.perplexity, args.tsne_seed)

    meta = {
        'kind': np.asarray([row['kind'] for row in meta_rows]),
        'label': np.asarray([row['label'] for row in meta_rows]),
        'name': np.asarray([row['name'] for row in meta_rows]),
        'virtual_idx': np.asarray([row['virtual_idx'] for row in meta_rows]),
    }

    overview_png = os.path.join(args.output_dir, f'retinaloct_{args.protocol_mode}_best_tsne_overview.png')
    save_overview_plot(coords, meta, overview_png, known_names, unknown_names)

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
        focus_meta_rows.append({'kind': 'virtual_anchor', 'label': args.known_class + idx, 'name': f'V{idx}', 'virtual_idx': idx})

    focus_png = None
    if focus_matrix:
        focus_matrix = np.asarray(focus_matrix, dtype=np.float32)
        focus_matrix = focus_matrix / np.clip(np.linalg.norm(focus_matrix, axis=1, keepdims=True), 1e-12, None)
        focus_coords = fit_tsne(focus_matrix, min(args.perplexity, 20.0), args.tsne_seed)
        focus_meta = {
            'kind': np.asarray([row['kind'] for row in focus_meta_rows]),
            'virtual_idx': np.asarray([row['virtual_idx'] for row in focus_meta_rows]),
            'name': np.asarray([row['name'] for row in focus_meta_rows]),
        }
        focus_png = os.path.join(args.output_dir, f'retinaloct_{args.protocol_mode}_best_tsne_open_virtual_focus.png')
        save_virtual_focus_plot(focus_coords, focus_meta, focus_png)

    virtual_hist = Counter(int(x) for x in open_virtual_idx if x >= 0)
    anchor_cos = F.normalize(torch.from_numpy(virtual_anchor), dim=1)
    anchor_cos = torch.mm(anchor_cos, anchor_cos.t()).numpy().tolist()
    summary = {
        'checkpoint': args.checkpoint,
        'protocol_mode': args.protocol_mode,
        'overview_png': overview_png,
        'focus_png': focus_png,
        'close_sample_counts': close_pack['sample_counts'],
        'open_sample_counts': open_pack['sample_counts'],
        'open_virtual_count': int(open_virtual_mask.sum()),
        'open_virtual_hist': {str(k): int(v) for k, v in sorted(virtual_hist.items())},
        'virtual_anchor_cosine_matrix': anchor_cos,
        'known_names': known_names,
        'unknown_names': unknown_names,
    }
    summary_path = os.path.join(args.output_dir, f'retinaloct_{args.protocol_mode}_best_tsne_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(parse_args())
