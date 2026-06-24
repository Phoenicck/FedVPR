#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone evaluator for FedVPR Stage-1 reserve checkpoints."""
import argparse
import json
import os
import os.path as osp
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib import simple_metrics as metrics
from lib.common import setup
from lib.stage1_reserve import (
    compute_known_logits,
    compute_virtual_logits,
    margin_summary,
)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def _load_yaml_config(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _parse_run_meta(path):
    meta = {}
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def _default_args_dict():
    return {
        "mode": "Pretrain",
        "lr": 5e-4,
        "model_type": "softmax",
        "backbone": "Resnet18",
        "dataset": "RetinalOCT",
        "known_class": 5,
        "unknown_class": 3,
        "device_id": 0,
        "virtue_num": 3,
        "seed": 0,
        "data_root": "./dataset/",
        "rotation": 45,
        "resize": 144,
        "cropsize": 128,
        "batchsize": 16,
        "epoches": 200,
        "client_num": 8,
        "worker_steps": 1,
        "dirichlet": 0.5,
        "protocol_mode": "random",
        "stage1_reserve_enable": True,
        "stage1_warmup_rounds": 25,
        "lambda_reserve": 0.1,
        "cosine_scale": 16.0,
        "anchor_freeze": True,
        "anchor_similarity_threshold": 0.95,
        "anchor_density_angles": [15.0, 20.0, 25.0],
        "anchor_init_file": "stage1_anchor_init.json",
        "max_train_batches": 0,
        "max_eval_batches": 0,
        "save_path": "",
    }


def _build_args(config_path=None, run_meta_path=None, overrides=None):
    args_dict = _default_args_dict()
    if config_path:
        args_dict.update(_load_yaml_config(config_path))
    if run_meta_path and osp.exists(run_meta_path):
        meta = _parse_run_meta(run_meta_path)
        if "protocol_mode" in meta:
            args_dict["protocol_mode"] = meta["protocol_mode"]
        if "lambda_reserve" in meta:
            args_dict["lambda_reserve"] = float(meta["lambda_reserve"])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                args_dict[key] = value
    if isinstance(args_dict.get("anchor_density_angles"), str):
        args_dict["anchor_density_angles"] = [
            float(item.strip())
            for item in args_dict["anchor_density_angles"].split(",")
            if item.strip()
        ]
    args_dict["anchor_freeze"] = _str2bool(args_dict.get("anchor_freeze", True))
    return SimpleNamespace(**args_dict)


def _get_dataloaders(args):
    param = {
        "dataset": args.dataset,
        "Known_class": args.known_class,
        "unKnown_class": args.unknown_class,
        "Rotation": args.rotation,
        "Resize": args.resize,
        "CropSize": args.cropsize,
        "Batchsize": args.batchsize,
        "dirichlet": args.dirichlet,
        "protocol_mode": args.protocol_mode,
    }
    if args.dataset == "RetinalOCT":
        from data.fed_retinal_oct_relabel import get_dataloaders
    elif args.dataset == "ISIC":
        from data.fed_isic_relabel import get_dataloaders
    elif args.dataset == "Bloodmnist":
        from data.fed_MedMINIST_relabel import get_dataloaders
    elif args.dataset == "OrganMNIST3D":
        from data.fed_MedMINIST3D_relabel import get_dataloaders
    elif args.dataset == "Hyperkvasir":
        from data.fed_hyper_kvasir_relabel import get_dataloaders
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return get_dataloaders(args.client_num, args.data_root, args.seed, param)


def _load_stage1_state(checkpoint, anchor_init_path=None):
    stage1_state = checkpoint.get("stage1_state")
    if stage1_state is not None:
        return stage1_state
    if not anchor_init_path or not osp.exists(anchor_init_path):
        raise KeyError("Stage-1 checkpoint does not contain stage1_state and no anchor init file was found.")
    with open(anchor_init_path, "r") as f:
        payload = json.load(f)
    if "virtual_anchors" in payload:
        payload["virtual_anchors"] = torch.tensor(payload["virtual_anchors"], dtype=torch.float32)
    else:
        raise KeyError(f"Anchor init file is missing virtual_anchors: {anchor_init_path}")
    return payload


def _normalized_entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float((-(p * np.log(p)).sum()) / np.log(len(counts)))


def evaluate_stage1_osr(args, model, device, closerloader, openloader, stage1_state):
    model.eval()
    anchors = stage1_state["virtual_anchors"].to(device).float()
    temperature = 1.0
    criterion = torch.nn.CrossEntropyLoss()

    close_ce_loss = 0.0
    close_pred_known = []
    close_target_known = []
    close_prob_known_list = []
    close_raw_preds = []
    close_targets = []
    close_virtual_flags = []
    close_true_virtual_margin = []
    close_virtual_prob_sum = []
    close_virtual_prob_mean_per_anchor = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(closerloader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            outs = model(inputs)
            features = outs["feature"]
            known_logits = compute_known_logits(features, model.main_cls.weight[:args.known_class], args.cosine_scale)
            virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)
            full_logits = torch.cat([known_logits, virtual_logits], dim=1)
            prob = F.softmax(full_logits / temperature, dim=-1).cpu().numpy()

            close_ce_loss += criterion(known_logits, targets).item()
            close_pred_known.extend(known_logits.argmax(1).detach().cpu().numpy().tolist())
            close_target_known.extend(targets.cpu().numpy().tolist())

            close_prob_known_list.append(prob[:, :args.known_class].max(1))
            close_virtual_prob_sum.append(prob[:, args.known_class:].sum(1))
            close_virtual_prob_mean_per_anchor.append(prob[:, args.known_class:].mean(0))

            full_pred = full_logits.argmax(1).detach().cpu().numpy()
            close_raw_preds.append(full_pred)
            close_targets.append(targets.cpu().numpy())
            close_virtual_flags.append(full_pred >= args.known_class)

            true_known = known_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            close_true_virtual_margin.append(
                (true_known - virtual_logits.max(1)[0]).detach().cpu().numpy()
            )

            if args.max_eval_batches > 0 and batch_idx + 1 >= args.max_eval_batches:
                break

    open_prob_known_list = []
    open_raw_preds = []
    open_targets = []
    open_virtual_prob_sum = []
    open_virtual_prob_mean_per_anchor = []
    open_known_virtual_margin = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(openloader):
            inputs = inputs.to(device)
            outs = model(inputs)
            features = outs["feature"]
            known_logits = compute_known_logits(features, model.main_cls.weight[:args.known_class], args.cosine_scale)
            virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)
            full_logits = torch.cat([known_logits, virtual_logits], dim=1)
            prob = F.softmax(full_logits / temperature, dim=-1).cpu().numpy()

            open_prob_known_list.append(prob[:, :args.known_class].max(1))
            open_virtual_prob_sum.append(prob[:, args.known_class:].sum(1))
            open_virtual_prob_mean_per_anchor.append(prob[:, args.known_class:].mean(0))

            pred = full_logits.argmax(1).detach().cpu().numpy()
            open_raw_preds.append(pred)
            open_targets.append(np.ones(pred.shape[0], dtype=np.int64) * args.known_class)
            open_known_virtual_margin.append(
                (known_logits.max(1)[0] - virtual_logits.max(1)[0]).detach().cpu().numpy()
            )

            if args.max_eval_batches > 0 and batch_idx + 1 >= args.max_eval_batches:
                break

    close_prob_known = np.concatenate(close_prob_known_list) if close_prob_known_list else np.zeros(0, dtype=np.float32)
    open_prob_known = np.concatenate(open_prob_known_list) if open_prob_known_list else np.zeros(0, dtype=np.float32)
    prob_known = np.concatenate([close_prob_known, open_prob_known]) if (close_prob_known.size or open_prob_known.size) else np.zeros(0, dtype=np.float32)

    close_virtual_prob_sum = np.concatenate(close_virtual_prob_sum) if close_virtual_prob_sum else np.zeros(0, dtype=np.float32)
    open_virtual_prob_sum = np.concatenate(open_virtual_prob_sum) if open_virtual_prob_sum else np.zeros(0, dtype=np.float32)
    close_raw_preds = np.concatenate(close_raw_preds) if close_raw_preds else np.zeros(0, dtype=np.int64)
    open_raw_preds = np.concatenate(open_raw_preds) if open_raw_preds else np.zeros(0, dtype=np.int64)
    close_targets = np.concatenate(close_targets) if close_targets else np.zeros(0, dtype=np.int64)
    open_targets = np.concatenate(open_targets) if open_targets else np.zeros(0, dtype=np.int64)
    close_virtual_flags = np.concatenate(close_virtual_flags) if close_virtual_flags else np.zeros(0, dtype=bool)
    close_true_virtual_margin = np.concatenate(close_true_virtual_margin) if close_true_virtual_margin else np.zeros(0, dtype=np.float32)
    open_known_virtual_margin = np.concatenate(open_known_virtual_margin) if open_known_virtual_margin else np.zeros(0, dtype=np.float32)

    targets_all = np.concatenate([close_targets, open_targets]) if (close_targets.size or open_targets.size) else np.zeros(0, dtype=np.int64)
    preds_all = np.concatenate([close_raw_preds, open_raw_preds]) if (close_raw_preds.size or open_raw_preds.size) else np.zeros(0, dtype=np.int64)

    binary_labels = (targets_all == args.known_class).astype(np.int64)
    novelty_score = 1.0 - prob_known
    try:
        auroc = 100.0 * metrics.roc_auc_score(binary_labels, novelty_score)
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(binary_labels, novelty_score)
        aupr = 100.0 * metrics.auc(recall_curve, precision_curve)
    except Exception:
        auroc = 0.0
        aupr = 0.0

    unknown_mask = targets_all == args.known_class
    unk = 100.0 * np.mean(preds_all[unknown_mask] >= args.known_class) if np.any(unknown_mask) else 0.0

    known_mask = targets_all < args.known_class
    if np.any(known_mask):
        os_star = 100.0 * metrics.recall_score(
            y_true=targets_all[known_mask],
            y_pred=preds_all[known_mask],
            labels=list(range(args.known_class)),
            average="macro",
            zero_division=0,
        )
    else:
        os_star = 0.0
    hos = 2.0 * os_star * unk / (os_star + unk) if (os_star + unk) > 0 else 0.0

    try:
        is_known = targets_all < args.known_class
        is_unknown = ~is_known
        correct = np.zeros_like(targets_all, dtype=bool)
        correct[is_known] = preds_all[is_known] == targets_all[is_known]
        sorted_idx = np.argsort(-prob_known)
        correct_sorted = correct[sorted_idx]
        unknown_sorted = is_unknown[sorted_idx]
        tp_cum = np.cumsum(correct_sorted)
        fp_cum = np.cumsum(unknown_sorted)
        n_known_total = int(is_known.sum())
        n_unknown_total = int(is_unknown.sum())
        if n_known_total == 0 or n_unknown_total == 0:
            oscr = 0.0
        else:
            ccr = tp_cum / n_known_total
            fpr = fp_cum / n_unknown_total
            fpr = np.concatenate([[0.0], fpr, [1.0]])
            ccr = np.concatenate([[0.0], ccr, [ccr[-1]]])
            oscr = 100.0 * metrics.auc(fpr, ccr)
    except Exception:
        oscr = 0.0

    preds_mapped = preds_all.copy()
    preds_mapped[preds_mapped >= args.known_class] = args.known_class
    osr_acc = 100.0 * metrics.accuracy_score(targets_all, preds_mapped) if targets_all.size > 0 else 0.0
    osr_f1 = 100.0 * metrics.f1_score(targets_all, preds_mapped, average="macro", zero_division=0) if targets_all.size > 0 else 0.0
    osr_recall = 100.0 * metrics.recall_score(targets_all, preds_mapped, average="macro", zero_division=0) if targets_all.size > 0 else 0.0
    osr_precision = 100.0 * metrics.precision_score(targets_all, preds_mapped, average="macro", zero_division=0) if targets_all.size > 0 else 0.0

    close_hist = np.bincount(
        close_raw_preds[close_raw_preds >= args.known_class] - args.known_class,
        minlength=args.virtue_num,
    ) if close_raw_preds.size > 0 else np.zeros(args.virtue_num, dtype=np.int64)
    open_hist = np.bincount(
        open_raw_preds[open_raw_preds >= args.known_class] - args.known_class,
        minlength=args.virtue_num,
    ) if open_raw_preds.size > 0 else np.zeros(args.virtue_num, dtype=np.int64)

    close_anchor_prob_mean = (
        np.mean(np.stack(close_virtual_prob_mean_per_anchor, axis=0), axis=0).tolist()
        if close_virtual_prob_mean_per_anchor else []
    )
    open_anchor_prob_mean = (
        np.mean(np.stack(open_virtual_prob_mean_per_anchor, axis=0), axis=0).tolist()
        if open_virtual_prob_mean_per_anchor else []
    )

    close_test_result = {
        "loss": close_ce_loss / max(len(close_target_known), 1),
        "acc": 100.0 * metrics.accuracy_score(close_target_known, close_pred_known) if close_target_known else 0.0,
        "f1": 100.0 * metrics.f1_score(close_target_known, close_pred_known, average="macro", zero_division=0) if close_target_known else 0.0,
        "recall": 100.0 * metrics.recall_score(close_target_known, close_pred_known, average="macro", zero_division=0) if close_target_known else 0.0,
        "precision": 100.0 * metrics.precision_score(close_target_known, close_pred_known, average="macro", zero_division=0) if close_target_known else 0.0,
    }

    osr_result = {
        "acc": osr_acc,
        "f1": osr_f1,
        "recall": osr_recall,
        "precision": osr_precision,
        "unk": unk,
        "os_star": os_star,
        "hos": hos,
        "auroc": auroc,
        "aupr": aupr,
        "oscr": oscr,
        "close_virtual_pred_rate": float(close_virtual_flags.mean() * 100.0) if close_virtual_flags.size > 0 else 0.0,
        "close_virtual_prob_mean": float(close_virtual_prob_sum.mean()) if close_virtual_prob_sum.size > 0 else 0.0,
        "close_known_virtual_margin_mean": float(close_true_virtual_margin.mean()) if close_true_virtual_margin.size > 0 else 0.0,
        "close_virtual_hist": close_hist.tolist(),
        "close_virtual_entropy": _normalized_entropy(close_hist),
        "close_virtual_prob_mean_per_anchor": close_anchor_prob_mean,
        "close_true_virtual_logit_margin": margin_summary(close_true_virtual_margin),
        "open_virtual_pred_rate": float((open_raw_preds >= args.known_class).mean() * 100.0) if open_raw_preds.size > 0 else 0.0,
        "open_virtual_prob_mean": float(open_virtual_prob_sum.mean()) if open_virtual_prob_sum.size > 0 else 0.0,
        "open_virtual_hist": open_hist.tolist(),
        "open_virtual_entropy": _normalized_entropy(open_hist),
        "open_virtual_prob_mean_per_anchor": open_anchor_prob_mean,
        "open_known_virtual_logit_margin": margin_summary(open_known_virtual_margin),
        "num_close_samples": int(close_targets.size),
        "num_open_samples": int(open_targets.size),
    }
    return osr_result, close_test_result


def _resolve_single_run_paths(args_cli):
    if args_cli.run_dir:
        run_dir = osp.abspath(args_cli.run_dir)
        checkpoint_path = args_cli.checkpoint or osp.join(
            run_dir,
            "result_dir_snapshot",
            f"best_ckpt_Pretrain_known_class_{args_cli.known_class or 5}_unknown_class_{args_cli.unknown_class or 3}_seed_{args_cli.seed or 0}.pth",
        )
        result_dir = osp.join(run_dir, "result_dir_snapshot")
        run_meta_path = osp.join(run_dir, "run_meta.txt")
        config_path = args_cli.config
        if not config_path and osp.exists(run_meta_path):
            meta = _parse_run_meta(run_meta_path)
            config_path = meta.get("source_config", "")
        if config_path and not osp.isabs(config_path):
            config_path = osp.abspath(osp.join("/workspace/Phoenic/claude0527/FedVPR", config_path.replace("./", "")))
        return run_dir, osp.abspath(checkpoint_path), osp.abspath(result_dir), run_meta_path, config_path

    checkpoint_path = osp.abspath(args_cli.checkpoint)
    result_dir = osp.dirname(checkpoint_path)
    run_dir = None
    run_meta_path = args_cli.run_meta
    config_path = args_cli.config
    return run_dir, checkpoint_path, result_dir, run_meta_path, config_path


def _evaluate_one(args_cli):
    run_dir, checkpoint_path, result_dir, run_meta_path, config_path = _resolve_single_run_paths(args_cli)
    overrides = {
        "dataset": args_cli.dataset,
        "known_class": args_cli.known_class,
        "unknown_class": args_cli.unknown_class,
        "seed": args_cli.seed,
        "client_num": args_cli.client_num,
        "data_root": args_cli.data_root,
        "batchsize": args_cli.batchsize,
        "device_id": args_cli.device_id,
        "protocol_mode": args_cli.protocol_mode,
        "cosine_scale": args_cli.cosine_scale,
        "max_eval_batches": args_cli.max_eval_batches,
    }
    args = _build_args(config_path=config_path, run_meta_path=run_meta_path, overrides=overrides)
    args.save_path = result_dir

    trainloaders, _valloader, closeloader, openloader, _train_val_loaders = _get_dataloaders(args)
    server_model, _models, device, _client_weights = setup(args, trainloaders)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    server_model.load_state_dict(checkpoint["net"], strict=True)
    stage1_state = _load_stage1_state(
        checkpoint,
        anchor_init_path=osp.join(result_dir, args.anchor_init_file),
    )

    osr_result, close_test_result = evaluate_stage1_osr(
        args, server_model, device, closeloader, openloader, stage1_state
    )

    output = {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "result_dir": result_dir,
        "config": config_path,
        "protocol_mode": args.protocol_mode,
        "dataset": args.dataset,
        "known_class": args.known_class,
        "unknown_class": args.unknown_class,
        "seed": args.seed,
        "lambda_reserve": float(args.lambda_reserve),
        "stage1_warmup_rounds": int(args.stage1_warmup_rounds),
        "virtue_num": int(args.virtue_num),
        "close_test": close_test_result,
        "osr": osr_result,
        "selected_anchor_pairs": stage1_state.get("selected_anchor_pairs", []),
        "anchor_pairwise_cosine": stage1_state.get("anchor_pairwise_cosine", []),
    }
    return output


def _find_run_dirs(grid_root):
    run_dirs = []
    for name in sorted(os.listdir(grid_root)):
        path = osp.join(grid_root, name)
        if not osp.isdir(path):
            continue
        snapshot = osp.join(path, "result_dir_snapshot")
        if osp.isdir(snapshot):
            run_dirs.append(path)
    return run_dirs


def _print_summary(result):
    osr = result["osr"]
    close_test = result["close_test"]
    print(
        f"[{osp.basename(result['run_dir']) if result['run_dir'] else osp.basename(result['result_dir'])}] "
        f"protocol={result['protocol_mode']} lambda={result['lambda_reserve']:.4f} "
        f"CloseACC={close_test['acc']:.3f} CloseF1={close_test['f1']:.3f} "
        f"AUROC={osr['auroc']:.3f} AUPR={osr['aupr']:.3f} OSCR={osr['oscr']:.3f} "
        f"UNK={osr['unk']:.3f} OS*={osr['os_star']:.3f} HOS={osr['hos']:.3f} "
        f"CloseK->V={osr['close_virtual_pred_rate']:.3f}% Open->V={osr['open_virtual_pred_rate']:.3f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate FedVPR Stage-1 reserve checkpoints with AUROC/AUPR/OSCR.")
    parser.add_argument("--checkpoint", type=str, default="", help="path to a Stage-1 best checkpoint")
    parser.add_argument("--run-dir", type=str, default="", help="archived run directory containing run_meta.txt and result_dir_snapshot/")
    parser.add_argument("--grid-root", type=str, default="", help="evaluate every archived run under this grid root")
    parser.add_argument("--config", type=str, default="", help="optional config YAML")
    parser.add_argument("--run-meta", type=str, default="", help="optional run_meta.txt")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--known_class", type=int, default=None)
    parser.add_argument("--unknown_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--client_num", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=None)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--protocol_mode", type=str, default=None)
    parser.add_argument("--cosine_scale", type=float, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="", help="optional json output path")
    args_cli = parser.parse_args()

    if args_cli.grid_root:
        grid_root = osp.abspath(args_cli.grid_root)
        results = []
        for run_dir in _find_run_dirs(grid_root):
            child_args = argparse.Namespace(**vars(args_cli))
            child_args.run_dir = run_dir
            child_args.grid_root = ""
            child_args.checkpoint = ""
            result = _evaluate_one(child_args)
            results.append(result)
            _print_summary(result)
        if args_cli.output_json:
            with open(args_cli.output_json, "w") as f:
                json.dump(results, f, indent=2)
        return

    if not args_cli.run_dir and not args_cli.checkpoint:
        raise SystemExit("Please provide either --run-dir, --checkpoint, or --grid-root.")

    result = _evaluate_one(args_cli)
    _print_summary(result)
    if args_cli.output_json:
        with open(args_cli.output_json, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
