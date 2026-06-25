#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sanity-check pseudo-unknown generation from Stage-1 reserve checkpoints."""
import argparse
import csv
import itertools
import json
import math
import os
import os.path as osp
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = osp.abspath(osp.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.common import setup
from lib.stage1_reserve import compute_known_logits, compute_virtual_logits


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
        "batchsize": 8,
        "epoches": 50,
        "client_num": 8,
        "worker_steps": 1,
        "dirichlet": 0.5,
        "protocol_mode": "random",
        "stage1_reserve_enable": True,
        "stage1_warmup_rounds": 15,
        "lambda_reserve": 0.05,
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


def _resolve_paths(args_cli):
    checkpoint_path = osp.abspath(args_cli.checkpoint)
    result_dir = osp.dirname(checkpoint_path)
    run_dir = args_cli.run_dir
    run_meta_path = args_cli.run_meta
    if run_dir:
        run_dir = osp.abspath(run_dir)
        result_dir = osp.join(run_dir, "result_dir_snapshot")
        if not run_meta_path:
            run_meta_path = osp.join(run_dir, "run_meta.txt")
    elif not run_meta_path:
        parent = osp.dirname(result_dir)
        maybe_meta = osp.join(parent, "run_meta.txt")
        if osp.exists(maybe_meta):
            run_meta_path = maybe_meta
            run_dir = parent
    config_path = args_cli.config
    if not config_path and run_meta_path and osp.exists(run_meta_path):
        meta = _parse_run_meta(run_meta_path)
        config_path = meta.get("source_config", "")
    if config_path and not osp.isabs(config_path):
        config_path = osp.abspath(osp.join(REPO_ROOT, config_path.replace("./", "")))
    return checkpoint_path, result_dir, run_dir, run_meta_path, config_path


def _load_stage1_state(checkpoint, anchor_init_path=None):
    stage1_state = checkpoint.get("stage1_state")
    if stage1_state is None:
        raise KeyError("Checkpoint is missing stage1_state.")
    if "virtual_anchors" not in stage1_state:
        raise KeyError("Checkpoint stage1_state is missing virtual_anchors.")
    if anchor_init_path and osp.exists(anchor_init_path):
        with open(anchor_init_path, "r") as f:
            anchor_meta = json.load(f)
        if anchor_meta.get("selected_anchor_pairs", []) != stage1_state.get("selected_anchor_pairs", []):
            raise ValueError("Selected anchor pairs mismatch between checkpoint and stage1_anchor_init.json")
        if anchor_meta.get("known_class_names", []) != stage1_state.get("known_class_names", []):
            raise ValueError("Known class names mismatch between checkpoint and stage1_anchor_init.json")
        left = np.asarray(anchor_meta.get("anchor_pairwise_cosine", []), dtype=np.float32)
        right = np.asarray(stage1_state.get("anchor_pairwise_cosine", []), dtype=np.float32)
        if left.shape != right.shape or not np.allclose(left, right, atol=1e-6):
            raise ValueError("Anchor cosine metadata mismatch between checkpoint and stage1_anchor_init.json")
    return stage1_state


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _pair_to_key(pair):
    return tuple(sorted(int(x) for x in pair))


def _summarize(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _parse_float_list(value):
    if not value:
        return []
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_int_list(value):
    if not value:
        return []
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _format_grid_token(value):
    if isinstance(value, int):
        return str(value)
    return str(value).replace(".", "p").replace("-", "m")


def collect_known_train_samples(args, model, device, train_val_loaders, stage1_state, analysis_batch_size, max_trainval_batches=0, analysis_num_workers=4):
    classifier_weight = model.main_cls.weight[:args.known_class].detach().to(device)
    anchors = stage1_state["virtual_anchors"].detach().to(device).float()
    model.eval()
    samples = []
    global_index = 0
    with torch.no_grad():
        for client_id, loader in enumerate(train_val_loaders):
            batch_loader = torch.utils.data.DataLoader(
                loader.dataset,
                batch_size=analysis_batch_size,
                shuffle=False,
                num_workers=max(0, int(analysis_num_workers)),
            )
            for batch_idx, (inputs, targets, img_paths) in enumerate(batch_loader):
                inputs = inputs.to(device)
                targets = targets.long().to(device)
                features = F.normalize(model(inputs)["feature"], dim=1)
                known_logits = compute_known_logits(features, classifier_weight, args.cosine_scale)
                virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)

                other_known_logits = known_logits.clone()
                other_known_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
                j_star = other_known_logits.argmax(1)
                true_known_logits = known_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                j_star_logits = other_known_logits.gather(1, j_star.unsqueeze(1)).squeeze(1)
                known_boundary_margin = true_known_logits - j_star_logits
                known_to_virtual_margin = true_known_logits - virtual_logits.max(1)[0]

                for row in range(inputs.shape[0]):
                    samples.append(
                        {
                            "sample_index": global_index,
                            "client_id": client_id,
                            "img_path": img_paths[row],
                            "label": int(targets[row].item()),
                            "j_star": int(j_star[row].item()),
                            "known_boundary_margin": float(known_boundary_margin[row].item()),
                            "known_to_virtual_margin": float(known_to_virtual_margin[row].item()),
                            "feature": features[row].detach().cpu(),
                            "true_known_logit": float(true_known_logits[row].item()),
                            "j_star_logit": float(j_star_logits[row].item()),
                            "max_virtual_logit": float(virtual_logits[row].max().item()),
                        }
                    )
                    global_index += 1
                if max_trainval_batches > 0 and batch_idx + 1 >= max_trainval_batches:
                    break
    return samples


def select_boundary_seeds(samples, selected_anchor_pairs, known_class, boundary_fraction):
    pair_to_anchor = {}
    for anchor_id, pair_info in enumerate(selected_anchor_pairs):
        pair_to_anchor[_pair_to_key(pair_info["pair"])] = {
            "anchor_id": anchor_id,
            "pair": pair_info["pair"],
            "pair_names": pair_info.get("pair_names", []),
        }

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)

    candidate_rows = []
    generation_seeds = []
    for label in range(known_class):
        group = sorted(grouped.get(label, []), key=lambda item: item["known_boundary_margin"])
        if not group:
            continue
        top_k = max(1, int(math.ceil(len(group) * boundary_fraction)))
        for rank, sample in enumerate(group[:top_k]):
            anchor_info = pair_to_anchor.get(_pair_to_key((sample["label"], sample["j_star"])))
            row = {
                "sample_index": sample["sample_index"],
                "client_id": sample["client_id"],
                "img_path": sample["img_path"],
                "label": sample["label"],
                "j_star": sample["j_star"],
                "class_rank": rank,
                "known_boundary_margin": sample["known_boundary_margin"],
                "known_to_virtual_margin": sample["known_to_virtual_margin"],
                "has_target_anchor": int(anchor_info is not None),
                "target_anchor_id": int(anchor_info["anchor_id"]) if anchor_info is not None else -1,
                "target_anchor_pair": "|".join(str(x) for x in anchor_info["pair"]) if anchor_info is not None else "",
                "target_anchor_pair_names": "|".join(anchor_info.get("pair_names", [])) if anchor_info is not None else "",
                "skip_reason": "" if anchor_info is not None else "pair_not_selected",
            }
            candidate_rows.append(row)
            if anchor_info is None:
                continue
            generation_seeds.append(
                {
                    **sample,
                    "target_anchor_id": int(anchor_info["anchor_id"]),
                    "target_anchor_pair": list(anchor_info["pair"]),
                    "target_anchor_pair_names": list(anchor_info.get("pair_names", [])),
                }
            )
    generation_seeds.sort(key=lambda item: (item["known_boundary_margin"], item["client_id"], item["sample_index"]))
    return candidate_rows, generation_seeds


def optimize_pseudo_feature(seed, classifier_weight, anchors, steps, step_size, beta, max_feature_distance, cosine_scale):
    label = int(seed["label"])
    anchor_id = int(seed["target_anchor_id"])
    z0 = seed["feature"].view(1, -1).clone().detach().to(classifier_weight.device)
    z = z0.clone().detach().requires_grad_(True)

    with torch.no_grad():
        known_before = compute_known_logits(z0, classifier_weight, cosine_scale)
        virtual_before = compute_virtual_logits(z0, anchors, cosine_scale)
        true_known_before = float(known_before[0, label].item())
        target_virtual_before = float(virtual_before[0, anchor_id].item())
        max_known_before = float(known_before.max(1)[0].item())
        max_virtual_before = float(virtual_before.max(1)[0].item())

    exceeded_distance = False
    for _step in range(int(steps)):
        known_logits = compute_known_logits(z, classifier_weight, cosine_scale)
        virtual_logits = compute_virtual_logits(z, anchors, cosine_scale)
        objective = virtual_logits[0, anchor_id] - known_logits[0, label] - float(beta) * ((z - z0) ** 2).sum()
        grad = torch.autograd.grad(objective, z, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            z.add_(float(step_size) * grad)
            z.copy_(F.normalize(z, dim=1))
            feature_distance = float(torch.norm(z - z0, p=2, dim=1).item())
            if feature_distance > float(max_feature_distance):
                exceeded_distance = True
                break

    with torch.no_grad():
        pseudo = z.detach()
        known_after = compute_known_logits(pseudo, classifier_weight, cosine_scale)
        virtual_after = compute_virtual_logits(pseudo, anchors, cosine_scale)
        true_known_after = float(known_after[0, label].item())
        target_virtual_after = float(virtual_after[0, anchor_id].item())
        max_known_after = float(known_after.max(1)[0].item())
        max_virtual_after = float(virtual_after.max(1)[0].item())
        feature_distance = float(torch.norm(pseudo - z0, p=2, dim=1).item())

    success = target_virtual_after > target_virtual_before and true_known_after < true_known_before and feature_distance <= float(max_feature_distance)
    return {
        "pseudo_feature": pseudo.detach().cpu().squeeze(0),
        "seed_feature": z0.detach().cpu().squeeze(0),
        "true_known_logit_before": true_known_before,
        "true_known_logit_after": true_known_after,
        "target_virtual_logit_before": target_virtual_before,
        "target_virtual_logit_after": target_virtual_after,
        "max_known_logit_before": max_known_before,
        "max_known_logit_after": max_known_after,
        "max_virtual_logit_before": max_virtual_before,
        "max_virtual_logit_after": max_virtual_after,
        "feature_distance": feature_distance,
        "known_logit_drop": true_known_before - true_known_after,
        "target_virtual_logit_gain": target_virtual_after - target_virtual_before,
        "generation_success": bool(success),
        "stopped_for_distance": bool(exceeded_distance),
    }


def _write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(processed_rows, selected_anchor_pairs, known_class):
    total = len(processed_rows)
    success_rows = [row for row in processed_rows if row["generation_success"]]
    per_anchor = []
    for anchor_id, pair_info in enumerate(selected_anchor_pairs):
        anchor_rows = [row for row in processed_rows if row["target_anchor_id"] == anchor_id]
        anchor_success = [row for row in anchor_rows if row["generation_success"]]
        per_anchor.append(
            {
                "anchor_id": anchor_id,
                "pair": pair_info["pair"],
                "pair_names": pair_info.get("pair_names", []),
                "pseudo_count": len(anchor_rows),
                "success_count": len(anchor_success),
                "success_rate": float(100.0 * len(anchor_success) / len(anchor_rows)) if anchor_rows else 0.0,
                "mean_feature_distance": float(np.mean([row["feature_distance"] for row in anchor_rows])) if anchor_rows else 0.0,
                "mean_known_logit_drop": float(np.mean([row["known_logit_drop"] for row in anchor_rows])) if anchor_rows else 0.0,
                "mean_target_virtual_logit_gain": float(np.mean([row["target_virtual_logit_gain"] for row in anchor_rows])) if anchor_rows else 0.0,
            }
        )

    per_class = []
    for label in range(known_class):
        class_rows = [row for row in processed_rows if row["label"] == label]
        class_success = [row for row in class_rows if row["generation_success"]]
        per_class.append(
            {
                "label": label,
                "pseudo_count": len(class_rows),
                "success_count": len(class_success),
                "success_rate": float(100.0 * len(class_success) / len(class_rows)) if class_rows else 0.0,
            }
        )

    return {
        "processed_seed_count": total,
        "success_count": len(success_rows),
        "success_rate": float(100.0 * len(success_rows) / total) if total else 0.0,
        "target_anchor_histogram": [sum(1 for row in processed_rows if row["target_anchor_id"] == idx) for idx in range(len(selected_anchor_pairs))],
        "mean_feature_distance": float(np.mean([row["feature_distance"] for row in processed_rows])) if processed_rows else 0.0,
        "mean_known_logit_drop": float(np.mean([row["known_logit_drop"] for row in processed_rows])) if processed_rows else 0.0,
        "mean_target_virtual_logit_gain": float(np.mean([row["target_virtual_logit_gain"] for row in processed_rows])) if processed_rows else 0.0,
        "feature_distance_summary": _summarize([row["feature_distance"] for row in processed_rows]),
        "known_logit_drop_summary": _summarize([row["known_logit_drop"] for row in processed_rows]),
        "target_virtual_logit_gain_summary": _summarize([row["target_virtual_logit_gain"] for row in processed_rows]),
        "per_anchor": per_anchor,
        "per_known_class": per_class,
        "anchors_without_success": [item["anchor_id"] for item in per_anchor if item["success_count"] == 0],
    }


def _generate_records(generation_seeds, classifier_weight, anchor_tensor, steps, step_size, beta, max_feature_distance, cosine_scale):
    processed_rows = []
    for seed in generation_seeds:
        result = optimize_pseudo_feature(seed, classifier_weight, anchor_tensor, steps, step_size, beta, max_feature_distance, cosine_scale)
        processed_rows.append(
            {
                "sample_index": seed["sample_index"],
                "client_id": seed["client_id"],
                "img_path": seed["img_path"],
                "label": seed["label"],
                "j_star": seed["j_star"],
                "known_boundary_margin": seed["known_boundary_margin"],
                "known_to_virtual_margin": seed["known_to_virtual_margin"],
                "target_anchor_id": seed["target_anchor_id"],
                "target_anchor_pair": "|".join(str(x) for x in seed["target_anchor_pair"]),
                "target_anchor_pair_names": "|".join(seed["target_anchor_pair_names"]),
                "true_known_logit_before": result["true_known_logit_before"],
                "true_known_logit_after": result["true_known_logit_after"],
                "target_virtual_logit_before": result["target_virtual_logit_before"],
                "target_virtual_logit_after": result["target_virtual_logit_after"],
                "max_known_logit_before": result["max_known_logit_before"],
                "max_known_logit_after": result["max_known_logit_after"],
                "max_virtual_logit_before": result["max_virtual_logit_before"],
                "max_virtual_logit_after": result["max_virtual_logit_after"],
                "feature_distance": result["feature_distance"],
                "known_logit_drop": result["known_logit_drop"],
                "target_virtual_logit_gain": result["target_virtual_logit_gain"],
                "generation_success": int(result["generation_success"]),
                "stopped_for_distance": int(result["stopped_for_distance"]),
                "_seed_feature": result["seed_feature"],
                "_pseudo_feature": result["pseudo_feature"],
            }
        )
    return processed_rows


def _save_feature_bank(path, processed_rows, summary, save_limit):
    ordered = sorted(processed_rows, key=lambda row: (not row["generation_success"], row["feature_distance"]))
    kept = ordered[: int(save_limit)] if save_limit > 0 else ordered
    payload = {
        "summary": summary,
        "count": len(kept),
        "seed_features": torch.stack([row["_seed_feature"] for row in kept], dim=0) if kept else torch.empty(0, 0),
        "pseudo_features": torch.stack([row["_pseudo_feature"] for row in kept], dim=0) if kept else torch.empty(0, 0),
        "labels": torch.tensor([row["label"] for row in kept], dtype=torch.long),
        "j_star": torch.tensor([row["j_star"] for row in kept], dtype=torch.long),
        "target_anchor_id": torch.tensor([row["target_anchor_id"] for row in kept], dtype=torch.long),
        "generation_success": torch.tensor([row["generation_success"] for row in kept], dtype=torch.bool),
        "img_paths": [row["img_path"] for row in kept],
        "target_anchor_pair_names": [row["target_anchor_pair_names"] for row in kept],
    }
    torch.save(payload, path)


def _save_generation_outputs(output_dir, candidate_rows, processed_rows, summary, args_cli):
    _ensure_dir(output_dir)
    _write_csv(osp.join(output_dir, "boundary_seed_candidates.csv"), candidate_rows)
    _write_csv(osp.join(output_dir, "generation_records.csv"), [{k: v for k, v in row.items() if not k.startswith("_")} for row in processed_rows])
    _write_csv(osp.join(output_dir, "summary_per_anchor.csv"), summary["per_anchor"])
    _write_csv(osp.join(output_dir, "summary_per_class.csv"), summary["per_known_class"])
    with open(osp.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(osp.join(output_dir, "stage2a_settings.json"), "w") as f:
        json.dump(
            {
                "boundary_fraction": args_cli.boundary_fraction,
                "stage2a_steps": summary["stage2a_steps"],
                "stage2a_step_size": summary["stage2a_step_size"],
                "stage2a_beta": summary["stage2a_beta"],
                "stage2a_max_feature_distance": summary["stage2a_max_feature_distance"],
                "analysis_batch_size": args_cli.analysis_batch_size,
                "analysis_num_workers": args_cli.analysis_num_workers,
                "max_trainval_batches": args_cli.max_trainval_batches,
                "max_generation_candidates": args_cli.max_generation_candidates,
            },
            f,
            indent=2,
        )
    _save_feature_bank(osp.join(output_dir, "pseudo_feature_bank.pt"), processed_rows, summary, save_limit=args_cli.save_feature_limit)


def _print_anchor_table(per_anchor):
    print("Anchor Summary")
    print("anchor_id | pair_names | seeds | success | success_rate | mean_dist | mean_drop | mean_gain")
    for item in per_anchor:
        print(
            f"{item['anchor_id']:>8} | "
            f"{'+'.join(item['pair_names']):<17} | "
            f"{item['pseudo_count']:>5} | "
            f"{item['success_count']:>7} | "
            f"{item['success_rate']:>12.3f} | "
            f"{item['mean_feature_distance']:>9.4f} | "
            f"{item['mean_known_logit_drop']:>9.4f} | "
            f"{item['mean_target_virtual_logit_gain']:>9.4f}"
        )


def _attach_common_summary(summary, args_cli, args, checkpoint_path, result_dir, run_dir, run_meta_path, config_path, candidate_rows, generation_seeds, steps, step_size, max_distance, grid_run_name=None):
    summary.update(
        {
            "checkpoint": checkpoint_path,
            "run_dir": run_dir,
            "result_dir": result_dir,
            "config": config_path,
            "run_meta": run_meta_path,
            "protocol_mode": args.protocol_mode,
            "dataset": args.dataset,
            "boundary_fraction": float(args_cli.boundary_fraction),
            "stage2a_steps": int(steps),
            "stage2a_step_size": float(step_size),
            "stage2a_beta": float(args_cli.stage2a_beta),
            "stage2a_max_feature_distance": float(max_distance),
            "candidate_seed_count": len(candidate_rows),
            "matched_seed_count": len(generation_seeds),
            "skipped_seed_count": int(sum(1 for row in candidate_rows if not row["has_target_anchor"])),
        }
    )
    if grid_run_name is not None:
        summary["grid_run_name"] = grid_run_name
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage-2a anchor-guided pseudo-unknown generation sanity check.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage-1 checkpoint path")
    parser.add_argument("--config", type=str, default="", help="optional config YAML")
    parser.add_argument("--run-dir", type=str, default="", help="optional archived run dir containing run_meta.txt")
    parser.add_argument("--run-meta", type=str, default="", help="optional run_meta.txt path")
    parser.add_argument("--output-dir", type=str, default="", help="optional output directory")
    parser.add_argument("--analysis-batch-size", type=int, default=64)
    parser.add_argument("--analysis-num-workers", type=int, default=4)
    parser.add_argument("--boundary-fraction", type=float, default=0.10)
    parser.add_argument("--stage2a_steps", type=int, default=5)
    parser.add_argument("--stage2a_step_size", type=float, default=0.05)
    parser.add_argument("--stage2a_beta", type=float, default=1.0)
    parser.add_argument("--stage2a_max_feature_distance", type=float, default=0.5)
    parser.add_argument("--grid-max-feature-distances", type=str, default="", help="comma-separated max feature distances for sweep")
    parser.add_argument("--grid-step-sizes", type=str, default="", help="comma-separated step sizes for sweep")
    parser.add_argument("--grid-steps", type=str, default="", help="comma-separated optimization step counts for sweep")
    parser.add_argument("--max-trainval-batches", type=int, default=0, help="smoke-test only: limit batches per client train_val loader")
    parser.add_argument("--max-generation-candidates", type=int, default=0, help="smoke-test only: limit processed seeds after anchor matching")
    parser.add_argument("--save-feature-limit", type=int, default=256)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--known_class", type=int, default=None)
    parser.add_argument("--unknown_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--client_num", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--protocol_mode", type=str, default=None)
    args_cli = parser.parse_args()

    checkpoint_path, result_dir, run_dir, run_meta_path, config_path = _resolve_paths(args_cli)
    overrides = {
        "device_id": args_cli.device_id,
        "dataset": args_cli.dataset,
        "known_class": args_cli.known_class,
        "unknown_class": args_cli.unknown_class,
        "seed": args_cli.seed,
        "client_num": args_cli.client_num,
        "data_root": args_cli.data_root,
        "protocol_mode": args_cli.protocol_mode,
    }
    args = _build_args(config_path=config_path, run_meta_path=run_meta_path, overrides=overrides)
    args.save_path = result_dir

    grid_mode = bool(args_cli.grid_max_feature_distances or args_cli.grid_step_sizes or args_cli.grid_steps)
    output_dir = args_cli.output_dir
    if not output_dir:
        dirname = "stage2a_anchor_generation_grid" if grid_mode else "stage2a_anchor_generation"
        output_dir = osp.join(result_dir, "analysis", dirname)
    output_dir = _ensure_dir(osp.abspath(output_dir))

    trainloaders, _valloader, _closeloader, _openloader, train_val_loaders = _get_dataloaders(args)
    server_model, _models, device, _client_weights = setup(args, trainloaders)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    server_model.load_state_dict(checkpoint["net"], strict=True)
    server_model.eval()
    for param in server_model.parameters():
        param.requires_grad_(False)

    stage1_state = _load_stage1_state(checkpoint, anchor_init_path=osp.join(result_dir, args.anchor_init_file))
    anchors = stage1_state["virtual_anchors"].detach().cpu()
    if anchors.shape[0] != args.virtue_num:
        raise ValueError(f"Expected {args.virtue_num} virtual anchors, found {anchors.shape[0]}")

    print("Selected anchor pairs")
    for anchor_id, pair_info in enumerate(stage1_state.get("selected_anchor_pairs", [])):
        print(f"  Anchor {anchor_id}: pair={pair_info['pair']} names={pair_info.get('pair_names', [])} score={pair_info.get('pair_score', 0.0):.6f}")

    samples = collect_known_train_samples(
        args,
        server_model,
        device,
        train_val_loaders,
        stage1_state,
        analysis_batch_size=args_cli.analysis_batch_size,
        max_trainval_batches=args_cli.max_trainval_batches,
        analysis_num_workers=args_cli.analysis_num_workers,
    )
    candidate_rows, generation_seeds = select_boundary_seeds(samples, stage1_state.get("selected_anchor_pairs", []), args.known_class, args_cli.boundary_fraction)
    if args_cli.max_generation_candidates > 0:
        generation_seeds = generation_seeds[: args_cli.max_generation_candidates]

    classifier_weight = server_model.main_cls.weight[:args.known_class].detach().to(device)
    anchor_tensor = stage1_state["virtual_anchors"].detach().to(device).float()
    step_values = _parse_int_list(args_cli.grid_steps) or [int(args_cli.stage2a_steps)]
    step_size_values = _parse_float_list(args_cli.grid_step_sizes) or [float(args_cli.stage2a_step_size)]
    max_distance_values = _parse_float_list(args_cli.grid_max_feature_distances) or [float(args_cli.stage2a_max_feature_distance)]

    if grid_mode:
        grid_rows = []
        grid_payload = []
        _write_csv(osp.join(output_dir, "boundary_seed_candidates.csv"), candidate_rows)
        for steps, step_size, max_distance in itertools.product(step_values, step_size_values, max_distance_values):
            run_name = f"steps{_format_grid_token(steps)}_step{_format_grid_token(step_size)}_dist{_format_grid_token(max_distance)}"
            combo_dir = osp.join(output_dir, run_name)
            processed_rows = _generate_records(generation_seeds, classifier_weight, anchor_tensor, steps, step_size, args_cli.stage2a_beta, max_distance, args.cosine_scale)
            summary = _build_summary(processed_rows, stage1_state.get("selected_anchor_pairs", []), args.known_class)
            summary = _attach_common_summary(summary, args_cli, args, checkpoint_path, result_dir, run_dir, run_meta_path, config_path, candidate_rows, generation_seeds, steps, step_size, max_distance, run_name)
            _save_generation_outputs(combo_dir, candidate_rows, processed_rows, summary, args_cli)
            row = {
                "grid_run_name": run_name,
                "stage2a_steps": int(steps),
                "stage2a_step_size": float(step_size),
                "stage2a_max_feature_distance": float(max_distance),
                "processed_seed_count": summary["processed_seed_count"],
                "success_count": summary["success_count"],
                "success_rate": summary["success_rate"],
                "mean_feature_distance": summary["mean_feature_distance"],
                "mean_known_logit_drop": summary["mean_known_logit_drop"],
                "mean_target_virtual_logit_gain": summary["mean_target_virtual_logit_gain"],
                "anchors_without_success": "|".join(str(x) for x in summary["anchors_without_success"]),
                "target_anchor_histogram": "|".join(str(x) for x in summary["target_anchor_histogram"]),
            }
            for item in summary["per_anchor"]:
                row[f"anchor{item['anchor_id']}_success_rate"] = item["success_rate"]
                row[f"anchor{item['anchor_id']}_count"] = item["pseudo_count"]
            grid_rows.append(row)
            grid_payload.append(summary)
            print(f"Grid {run_name}: success={summary['success_rate']:.3f}% count={summary['success_count']}/{summary['processed_seed_count']} mean_dist={summary['mean_feature_distance']:.4f}")

        _write_csv(osp.join(output_dir, "grid_summary.csv"), grid_rows)
        with open(osp.join(output_dir, "grid_summary.json"), "w") as f:
            json.dump(grid_payload, f, indent=2)
        best = max(grid_rows, key=lambda row: (row["success_rate"], -row["mean_feature_distance"])) if grid_rows else None
        print()
        print("Stage-2a Grid Summary")
        print(f"Output dir: {output_dir}")
        print(f"Boundary candidates: {len(candidate_rows)}")
        print(f"Matched seeds: {len(generation_seeds)}")
        if best:
            print(f"Best run: {best['grid_run_name']} success={best['success_rate']:.3f}% ({best['success_count']}/{best['processed_seed_count']})")
        return

    processed_rows = _generate_records(generation_seeds, classifier_weight, anchor_tensor, args_cli.stage2a_steps, args_cli.stage2a_step_size, args_cli.stage2a_beta, args_cli.stage2a_max_feature_distance, args.cosine_scale)
    summary = _build_summary(processed_rows, stage1_state.get("selected_anchor_pairs", []), args.known_class)
    summary = _attach_common_summary(summary, args_cli, args, checkpoint_path, result_dir, run_dir, run_meta_path, config_path, candidate_rows, generation_seeds, args_cli.stage2a_steps, args_cli.stage2a_step_size, args_cli.stage2a_max_feature_distance)
    _save_generation_outputs(output_dir, candidate_rows, processed_rows, summary, args_cli)

    print()
    print("Stage-2a Summary")
    print(f"Output dir: {output_dir}")
    print(f"Boundary candidates: {summary['candidate_seed_count']}")
    print(f"Matched seeds: {summary['matched_seed_count']}")
    print(f"Processed seeds: {summary['processed_seed_count']}")
    print(f"Success count: {summary['success_count']}")
    print(f"Success rate: {summary['success_rate']:.3f}%")
    print(f"Mean feature distance: {summary['mean_feature_distance']:.4f}")
    print(f"Mean known logit drop: {summary['mean_known_logit_drop']:.4f}")
    print(f"Mean target virtual gain: {summary['mean_target_virtual_logit_gain']:.4f}")
    print(f"Anchors without success: {summary['anchors_without_success']}")
    _print_anchor_table(summary["per_anchor"])


if __name__ == "__main__":
    main()
