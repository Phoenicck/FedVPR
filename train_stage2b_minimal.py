#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Stage-2b anchor-guided pseudo-feature fine-tuning.

This script intentionally keeps the baseline FedVPR entrypoints untouched. It
starts from a Stage-1 reserve checkpoint, builds per-client pseudo-feature banks
from local known training data, and runs a small FedAvg fine-tuning loop.
"""
import argparse
import json
import os
import os.path as osp
import random
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = osp.abspath(osp.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TOOLS_DIR = osp.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from analyze_stage2a_generation import (
    _build_args,
    _build_summary,
    _ensure_dir,
    _generate_records,
    _get_dataloaders,
    _load_stage1_state,
    _resolve_paths,
    collect_known_train_samples,
    select_boundary_seeds,
)
from eval_stage1_osr import evaluate_stage1_osr
from lib.common import setup
from lib.communication import communication_Pretrain
from lib.stage1_reserve import compute_known_logits, compute_virtual_logits


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _format_token(value):
    return str(value).replace(".", "p").replace("-", "m")


def _client_checkpoint_path(server_checkpoint, client_idx):
    root, ext = osp.splitext(server_checkpoint)
    return f"{root}_C_{client_idx}{ext}"


def _load_stage1_weights(server_model, client_models, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    server_model.load_state_dict(checkpoint["net"], strict=True)
    loaded_client_paths = []
    for client_idx, model in enumerate(client_models):
        client_path = _client_checkpoint_path(checkpoint_path, client_idx)
        if osp.exists(client_path):
            client_checkpoint = torch.load(client_path, map_location=device)
            model.load_state_dict(client_checkpoint["net"], strict=True)
            loaded_client_paths.append(client_path)
        else:
            model.load_state_dict(checkpoint["net"], strict=True)
    return checkpoint, loaded_client_paths


def _freeze_anchors(stage1_state, device):
    anchors = stage1_state["virtual_anchors"].detach().clone().float().to(device)
    anchors.requires_grad_(False)
    stage1_state = dict(stage1_state)
    stage1_state["virtual_anchors"] = anchors.detach().cpu()
    return anchors, stage1_state


def _cap_rows_balanced(rows, cap_per_anchor, seed):
    if cap_per_anchor <= 0:
        return rows
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["target_anchor_id"])].append(row)
    kept = []
    for anchor_id in sorted(grouped):
        anchor_rows = grouped[anchor_id]
        rng.shuffle(anchor_rows)
        kept.extend(anchor_rows[:cap_per_anchor])
    kept.sort(key=lambda row: (row["client_id"], row["target_anchor_id"], row["feature_distance"]))
    return kept


def _build_client_pseudo_bank(
    args,
    model,
    device,
    client_idx,
    train_val_loader,
    stage1_state,
    anchor_tensor,
    args_cli,
):
    model.eval()
    classifier_weight = model.main_cls.weight[: args.known_class].detach().to(device)
    samples = collect_known_train_samples(
        args,
        model,
        device,
        [train_val_loader],
        stage1_state,
        analysis_batch_size=args_cli.analysis_batch_size,
        max_trainval_batches=args_cli.max_trainval_batches,
        analysis_num_workers=args_cli.analysis_num_workers,
    )
    for sample in samples:
        sample["client_id"] = client_idx

    candidate_rows, generation_seeds = select_boundary_seeds(
        samples,
        stage1_state.get("selected_anchor_pairs", []),
        args.known_class,
        args_cli.boundary_fraction,
    )
    if args_cli.max_generation_candidates > 0:
        generation_seeds = generation_seeds[: args_cli.max_generation_candidates]

    raw_processed_rows = _generate_records(
        generation_seeds,
        classifier_weight,
        anchor_tensor,
        args_cli.generator_steps,
        args_cli.generator_step_size,
        args_cli.generator_beta,
        args_cli.generator_max_feature_distance,
        args.cosine_scale,
    )
    raw_summary = _build_summary(raw_processed_rows, stage1_state.get("selected_anchor_pairs", []), args.known_class)
    processed_rows = raw_processed_rows
    if not args_cli.use_failed_pseudo:
        processed_rows = [row for row in processed_rows if bool(row["generation_success"])]
    processed_rows = _cap_rows_balanced(processed_rows, args_cli.pseudo_per_anchor_cap, args.seed + client_idx)

    bank = {}
    row_summaries = []
    for anchor_id in range(len(stage1_state.get("selected_anchor_pairs", []))):
        anchor_rows = [row for row in processed_rows if int(row["target_anchor_id"]) == anchor_id]
        if anchor_rows:
            features = torch.stack([row["_pseudo_feature"] for row in anchor_rows], dim=0).float()
            bank[anchor_id] = features
        row_summaries.append(
            {
                "client_id": client_idx,
                "anchor_id": anchor_id,
                "pseudo_count": len(anchor_rows),
                "pair_names": stage1_state.get("selected_anchor_pairs", [])[anchor_id].get("pair_names", []),
            }
        )

    summary = _build_summary(processed_rows, stage1_state.get("selected_anchor_pairs", []), args.known_class)
    summary.update(
        {
            "client_id": client_idx,
            "candidate_seed_count": len(candidate_rows),
            "matched_seed_count": len(generation_seeds),
            "raw_processed_seed_count": raw_summary["processed_seed_count"],
            "raw_success_count": raw_summary["success_count"],
            "raw_success_rate": raw_summary["success_rate"],
            "raw_target_anchor_histogram": raw_summary["target_anchor_histogram"],
            "kept_pseudo_count": len(processed_rows),
            "pseudo_per_anchor_cap": int(args_cli.pseudo_per_anchor_cap),
        }
    )
    return bank, summary, row_summaries


class BalancedPseudoSampler:
    def __init__(self, bank, known_class, pseudo_ratio, pseudo_per_anchor_per_batch=0):
        self.bank = {int(k): v for k, v in bank.items() if v is not None and v.shape[0] > 0}
        self.anchor_ids = sorted(self.bank)
        self.known_class = int(known_class)
        self.pseudo_ratio = float(pseudo_ratio)
        self.pseudo_per_anchor_per_batch = int(pseudo_per_anchor_per_batch)

    def __len__(self):
        return int(sum(self.bank[anchor_id].shape[0] for anchor_id in self.anchor_ids))

    def sample(self, known_batch_size, device):
        if not self.anchor_ids or self.pseudo_ratio <= 0:
            return None, None
        if self.pseudo_per_anchor_per_batch > 0:
            per_anchor = self.pseudo_per_anchor_per_batch
        else:
            target_total = max(1, int(round(int(known_batch_size) * self.pseudo_ratio)))
            per_anchor = max(1, int(round(target_total / max(len(self.anchor_ids), 1))))

        features = []
        targets = []
        for anchor_id in self.anchor_ids:
            anchor_bank = self.bank[anchor_id]
            choice = torch.randint(0, anchor_bank.shape[0], (per_anchor,))
            features.append(anchor_bank[choice])
            targets.extend([self.known_class + anchor_id] * per_anchor)
        return torch.cat(features, dim=0).to(device), torch.tensor(targets, dtype=torch.long, device=device)


def _train_one_client(args, args_cli, model, trainloader, pseudo_sampler, anchors, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args_cli.stage2b_lr))
    criterion = nn.CrossEntropyLoss()
    totals = defaultdict(float)
    steps = 0
    correct = 0
    total = 0

    for local_epoch in range(int(args.worker_steps)):
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            optimizer.zero_grad()

            features = model(inputs)["feature"]
            known_logits = compute_known_logits(features, model.main_cls.weight[: args.known_class], args.cosine_scale)
            virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)
            full_logits = torch.cat([known_logits, virtual_logits], dim=1)

            loss_known = criterion(known_logits, targets)
            loss_reserve = criterion(full_logits, targets)
            loss_pseudo = known_logits.new_tensor(0.0)
            pseudo_count = 0
            if pseudo_sampler is not None:
                pseudo_features, pseudo_targets = pseudo_sampler.sample(inputs.shape[0], device)
                if pseudo_features is not None:
                    pseudo_features = pseudo_features.detach()
                    pseudo_known_logits = compute_known_logits(
                        pseudo_features, model.main_cls.weight[: args.known_class], args.cosine_scale
                    )
                    pseudo_virtual_logits = compute_virtual_logits(pseudo_features, anchors, args.cosine_scale)
                    pseudo_full_logits = torch.cat([pseudo_known_logits, pseudo_virtual_logits], dim=1)
                    loss_pseudo = criterion(pseudo_full_logits, pseudo_targets)
                    pseudo_count = int(pseudo_targets.numel())

            loss = (
                loss_known
                + float(args_cli.lambda_reserve) * loss_reserve
                + float(args_cli.lambda_pseudo) * loss_pseudo
            )
            loss.backward()
            optimizer.step()

            steps += 1
            totals["loss"] += float(loss.item())
            totals["loss_known"] += float(loss_known.item())
            totals["loss_reserve"] += float(loss_reserve.item())
            totals["loss_pseudo"] += float(loss_pseudo.item())
            totals["pseudo_count"] += pseudo_count
            pred = known_logits.argmax(1)
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())

            if args.max_train_batches > 0 and batch_idx + 1 >= args.max_train_batches:
                break

    denom = max(steps, 1)
    return {
        "loss": totals["loss"] / denom,
        "loss_known": totals["loss_known"] / denom,
        "loss_reserve": totals["loss_reserve"] / denom,
        "loss_pseudo": totals["loss_pseudo"] / denom,
        "pseudo_per_step": totals["pseudo_count"] / denom,
        "known_acc": 100.0 * correct / max(total, 1),
        "steps": steps,
    }


def _full_head_known_accuracy(args, model, device, loader, anchors):
    model.eval()
    correct = 0
    total = 0
    virtual_pred = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.long().to(device)
            features = model(inputs)["feature"]
            known_logits = compute_known_logits(features, model.main_cls.weight[: args.known_class], args.cosine_scale)
            virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)
            full_pred = torch.cat([known_logits, virtual_logits], dim=1).argmax(1)
            correct += int((full_pred == targets).sum().item())
            virtual_pred += int((full_pred >= args.known_class).sum().item())
            total += int(targets.numel())
            if args.max_eval_batches > 0 and batch_idx + 1 >= args.max_eval_batches:
                break
    return {
        "full_head_known_acc": 100.0 * correct / max(total, 1),
        "known_to_virtual_rate": 100.0 * virtual_pred / max(total, 1),
        "num_known_eval": total,
    }


def _per_unknown_virtual_recall(args, model, device, openloader, anchors):
    model.eval()
    counts = defaultdict(int)
    detected = defaultdict(int)
    anchor_hist = defaultdict(lambda: [0 for _ in range(args.virtue_num)])
    with torch.no_grad():
        for batch_idx, (inputs, targets, _img_dirs) in enumerate(openloader):
            inputs = inputs.to(device)
            targets = targets.long().cpu().numpy()
            features = model(inputs)["feature"]
            known_logits = compute_known_logits(features, model.main_cls.weight[: args.known_class], args.cosine_scale)
            virtual_logits = compute_virtual_logits(features, anchors, args.cosine_scale)
            full_pred = torch.cat([known_logits, virtual_logits], dim=1).argmax(1).detach().cpu().numpy()
            for target, pred in zip(targets, full_pred):
                unknown_id = int(target - args.known_class)
                counts[unknown_id] += 1
                if pred >= args.known_class:
                    detected[unknown_id] += 1
                    anchor_hist[unknown_id][int(pred - args.known_class)] += 1
            if args.max_eval_batches > 0 and batch_idx + 1 >= args.max_eval_batches:
                break
    rows = []
    for unknown_id in range(args.unknown_class):
        total = counts[unknown_id]
        rows.append(
            {
                "unknown_id": unknown_id,
                "count": total,
                "virtual_recall": 100.0 * detected[unknown_id] / max(total, 1),
                "pred_anchor_hist": anchor_hist[unknown_id],
            }
        )
    return rows


def _evaluate(args, model, device, closeloader, openloader, stage1_state, anchors):
    osr_result, close_test = evaluate_stage1_osr(args, model, device, closeloader, openloader, stage1_state)
    full_head = _full_head_known_accuracy(args, model, device, closeloader, anchors)
    per_unknown = _per_unknown_virtual_recall(args, model, device, openloader, anchors)
    return {
        "close_test": close_test,
        "osr": osr_result,
        "full_head": full_head,
        "per_unknown_recall": per_unknown,
    }


def _append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def _print_eval_row(prefix, metrics):
    close = metrics["close_test"]
    osr = metrics["osr"]
    full_head = metrics["full_head"]
    print(
        f"{prefix} "
        f"CloseACC={close['acc']:.3f} CloseF1={close['f1']:.3f} "
        f"AUROC={osr['auroc']:.3f} AUPR={osr['aupr']:.3f} OSCR={osr['oscr']:.3f} "
        f"UNK={osr['unk']:.3f} HOS={osr['hos']:.3f} "
        f"FullHeadKnownACC={full_head['full_head_known_acc']:.3f} "
        f"CloseK->V={osr['close_virtual_pred_rate']:.3f}% Open->V={osr['open_virtual_pred_rate']:.3f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="Minimal Stage-2b pseudo-feature FedAvg fine-tuning.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage-1 lambda=0.05 checkpoint")
    parser.add_argument("--config", type=str, default="", help="optional Stage-1 config YAML")
    parser.add_argument("--run-dir", type=str, default="", help="archived Stage-1 run directory")
    parser.add_argument("--run-meta", type=str, default="", help="optional run_meta.txt")
    parser.add_argument("--output-dir", type=str, default="", help="optional output directory")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--stage2b-lr", type=float, default=1e-4)
    parser.add_argument("--lambda-reserve", type=float, default=0.05)
    parser.add_argument("--lambda-pseudo", type=float, default=0.1)
    parser.add_argument("--pseudo-ratio", type=float, default=0.25)
    parser.add_argument("--pseudo-per-anchor-per-batch", type=int, default=0)
    parser.add_argument("--pseudo-per-anchor-cap", type=int, default=64)
    parser.add_argument("--boundary-fraction", type=float, default=0.10)
    parser.add_argument("--generator-steps", type=int, default=5)
    parser.add_argument("--generator-step-size", type=float, default=0.005)
    parser.add_argument("--generator-beta", type=float, default=1.0)
    parser.add_argument("--generator-max-feature-distance", type=float, default=0.5)
    parser.add_argument("--use-failed-pseudo", action="store_true")
    parser.add_argument("--analysis-batch-size", type=int, default=128)
    parser.add_argument("--analysis-num-workers", type=int, default=4)
    parser.add_argument("--max-trainval-batches", type=int, default=0)
    parser.add_argument("--max-generation-candidates", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--known_class", type=int, default=None)
    parser.add_argument("--unknown_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--client_num", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--protocol_mode", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=None)
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
        "batchsize": args_cli.batchsize,
        "max_train_batches": args_cli.max_train_batches,
        "max_eval_batches": args_cli.max_eval_batches,
        "lambda_reserve": args_cli.lambda_reserve,
    }
    args = _build_args(config_path=config_path, run_meta_path=run_meta_path, overrides=overrides)
    args.save_path = result_dir
    args.stage1_reserve_enable = True
    args.lambda_reserve = float(args_cli.lambda_reserve)
    _set_seed(int(args.seed))

    if not args_cli.output_dir:
        run_name = (
            f"stage2b_minimal_rounds{args_cli.rounds}_ratio{_format_token(args_cli.pseudo_ratio)}"
            f"_lp{_format_token(args_cli.lambda_pseudo)}_lr{_format_token(args_cli.stage2b_lr)}"
        )
        output_dir = osp.join(result_dir, "analysis", run_name)
    else:
        output_dir = args_cli.output_dir
    output_dir = _ensure_dir(osp.abspath(output_dir))
    log_jsonl = osp.join(output_dir, "stage2b_log.jsonl")
    if osp.exists(log_jsonl):
        os.remove(log_jsonl)

    trainloaders, _valloader, closeloader, openloader, train_val_loaders = _get_dataloaders(args)
    server_model, client_models, device, client_weights = setup(args, trainloaders)
    checkpoint, loaded_client_paths = _load_stage1_weights(server_model, client_models, checkpoint_path, device)
    stage1_state = _load_stage1_state(checkpoint, anchor_init_path=osp.join(result_dir, args.anchor_init_file))
    anchors, stage1_state = _freeze_anchors(stage1_state, device)

    print("Selected anchor pairs:")
    for anchor_id, pair_info in enumerate(stage1_state.get("selected_anchor_pairs", [])):
        print(f"  anchor {anchor_id}: {pair_info.get('pair_names', pair_info.get('pair'))}")
    print(f"Loaded {len(loaded_client_paths)} client checkpoints.")
    print(f"Output: {output_dir}")

    pseudo_banks = []
    pseudo_summaries = []
    pseudo_rows = []
    for client_idx, client_model in enumerate(client_models):
        bank, summary, row_summaries = _build_client_pseudo_bank(
            args,
            client_model,
            device,
            client_idx,
            train_val_loaders[client_idx],
            stage1_state,
            anchors,
            args_cli,
        )
        pseudo_banks.append(bank)
        pseudo_summaries.append(summary)
        pseudo_rows.extend(row_summaries)
        print(
            f"Client {client_idx}: kept={summary['kept_pseudo_count']} "
            f"hist={summary['target_anchor_histogram']} "
            f"raw_success={summary['raw_success_rate']:.2f}%"
        )

    with open(osp.join(output_dir, "pseudo_bank_summary.json"), "w") as f:
        json.dump(
            {
                "client_summaries": pseudo_summaries,
                "per_client_anchor_counts": pseudo_rows,
                "settings": vars(args_cli),
            },
            f,
            indent=2,
            default=_json_default,
        )

    with open(osp.join(output_dir, "stage2b_settings.json"), "w") as f:
        json.dump(
            {
                "checkpoint": checkpoint_path,
                "run_dir": run_dir,
                "result_dir": result_dir,
                "config": config_path,
                "run_meta": run_meta_path,
                "protocol_mode": args.protocol_mode,
                "lambda_reserve": args_cli.lambda_reserve,
                "lambda_pseudo": args_cli.lambda_pseudo,
                "pseudo_ratio": args_cli.pseudo_ratio,
                "stage2b_lr": args_cli.stage2b_lr,
                "rounds": args_cli.rounds,
                "selected_anchor_pairs": stage1_state.get("selected_anchor_pairs", []),
                "anchor_pairwise_cosine": stage1_state.get("anchor_pairwise_cosine", []),
            },
            f,
            indent=2,
            default=_json_default,
        )

    baseline_metrics = _evaluate(args, server_model, device, closeloader, openloader, stage1_state, anchors)
    _print_eval_row("[Round 0 / Stage-1]", baseline_metrics)
    _append_jsonl(log_jsonl, {"round": 0, "phase": "baseline", "metrics": baseline_metrics})

    best_oscr = baseline_metrics["osr"]["oscr"]
    best_round = 0
    best_path = osp.join(output_dir, "best_stage2b_server.pth")
    final_path = osp.join(output_dir, "final_stage2b_server.pth")
    torch.save({"net": deepcopy(server_model.state_dict()), "stage1_state": stage1_state}, best_path)

    for round_idx in range(1, int(args_cli.rounds) + 1):
        train_stats = []
        for client_idx, (client_model, trainloader) in enumerate(zip(client_models, trainloaders)):
            sampler = BalancedPseudoSampler(
                pseudo_banks[client_idx],
                args.known_class,
                args_cli.pseudo_ratio,
                pseudo_per_anchor_per_batch=args_cli.pseudo_per_anchor_per_batch,
            )
            stat = _train_one_client(args, args_cli, client_model, trainloader, sampler, anchors, device)
            stat["client_id"] = client_idx
            stat["pseudo_bank_size"] = len(sampler)
            train_stats.append(stat)

        server_model, client_models = communication_Pretrain(args, server_model, client_models, client_weights)
        train_mean = {
            key: float(np.mean([row[key] for row in train_stats]))
            for key in ["loss", "loss_known", "loss_reserve", "loss_pseudo", "pseudo_per_step", "known_acc"]
        }

        do_eval = (round_idx % max(1, int(args_cli.eval_every)) == 0) or (round_idx == int(args_cli.rounds))
        row = {"round": round_idx, "phase": "train", "train": train_mean, "client_train": train_stats}
        if do_eval:
            metrics = _evaluate(args, server_model, device, closeloader, openloader, stage1_state, anchors)
            row["metrics"] = metrics
            _print_eval_row(f"[Round {round_idx}]", metrics)
            if metrics["osr"]["oscr"] > best_oscr:
                best_oscr = metrics["osr"]["oscr"]
                best_round = round_idx
                torch.save({"net": deepcopy(server_model.state_dict()), "stage1_state": stage1_state}, best_path)
        else:
            print(
                f"[Round {round_idx}] loss={train_mean['loss']:.4f} "
                f"known={train_mean['loss_known']:.4f} reserve={train_mean['loss_reserve']:.4f} "
                f"pseudo={train_mean['loss_pseudo']:.4f}"
            )
        _append_jsonl(log_jsonl, row)

    torch.save({"net": server_model.state_dict(), "stage1_state": stage1_state}, final_path)
    final_metrics = _evaluate(args, server_model, device, closeloader, openloader, stage1_state, anchors)
    summary = {
        "output_dir": output_dir,
        "best_round": best_round,
        "best_oscr": best_oscr,
        "baseline": baseline_metrics,
        "final": final_metrics,
        "best_checkpoint": best_path if osp.exists(best_path) else "",
        "final_checkpoint": final_path,
        "pseudo_bank_summary": pseudo_summaries,
    }
    with open(osp.join(output_dir, "stage2b_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    _print_eval_row("[Final]", final_metrics)
    print(f"Best OSCR round: {best_round} ({best_oscr:.3f})")
    print(f"Saved summary: {osp.join(output_dir, 'stage2b_summary.json')}")


if __name__ == "__main__":
    main()
