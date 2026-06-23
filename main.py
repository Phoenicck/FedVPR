# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 10:05:53 2022

@author: ZML
"""

import argparse
import os
import os.path as osp
import sys
import numpy as np
import torch
import random
import warnings
import yaml
from models.utils import pprint, ensure_path
from utils import Tee
warnings.filterwarnings('ignore')


def main(args):
    if args.mode == 'Pretrain':
        from lib.Pretrain import run
        run(args)
    elif args.mode == 'Finetune':
        from lib.Finetune import run
        run(args)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Cannot parse boolean value from: {value}')


def _load_config_defaults(argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default='')
    pre_args, _ = pre_parser.parse_known_args(argv)
    if not pre_args.config:
        return {}
    with open(pre_args.config, 'r') as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f'Config file must contain a mapping: {pre_args.config}')
    return config


def _normalize_args(args):
    if isinstance(args.anchor_density_angles, str):
        args.anchor_density_angles = [float(item.strip()) for item in args.anchor_density_angles.split(',') if item.strip()]
    elif isinstance(args.anchor_density_angles, (list, tuple)):
        args.anchor_density_angles = [float(item) for item in args.anchor_density_angles]
    else:
        args.anchor_density_angles = [15.0, 20.0, 25.0]
    args.anchor_freeze = _str2bool(args.anchor_freeze)
    if args.stage1_reserve_enable and args.mode != 'Pretrain':
        raise ValueError('stage1_reserve_enable is only supported in Pretrain mode.')
    return args


if __name__=="__main__":
    config_defaults = _load_config_defaults(sys.argv[1:])
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='', type=str, help='optional YAML config file')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', default='', type=str, help='checkpoint path for resume')
    parser.add_argument('--model_type', default='softmax', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='Resnet18', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='Hyperkvasir',type=str,help='dataset configuration')
    parser.add_argument('--known_class', default=5,type=int,help='number of known class')
    parser.add_argument('--unknown_class', default=3,type=int,help='number of unknown class')
    parser.add_argument('--device_id', default=0, type=int, help='CUDA device index')
    parser.add_argument('--virtue_num', default=3, type=int, help='number of virtual classes')
    parser.add_argument('--seed', default='0',type=int,help='random seed for dataset generation.')

    parser.add_argument('--data_root', default='./dataset/', type=str, help='data_root')
    parser.add_argument('--rotation', default=45,type=int,help='Rotation Angle')
    parser.add_argument('--resize', default=144,type=int,help='resize')
    parser.add_argument('--cropsize', default=128,type=int,help='crop size')
    parser.add_argument('--batchsize', default=16,type=int,help='minibatch size')
    parser.add_argument('--epoches', default=200,type=int,help='epoches')
    parser.add_argument('--save_interval', default=0, type=int, help='save checkpoint every N epochs (0=disabled)')
    parser.add_argument('--log_dir', default='./logs', type=str, help='directory for log files (set to empty string to disable)')
    parser.add_argument('--max_train_batches', default=0, type=int, help='debug only: max train batches per client per epoch (0=all)')
    parser.add_argument('--max_eval_batches', default=0, type=int, help='debug only: max eval batches per loader (0=all)')

    parser.add_argument('--client_num', type=int, default=8, help='the number of clients')
    parser.add_argument('--worker_steps', type=int, default=1, help='step of worker')
    parser.add_argument('--mode', type=str, default='Pretrain', help='Pretrain, Finetune')

    parser.add_argument('--dirichlet', type=float, default=0.5,help='dirichlet alpha')
    parser.add_argument('--vir_weight_warmup', default=0.5, type=float,
                        help='stage-1 virtual loss weight before warmup boundary')
    parser.add_argument('--vir_weight_main', default=0.01, type=float,
                        help='stage-1 virtual loss weight after annealing finishes')
    parser.add_argument('--vir_warmup_epochs', default=4, type=int,
                        help='number of epochs to keep the warmup virtual-loss weight before annealing')
    parser.add_argument('--vir_anneal_epochs', default=0, type=int,
                        help='number of epochs to linearly anneal virtual-loss weight from warmup to main (0 keeps the legacy hard switch)')
    parser.add_argument('--vir_margin', default=1.0, type=float,
                        help='target margin between the true known logit and the strongest virtual logit during stage-1')
    parser.add_argument('--vir_margin_weight', default=0.0, type=float,
                        help='extra weight for the stage-1 virtual margin term (0 disables the margin add-on)')
    parser.add_argument('--protocol_mode', default='random', choices=['random', 'easy', 'hard'],
                        help='class protocol mode: random selection, fixed practical-easy protocol, or fixed hard protocol')
    parser.add_argument('--anchor_log_interval', default=1, type=int,
                        help='log virtual-anchor diagnostics every N epochs during pretrain')
    parser.add_argument('--anchor_log_file', default='anchor_diagnostics.jsonl', type=str,
                        help='jsonl file for per-epoch virtual-anchor diagnostics')
    parser.add_argument('--stage1_boundary_quantile', default=0.2, type=float,
                        help='fraction of lowest known-logit-margin validation samples treated as boundary-like for stage-1 diagnostics')
    parser.add_argument('--stage1_diag_eps', default=1e-12, type=float,
                        help='small epsilon used in stage-1 compactness diagnostics')

    parser.add_argument('--stage1_reserve_enable', default=False, type=_str2bool,
                        help='enable warm-up + fixed virtual-anchor reserve training')
    parser.add_argument('--stage1_warmup_rounds', default=25, type=int,
                        help='warm-up communication rounds before initializing reserve anchors')
    parser.add_argument('--lambda_reserve', default=0.1, type=float,
                        help='reserve loss weight after anchor initialization')
    parser.add_argument('--cosine_scale', default=16.0, type=float,
                        help='cosine logit scale for known and virtual anchor logits')
    parser.add_argument('--anchor_freeze', default=True, type=_str2bool,
                        help='freeze virtual anchors after server-side initialization')
    parser.add_argument('--anchor_similarity_threshold', default=0.95, type=float,
                        help='max allowed cosine similarity between selected reserve anchors')
    parser.add_argument('--anchor_density_angles', default='15,20,25', type=str,
                        help='comma-separated angle thresholds for anchor density diagnostics')
    parser.add_argument('--anchor_init_file', default='stage1_anchor_init.json', type=str,
                        help='json file storing initialized anchor metadata')

    #Attack
    parser.add_argument('--eps', type=float, default=1.,help='eps')
    parser.add_argument('--num_steps', type=int, default=10,help='num_steps')
    parser.add_argument('--unknown_weight', type=float, default=1.,help='unknown_weight')
    parser.add_argument('--rank_weight', default=0.05, type=float, help='ranking loss weight')
    parser.add_argument('--rank_margin', default=0.2, type=float, help='ranking loss margin')

    parser.add_argument('--start_epoch', type=str, default='[5, 10, 15, 20, 25]', help='start_epoch')
    parser.add_argument('--sample_from', type=int, default=8, help='sample_from')

    # LUPS parameters
    parser.add_argument('--lups_mode', default='diag', choices=['fullcov', 'diag'], help='FOSS mode: fullcov or diag')
    parser.add_argument('--lups_space', default='pooled', choices=['fullmap', 'pooled', 'gap'], help='feature space for LUPS')
    parser.add_argument('--lups_pool_size', default=2, type=int, help='pooled spatial size')
    parser.add_argument('--lups_min_count', default=10, type=int, help='min samples to build distribution')
    parser.add_argument('--lups_min_var', default=1e-4, type=float, help='min diagonal variance')
    parser.add_argument('--lups_var_scale', default=1.0, type=float, help='variance scale factor')
    parser.add_argument('--lups_candidates', default=100, type=int, help='candidate samples for low_density')
    parser.add_argument('--lups_sample_strategy', default='low_density', choices=['random', 'low_density'])
    parser.add_argument('--lups_local_weight', default=0.1, type=float, help='local i-DUS loss weight')
    parser.add_argument('--lups_global_weight', default=0.01, type=float, help='global LUPS loss weight')

    parser.set_defaults(**config_defaults)

    import time
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print('The starting time ：{}'.format(now))
    args = parser.parse_args()
    args = _normalize_args(args)
    client_names = ['Client'+str(i) for i in range(args.client_num)]
    args.client_names = client_names

    args.start_epoch = eval(args.start_epoch)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)

    pprint(vars(args))

    set_seed(args.seed)

    save_path1 = osp.join('results','M{}-D{}-M{}-B{}'.format(args.mode, args.dataset,args.model_type, args.backbone))
    save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr),str(args.known_class),str(args.unknown_class),str(args.seed))
    if args.stage1_reserve_enable and args.mode == 'Pretrain':
        save_path2 += '-RsvW{}-V{}'.format(str(args.stage1_warmup_rounds), str(args.virtue_num))
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path, remove=False)

    tee = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_name = f"{args.dataset}_{args.mode}_K{args.known_class}_U{args.unknown_class}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log_path = osp.join(args.log_dir, log_name)
        tee = Tee(log_path)
        print(f"Logging to: {log_path}")

    try:
        main(args)
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        print('The ending time ：{}'.format(now))
    finally:
        if tee:
            tee.close()
