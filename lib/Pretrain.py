# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:44:05 2022

@author: ZML
"""
import json
import os.path as osp

import torch
import torch.nn.functional as F

from .communication import communication_Pretrain
from .common import setup, update_lr
from .Pretrain_library import train, val, test


def _pairwise_cosine_stats(weights):
    if weights.shape[0] <= 1:
        return {
            'matrix': [[1.0]] if weights.shape[0] == 1 else [],
            'min': 1.0,
            'mean': 1.0,
            'max': 1.0,
        }
    normed = F.normalize(weights, dim=1)
    sim = torch.mm(normed, normed.t())
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    vals = sim[mask]
    return {
        'matrix': sim.cpu().tolist(),
        'min': float(vals.min().item()),
        'mean': float(vals.mean().item()),
        'max': float(vals.max().item()),
    }


def _cross_cosine_stats(a, b):
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {
            'matrix': [],
            'min': 0.0,
            'mean': 0.0,
            'max': 0.0,
        }
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    sim = torch.mm(a_norm, b_norm.t())
    return {
        'matrix': sim.cpu().tolist(),
        'min': float(sim.min().item()),
        'mean': float(sim.mean().item()),
        'max': float(sim.max().item()),
    }


def _append_anchor_log(args, diagnostics):
    if not args.anchor_log_file:
        return
    path = osp.join(args.save_path, args.anchor_log_file)
    with open(path, 'a') as f:
        f.write(json.dumps(diagnostics) + '\n')


def _build_anchor_diagnostics(args, epoch, server_model, initial_virtual_weights, val_result, osr_result):
    weight = server_model.main_cls.weight.detach().float().cpu()
    known_weights = weight[:args.known_class]
    virtual_weights = weight[args.known_class:]

    virtual_pairwise = _pairwise_cosine_stats(virtual_weights)
    known_virtual = _cross_cosine_stats(virtual_weights, known_weights)
    virtual_norms = virtual_weights.norm(dim=1) if virtual_weights.numel() > 0 else torch.zeros(0)
    drift = (virtual_weights - initial_virtual_weights).norm(dim=1) if virtual_weights.shape == initial_virtual_weights.shape else torch.zeros(virtual_weights.shape[0])

    diagnostics = {
        'epoch': epoch,
        'protocol_mode': args.protocol_mode,
        'dataset': args.dataset,
        'known_class': args.known_class,
        'unknown_class': args.unknown_class,
        'virtue_num': args.virtue_num,
        'virtual_anchor_norms': virtual_norms.tolist(),
        'virtual_anchor_drift_l2': drift.tolist(),
        'virtual_anchor_drift_l2_max': float(drift.max().item()) if drift.numel() > 0 else 0.0,
        'virtual_pairwise_cosine': virtual_pairwise,
        'known_virtual_cosine': known_virtual,
        'val_known_virtual_pred_rate': float(val_result.get('known_virtual_pred_rate', 0.0)),
        'val_known_virtual_prob_mean': float(val_result.get('known_virtual_prob_mean', 0.0)),
        'val_known_virtual_margin_mean': float(val_result.get('known_virtual_margin_mean', 0.0)),
        'val_known_true_virtual_logit_margin': val_result.get('known_true_virtual_logit_margin', {}),
        'test_close_virtual_pred_rate': float(osr_result.get('close_virtual_pred_rate', 0.0)),
        'test_close_virtual_prob_mean': float(osr_result.get('close_virtual_prob_mean', 0.0)),
        'test_close_known_virtual_margin_mean': float(osr_result.get('close_known_virtual_margin_mean', 0.0)),
        'test_close_virtual_hist': osr_result.get('close_virtual_hist', []),
        'test_close_virtual_entropy': float(osr_result.get('close_virtual_entropy', 0.0)),
        'test_close_virtual_prob_mean_per_anchor': osr_result.get('close_virtual_prob_mean_per_anchor', []),
        'test_close_true_virtual_logit_margin': osr_result.get('close_true_virtual_logit_margin', {}),
        'test_open_virtual_pred_rate': float(osr_result.get('open_virtual_pred_rate', 0.0)),
        'test_open_virtual_prob_mean': float(osr_result.get('open_virtual_prob_mean', 0.0)),
        'test_open_virtual_hist': osr_result.get('open_virtual_hist', []),
        'test_open_virtual_entropy': float(osr_result.get('open_virtual_entropy', 0.0)),
        'test_open_virtual_prob_mean_per_anchor': osr_result.get('open_virtual_prob_mean_per_anchor', []),
        'test_open_known_virtual_logit_margin': osr_result.get('open_known_virtual_logit_margin', {}),
        'test_unk': float(osr_result.get('unk', 0.0)),
        'test_oscr': float(osr_result.get('oscr', 0.0)),
        'test_auroc': float(osr_result.get('auroc', 0.0)),
    }
    return diagnostics


def _print_anchor_diagnostics(epoch, args, diagnostics):
    print(
        f"AnchorDiag [{epoch}/{args.epoches}] "
        f"VNorm={diagnostics['virtual_anchor_norms']} "
        f"VDriftMax={diagnostics['virtual_anchor_drift_l2_max']:.6f} "
        f"Vcos(max/mean)=({diagnostics['virtual_pairwise_cosine']['max']:.4f}/{diagnostics['virtual_pairwise_cosine']['mean']:.4f}) "
        f"KVcos(max/mean)=({diagnostics['known_virtual_cosine']['max']:.4f}/{diagnostics['known_virtual_cosine']['mean']:.4f}) "
        f"ValK->V={diagnostics['val_known_virtual_pred_rate']:.2f}% "
        f"CloseK->V={diagnostics['test_close_virtual_pred_rate']:.2f}% "
        f"Open->V={diagnostics['test_open_virtual_pred_rate']:.2f}% "
        f"OpenVEntropy={diagnostics['test_open_virtual_entropy']:.4f}"
    )


def run(args):

    best_oscr = 0
    best_epoch = 0
    best_osr_acc = best_osr_f1 = best_osr_recall = best_osr_precision = 0
    best_osr_unk = best_osr_os_star = best_osr_hos = best_osr_auroc = best_osr_aupr = best_osr_oscr = 0
    print('==> Preparing data..')
    param = {'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet, 'protocol_mode': args.protocol_mode}
    if args.dataset=='Hyperkvasir':
        from data.fed_hyper_kvasir_relabel import get_dataloaders
    elif args.dataset=='RetinalOCT':
        param = {'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet, 'protocol_mode': args.protocol_mode}
        from data.fed_retinal_oct_relabel import get_dataloaders
    elif args.dataset=='ISIC':
        param = {'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet, 'protocol_mode': args.protocol_mode}
        from data.fed_isic_relabel import get_dataloaders
    elif args.dataset=='Bloodmnist':
        param = {'dataset': args.dataset, 'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet, 'protocol_mode': args.protocol_mode}
        from data.fed_MedMINIST_relabel import get_dataloaders
    elif args.dataset=='OrganMNIST3D':
        param = {'dataset': args.dataset, 'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet, 'protocol_mode': args.protocol_mode}
        from data.fed_MedMINIST3D_relabel import get_dataloaders
    else:
        assert False
    trainloaders, valloader, closerloader, openloader, train_val_loaders = get_dataloaders(args.client_num, args.data_root, args.seed, param)
    server_model, models, device, client_weights  = setup(args, trainloaders)
    initial_virtual_weights = server_model.main_cls.weight.detach().float().cpu()[args.known_class:].clone()
    epoch = 0
    for epoch_it in range(args.epoches // args.worker_steps):
        args.lr = update_lr(args.lr, epoch, args.epoches, lr_step=20, lr_gamma=0.5)
        optimizers = [torch.optim.Adam(params=models[idx].parameters(), lr=args.lr, betas=(0.9, 0.99), amsgrad=False) for idx in range(args.client_num)]
        for ws in range(args.worker_steps):
            for client_idx in range(args.client_num):
                client_name = args.client_names[client_idx]
                model, train_loader, optimizer= models[client_idx], trainloaders[client_idx], optimizers[client_idx]
                # Do training and validation loops
                train_result = train(args, device, epoch, model, train_loader, optimizer)
                train_loss, train_acc, train_f1, train_recall, train_precision = train_result['loss'], train_result['acc'],train_result['f1'],train_result['recall'], train_result['precision']
                train_loss_ce = train_result.get('loss_ce', 0.0)
                train_loss_vir = train_result.get('loss_vir', 0.0)
                train_loss_vir_weighted = train_result.get('loss_vir_weighted', 0.0)
                train_loss_aux = train_result.get('loss_aux', 0.0)
                train_vir_weight = train_result.get('vir_weight', 0.0)
                print(
                    f"Train {client_name} [{epoch}/{args.epoches}] LR={args.lr:.7f} loss={train_loss:.3f} "
                    f"(CE={train_loss_ce:.3f} VIR={train_loss_vir:.3f} wVIR={train_loss_vir_weighted:.3f} AUX={train_loss_aux:.3f} w={train_vir_weight:.3f}) "
                    f"ACC={train_acc:.3f} F1={train_f1:.3f} Rec={train_recall:.3f} Prec={train_precision:.3f}"
                )
        server_model, models = communication_Pretrain(args, server_model, models, client_weights)
        val_result = val(args, device, epoch, server_model, valloader)
        val_loss, val_acc, val_f1, val_recall, val_prec = val_result['loss'], val_result['acc'],val_result['f1'],val_result['recall'], val_result['precision']
        print()
        print(f"Val    [{epoch}/{args.epoches}] LR={args.lr:.7f} loss={val_loss:.3f} ACC={val_acc:.3f} F1={val_f1:.3f} Rec={val_recall:.3f} Prec={val_prec:.3f}")
        print(f"Val-Known->Virtual [{epoch}/{args.epoches}] rate={val_result.get('known_virtual_pred_rate', 0.0):.3f}% prob_mean={val_result.get('known_virtual_prob_mean', 0.0):.6f} margin_mean={val_result.get('known_virtual_margin_mean', 0.0):.6f}")
        val_logit_margin = val_result.get('known_true_virtual_logit_margin', {})
        print(
            f"Val-LogitMargin [{epoch}/{args.epoches}] TrueKnown-VMax "
            f"mean={val_logit_margin.get('mean', 0.0):.4f} std={val_logit_margin.get('std', 0.0):.4f} "
            f"p10={val_logit_margin.get('p10', 0.0):.4f} p50={val_logit_margin.get('p50', 0.0):.4f} "
            f"p90={val_logit_margin.get('p90', 0.0):.4f}"
        )
        print()

        osr_result, close_test_result = test(args, device, epoch, server_model, closerloader, openloader)
        osr_acc, osr_f1, osr_recall, osr_precision = osr_result['acc'],osr_result['f1'],osr_result['recall'],osr_result['precision']
        osr_unk, osr_os_star, osr_hos, osr_auroc, osr_aupr, osr_oscr = osr_result['unk'], osr_result['os_star'], osr_result['hos'], osr_result['auroc'], osr_result['aupr'], osr_result['oscr']
        test_loss, test_acc, test_f1, test_recall, test_precision = close_test_result['loss'], close_test_result['acc'],close_test_result['f1'],close_test_result['recall'],close_test_result['precision']
        print(f"Test-  OSR [{epoch}/{args.epoches}] LR={args.lr:.7f} ACC={osr_acc:.3f} F1={osr_f1:.3f} Rec={osr_recall:.3f} Prec={osr_precision:.3f} UNK={osr_unk:.3f} OS*={osr_os_star:.3f} HOS={osr_hos:.3f} AUROC={osr_auroc:.3f} AUPR={osr_aupr:.3f} OSCR={osr_oscr:.3f}")
        print(f"Test-Close [{epoch}/{args.epoches}] LR={args.lr:.7f} loss={test_loss:.3f} ACC={test_acc:.3f} F1={test_f1:.3f} Rec={test_recall:.3f} Prec={test_precision:.3f}")
        print(f"Test-Virtual [{epoch}/{args.epoches}] CloseK->V={osr_result.get('close_virtual_pred_rate', 0.0):.3f}% Open->V={osr_result.get('open_virtual_pred_rate', 0.0):.3f}% OpenVEntropy={osr_result.get('open_virtual_entropy', 0.0):.6f} CloseHist={osr_result.get('close_virtual_hist', [])} OpenHist={osr_result.get('open_virtual_hist', [])}")
        close_logit_margin = osr_result.get('close_true_virtual_logit_margin', {})
        open_logit_margin = osr_result.get('open_known_virtual_logit_margin', {})
        print(
            f"Test-LogitMargin [{epoch}/{args.epoches}] CloseTrue-VMax "
            f"mean={close_logit_margin.get('mean', 0.0):.4f} std={close_logit_margin.get('std', 0.0):.4f} "
            f"p10={close_logit_margin.get('p10', 0.0):.4f} p50={close_logit_margin.get('p50', 0.0):.4f} "
            f"p90={close_logit_margin.get('p90', 0.0):.4f} | "
            f"OpenKnown-VMax mean={open_logit_margin.get('mean', 0.0):.4f} std={open_logit_margin.get('std', 0.0):.4f} "
            f"p10={open_logit_margin.get('p10', 0.0):.4f} p50={open_logit_margin.get('p50', 0.0):.4f} "
            f"p90={open_logit_margin.get('p90', 0.0):.4f}"
        )

        if args.anchor_log_interval > 0 and epoch % args.anchor_log_interval == 0:
            diagnostics = _build_anchor_diagnostics(args, epoch, server_model, initial_virtual_weights, val_result, osr_result)
            _append_anchor_log(args, diagnostics)
            _print_anchor_diagnostics(epoch, args, diagnostics)

        if osr_oscr > best_oscr:
            best_oscr = osr_oscr
            best_epoch = epoch
            best_osr_acc, best_osr_f1, best_osr_recall, best_osr_precision = osr_acc, osr_f1, osr_recall, osr_precision
            best_osr_unk, best_osr_os_star, best_osr_hos, best_osr_auroc, best_osr_aupr, best_osr_oscr = osr_unk, osr_os_star, osr_hos, osr_auroc, osr_aupr, osr_oscr
            #server
            state = {
                'net': server_model.state_dict(),
                }
            name_model = 'best_ckpt_'+args.mode+'_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'.pth'
            torch.save(state, osp.join(args.save_path,name_model))
            #clients
            for clint_idx, mo in enumerate(models):
                state = {
                    'net': mo.state_dict(),
                    }
                name_model = 'best_ckpt_'+args.mode+'_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'_C_'+str(clint_idx)+'.pth'
                torch.save(state, osp.join(args.save_path,name_model))
            print(f'Saving best model . . . . . . . .')
            print()

        # Save checkpoint at specified interval
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            #server
            state = {
                'net': server_model.state_dict(),
                }
            name_model = 'ckpt_'+args.mode+'_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'_epoch_'+str(epoch)+'.pth'
            torch.save(state, osp.join(args.save_path,name_model))
            #clients
            for clint_idx, mo in enumerate(models):
                state = {
                    'net': mo.state_dict(),
                    }
                name_model = 'ckpt_'+args.mode+'_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'_C_'+str(clint_idx)+'_epoch_'+str(epoch)+'.pth'
                torch.save(state, osp.join(args.save_path,name_model))
            print(f'Saving model at epoch {epoch} . . . . . . . .')
            print()

        epoch += 1

    print('------>Best performance (by OSCR)--->>>>>>>')
    print()
    print(f"Test-  OSR [{best_epoch}/{args.epoches}] ACC={best_osr_acc:.3f} F1={best_osr_f1:.3f} Rec={best_osr_recall:.3f} Prec={best_osr_precision:.3f} UNK={best_osr_unk:.3f} OS*={best_osr_os_star:.3f} HOS={best_osr_hos:.3f} AUROC={best_osr_auroc:.3f} AUPR={best_osr_aupr:.3f} OSCR={best_osr_oscr:.3f}")
    print('=====================================================================================================================================')
