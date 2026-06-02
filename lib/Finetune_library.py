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
from random import sample
import gc


def known_unknown_rank_loss(args, outputs_known, outputs_unknown):
    """Penalize when known samples have lower max-known-prob than unknown samples."""
    prob_known = torch.softmax(outputs_known, dim=-1)
    prob_unknown = torch.softmax(outputs_unknown, dim=-1)
    known_score = prob_known[:, :args.known_class].max(dim=1)[0]
    unknown_known_score = prob_unknown[:, :args.known_class].max(dim=1)[0]
    return torch.relu(args.rank_margin - known_score.mean() + unknown_known_score.mean())


def prepare_lups_feature(args, feats):
    """Pool feature maps for LUPS to reduce dimension."""
    if args.lups_space == 'fullmap':
        return feats
    if args.lups_space == 'pooled':
        return F.adaptive_avg_pool2d(feats, (args.lups_pool_size, args.lups_pool_size))
    if args.lups_space == 'gap':
        return F.adaptive_avg_pool2d(feats, (1, 1))
    raise ValueError(args.lups_space)

def train(args, device, epoch, net, trainloader, optimizer, net_peers=None, attack = None, unknown_dis = None):
    net.train()
    for peer_net in net_peers:        
        peer_net.eval()    
    train_loss = 0
    pred_list = []
    label_list = []
    output_list = []
    criterion = nn.CrossEntropyLoss()
    
    net_peers_sample_number = args.client_num-1
    if args.dataset == 'Hyperkvasir':
        if args.unknown_class == 3:
            p_lower = 0
            p_upper = 1.
        if args.unknown_class == 9:
            p_lower = 0
            p_upper = 1.

    if args.dataset == 'RetinalOCT':
        if args.unknown_class == 3:
            p_lower = 0
            p_upper = 1.
        if args.unknown_class == 5:
            p_lower = 0
            p_upper = 1.

    if args.dataset == 'ISIC':
        if args.unknown_class == 3:
            p_lower = 0
            p_upper = 1.
        if args.unknown_class == 5:
            p_lower = 0
            p_upper = 1.

    if args.dataset == 'Bloodmnist':
        if args.unknown_class == 3:
            p_lower = 0
            p_upper = 13./16
        if args.unknown_class == 5:            
            p_lower = 0
            p_upper = 14./16
    
    if args.dataset == 'OrganMNIST3D':
        if args.unknown_class == 4:
            p_lower = 0
            p_upper = 1.
        if args.unknown_class == 7:            
            p_lower = 0
            p_upper = 1.    
    unknown_dict = [None for i in range(args.virtue_num)]
    mean_dict = [None for i in range(args.virtue_num)]
    cov_dict = [None for i in range(args.virtue_num)]
    var_dict = [None for i in range(args.virtue_num)]
    number_dict = torch.zeros(args.virtue_num)
    for batch_idx, (inputs, targets, img_dirs) in enumerate(trainloader):
        gc.collect()
        torch.cuda.empty_cache()
        inputs, targets = inputs.to(device), targets.long().to(device)        
        outs = net(inputs)
        outputs = outs['outputs']    
        aux_outputs = outs['aux_out']
        boundary_feats = outs['boundary_feats'] 
        discrete_feats = outs['discrete_feats']
        loss = criterion(outputs, targets)        
        loss += criterion(aux_outputs, targets) 
        if epoch>=0:      
            #Client Inconsistency-based Boundary Samples Recognition 
            net_peers_sample = sample(net_peers, net_peers_sample_number) # 随机抽取n个peer        
            _, aux_pred = aux_outputs.max(1)
            aux_preds_peers = torch.eq(aux_pred, targets).float()
            assert len(net_peers)== (args.client_num-1)        
            for idx, peer_net in enumerate(net_peers_sample):
                with torch.no_grad():
                    outs_peer = peer_net.aux_forward(boundary_feats.clone().detach())
                    aux_out_peer = outs_peer['aux_out']
                    _, aux_pred_peer = aux_out_peer.max(1)
                    aux_preds_peers += torch.eq(aux_pred_peer, targets).float()
            is_boundary_upper = torch.lt(aux_preds_peers/(net_peers_sample_number+1), p_upper)
            is_boundary_lower = torch.gt(aux_preds_peers/(net_peers_sample_number+1), p_lower)
            is_boundary = is_boundary_lower & is_boundary_upper    

            if (is_boundary.sum()>0 and (args.dataset=='Hyperkvasir' or args.dataset=='RetinalOCT' or args.dataset=='ISIC')) or (is_boundary.sum()>1 and (args.dataset=='Bloodmnist' or args.dataset=='OrganMNIST3D')): #  batchnorm error when batchsize = 1               
                discrete_feats = discrete_feats[is_boundary]
                discrete_targets = targets[is_boundary]
                inputs_unknown, targets_unknown = attack.i_DUS(net, discrete_feats, discrete_targets, net_peers_sample)
                if inputs_unknown is not None:                                        
                    outs_unknown = net.discrete_forward(inputs_unknown.clone().detach()) 
                    outputs_unknown = outs_unknown['outputs']
                    # probabilistic distance
                    prob_unknown = torch.softmax(outputs_unknown,dim=-1)
                    PDs = prob_unknown[:,-1] - prob_unknown[:,:-1].max(-1)[0]
                    # Multi-virtual target assignment (deterministic)
                    virtual_idx = targets_unknown % args.virtue_num
                    virtual_targets = (args.known_class + virtual_idx).long().to(device)
                    loss += criterion(outputs_unknown, virtual_targets) * args.lups_local_weight
                    loss += args.rank_weight * known_unknown_rank_loss(args, outputs, outputs_unknown)          
                    
                    #start save unknown data
                    if epoch in args.start_epoch:
                        targets_unknown_numpy = targets_unknown.cpu().data.numpy() 
                        for index in range(len(targets_unknown)):
                            if ((args.dataset=='Hyperkvasir' or args.dataset=='RetinalOCT' or args.dataset=='ISIC') and PDs[index]>0) or ((args.dataset=='Bloodmnist' or args.dataset=='OrganMNIST3D') and PDs[index]>-1):
                                dict_key = int(targets_unknown_numpy[index]) % args.virtue_num
                                unknown_feat = inputs_unknown[index].clone().detach()
                                if args.lups_space != 'fullmap':
                                    unknown_feat = prepare_lups_feature(args, unknown_feat)
                                unknown_sample = unknown_feat.view(1, -1)
                                if unknown_dict[dict_key] is None:
                                    unknown_dict[dict_key] = unknown_sample
                                else:
                                    unknown_dict[dict_key] = torch.cat((unknown_dict[dict_key], unknown_sample),dim=0)
                    if unknown_dis is not None:
                        sample_c = torch.randint(0, args.virtue_num, (args.sample_from,))
                        sample_num = {index: 0 for index in range(args.virtue_num)}
                        for it in sample_c:
                            sample_num[it.item()] = sample_num[it.item()] + 1
                        ood_samples = None
                        ood_targets = None
                        for index in range(args.virtue_num):
                            if sample_num[index] > 0 and unknown_dis[index] is not None:
                                if args.lups_mode == 'diag':
                                    d = unknown_dis[index]
                                    mean = d['mean'].to(device)
                                    var = d['var'].to(device)
                                    eps = torch.randn(args.lups_candidates, mean.shape[0], device=device)
                                    z = mean + torch.sqrt(var + 1e-8) * eps
                                    if args.lups_sample_strategy == 'low_density':
                                        score = ((z - mean) ** 2 / (var + 1e-8)).sum(dim=1)
                                        _, idx = torch.topk(score, sample_num[index])
                                    else:
                                        idx = torch.randperm(args.lups_candidates, device=device)[:sample_num[index]]
                                    generated_unknown_samples = z[idx]
                                else:
                                    generated_unknown_samples = unknown_dis[index].rsample((args.lups_candidates,))
                                    prob_density = unknown_dis[index].log_prob(generated_unknown_samples)
                                    _, index_prob = torch.topk(-prob_density, sample_num[index])
                                    generated_unknown_samples = generated_unknown_samples[index_prob].to(device)

                                # Reshape for discrete_forward
                                if args.dataset == 'OrganMNIST3D':
                                    p = args.lups_pool_size if args.lups_space == 'pooled' else 2
                                    generated_unknown_samples = generated_unknown_samples.reshape(sample_num[index], 256, p, p, p)
                                else:
                                    p = args.lups_pool_size if args.lups_space == 'pooled' else (2 if args.dataset == 'Bloodmnist' else 8)
                                    generated_unknown_samples = generated_unknown_samples.reshape(sample_num[index], 256, p, p)
                                generated_unknown_targets = (torch.ones(sample_num[index]) * index).long().to(device) 
                                if ood_samples is None:
                                    ood_samples = generated_unknown_samples
                                    ood_targets = generated_unknown_targets
                                else:
                                    ood_samples = torch.cat((ood_samples, generated_unknown_samples), 0) 
                                    ood_targets = torch.cat((ood_targets, generated_unknown_targets), 0)
                                del generated_unknown_samples
                        if ood_samples is not None and ood_samples.shape[0]>1:
                            outs_unknown = net.discrete_forward(ood_samples.clone().detach())
                            outputs_unknown = outs_unknown['outputs']
                            # Multi-virtual target for global FOSS samples
                            global_virtual_targets = (args.known_class + ood_targets).long().to(device)
                            loss += criterion(outputs_unknown, global_virtual_targets) * args.lups_global_weight
                            loss += args.rank_weight * known_unknown_rank_loss(args, outputs, outputs_unknown)                                
                                
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()     
        train_loss += loss.item()
        _, predicted = outputs[:, :args.known_class].max(1)
        
        pred_list.extend(predicted.cpu().numpy().tolist())
        label_list.extend(targets.cpu().numpy().tolist())    
        output_list.append(torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy())
    
        del inputs, loss
        gc.collect()
        if args.max_train_batches > 0 and (batch_idx + 1) >= args.max_train_batches:
            break

    if epoch in args.start_epoch:
        fallback_dim = None
        for item in unknown_dict:
            if item is not None:
                fallback_dim = item.shape[1]
                break
        if fallback_dim is None:
            with torch.no_grad():
                fallback_feat = prepare_lups_feature(args, discrete_feats[:1])
                fallback_dim = fallback_feat.reshape(1, -1).shape[1]

        for index in range(args.virtue_num):
            if unknown_dict[index] is not None:
                mean_dict[index] = unknown_dict[index].mean(0).cpu()
                number_dict[index] = len(unknown_dict[index])
                if args.lups_mode == 'diag':
                    X = unknown_dict[index] - unknown_dict[index].mean(0)
                    var_dict[index] = X.var(0, unbiased=False).cpu()
                    del X
                else:
                    X = unknown_dict[index] - unknown_dict[index].mean(0)
                    cov_matrix = torch.mm(X.t(), X) / len(X)
                    cov_dict[index] = cov_matrix.cpu()
                    del cov_matrix, X
            else:
                mean_dict[index] = torch.zeros(fallback_dim)
                if args.lups_mode == 'diag':
                    var_dict[index] = torch.ones(fallback_dim)
                else:
                    cov_dict[index] = torch.zeros(fallback_dim, fallback_dim)
        del unknown_dict
        gc.collect()

        mean_dict = torch.stack(mean_dict, dim=0)
        if args.lups_mode == 'diag':
            var_dict = torch.stack(var_dict, dim=0)
        else:
            cov_dict = torch.stack(cov_dict, dim=0)

    for peer_net in net_peers:        
        peer_net.train()          

    loss_avg = train_loss/(batch_idx+1)
    mean_acc = 100*metrics.accuracy_score(label_list, pred_list)
    precision = 100*metrics.precision_score(label_list, pred_list, average='macro')    
    recall_macro = 100*metrics.recall_score(y_true=label_list, y_pred=pred_list, average='macro')      
    f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')    

    result = {'loss':loss_avg,
              'acc':mean_acc,
              'f1': f1_macro,
              'recall':recall_macro,
              'precision': precision,
              'mean_dict': mean_dict,
              'cov_dict': cov_dict if args.lups_mode == 'fullcov' else None,
              'var_dict': var_dict if args.lups_mode == 'diag' else None,
              'number_dict': number_dict
              }
    return result

def val(args, device, epoch, net, valloader):
    net.eval()
    
    val_loss = 0
    pred_list = []
    label_list = []
    criterion = nn.CrossEntropyLoss()
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
            if args.max_eval_batches > 0 and (batch_idx + 1) >= args.max_eval_batches:
                break

        loss_avg = val_loss/(batch_idx+1)
        mean_acc = 100*metrics.accuracy_score(label_list, pred_list)
        precision = 100*metrics.precision_score(label_list, pred_list, average='macro')        
        recall_macro = 100*metrics.recall_score(y_true=label_list, y_pred=pred_list, average='macro')      
        f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')    
        confusion_matrix = metrics.confusion_matrix(y_true=label_list, y_pred=pred_list)   
        
        result = {'loss':loss_avg,
                      'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision': precision,
                      'confusion_matrix':confusion_matrix,
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
            if args.max_eval_batches > 0 and (batch_idx + 1) >= args.max_eval_batches:
                break

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
        
        prob_total = None
        prob_known_list = []  # Store known class probabilities for AUROC/AUPR

        for batch_idx, (inputs, targets, img_dirs) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            prob=nn.functional.softmax(outputs/temperature,dim=-1)
            if prob_total == None:
                prob_total = prob
            else:
                prob_total = torch.cat([prob_total, prob])
            # Store max probability over known classes for AUROC/AUPR
            prob_known_list.append(prob[:, :args.known_class].max(1)[0].cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            if args.max_eval_batches > 0 and (batch_idx + 1) >= args.max_eval_batches:
                break

        for batch_idx, (inputs, targets, img_dirs) in enumerate(openloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            prob=nn.functional.softmax(outputs/temperature,dim=-1)
            prob_total = torch.cat([prob_total, prob])
            # Store max probability over known classes for AUROC/AUPR
            prob_known_list.append(prob[:, :args.known_class].max(1)[0].cpu().numpy())

            targets = np.ones_like(targets.cpu().numpy())*args.known_class
            targets_list.append(targets)
            if args.max_eval_batches > 0 and (batch_idx + 1) >= args.max_eval_batches:
                break

        # openset recognition
        targets_list=np.reshape(np.array(targets_list),(-1))
        _, pred_list = prob_total.max(1)
        pred_list = pred_list.cpu().numpy()

        # Concatenate all probabilities
        prob_known_array = np.concatenate(prob_known_list)

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

        osr_result = {'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision':precision,
                      'unk': unk_recall,
                      'os_star': os_star,
                      'hos': hos,
                      'auroc': auroc,
                      'aupr': aupr,
                      'oscr': oscr}

    return osr_result, close_test_result

