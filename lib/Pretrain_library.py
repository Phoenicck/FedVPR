# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:03:23 2022

@author: ZML
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics

def train(args, device, epoch, net, trainloader, optimizer):
    net.train()  
    train_loss = 0
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
        loss = criterion(outputs, targets)        
        loss += criterion(aux_outputs, targets)  
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs[:, :args.known_class].max(1)

        pred_list.extend(predicted.cpu().numpy().tolist())
        label_list.extend(targets.cpu().numpy().tolist())    
        output_list.append(torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy())
        
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



