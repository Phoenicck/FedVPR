# -*- coding: utf-8 -*-
"""
ISIC 2019 dataset loader for FedOSS.
Structure: ISIC_2019_Training_Input/*.jpg + ISIC_2019_Training_GroundTruth.csv
8 classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC (UNK has 0 samples, excluded)
"""
import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import csv
import platform
from collections import Counter

ISIC_CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
TOTAL_CLASS = 8


def dirichlet_split_noniid(train_labels, alpha, n_clients, state):
    n_classes = train_labels.max() + 1
    np.random.set_state(state)
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


def get_dataloaders(client_num, data_root, seed, param=None):
    if param is None:
        param = {'Known_class': 5, 'unKnown_class': 3, 'Rotation': 45,
                 'Resize': 144, 'CropSize': 128, 'Batchsize': 8, 'dirichlet': 0.5}

    known_class = param['Known_class']
    unknown_class = param['unKnown_class']
    batchsize = param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    assert known_class + unknown_class <= TOTAL_CLASS

    np.random.seed(seed)
    state = np.random.get_state()

    # Randomly pick known and unknown classes
    class_candidates = np.arange(TOTAL_CLASS)
    np.random.shuffle(class_candidates)
    known_class_list = class_candidates[:known_class]
    unknown_class_list = class_candidates[known_class:known_class + unknown_class]

    print('Known class list:', known_class_list, 'Unknown class list', unknown_class_list)

    # Read CSV ground truth
    csv_path = osp.join(data_root, 'ISIC_2019_Training_GroundTruth.csv')
    all_images = []
    all_labels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_idx = None
            for i, c in enumerate(ISIC_CLASSES):
                if float(row[c]) == 1.0:
                    cls_idx = i
                    break
            if cls_idx is None:
                continue
            if cls_idx in known_class_list or cls_idx in unknown_class_list:
                all_images.append(row['image'] + '.jpg')
                all_labels.append(cls_idx)

    print('Total selected images: {}'.format(len(all_images)))

    # Shuffle and split 70/10/20
    np.random.set_state(state)
    idx = np.arange(len(all_images))
    np.random.shuffle(idx)
    all_images = np.array(all_images)[idx]
    all_labels = np.array(all_labels)[idx]

    train_n = int(len(all_images) * 0.70)
    val_n = int(len(all_images) * 0.10)
    trainx, trainy = all_images[:train_n], all_labels[:train_n]
    valx, valy = all_images[train_n:train_n + val_n], all_labels[train_n:train_n + val_n]
    testx, testy = all_images[train_n + val_n:], all_labels[train_n + val_n:]

    print('Train: {}, Val: {}, Test: {}'.format(len(trainx), len(valx), len(testx)))
    d = Counter(trainy)
    print('Train class distribution:', sorted(d.items()))

    # Relabel to contiguous indices
    knowndict = {known_class_list[i]: i for i in range(known_class)}
    unknowndict = {unknown_class_list[j]: j + known_class for j in range(len(unknown_class_list))}
    print(knowndict, unknowndict)

    trainy = np.array(trainy)
    valy = np.array(valy)
    testy = np.array(testy)
    copytrainy = copy.deepcopy(trainy)
    copyvaly = copy.deepcopy(valy)
    copytesty = copy.deepcopy(testy)

    for orig, new in knowndict.items():
        trainy[copytrainy == orig] = new
        valy[copyvaly == orig] = new
        testy[copytesty == orig] = new
    for orig, new in unknowndict.items():
        trainy[copytrainy == orig] = new
        valy[copyvaly == orig] = new
        testy[copytesty == orig] = new

    new_known = np.arange(known_class)
    print('Known origin:', known_class_list, ', new:', new_known)
    print('Unknown origin:', unknown_class_list, ', new:', np.arange(known_class, known_class + len(unknown_class_list)))

    # Build known-class indices
    def known_indices(labels_arr):
        idxs = []
        for item in new_known:
            idxs.extend(list(np.where(labels_arr == item)[0]))
        return idxs

    train_known_idx = known_indices(trainy)
    val_known_idx = known_indices(valy)
    test_known_idx = known_indices(testy)

    train_unknown_idx = list(set(range(len(trainy))) - set(train_known_idx))
    val_unknown_idx = list(set(range(len(valy))) - set(val_known_idx))
    test_unknown_idx = list(set(range(len(testy))) - set(test_known_idx))

    print('Known/Unknown in Train: {}/{}'.format(len(train_known_idx), len(train_unknown_idx)))
    print('Known/Unknown in Val:   {}/{}'.format(len(val_known_idx), len(val_unknown_idx)))
    print('Known/Unknown in Test:  {}/{}'.format(len(test_known_idx), len(test_unknown_idx)))

    assert len(test_unknown_idx) + len(test_known_idx) == len(testy)

    num_workers = 0 if platform.system() == 'Windows' else 4

    train_labels_np = trainy[train_known_idx]
    train_images = [trainx[idx] for idx in train_known_idx]
    client_idcs = dirichlet_split_noniid(train_labels_np, alpha=dirichlet_alpha, n_clients=client_num, state=state)

    img_dir = osp.join(data_root, 'ISIC_2019_Training_Input')
    trainloaders = []
    train_val_loaders = []

    for i in range(client_num):
        print('Client{} sample num: {}'.format(i, len(client_idcs[i])))
        sub_idx = client_idcs[i]
        client_trainset = ISICDataset(img_dir, sub_idx, 'train', train_images, train_labels_np, param)
        trainloaders.append(torch.utils.data.DataLoader(
            client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True))

        client_valset = ISICDataset(img_dir, sub_idx, 'train_val', train_images, train_labels_np, param)
        train_val_loaders.append(torch.utils.data.DataLoader(
            client_valset, batch_size=1, shuffle=False, num_workers=num_workers))

    valset = ISICDataset(img_dir, val_known_idx, 'valclose', valx, valy, param)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)

    closeset = ISICDataset(img_dir, test_known_idx, 'testclose', testx, testy, param)
    closeloader = torch.utils.data.DataLoader(closeset, batch_size=1, shuffle=False, num_workers=num_workers)

    openset = ISICDataset(img_dir, test_unknown_idx, 'testopen', testx, testy, param)
    openloader = torch.utils.data.DataLoader(openset, batch_size=1, shuffle=False, num_workers=num_workers)

    return trainloaders, valloader, closeloader, openloader, train_val_loaders


class ISICDataset(Dataset):
    def __init__(self, img_dir, data_index, setname, datax, datay, param=None):
        if param is None:
            param = {'Known_class': 5, 'unKnown_class': 3, 'Rotation': 45, 'Resize': 144, 'CropSize': 128}
        self.img_dir = img_dir
        self.data_index = data_index
        self.setname = setname
        self.datax = datax
        self.datay = datay

        self.transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=param['Rotation'], shear=5.729578),
            transforms.Resize((param['Resize'], param['Resize'])),
            transforms.RandomCrop((param['CropSize'], param['CropSize'])),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((param['CropSize'], param['CropSize'])),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        img_id = self.data_index[index]
        img_name = self.datax[img_id]
        label = self.datay[img_id]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        if self.setname == 'train':
            image = self.transform_train(img)
        else:
            image = self.transform_test(img)
        return image, label, img_name
