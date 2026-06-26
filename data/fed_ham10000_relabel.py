# -*- coding: utf-8 -*-
"""
HAM10000 dataset loader for FedOSS.
Structure: HAM10000/{HAM10000_images_part_1,HAM10000_images_part_2,HAM10000_metadata.csv}
7 classes: nv, mel, bkl, bcc, akiec, vasc, df.
Lesion-level stratified 70/10/20 split to prevent data leakage.
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
import random
import platform
from collections import Counter

HAM10000_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
HAM10000_CLASS_TO_IDX = {name: i for i, name in enumerate(HAM10000_CLASSES)}
TOTAL_CLASS = 7

HAM10000_PROTOCOLS = {
    'easy': {
        # Easy: keep the five largest classes as known, rare classes as unknown
        'known': ['nv', 'mel', 'bkl', 'bcc', 'akiec'],
        'unknown': ['vasc', 'df'],
    },
    'hard': {
        # Hard: melanoma (most important clinically) becomes unknown
        'known': ['nv', 'bkl', 'bcc', 'akiec', 'vasc'],
        'unknown': ['mel', 'df'],
    },
}


def _build_image_folder_lookup(data_root):
    """Map image_id -> folder name (HAM10000_images_part_1 or part_2)."""
    lookup = {}
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        folder_path = osp.join(data_root, folder)
        if not osp.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith('.jpg'):
                image_id = fname.replace('.jpg', '')
                lookup[image_id] = folder
    return lookup


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


def resolve_protocol(param):
    protocol_mode = param.get('protocol_mode', 'random')
    known_class = param['Known_class']
    unknown_class = param['unKnown_class']

    if protocol_mode in HAM10000_PROTOCOLS:
        spec = HAM10000_PROTOCOLS[protocol_mode]
        known_class_list = np.array([HAM10000_CLASS_TO_IDX[name] for name in spec['known']], dtype=np.int64)
        unknown_class_list = np.array([HAM10000_CLASS_TO_IDX[name] for name in spec['unknown']], dtype=np.int64)
        if len(known_class_list) != known_class or len(unknown_class_list) != unknown_class:
            raise ValueError(
                f"HAM10000 {protocol_mode} protocol expects K={len(known_class_list)}, U={len(unknown_class_list)} "
                f"but got K={known_class}, U={unknown_class}"
            )
        print(f'Protocol mode: {protocol_mode}')
        print('Fixed known classes:', spec['known'])
        print('Fixed unknown classes:', spec['unknown'])
        return known_class_list, unknown_class_list

    class_candidates = np.arange(TOTAL_CLASS)
    np.random.shuffle(class_candidates)
    known_class_list = class_candidates[:known_class]
    unknown_class_list = class_candidates[known_class:known_class + unknown_class]
    print('Protocol mode: random')
    return known_class_list, unknown_class_list


def get_dataloaders(client_num, data_root, seed, param=None):
    if param is None:
        param = {'Known_class': 5, 'unKnown_class': 2, 'Rotation': 45,
                 'Resize': 144, 'CropSize': 128, 'Batchsize': 8, 'dirichlet': 0.5,
                 'protocol_mode': 'random'}

    known_class = param['Known_class']
    unknown_class = param['unKnown_class']
    batchsize = param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    assert known_class + unknown_class <= TOTAL_CLASS

    np.random.seed(seed)
    state = np.random.get_state()
    rng = random.Random(seed)

    known_class_list, unknown_class_list = resolve_protocol(param)
    selected_classes = set(known_class_list) | set(unknown_class_list)
    print('Known class list:', known_class_list, 'Unknown class list', unknown_class_list)

    # ---- 1. Read metadata and build image folder lookup ----
    folder_lookup = _build_image_folder_lookup(data_root)

    csv_path = osp.join(data_root, 'HAM10000_metadata.csv')
    lesions = {}  # lesion_id -> list of (image_id, class_idx)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dx = row['dx']
            if dx not in HAM10000_CLASS_TO_IDX:
                continue
            cls_idx = HAM10000_CLASS_TO_IDX[dx]
            if cls_idx not in selected_classes:
                continue
            image_id = row['image_id']
            if image_id not in folder_lookup:
                continue
            lesion_id = row['lesion_id']
            if lesion_id not in lesions:
                lesions[lesion_id] = []
            lesions[lesion_id].append((image_id, cls_idx))

    print(f'Loaded {sum(len(v) for v in lesions.values())} images from {len(lesions)} lesions')
    print(f'Class distribution: {sorted(Counter(c for v in lesions.values() for _, c in v).items())}')

    # ---- 2. Lesion-level stratified 70/10/20 split ----
    lesion_list = list(lesions.items())
    rng.shuffle(lesion_list)

    # Group lesions by class of first image
    by_class = {c: [] for c in HAM10000_CLASS_TO_IDX.values() if c in selected_classes}
    for lesion_id, images in lesion_list:
        cls_idx = images[0][1]
        if cls_idx in by_class:
            by_class[cls_idx].append((lesion_id, images))

    train_entries, val_entries, test_entries = [], [], []
    for cls_idx in sorted(by_class.keys()):
        items = by_class[cls_idx]
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.10)) if n >= 3 else 0
        train_entries.extend(items[:n_train])
        val_entries.extend(items[n_train:n_train + n_val])
        test_entries.extend(items[n_train + n_val:])
    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    rng.shuffle(test_entries)

    # Flatten: (image_id, folder, class_idx)
    def flatten(entries):
        result = []
        for lesion_id, images in entries:
            for image_id, cls_idx in images:
                result.append((image_id, folder_lookup[image_id], cls_idx))
        return result

    train_items = flatten(train_entries)
    val_items = flatten(val_entries)
    test_items = flatten(test_entries)

    print(f'Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}')
    print(f'Train class distribution: {sorted(Counter(c for _, _, c in train_items).items())}')

    train_labels_all = np.array([c for _, _, c in train_items])
    val_labels_all = np.array([c for _, _, c in val_items])
    test_labels_all = np.array([c for _, _, c in test_items])

    # ---- 3. Relabel to contiguous indices ----
    knowndict = {known_class_list[i]: i for i in range(known_class)}
    unknowndict = {unknown_class_list[j]: j + known_class for j in range(len(unknown_class_list))}
    print(knowndict, unknowndict)

    copytrainy = copy.deepcopy(train_labels_all)
    copyvaly = copy.deepcopy(val_labels_all)
    copytesty = copy.deepcopy(test_labels_all)

    for orig, new in knowndict.items():
        train_labels_all[copytrainy == orig] = new
        val_labels_all[copyvaly == orig] = new
        test_labels_all[copytesty == orig] = new
    for orig, new in unknowndict.items():
        train_labels_all[copytrainy == orig] = new
        val_labels_all[copyvaly == orig] = new
        test_labels_all[copytesty == orig] = new

    new_known = np.arange(known_class)
    print('Known origin:', known_class_list, ', new:', new_known)
    print('Unknown origin:', unknown_class_list, ', new:', np.arange(known_class, known_class + len(unknown_class_list)))

    # Build known-class indices
    def known_indices(labels_arr):
        idxs = []
        for item in new_known:
            idxs.extend(list(np.where(labels_arr == item)[0]))
        return idxs

    train_known_idx = known_indices(train_labels_all)
    val_known_idx = known_indices(val_labels_all)
    test_known_idx = known_indices(test_labels_all)

    train_unknown_idx = list(set(range(len(train_labels_all))) - set(train_known_idx))
    val_unknown_idx = list(set(range(len(val_labels_all))) - set(val_known_idx))
    test_unknown_idx = list(set(range(len(test_labels_all))) - set(test_known_idx))

    print('Known/Unknown in Train: {}/{}'.format(len(train_known_idx), len(train_unknown_idx)))
    print('Known/Unknown in Val:   {}/{}'.format(len(val_known_idx), len(val_unknown_idx)))
    print('Known/Unknown in Test:  {}/{}'.format(len(test_known_idx), len(test_unknown_idx)))

    assert len(test_unknown_idx) + len(test_known_idx) == len(test_labels_all)

    num_workers = 0 if platform.system() == 'Windows' else 4

    train_labels_known = train_labels_all[train_known_idx]
    train_items_known = [train_items[idx] for idx in train_known_idx]
    client_idcs = dirichlet_split_noniid(train_labels_known, alpha=dirichlet_alpha, n_clients=client_num, state=state)

    trainloaders = []
    train_val_loaders = []

    for i in range(client_num):
        print('Client{} sample num: {}'.format(i, len(client_idcs[i])))
        sub_idx = client_idcs[i]
        client_trainset = HAM10000Dataset(data_root, sub_idx, 'train', train_items_known, train_labels_known, param)
        trainloaders.append(torch.utils.data.DataLoader(
            client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True))

        client_valset = HAM10000Dataset(data_root, sub_idx, 'train_val', train_items_known, train_labels_known, param)
        train_val_loaders.append(torch.utils.data.DataLoader(
            client_valset, batch_size=1, shuffle=False, num_workers=num_workers))

    valset = HAM10000Dataset(data_root, val_known_idx, 'valclose', val_items, val_labels_all, param)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)

    closeset = HAM10000Dataset(data_root, test_known_idx, 'testclose', test_items, test_labels_all, param)
    closeloader = torch.utils.data.DataLoader(closeset, batch_size=1, shuffle=False, num_workers=num_workers)

    openset = HAM10000Dataset(data_root, test_unknown_idx, 'testopen', test_items, test_labels_all, param)
    openloader = torch.utils.data.DataLoader(openset, batch_size=1, shuffle=False, num_workers=num_workers)

    return trainloaders, valloader, closeloader, openloader, train_val_loaders


class HAM10000Dataset(Dataset):
    def __init__(self, data_root, data_index, setname, data_items, datay, param=None):
        if param is None:
            param = {'Known_class': 5, 'unKnown_class': 2, 'Rotation': 45, 'Resize': 144, 'CropSize': 128}
        self.data_root = data_root
        self.data_index = data_index
        self.setname = setname
        self.data_items = data_items  # list of (image_id, folder, class_idx)
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
        img_idx = self.data_index[index]
        image_id, folder, _label = self.data_items[img_idx]
        label = self.datay[img_idx]
        img_path = osp.join(self.data_root, folder, f'{image_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.setname in ('train', 'train_val'):
            image = self.transform_train(img)
        else:
            image = self.transform_test(img)
        return image, label, img_path


if __name__ == '__main__':
    from collections import Counter
    datadir = './datasets/HAM10000'
    trainloaders, valloader, closeloader, openloader, train_val_loaders = get_dataloaders(8, datadir, 0)

    for c, trainloader in enumerate(trainloaders):
        print(trainloader.dataset.__len__())
        labels = []
        for i, data_ in enumerate(trainloader):
            img, label, img_dir = data_
            labels += label.data.tolist()
        d = Counter(labels)
        d_s = sorted(d.items(), key=lambda x: x[1], reverse=True)
        print('Client {} class distribution:'.format(c), d_s)
