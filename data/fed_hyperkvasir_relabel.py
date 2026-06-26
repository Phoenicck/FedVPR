# -*- coding: utf-8 -*-
"""
HyperKvasir 15-class dataset loader for FedOSS.
Reads preprocessed CSV (hyperkvasir_15class.csv) with train/val/test split.
15 merged classes from the FedOSS protocol.
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

HK_CLASSES = [
    'bbps', 'polyps', 'cecum', 'dyed-lifted-polyps', 'pylorus',
    'dyed-resection-margins', 'z-line', 'ulcerative-colitis',
    'retroflex-stomach', 'esophagitis', 'retroflex-rectum',
    'impacted-stool', 'barretts', 'ileum', 'hemorrhoids',
]
HK_CLASS_TO_IDX = {name: i for i, name in enumerate(HK_CLASSES)}
TOTAL_CLASS = 15

HK_PROTOCOLS = {
    'easy': {
        # Easy: 6 large classes as known, 9 as unknown (including all tail classes)
        'known': ['bbps', 'polyps', 'cecum', 'dyed-lifted-polyps', 'pylorus', 'dyed-resection-margins'],
        'unknown': ['z-line', 'ulcerative-colitis', 'retroflex-stomach', 'esophagitis',
                     'retroflex-rectum', 'impacted-stool', 'barretts', 'ileum', 'hemorrhoids'],
    },
    'hard': {
        # Hard: anatomically similar classes split across known/unknown
        'known': ['bbps', 'polyps', 'cecum', 'z-line', 'pylorus', 'retroflex-stomach'],
        'unknown': ['dyed-lifted-polyps', 'dyed-resection-margins', 'ulcerative-colitis',
                     'esophagitis', 'retroflex-rectum', 'impacted-stool', 'barretts', 'ileum', 'hemorrhoids'],
    },
}


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

    if protocol_mode in HK_PROTOCOLS:
        spec = HK_PROTOCOLS[protocol_mode]
        known_class_list = np.array([HK_CLASS_TO_IDX[name] for name in spec['known']], dtype=np.int64)
        unknown_class_list = np.array([HK_CLASS_TO_IDX[name] for name in spec['unknown']], dtype=np.int64)
        if len(known_class_list) != known_class or len(unknown_class_list) != unknown_class:
            raise ValueError(
                f"HyperKvasir {protocol_mode} protocol expects K={len(known_class_list)}, U={len(unknown_class_list)} "
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
        param = {'Known_class': 6, 'unKnown_class': 9, 'Rotation': 45,
                 'Resize': 144, 'CropSize': 128, 'Batchsize': 8, 'dirichlet': 0.5,
                 'protocol_mode': 'random'}

    known_class = param['Known_class']
    unknown_class = param['unKnown_class']
    batchsize = param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    assert known_class + unknown_class <= TOTAL_CLASS

    np.random.seed(seed)
    state = np.random.get_state()

    known_class_list, unknown_class_list = resolve_protocol(param)
    selected_classes = set(known_class_list) | set(unknown_class_list)
    print('Known class list:', known_class_list, 'Unknown class list', unknown_class_list)

    # ---- 1. Read preprocessed CSV ----
    csv_path = osp.join(data_root, 'hyperkvasir_15class.csv')
    all_items = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_idx = int(row['label_15class_id'])
            if cls_idx not in selected_classes:
                continue
            split = row['split']
            all_items.setdefault(split, []).append({
                'img_path': row['img_path'],
                'label_id': cls_idx,
                'label_name': row['label_15class'],
            })

    train_items = all_items.get('train', [])
    val_items = all_items.get('val', [])
    test_items = all_items.get('test', [])

    print(f'Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}')
    print(f'Train class dist: {sorted(Counter(r["label_id"] for r in train_items).items())}')

    train_labels_all = np.array([r['label_id'] for r in train_items])
    val_labels_all = np.array([r['label_id'] for r in val_items])
    test_labels_all = np.array([r['label_id'] for r in test_items])

    # ---- 2. Relabel to contiguous indices ----
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
        client_trainset = HKDataset(data_root, sub_idx, 'train', train_items_known, train_labels_known, param)
        trainloaders.append(torch.utils.data.DataLoader(
            client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True))

        client_valset = HKDataset(data_root, sub_idx, 'train_val', train_items_known, train_labels_known, param)
        train_val_loaders.append(torch.utils.data.DataLoader(
            client_valset, batch_size=1, shuffle=False, num_workers=num_workers))

    valset = HKDataset(data_root, val_known_idx, 'valclose', val_items, val_labels_all, param)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)

    closeset = HKDataset(data_root, test_known_idx, 'testclose', test_items, test_labels_all, param)
    closeloader = torch.utils.data.DataLoader(closeset, batch_size=1, shuffle=False, num_workers=num_workers)

    openset = HKDataset(data_root, test_unknown_idx, 'testopen', test_items, test_labels_all, param)
    openloader = torch.utils.data.DataLoader(openset, batch_size=1, shuffle=False, num_workers=num_workers)

    return trainloaders, valloader, closeloader, openloader, train_val_loaders


class HKDataset(Dataset):
    def __init__(self, data_root, data_index, setname, data_items, datay, param=None):
        if param is None:
            param = {'Known_class': 6, 'unKnown_class': 9, 'Rotation': 45, 'Resize': 144, 'CropSize': 128}
        self.data_root = data_root
        self.data_index = data_index
        self.setname = setname
        self.data_items = data_items  # list of dicts with img_path, label_id, label_name
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
        item = self.data_items[img_idx]
        label = self.datay[img_idx]

        img_path = osp.join(self.data_root, item['img_path'])
        img = Image.open(img_path).convert('RGB')
        if self.setname in ('train', 'train_val'):
            image = self.transform_train(img)
        else:
            image = self.transform_test(img)
        return image, label, item['img_path']


if __name__ == '__main__':
    from collections import Counter
    datadir = './datasets/HyperKvasir'
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
