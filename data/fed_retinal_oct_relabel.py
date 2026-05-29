# -*- coding: utf-8 -*-
"""
RetinalOCT dataset loader for FedOSS.
Structure: RetinalOCT_Dataset/{train,val,test}/{AMD,CNV,CSR,DME,DR,DRUSEN,MH,NORMAL}/*.jpg
"""
import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import platform

# 8 classes in fixed order
OCT_CLASSES = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
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

    # Load pre-split data
    def load_split(subset):
        imgs, labels = [], []
        split_dir = osp.join(data_root, subset)
        for cls_idx in np.concatenate([known_class_list, unknown_class_list]):
            cls_name = OCT_CLASSES[cls_idx]
            cls_dir = osp.join(split_dir, cls_name)
            if not osp.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                imgs.append(osp.join(subset, cls_name, fname))
                labels.append(cls_idx)
        return imgs, labels

    trainx, trainy = load_split('train')
    valx, valy = load_split('val')
    testx, testy = load_split('test')

    print('Total images — Train: {}, Val: {}, Test: {}'.format(len(trainx), len(valx), len(testx)))

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
    new_unknown = np.arange(known_class, known_class + len(unknown_class_list))
    print('Known origin:', known_class_list, ', new:', new_known)
    print('Unknown origin:', unknown_class_list, ', new:', new_unknown)

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

    default_workers = 0 if platform.system() == 'Windows' else 4
    num_workers = int(param.get('num_workers', default_workers))
    pin_memory = bool(param.get('pin_memory', False))
    prefetch_factor = int(param.get('prefetch_factor', 2))
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    print('DataLoader settings: num_workers={}, pin_memory={}, prefetch_factor={}'.format(
        num_workers, pin_memory, prefetch_factor if num_workers > 0 else 'disabled'))

    train_labels = trainy[train_known_idx]
    train_images = [trainx[idx] for idx in train_known_idx]
    client_idcs = dirichlet_split_noniid(train_labels, alpha=dirichlet_alpha, n_clients=client_num, state=state)

    trainloaders = []
    train_val_loaders = []

    for i in range(client_num):
        print('Client{} sample num: {}'.format(i, len(client_idcs[i])))
        sub_idx = client_idcs[i]
        client_trainset = RetinalOCT(data_root, sub_idx, 'train', train_images, train_labels, param)
        trainloaders.append(torch.utils.data.DataLoader(
            client_trainset, batch_size=batchsize, shuffle=True, drop_last=True, **loader_kwargs))

        client_valset = RetinalOCT(data_root, sub_idx, 'train_val', train_images, train_labels, param)
        train_val_loaders.append(torch.utils.data.DataLoader(
            client_valset, batch_size=1, shuffle=False, **loader_kwargs))

    valset = RetinalOCT(data_root, val_known_idx, 'valclose', valx, valy, param)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, **loader_kwargs)

    closeset = RetinalOCT(data_root, test_known_idx, 'testclose', testx, testy, param)
    closeloader = torch.utils.data.DataLoader(closeset, batch_size=1, shuffle=False, **loader_kwargs)

    openset = RetinalOCT(data_root, test_unknown_idx, 'testopen', testx, testy, param)
    openloader = torch.utils.data.DataLoader(openset, batch_size=1, shuffle=False, **loader_kwargs)

    return trainloaders, valloader, closeloader, openloader, train_val_loaders


class RetinalOCT(Dataset):
    def __init__(self, data_root, data_index, setname, datax, datay, param=None):
        if param is None:
            param = {'Known_class': 5, 'unKnown_class': 3, 'Rotation': 45, 'Resize': 144, 'CropSize': 128}
        self.data_root = data_root
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
        img_path = self.datax[img_id]
        label = self.datay[img_id]

        img = Image.open(os.path.join(self.data_root, img_path)).convert('RGB')
        if self.setname == 'train':
            image = self.transform_train(img)
        else:
            image = self.transform_test(img)
        return image, label, img_path


if __name__ == '__main__':
    from collections import Counter
    datadir = './datasets/RetinalOCT_Dataset'
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
