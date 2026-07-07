import numpy as np


def _balanced_iid_split(train_labels, n_clients, state):
    """Split data across clients with balanced class distribution (IID).

    Each class's samples are shuffled and evenly distributed across clients,
    so every client sees approximately the same class distribution.
    """
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    total_samples = len(train_labels)

    if total_samples < n_clients:
        raise ValueError(
            f'Balanced IID split requires at least {n_clients} training samples '
            f'(one per client), but got only {total_samples}.'
        )

    n_classes = int(train_labels.max()) + 1
    rng = np.random.RandomState()
    rng.set_state(state)

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]

    for c_idcs in class_idcs:
        rng.shuffle(c_idcs)
        splits = np.array_split(c_idcs, n_clients)
        for i, split in enumerate(splits):
            client_idcs[i].append(split)

    client_idcs = [
        np.concatenate(idcs).astype(np.int64, copy=False) if idcs else np.array([], dtype=np.int64)
        for idcs in client_idcs
    ]

    return client_idcs
