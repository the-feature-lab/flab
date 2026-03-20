import numpy as np
import torch as torch
import torchvision

import os
from einops import reduce, rearrange


def download_dataset(dataset_name, datapath=None):
    """Download a torchvision dataset and save it to disk as a single npz file.

    Saves the full dataset (train + test concatenated) as uint8 NCHW arrays
    with keys 'X' and 'y'. Compatible with get_image_data. Imagenet datasets
    are not available via torchvision and must be placed manually.

    Args:
        dataset_name (str): one of 'mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'.
        datapath (str): root directory for dataset storage. Files are saved to
            {datapath}/{dataset_name}/{dataset_name}.npz.
    """
    if datapath is None:
        datapath = os.getenv('DATASETPATH')
        if datapath is None:
            raise ValueError("datapath must be provided")
    dataset_cls = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist': torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'svhn': torchvision.datasets.SVHN,
    }
    if dataset_name not in dataset_cls:
        raise ValueError(f"dataset '{dataset_name}' not supported. Choose from {list(dataset_cls)}")

    cls = dataset_cls[dataset_name]
    if dataset_name == 'svhn':
        raw_train = cls(root=datapath, split='train', download=True)
        raw_test = cls(root=datapath, split='test', download=True)
        X_train, y_train = raw_train.data, np.array(raw_train.labels)
        X_test, y_test = raw_test.data, np.array(raw_test.labels)
    elif dataset_name in ['mnist', 'fmnist']:
        raw_train = cls(root=datapath, train=True, download=True)
        raw_test = cls(root=datapath, train=False, download=True)
        X_train = rearrange(raw_train.data.numpy(), 'b h w -> b 1 h w')
        y_train = raw_train.targets.numpy()
        X_test = rearrange(raw_test.data.numpy(), 'b h w -> b 1 h w')
        y_test = raw_test.targets.numpy()
    else:  # cifar10, cifar100
        raw_train = cls(root=datapath, train=True, download=True)
        raw_test = cls(root=datapath, train=False, download=True)
        X_train = rearrange(np.array(raw_train.data), 'b h w c -> b c h w')
        y_train = np.array(raw_train.targets)
        X_test = rearrange(np.array(raw_test.data), 'b h w c -> b c h w')
        y_test = np.array(raw_test.targets)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    out_dir = os.path.join(datapath, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    fname = 'svhn.npz' if dataset_name == 'svhn' else f'{dataset_name}.npz'
    fn = os.path.join(out_dir, fname)
    np.savez(fn, X=X, y=y)
    print(f"Saved {len(X)} samples to {fn}")


def get_image_data(dataset, n_samples, datapath=None, classes=None):
    """Load an image dataset from disk, apply class grouping, and return balanced X, y.

    Draws n_samples // len(classes) samples per class group. X is float32 NCHW
    in [0, 1]. y is one-hot float32.

    Args:
        dataset (str): one of 'mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn',
            'imagenet32', 'cifar5m'. cifar5m and imagenet32 must be placed manually.
            All others require a prior call to download_dataset.
        n_samples (int): total number of samples to return. Must be divisible
            by len(classes).
        datapath (str): root directory for dataset storage.
        classes (list of lists or None): class groupings, e.g. [[0,1],[2,3]]
            merges original classes 0 and 1 into new class 0, etc. If None,
            each original class becomes its own group.

    Returns:
        X (ndarray): float32 of shape (n_samples, c, h, w) in [0, 1].
        y (ndarray): float32 one-hot of shape (n_samples, n_classes).

    Raises:
        FileNotFoundError: If the dataset file is not found on disk.
    """
    if datapath is None:
        datapath = os.getenv('DATASETPATH')
        if datapath is None:
            raise ValueError("datapath must be provided")
    # Load raw X (uint8, NCHW) and integer y labels from disk
    if dataset == 'imagenet32':
        fn = os.path.join(datapath, 'imagenet', 'imagenet32.npz')
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"imagenet32 not found at {fn}")
        data = np.load(fn)
        X = rearrange(data['data'], 'n (c h w) -> n c h w', c=3, h=32, w=32)
        y = data['labels'].astype(int) - 1  # stored 1-indexed
    elif dataset == 'cifar5m':
        fn = os.path.join(datapath, 'cifar5m', 'part0.npz')
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"cifar5m not found at {fn}")
        data = np.load(fn)
        X = rearrange(data['X'], 'b h w c -> b c h w')
        y = data['Y'].astype(int)
    elif dataset in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
        fn = os.path.join(datapath, dataset, f'{dataset}.npz')
        if not os.path.isfile(fn):
            print(f"Dataset file {fn} not found. Attempting to download raw dataset...")
            download_dataset(dataset, datapath)
            fn = os.path.join(datapath, dataset, f'{dataset}.npz')
            if not os.path.isfile(fn):
                raise FileNotFoundError(f"Failed to download {dataset} at {fn}.")
        data = np.load(fn)
        X, y = data['X'], data['y'].astype(int)
    else:
        raise ValueError(f"dataset '{dataset}' not supported")

    if classes is None:
        classes = [[c] for c in range(int(y.max()) + 1)]

    # Draw n_per_class samples from each class group (balanced)
    n_per_class = n_samples // len(classes)
    idxs = np.concatenate([
        np.flatnonzero(np.isin(y, group))[:n_per_class]
        for group in classes
    ])
    idxs = np.random.permutation(idxs)
    assert len(idxs) == n_samples, "not enough samples of specified classes"

    X = X[idxs] / 255.0

    # Map original labels to new class indices
    new_y = np.empty(len(idxs), dtype=int)
    for new_class, group in enumerate(classes):
        new_y[np.isin(y[idxs], group)] = new_class

    y_onehot = np.zeros((n_samples, len(classes)), dtype=np.float32)
    y_onehot[np.arange(n_samples), new_y] = 1.0

    return X.astype(np.float32), y_onehot


def preprocess(X, **kwargs):
    """
    Process image dataset. Returns vectorized (flattened) images.
    Operates in numpy if X is an ndarray, torch otherwise.

    X (ndarray or tensor): image dataset, shape (N, c, h, w)
    kwargs:
        "grayscale" (bool, False): If true, average over channels. Eliminates channel dim.
        "center" (bool, False): If true, center image vector distribution.
        "normalize" (bool, False): If true, make all image vectors unit norm.
        "zca_strength" (float, 0): Flatten covariance spectrum according to S_new = S / sqrt(zca_strength * S^2 + 1)

    returns: array of shape (N, d), same type as input
    """
    is_numpy = isinstance(X, np.ndarray)
    la = np if is_numpy else torch

    if kwargs.get('grayscale', False):
        X = reduce(X, 'N c h w -> N (h w)', 'mean')
    else:
        X = rearrange(X, 'N c h w -> N (c h w)')

    if kwargs.get('center', False):
        X_mean = reduce(X, 'N d -> d', 'mean')
        X -= X_mean

    if kwargs.get('normalize', False):
        if is_numpy:
            norms = la.linalg.norm(X, axis=1, keepdims=True)
        else:
            norms = la.linalg.norm(X, dim=1, keepdim=True)
        X /= norms

    zca_strength = kwargs.get('zca_strength', 0)
    if zca_strength:
        U, S, Vt = la.linalg.svd(X, full_matrices=False)
        zca_strength /= la.mean(S**2)
        S_zca = S / la.sqrt(zca_strength * S**2 + 1)
        S_zca /= la.linalg.norm(S_zca)
        X = U @ la.diag(S_zca) @ Vt

    if kwargs.get('center', False):
        X_mean = reduce(X, 'N d -> d', 'mean')
        X -= X_mean

    return X
