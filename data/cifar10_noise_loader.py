import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def add_label_noise(dataset, noise_ratio, seed=42):
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    n = len(targets)
    n_noisy = int(noise_ratio * n)
    noisy_idx = np.random.choice(n, n_noisy, replace=False)

    for idx in noisy_idx:
        true = targets[idx]
        new = np.random.choice([i for i in range(10) if i != true])
        dataset.targets[idx] = new
    return dataset

def get_dataloaders(batch_size=128, noise_ratio=0.2, augment=False):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    train = add_label_noise(train, noise_ratio)
    val = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
