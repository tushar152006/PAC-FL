import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple

def load_cifar10_data(data_dir: str = "./data") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Download and return CIFAR-10 train and test datasets with standard transforms."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return trainset, testset

def create_synthetic_client_data(num_clients: int = 10, samples_per_client: int = 50, batch_size: int = 16) -> Dict[int, DataLoader]:
    """
    Creates lightweight synthetic client datasets for fast local unit testing without downloading CIFAR-10.
    """
    loaders = {}
    for cid in range(num_clients):
        data = torch.randn(samples_per_client, 3, 32, 32)
        labels = torch.randint(0, 10, (samples_per_client,))
        dataset = TensorDataset(data, labels)
        loaders[cid] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loaders
