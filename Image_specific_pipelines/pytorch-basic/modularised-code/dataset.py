# dataset.py

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

def get_custom_data_loaders(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    entire_dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    train_ds, val_ds = random_split(entire_dataset, [50000, 10000])

    test_ds = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
