import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Dataset:
    def __init__(
        self,
        dataset_class,
        root: str = "data",
        batch_size: int = 32,
        shuffle: bool = True,
        transform=None,
        download: bool = True,
    ):
        self.transform = transform if transform else transforms.ToTensor()

        self.train = DataLoader(
            dataset_class(
                root=root,
                train=True,
                download=download,
                transform=self.transform,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.test = DataLoader(
            dataset_class(
                root=root,
                train=False,
                download=download,
                transform=self.transform,
            ),
            batch_size=batch_size,
            shuffle=False,
        )


class MNIST(Dataset):
    def __init__(self, batch_size: int = 32, shuffle: bool = True):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        )

        super().__init__(
            dataset_class=datasets.MNIST,
            root="data",
            batch_size=batch_size,
            shuffle=shuffle,
            transform=transform,
            download=True,
        )
