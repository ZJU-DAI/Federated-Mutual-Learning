import torch
import torchvision
from torchvision import transforms


class Data(object):
    def __init__(self, args):
        tra_transformer = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        val_transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(
            root="~/data", train=True, download=False, transform=tra_transformer
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batchsize, shuffle=True, num_workers=4
        )
        test_set = torchvision.datasets.CIFAR10(
            root="~/data", train=False, download=False, transform=val_transformer
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batchsize, shuffle=False, num_workers=4
        )
