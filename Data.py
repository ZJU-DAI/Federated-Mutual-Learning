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

        train_set = torchvision.datasets.EMNIST(
            root="~/data", split='balanced', train=True, download=True, transform=tra_transformer
        )
        l = len(train_set)
        if args.all_data:
            data_num = [len(train_set) for _ in range(1)]
        else:
            data_num = [int(len(train_set) / args.split) for _ in range(args.split)]
        splited_set = torch.utils.data.random_split(train_set, data_num)

        self.train_loader = []
        for i in range(args.node_num):
            self.train_loader.append(torch.utils.data.DataLoader(
                splited_set[i], batch_size=args.batchsize, shuffle=True, num_workers=4
            ))

        test_set = torchvision.datasets.EMNIST(
            root="~/data", split='byclass', train=False, download=True, transform=val_transformer
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batchsize, shuffle=False, num_workers=4
        )
