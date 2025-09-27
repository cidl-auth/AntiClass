from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip,  CenterCrop


class ClosedSetDataset(Dataset):
    def __init__(self, root='data', train=True, transform=None, args=None):
        self.train = train
        self.train_bs = args.train_bs
        self.test_bs = args.test_bs
        if args.dataset == 'mnist':
            self.classes = 10
            self.transform = transform if transform else Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
            self.dataset = datasets.MNIST(root=root, train=train, transform=self.transform, download=True)
        elif args.dataset == 'cifar10':
            self.classes = 10
            if self.train:
                self.transform = transform if transform else Compose([
                    RandomHorizontalFlip(),
                    RandomCrop(32, 4, padding_mode='reflect'),
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            else:
                self.transform = transform if transform else Compose([
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            self.dataset = datasets.CIFAR10(root=root, train=train, transform=self.transform, download=True)
        elif args.dataset == 'cifar100':
            self.classes = 100
            if self.train:
                self.transform = transform if transform else Compose([
                    RandomHorizontalFlip(),
                    RandomCrop(32, 4, padding_mode='reflect'),
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            else:
                self.transform = transform if transform else Compose([
                    CenterCrop(32),
                    ToTensor(),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            self.dataset = datasets.CIFAR100(root=root, train=train, transform=self.transform, download=True)
        else:
            raise ValueError("Dataset not available for this demo!")

    def get_loader(self):
        if self.train:
            return DataLoader(self.dataset, batch_size=self.train_bs, shuffle=True)
        else:
            return DataLoader(self.dataset, batch_size=self.test_bs, shuffle=False)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)