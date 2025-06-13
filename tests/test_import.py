from torch import nn
import torch
from torch.utils.data import Dataset

def test_import_mammoth():
    from mammoth import train, load_runner, get_avail_args

    required_args, optional_args = get_avail_args(dataset='seq-cifar10', model='sgd')

    print("Required arguments:")
    for arg, info in required_args.items():
        print(f"  {arg}: {info['description']}")

    print("\nOptional arguments:")
    for arg, info in optional_args.items():
        print(f"  {arg}: {info['default']} - {info['description']}")

    model, dataset = load_runner('sgd','seq-cifar10', {'lr': 0.1, 'n_epochs': 1, 'batch_size': 4, 'num_workers': 0, 'non_verbose': True, 'debug_mode': True})

    train(model, dataset)

def test_import_after_register():
    from mammoth import train, load_runner, get_avail_args, register_backbone, register_dataset, register_model, ContinualDataset, store_masked_loaders, ContinualModel

    @register_backbone('dummy-backbone')
    def dummy_backbone():
        return nn.Linear(10, 10)

    class DummyDataset(Dataset):
        def __init__(self, data, targets, train=False):
            self.data = data
            self.targets = targets
            self.train = train
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            if self.train:
                return self.data[idx], self.targets[idx], self.data[idx]
            return self.data[idx], self.targets[idx]

    @register_dataset('dummy-dataset')
    class DummySeqDataset(ContinualDataset):
        NAME = 'dummy-dataset'
        SETTING = 'class-il'
        N_CLASSES_PER_TASK = 2
        N_TASKS = 5
        N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
        SIZE = (10,)
        TRANSFORM = None
        def __init__(self, args):
            super().__init__(args)
            self.train_data = torch.rand(500, 10)
            self.train_labels = torch.randint(0, self.N_CLASSES, (500,))
            self.test_data = torch.rand(200, 10)
            self.test_labels = torch.randint(0, self.N_CLASSES, (200,))

        def get_data_loaders(self):
            train_dataset = DummyDataset(self.train_data, self.train_labels, train=True)
            test_dataset = DummyDataset(self.test_data, self.test_labels, train=False)
            return store_masked_loaders(train_dataset, test_dataset, self)

        def get_loss(self):
            return nn.CrossEntropyLoss()

        def get_transform(self):
            return self.TRANSFORM

    @register_model('dummy_model')
    class DummyModel(ContinualModel):
        NAME = 'dummy-model'
        COMPATIBILITY = ['class-il', 'task-il']

        def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
            return 0
            
    required_args, optional_args = get_avail_args(dataset='seq-cifar10', model='sgd')

    print("Required arguments:")
    for arg, info in required_args.items():
        print(f"  {arg}: {info['description']}")

    print("\nOptional arguments:")
    for arg, info in optional_args.items():
        print(f"  {arg}: {info['default']} - {info['description']}")

    model, dataset = load_runner('dummy-model','dummy-dataset', {'lr': 0.1, 'n_epochs': 1, 'batch_size': 4, 'num_workers': 0, 'non_verbose': True, 'debug_mode': True, 'backbone': 'dummy-backbone'})

    train(model, dataset)