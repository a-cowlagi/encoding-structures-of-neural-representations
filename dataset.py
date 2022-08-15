from json import load
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def two_class(dataset):
    targets = dataset.targets
    m = torch.max(targets)
    targets = torch.where(targets <= torch.div(
        m, 2, rounding_mode='floor'), 0, 1)
    dataset.targets = targets


def sample_balance(dataset, n_balance):
    '''
    dataset: dataset to subsample from.
    n_balance: number of sample to use in total 
    return: balanced sampled data and targets

    '''
    data_list = []
    targets_list = []
    d = torch.max(dataset.targets)

    for i in range(d+1):
        idx = dataset.targets == i
        data_list.append(dataset.data[idx][:n_balance//(d+1)])
        targets_list.append(dataset.targets[idx][:n_balance//(d+1)])

    data = torch.cat(data_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    return data, targets


def split_balance(dataset, n1, n2):

    data1_list, targets1_list, data2_list, targets2_list = [], [], [], []
    d = torch.max(dataset.targets)

    for i in range(d+1):
        idx = dataset.targets == i
        data1_list.append(dataset.data[idx][:n1//(d+1)])
        targets1_list.append(dataset.targets[idx][:n1//(d+1)])
        data2_list.append(dataset.data[idx][-n2//(d+1):])
        targets2_list.append(dataset.targets[idx][-n2//(d+1):])

    data1 = torch.cat(data1_list, dim=0)
    targets1 = torch.cat(targets1_list, dim=0)
    data2 = torch.cat(data2_list, dim=0)
    targets2 = torch.cat(targets2_list, dim=0)

    return data1, targets1, data2, targets2


def split(dataset, n1, n2):

    data1 = dataset.data[:n1]
    targets1 = dataset.targets[:n1]
    data2 = dataset.data[-n2:]
    targets2 = dataset.targets[-n2:]

    return data1, targets1, data2, targets2


def sample_combined(dataset, n_true, n_random, balance=True):
    '''
    dataset: dataset to subsample from.
    n_true: number of samples with true labels
    n_ramdom: number of samples with random labels 
    return: data and targets with combined true and random data

    '''

    d = torch.max(dataset.targets)

    if balance == True:

        data_true_list = []
        targets_true_list = []
        data_random_list = []

        for i in range(d+1):
            idx = dataset.targets == i
            data_true_list.append(dataset.data[idx][:n_true//(d+1)])
            targets_true_list.append(dataset.targets[idx][:n_true//(d+1)])
            data_random_list.append(dataset.data[idx][-n_random//(d+1):])

        data_true = torch.cat(data_true_list, dim=0)
        targets_true = torch.cat(targets_true_list, dim=0)
        data_random = torch.cat(data_random_list, dim=0)
        targets_random = torch.randint(0, d+1, (n_random,))

        data_combined = torch.cat((data_true, data_random), dim=0)
        targets_combined = torch.cat((targets_true, targets_random), dim=0)

    if balance == False:

        data_true = dataset.data[:n_true]
        targets_true = dataset.targets[:n_true]
        data_random = dataset.data[-n_random:]
        targets_random = torch.randint(0, d+1, (n_random,))

        data_combined = torch.cat((data_true, data_random), dim=0)
        targets_combined = torch.cat((targets_true, targets_random), dim=0)

    if n_random == 0:

        data_combined = data_true
        targets_combined = targets_true

    return data_true, targets_true, data_combined, targets_combined


def sample_combined2(dataset, n_true, n_random, n_approx):

    d = torch.max(dataset.targets)
    data_true = dataset.data[:n_true]
    targets_true = dataset.targets[:n_true]
    data_random = dataset.data[-n_random:]
    targets_random = torch.randint(0, d+1, (n_random,))

    data_combined = torch.cat((data_true, data_random), dim=0)
    targets_combined = torch.cat((targets_true, targets_random), dim=0)

    n_true_approx = int(n_true * (n_approx / (n_true + n_random)))
    n_random_approx = int(n_random * (n_approx / (n_true + n_random)))

    data_true_approx = dataset.data[:n_true_approx]
    targets_true_approx = dataset.targets[:n_true_approx]
    data_random_approx = dataset.data[-n_random_approx:]
    targets_random_approx = torch.randint(0, d+1, (n_random_approx,))

    data_combined_approx = torch.cat(
        (data_true_approx, data_random_approx), dim=0)
    targets_combined_approx = torch.cat(
        (targets_true_approx, targets_random_approx), dim=0)

    if n_random == 0:
        data_combined = data_true
        targets_combined = targets_true
        data_combined_approx = data_true_approx
        targets_combined_approx = targets_true_approx

    return data_combined, targets_combined, data_combined_approx, targets_combined_approx


def create_dataset(dataset, num_classes, num_true, num_random, balance=False):

    if dataset == 'mnist':

        mean = [0.131]
        std = [0.289]
        normalize = transforms.Normalize(mean, std)
        transform = transforms.Compose([transforms.ToTensor()])

        train_set = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transform)

        train_set_combined = datasets.MNIST('./data',
                                            train=True,
                                            download=True,
                                            transform=transform)

        test_set = datasets.MNIST('./data',
                                  train=False,
                                  download=True,
                                  transform=transform)

        test_balance = int(len(test_set.data) * 0.8)

    if dataset == 'cifar10':

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform1 = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4)
        ])

        transform2 = transforms.Compose([transforms.ToTensor(), normalize])

        train_set = datasets.CIFAR10('./data',
                                     train=True,
                                     download=True,
                                     transform=transform1)

        train_set_combined = datasets.CIFAR10('./data',
                                              train=True,
                                              download=True,
                                              transform=transform1)

        test_set = datasets.CIFAR10('./data',
                                    train=False,
                                    download=True,
                                    transform=transform2)

        train_set.data, train_set.targets = torch.tensor(
            train_set.data), torch.tensor(train_set.targets)
        train_set_combined.data, train_set_combined.targets = torch.tensor(
            train_set_combined.data), torch.tensor(train_set_combined.targets)
        test_set.data, test_set.targets = torch.tensor(
            test_set.data), torch.tensor(test_set.targets)

        test_balance = len(test_set.data)

    if dataset == 'cifar100':

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean, std)
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        train_set = datasets.CIFAR100('./data',
                                      train=True,
                                      download=True,
                                      transform=transform)

        train_set_combined = datasets.CIFAR100('./data',
                                               train=True,
                                               download=True,
                                               transform=transform)

        test_set = datasets.CIFAR100('./data',
                                     train=False,
                                     download=True,
                                     transform=transform)

        train_set.data, train_set.targets = torch.tensor(
            train_set.data), torch.tensor(train_set.targets)
        train_set_combined.data, train_set_combined.targets = torch.tensor(
            train_set_combined.data), torch.tensor(train_set_combined.targets)
        test_set.data, test_set.targets = torch.tensor(
            test_set.data), torch.tensor(test_set.targets)

        test_balance = len(test_set.data)

    data_true, targets_true, data_combined, targets_combined = sample_combined(
        train_set, num_true, num_random, balance)

    train_set.data = data_true
    train_set.targets = targets_true
    train_set_combined.data = data_combined
    train_set_combined.targets = targets_combined

    if balance == True:

        data_test, targets_test = sample_balance(test_set, test_balance)

        test_set.data = data_test
        test_set.targets = targets_test

    if num_classes == 2:

        two_class(train_set)
        two_class(train_set_combined)
        two_class(test_set)

    if dataset != "mnist":
        train_set.data = train_set.data.numpy()
        train_set_combined.data = train_set_combined.data.numpy()
        test_set.data = test_set.data.numpy()

    return train_set, train_set_combined, test_set


def create_cifar(num_classes, n1, n2, num_approx):

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(mean, std)
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4)
    ])

    transform2 = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = datasets.CIFAR10('./data',
                                 train=True,
                                 download=True,
                                 transform=transform1)

    train_set_approx = datasets.CIFAR10('./data',
                                        train=True,
                                        download=True,
                                        transform=transform2)

    train_set_prior = datasets.CIFAR10('./data',
                                       train=True,
                                       download=True,
                                       transform=transform2)

    test_set = datasets.CIFAR10('./data',
                                train=False,
                                download=True,
                                transform=transform2)

    train_set.data, train_set.targets = torch.tensor(
        train_set.data), torch.tensor(train_set.targets)
    train_set_prior.data, train_set_prior.targets = torch.tensor(
        train_set_prior.data), torch.tensor(train_set_prior.targets)
    test_set.data, test_set.targets = torch.tensor(
        test_set.data), torch.tensor(test_set.targets)

    data1, targets1, data2, targets2 = split_balance(train_set, n1, n2)

    data3, targets3, _, _ = split_balance(train_set, num_approx, 0)

    train_set.data = data1
    train_set.targets = targets1
    train_set_prior.data = data2
    train_set_prior.targets = targets2
    train_set_approx.data = data3
    train_set_approx.targets = targets3

    if num_classes == 2:
        two_class(train_set)
        two_class(train_set_prior)
        two_class(test_set)
        two_class(train_set_approx)

    train_set.data = train_set.data.numpy()

    train_set_prior.data = train_set_prior.data.numpy()
    train_set_approx.data = train_set_approx.data.numpy()
    test_set.data = test_set.data.numpy()

    return train_set, train_set_prior, train_set_approx, test_set


def create_mnist(num_classes, n1, n2, num_approx):
    mean = [0.131]
    std = [0.289]
    # normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transform)

    train_set_approx = datasets.MNIST('./data',
                                      train=True,
                                      download=True,
                                      transform=transform)

    train_set_prior = datasets.MNIST('./data',
                                     train=True,
                                     download=True,
                                     transform=transform)

    test_set = datasets.MNIST('./data',
                              train=False,
                              download=True,
                              transform=transform)

    data1, targets1, data2, targets2 = split(train_set, n1, n2)

    data3, targets3, _, _ = split(train_set, num_approx, 0)

    train_set.data = data1
    train_set.targets = targets1
    train_set_prior.data = data2
    train_set_prior.targets = targets2
    train_set_approx.data = data3
    train_set_approx.targets = targets3

    if num_classes == 2:
        two_class(train_set)
        two_class(train_set_prior)
        two_class(test_set)
        two_class(train_set_approx)

    return train_set, train_set_prior, train_set_approx, test_set

def get_custom_transform(dataset_name):
    if dataset_name == "mnist":
        transform = transforms.Compose([])
    elif dataset_name == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform = transforms.Compose([normalize])
    elif dataset_name == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        normalize = transforms.Normalize(mean, std)
        transform = transforms.Compose([normalize])
    else:
        raise NotImplementedError

    return transform


def create_mnist_random(num_classes, num_true, num_random, num_approx):
    mean = [0.131]
    std = [0.289]
    # normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transform)

    train_set_combined = datasets.MNIST('./data',
                                        train=True,
                                        download=True,
                                        transform=transform)

    train_set_approx = datasets.MNIST('./data',
                                      train=True,
                                      download=True,
                                      transform=transform)
    test_set = datasets.MNIST('./data',
                              train=False,
                              download=True,
                              transform=transform)

    data_combined, targets_combined, data_combined_approx, targets_combined_approx = sample_combined2(
        train_set, num_true, num_random, num_approx)

    train_set_combined.data = data_combined
    train_set_combined.targets = targets_combined
    train_set_approx.data = data_combined_approx
    train_set_approx.targets = targets_combined_approx

    if num_classes == 2:
        two_class(train_set)
        two_class(train_set_combined)
        two_class(train_set_approx)
        two_class(test_set)

    return train_set, train_set_combined, train_set_approx, test_set



def create_dataset_approx(dataset_name = "cifar10", num_approx = None):
    x_train_approx = y_train_approx = None
    
    if (dataset_name == "cifar10"):
        num_true = 55000
        num_prior = 5000
        num_approx = 10000 if num_approx is None else num_approx
        num_classes = 10
        
        train_set, _, train_set_approx, test_set = create_cifar(num_classes, num_true, num_prior, num_approx)

        # Separating trainset/testset data/labels
        x_train_approx = train_set_approx.data
        y_train_approx = train_set_approx.targets
    
    if (dataset_name == "mnist"):
        num_true = 55000
        num_prior = 5000
        num_approx = 10000 if num_approx is None else num_approx
        num_classes = 10
        
        train_set, _, train_set_approx, test_set = create_mnist(num_classes, num_true, num_prior, num_approx)

        # Separating trainset/testset data/labels
        x_train_approx = train_set_approx.data
        y_train_approx = train_set_approx.targets
        
        
    return train_set, x_train_approx, y_train_approx, test_set

def get_class_i(x, y, i, dataset = "mnist"):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)

    
    # Collect all data that match the desired label
    x_i = x[pos_i]

    if (dataset == "cifar10"):
        x_i = np.transpose(x_i, (0, 4, 2, 3, 1))
        x_i = np.squeeze(x_i, axis = 4)
   

    return np.array(x_i).astype('float32'), i*np.ones(len(x_i))


def get_bin_task(x, y, cls1, cls2, dataset_name):
    y = np.array(y)
    x_task = x[np.logical_or(y == cls1,y == cls2) != 0]
    y_task = y[np.logical_or(y == cls1,y == cls2) != 0]

    if (dataset_name == "cifar10"):
        x_task = np.transpose(x_task, (0, 3, 2, 1))
   

    return np.array(x_task).astype('float32'), y_task


def per_class_loader(dataset_name, x_train, y_train, num_classes = 10, batch_size = 1, **kwargs):
    loaders = []
    num_samples = []
    for cls in range(num_classes):
        X, y = get_class_i(x_train, y_train, cls, dataset_name)
        X, y = torch.from_numpy(X), torch.from_numpy(y)
        cls_trainset = TensorDataset(X, y)
        cls_loader = DataLoader(cls_trainset,
                                          batch_size=batch_size,
                                          shuffle = True,
                                          **kwargs)

        num_samples.append((len(y)))
        loaders.append(cls_loader)
    
    return num_samples, loaders

def per_task_loader(dataset_name, x_train, y_train, tasks, batch_size = 1):
    loaders = []
    num_samples = []
    for i, (cls1, cls2) in enumerate(tasks):
        X, y = get_bin_task(x =x_train, y = y_train, cls1 = cls1, cls2 = cls2, dataset_name = dataset_name)
        X, y = torch.from_numpy(X), torch.from_numpy(y)
        transform = get_custom_transform(dataset_name)
        task_trainset = CustomTensorDataset((X, y), transform=transform)
        task_loader = DataLoader(task_trainset,
                                          batch_size=batch_size,
                                          shuffle = False)

        num_samples.append((len(y)))
        loaders.append(task_loader)

    return num_samples, loaders


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)