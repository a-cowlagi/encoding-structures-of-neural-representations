import argparse
import sys 
sys.path.append("./" )
import os
from utils import get_model_class, mkdir, get_args
import torch
from torch import nn, optim
from functions import train, val
from torch.optim.lr_scheduler import CosineAnnealingLR
from suppressed_dataset import SuppressDataset
from torch.utils.data import DataLoader
import pickle


def create_suppressed_model_path(model_name, args, start, end, dataset):
    path = model_name
    for item in args:
        path += "_" + str(item)

    path += "_" + str(start) + "_" + str(end) + "_" + dataset

    path = os.path.join("./runtime_generated/suppressed_models/", path, "")

    return path

def train_all_suppressed_models(dataset, model_name, epochs = 20, batch_sz = 128, window_size = 30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} 
    suppressed_dataset_generator = SuppressDataset(dataset, num_approx=10000, retain_var=0.96)
    _, args = get_args(model_name = model_name)  
    scores_dict = {}
    for start in range(0, suppressed_dataset_generator.retained_vals - window_size, window_size):
        print(f"Start: {start}")
        trainset, testset = suppressed_dataset_generator.generate_suppressed(start_ind= start, window_size= window_size)
        trainloader = DataLoader(trainset,
                                          batch_size=batch_sz,
                                          shuffle = True,
                                          **kwargs)
        testloader = DataLoader(testset,
                                        batch_size=batch_sz,
                                        shuffle = True,
                                        **kwargs)

        path = create_suppressed_model_path(model_name = str(model_name), args = args, start = start, end = start + window_size, dataset = dataset)
        
        mkdir(path)
        mkdir(path + "tensors")

        mc = get_model_class(model_name= model_name)
        model = nn.DataParallel(mc(*args))
        model = model.to(device)

        train_scores, val_scores = train_one_suppressed_model(model, epochs, device, trainloader, testloader, path, save = False)

        scores_dict[(start, start + window_size)] = (train_scores, val_scores)

    with open('runtime_generated/suppressed_models/scores_dict.pkl', 'wb') as handle:
        pickle.dump(scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def train_one_suppressed_model(model, epochs, device, train_loader, test_loader, path, save = False):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 2e-3)
    scheduler = CosineAnnealingLR(optimizer,T_max= epochs, eta_min = 5e-4)
    train_scores, val_scores = [], []

    tr_err, tr_loss = val(model, device, train_loader, criterion)
    val_err, val_loss = val(model, device, test_loader, criterion)      
    print('Init: ', 'tr_loss', tr_loss, 'val_loss', val_loss,  'tr_err', tr_err, 'val_err', val_err)
    train_scores.append((tr_err, tr_loss))
    val_scores.append((val_err, val_loss))

    for epoch in range(epochs):
        train(model, device, train_loader, criterion, optimizer, epoch)
        
        tr_err, tr_loss = val(model, device, train_loader, criterion)
        val_err, val_loss = val(model, device, test_loader, criterion)

        # using cosine annealing scheduler
        scheduler.step()

        train_scores.append((tr_err, tr_loss))
        val_scores.append((val_err, val_loss))
        print(epoch, 'tr_loss', tr_loss, 'val_loss', val_loss,  'tr_err', tr_err, 'val_err', val_err)
        print("lr", optimizer.param_groups[0]['lr'])

    if save:
        torch.save(model.state_dict(), path + "tensors/model.pt")
    return train_scores, val_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a sequence of models on the desired dataset by suppressing projections of the input\
        along certain eigendirections. Specify dataset, model name, epochs, batch size, and window size for sweep')
    parser.add_argument('--dataset', help='Dataset name, can be mnist" or cifar10', choices = ["mnist", "cifar10"], required= True)
    parser.add_argument('--model_name', help='Model name, can be fc, all_cnn, wr, or lenet', choices = ["fc", "all_cnn", "wr", "lenet"], required=True)
    parser.add_argument('--epochs', help='Number of epochs to train for', default = 20, type = int)
    parser.add_argument('--batch_size', help='Batch size used for training examples', default = 128, type = int)
    parser.add_argument('--window_size', help='Sweep window size used for eigenvector suppression', default = 30, type = int)

    args = parser.parse_args()

    train_all_suppressed_models(args.dataset, args.model_name, int(args.epochs), int(args.batch_size), int(args.window_size))