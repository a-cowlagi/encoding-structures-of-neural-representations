import argparse
import sys 
sys.path.append("./" )
import os
from utils import get_model_class, mkdir, get_args
import torch
from torch import nn, optim
from functions import train, val
from torch.optim.lr_scheduler import CosineAnnealingLR
from projecting_dataset import ProjectDataset
from torch.utils.data import DataLoader
import pickle


def create_model_path(model_name, cmd_args, model_data_args, start, end, dataset):
    model_data_path = model_name

    model_data_path += "_" + str(start) + "_" + str(end)

    suppressed_path = f"./runtime_generated/{dataset}/compressed_models/{cmd_args.project_suppress}_{cmd_args.suppress_mode}_window_sz_{args.window_size}/"

    curr_model_path = suppressed_path + model_data_path + "/"
    return suppressed_path, curr_model_path

def train_all_suppressed_models(args):
    dataset, model_name, epochs, batch_sz, window_size = str(args.dataset), str(args.model_name), int(args.epochs), int(args.batch_size), int(args.window_size)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} 
    print(device)
    compressed_dataset_generator = ProjectDataset(dataset, num_approx= 10000, retain_var=0.96, mode = args.suppress_mode)
    _, model_data_args = get_args(model_name = model_name, dataset= dataset)  
    scores_dict = {}
    for start in range(0, compressed_dataset_generator.retained_vals - window_size, window_size):
        print(f"Start: {start}")
        if (args.project_suppress == "suppress"):
            trainset, testset = compressed_dataset_generator.generate_suppressed(start_ind= start, window_size= window_size)
        elif (args.project_suppress == "project"):
            trainset, testset = compressed_dataset_generator.generate_projected(start_ind= start, window_size= window_size)
        
        trainloader = DataLoader(trainset,
                                          batch_size=batch_sz,
                                          shuffle = True,
                                          **kwargs)
        testloader = DataLoader(testset,
                                        batch_size=batch_sz,
                                        shuffle = True,
                                        **kwargs)

        compressed_path, curr_model_path = create_model_path(model_name = str(model_name), cmd_args = args, model_data_args = model_data_args, start = start, end = start + window_size, dataset = dataset)
        
        # make directory for model if it doesn't exist
        if not os.path.exists(curr_model_path):
            mkdir(curr_model_path)
        
        mc = get_model_class(model_name= model_name)
        model = mc(*model_data_args)
        model = model.to(device)

        train_scores, val_scores = train_one_suppressed_model(model, epochs, device, trainloader, testloader, curr_model_path, save = True)

        scores_dict[(start, start + window_size)] = (train_scores, val_scores)

    with open(compressed_path + f'scores_dict_{window_size}.pkl', 'wb') as handle:
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
        torch.save(model.state_dict(), path + "model.pt")
    return train_scores, val_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a sequence of models on the desired dataset by suppressing projections of the input\
        along certain eigendirections. Specify dataset, model name, epochs, batch size, and window size for sweep')
    parser.add_argument('--dataset', help='Dataset name, can be mnist" or cifar10', choices = ["mnist", "cifar10"], required= True)
    parser.add_argument('--model_name', help='Model name, can be fc, all_cnn, wr, or lenet', choices = ["fc", "all_cnn", "wr", "lenet"], required=True)
    parser.add_argument('--epochs', help='Number of epochs to train for', default = 20, type = int)
    parser.add_argument('--batch_size', help='Batch size used for training examples', default = 128, type = int)
    parser.add_argument('--window_size', help='Sweep window size used for eigenvector suppression', default = 200, type = int)
    parser.add_argument('--compress_mode', help='Compress data using SVD or Wavelet transform. Can be covariance or wavelet.', choices =  ["covariance", "wavelet"], default = "covariance")
    parser.add_argument('--project_suppress', help='Project or suppress directions. Can be project or suppress.', choices =  ["project", "suppress"], default = "suppress")

    args = parser.parse_args()

    train_all_suppressed_models(args)