import sys 
sys.path.append("./" )
from utils import *
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from functions import FIM_true, overlap
from dataset import create_dataset_approx, per_class_loader


# prepare dataset
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

model_name = "all_cnn"
dataset = "cifar10"

train_set, x_train_approx, y_train_approx, test_set = create_dataset_approx(dataset)
x_train, y_train = train_set.data, train_set.targets

(num_true, num_prior, num_random, num_approx, num_classes), args = get_args(num_approx=55000)    
path = create_path(model_name, args, num_true, num_random, dataset)

mc = get_model_class(model_name)
model = mc(*args).to(device)
criterion = nn.CrossEntropyLoss().to(device)
num_params = sum(param.numel() for param in model.parameters())

if __name__ == "__main__":
    model = mc(*args)
    model.load_state_dict(torch.load(path + "tensors/model.pt", map_location='cpu'))
    model_init = mc(*args)
    model_init.load_state_dict(torch.load(path + "tensors/model_init.pt", map_location='cpu'))
    model, model_init= model.to(device), model_init.to(device)

    _, loaders = per_class_loader(dataset, x_train, y_train, num_classes)

    evecs_init = []
    evecs_end = []

    for i, loader in enumerate(loaders):

        if (model_name == "lenet"):
            FIM_i, L_i, u_i = FIM_true(model_init, criterion, loader, device, 600)
            FIM_tt, L_tt, u_tt = FIM_true(model, criterion, loader, device, 600)
        else:
            FIM_i, L_i, u_i = FIM_true(model_init, criterion, loader, "cpu", 1000)
            FIM_tt, L_tt, u_tt = FIM_true(model, criterion, loader, "cpu", 1000)

        torch.save((FIM_i, L_i, u_i), f"{path}cls_{i}_FIM_true_init.pt")
        torch.save((FIM_tt, L_tt, u_tt), f"{path}cls_{i}_FIM_true_end.pt")
        evecs_init.append(u_i)
        evecs_end.append(u_tt)
    
    k = 500
    overlaps_init = torch.zeros((num_classes, num_classes, k))
    overlaps_end = torch.zeros((num_classes, num_classes, k))
    
    for cls1 in range(num_classes):
        cls1_init = evecs_init[cls1]
        cls1_end = evecs_end[cls1]
        for cls2 in range(num_classes):
            print(f"class pair: {(cls1, cls2)}")
            cls2_init = evecs_init[cls2]
            cls2_end = evecs_end[cls2]
            overlaps_init[cls1, cls2, :] = overlap(cls1_init, cls2_init, k, device)
            overlaps_end[cls1, cls2, :] = overlap(cls1_end, cls2_end, k, device)
            

    torch.save((overlaps_init, overlaps_end), f"{path}overlaps_init_end.pt")