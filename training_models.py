from utils import *
import torch
from torch import nn, optim
from functions import train, val, FIM_true
from dataset import create_cifar, create_mnist

from torch.optim.lr_scheduler import CosineAnnealingLR

# prepare dataset
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

model_name = "all_cnn"
dataset = "cifar10"
(num_true, num_prior, num_random, num_approx, num_classes), args = get_args()    
path = create_path(model_name, args, num_true, num_random, dataset)

print(path)
mkdir(path)
mkdir(path + "tensors")

mc = get_model_class(model_name)
model = mc(*args).to(device)

criterion = nn.CrossEntropyLoss().to(device)
num_params = sum(param.numel() for param in model.parameters())
print(num_params)

if dataset == "mnist":
    train_set, train_set_prior, train_set_approx, test_set = create_mnist(num_classes, num_true, num_prior, num_approx)

if dataset == "cifar10":
    train_set, train_set_prior, train_set_approx, test_set = create_cifar(num_classes, num_true, num_prior, num_approx)

train_loader = torch.utils.data.DataLoader(train_set,
                                          batch_size=500,
                                          shuffle = True,
                                          **kwargs)

train_loader_approx = torch.utils.data.DataLoader(train_set_approx,
                                          batch_size=len(train_set_approx.data),
                                          shuffle = True,
                                          **kwargs)

train_loader_FIM = torch.utils.data.DataLoader(train_set_approx,
                                          batch_size=1,
                                          shuffle = True,
                                          **kwargs)

train_loader_prior = torch.utils.data.DataLoader(train_set_prior,
                                          batch_size=len(train_set_prior.data),
                                          shuffle = True,
                                          **kwargs)

test_loader = torch.utils.data.DataLoader(test_set,
                                         batch_size=500,
                                         shuffle = True,
                                         **kwargs)


epochs = 20
torch.save(model.state_dict(), path + "tensors/model_init.pt")

optimizer = optim.Adam(model.parameters(), lr = 2e-3)
scheduler = CosineAnnealingLR(optimizer,T_max=epochs, eta_min = 5e-4)

tr_err, tr_loss = val(model, device, train_loader, criterion)
val_err, val_loss = val(model, device, test_loader, criterion)
print("initial", 'tr_loss', tr_loss, 'val_loss', val_loss,  'tr_err', tr_err, 'val_err', val_err)


for epoch in range(epochs):

    train(model, device, train_loader, criterion, optimizer, epoch)
    
    tr_err, tr_loss = val(model, device, train_loader, criterion)
    val_err, val_loss = val(model, device, test_loader, criterion)

    # using cosine annealing scheduler
    scheduler.step()

    print(epoch, 'tr_loss', tr_loss, 'val_loss', val_loss,  'tr_err', tr_err, 'val_err', val_err)
    print("lr", optimizer.param_groups[0]['lr'])

torch.save(model.state_dict(), path + "tensors/model.pt")


model.load_state_dict(torch.load(path + "tensors/model.pt", map_location='cpu'))
FIM_tt, L_tt, u_tt = FIM_true(model, criterion, train_loader_FIM, "cpu", 1000)
torch.save((FIM_tt, L_tt, u_tt), path + "tensors/FIM_true_end_full.pt")
