import argparse
import sys 
sys.path.append("./" )
from dataset import create_dataset_approx, per_task_loader
from functions import FIM_true, block_overlap
from utils import *


def compute_task_overlaps(dataset_name, model_name, model_init_path, model_fin_path, dest_path, window_size, tasks):

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    train_set, x_train_approx, y_train_approx, test_set = create_dataset_approx(dataset_name)
    x_train, y_train = train_set.data, train_set.targets
    
    _, args = get_args(model_name, dataset = dataset_name)   
    mc = get_model_class(model_name) 
    model = mc(*args).to(device)
    criterion = nn.CrossEntropyLoss().to(device) 

    model.load_state_dict(torch.load(model_fin_path, map_location='cpu'))
    model_init = mc(*args)
    model_init.load_state_dict(torch.load(model_init_path, map_location='cpu'))
    model, model_init= model.to(device), model_init.to(device)

    _, loaders = per_task_loader(dataset_name, x_train, y_train, tasks)


    evecs_init = []
    evecs_end = []

    for i, loader in enumerate(loaders):
        print(f"Currently Computing FIM for Task: {tasks[i]}")
        if (model_name == "lenet"):
            if (model_init_path != None):
                FIM_i, L_i, u_i = FIM_true(model_init, criterion, loader, device, 600)
            FIM_tt, L_tt, u_tt = FIM_true(model, criterion, loader, device, 600)
        else:
            if (model_init_path != None):
                FIM_i, L_i, u_i = FIM_true(model_init, criterion, loader, "cpu", 1000)
            FIM_tt, L_tt, u_tt = FIM_true(model, criterion, loader, "cpu", 1000)

        if (model_init_path != None):
            torch.save((FIM_i, L_i, u_i), f"{dest_path}task_{tasks[i][0]}_{tasks[i][1]}_FIM_true_init.pt")
        
        torch.save((FIM_tt, L_tt, u_tt), f"{dest_path}task_{tasks[i][0]}_{tasks[i][1]}_FIM_true_end.pt")
        evecs_init.append(u_i)
        evecs_end.append(u_tt)
    
    k = 600 if model_name == "lenet" else 1000
    overlaps_init = torch.zeros((len(tasks), len(tasks), k//window_size))
    overlaps_end = torch.zeros((len(tasks), len(tasks), k//window_size))
    
    for task1 in range(len(tasks)):
        task1_init = evecs_init[task1]
        task1_end = evecs_end[task1]
        for task2 in range(task1, len(tasks)):
            print(f"Task pair: {(tasks[task1], tasks[task2])}")
            task2_init = evecs_init[task2]
            task2_end = evecs_end[task2]
            overlaps_init[task1, task2, :] = block_overlap(task1_init, task2_init, k, window_size, device)
            overlaps_end[task1, task2, :] = block_overlap(task1_end, task2_end, k, window_size, device)
            

    torch.save((overlaps_init, overlaps_end), f"{dest_path}task_overlaps_init_end_window_size_{window_size}.pt")

def task_type(s):
    try:
        t1, t2 = map(int, s.split(','))
        return t1, t2
    except:
        raise argparse.ArgumentTypeError("Task must be t1, t2")

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Computes sweeping window Fisher eigenvector overlap for trained networks on the binary-classification task level.\
        Expects dataset name, path to model initial and final weights, and destination path for saving overlap scores. Expects window size to be a positive integer ')
    parser.add_argument('--dataset', help='Dataset name, can be mnist or cifar10', choices = ["mnist", "cifar10"], required=True)
    parser.add_argument('--model_name', help = "Model name, can be fc, all_cnn, wr, or lenet (must also have trained model).", choices = ["fc", "all_cnn", "wr", "lenet"], required= True)
    parser.add_argument('--model_init_path', help='Relative path from this script to the set of initial model weights, optional')
    parser.add_argument('--model_fin_path', help='Relative path from this script to the set of final model weights', required= True)
    parser.add_argument('--dest_path', help='Relative path from this script to the destination for the output overlap scores', required=True)
    parser.add_argument('--window_size', help='Sweep window size for overlap computation', default = 30)
    parser.add_argument('--tasks', help = 'Labels of tasks to compare Fisher overlap scores, expects list of 2-tuples. E.g. --tasks 0,1 2,3]', nargs="+", required = True, type = task_type)

    args = parser.parse_args()

    compute_task_overlaps(args.dataset, args.model_name, args.model_init_path, args.model_fin_path, args.dest_path, int(args.window_size), args.tasks)