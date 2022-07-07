import sys 
sys.path.append("./" )
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
import torch
from utils import *
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model_name = "lenet"
dataset = "mnist"
(num_true, num_prior, num_random, num_approx, num_classes), args = get_args()    
path = create_path(model_name, args, num_true, num_random, dataset)

overlaps_init, overlaps_end = torch.load(path + "overlaps_init_end.pt", map_location='cpu')

mkdir(path + "correlation_matrices_top_k")

k = overlaps_init.size(dim = -1)
incr = [0, 9, 19, 39, 79, 159, 319, k-1]
# Get correlation matrix for every 20th 
plt.figure()

for i in incr:
    overlap_init_i = overlaps_init[:, :, i]
    overlap_end_i = overlaps_end[:, :, i]
    
    corr_init = torch.round(overlap_init_i, decimals = 2)
    corr_end = torch.round(overlap_end_i, decimals = 2)
    
    plt.clf()
    sns.heatmap(corr_end, annot = True)
    plt.savefig(path + f"correlation_matrices_top_k/top_{i + 1}_end")

    plt.clf()
    sns.heatmap(corr_init, annot = True)
    plt.savefig(path + f"correlation_matrices_top_k/top_{i + 1}_start")



