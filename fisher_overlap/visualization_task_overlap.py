import sys
sys.path.append("./" )
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from utils import *
import argparse
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def construct_visualization(overlap_path, dest_path, window_size, tasks): 
    overlaps_init, overlaps_end = torch.load(overlap_path, map_location='cpu')

    mkdir(dest_path)
    mkdir(dest_path + "correlation_matrices_top_k")

    k = overlaps_end.size(dim = -1)
    plt.figure()

    for i in range(k):
        overlap_init_i = overlaps_init[:, :, i]
        overlap_end_i = overlaps_end[:, :, i]
        
        corr_init = torch.round(overlap_init_i, decimals = 2)
        corr_end = torch.round(overlap_end_i, decimals = 2)
        
        plt.clf()
        sns.heatmap(corr_end, annot = True)
        plt.savefig(dest_path + f"correlation_matrices_top_k/{i*window_size}_to_{(i+1)*window_size}_end")

        plt.clf()
        sns.heatmap(corr_init, annot = True)
        plt.savefig(dest_path + f"correlation_matrices_top_k/{i*window_size}_to_{(i+1)*window_size}_start")

    fig, ax = plt.subplots(len(tasks), len(tasks), figsize = (20, 20))
    for i in range(len(tasks)):
        for j in range(i, len(tasks)):
            to_plot_end = np.clip(overlaps_end[i, j, :], 0, 1)
            to_plot_init = np.clip(overlaps_init[i, j, :], 0, 1)
            
            if (i == j):
                to_plot_end = np.ones(len(to_plot_end))
                to_plot_init = np.ones(len(to_plot_init))

            inds = window_size * np.arange(0, len(to_plot_end))
            
            ax[i, j].plot(inds, to_plot_end)
            ax[i, j].plot(inds, to_plot_init, color = ("black" if i == j else "orange"))
            ax[i, j].set_ylabel(f"Task {tasks[i]}")
            ax[i, j].set_xlabel(f"Task {tasks[j]}")
            ax[i, j].set_ylim([0, 1.05])


            if (i != j):
                ax[j, i].plot(inds, to_plot_end)
                ax[j, i].plot(inds, to_plot_init)
                ax[j, i].set_ylabel(f"Task {tasks[i]}")
                ax[j, i].set_xlabel(f"Task {tasks[j]}")
                ax[j, i].set_ylim([0, 1.05])
            
            
            

    fig.savefig(dest_path + f"overlap_over_inds.png", bbox_inches="tight")


def task_type(s):
    try:
        t1, t2 = map(int, s.split(','))
        return t1, t2
    except:
        raise argparse.ArgumentTypeError("Task must be t1, t2")

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Constructs Fisher overlap visualization plots on a binary classification task basis.\
        Expects path to overlap tensor and window size used in overlap computation')
    parser.add_argument('--overlap_path', help='Relative path from root to the set of overlap tensors', required= True)
    parser.add_argument('--dest_path', help='Relative directory from this script to save plots', required=True)
    parser.add_argument('--window_size', help='Sweep window size used for overlap computation', default = 30)
    parser.add_argument('--tasks', help = 'Labels of tasks to compare Fisher overlap scores, expects list of 2-tuples. E.g. --tasks 0,1 2,3]', nargs="+", required = True, type = task_type)

    args = parser.parse_args()

    construct_visualization(args.overlap_path, args.dest_path, int(args.window_size), args.tasks)