import sys 
sys.path.append("./" )

from utils import *
from dataset import *
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model_name = "lenet"
dataset = "mnist"
projection_mode = "single_layer"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

(num_true, num_prior, num_random, num_approx, num_classes), args = get_args(model_name, dataset = dataset)
path = create_path(model_name, args, num_true, num_random, dataset)

if model_name == "all_cnn":
    possible_ks = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]
else: 
    possible_ks = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100]


FIM_end, evals, evecs = torch.load(path + "tensors/FIM_true_end_full.pt", map_location= "cpu")
trace = sum(evals)

if projection_mode == "all":
    num_layers = {"all_cnn": 1, "lenet": 1}
elif projection_mode == "single_layer":
    num_layers = {"all_cnn": 5, "lenet": 5}

max_accuracies = []
for curr_layer in range(num_layers[model_name]):
    print(curr_layer)
    if projection_mode == "all":
        fig_path = ""
        train_proj_path = f"{path}tensors/embedding_projections_targets.pt"
        test_proj_path = f"{path}tensors/test_embedding_projections_targets.pt"
    elif projection_mode == "single_layer":
        fig_path = f"layer_{curr_layer}_"
        train_proj_path = f"{path}tensors/layer_{curr_layer}_embedding_projections_targets.pt"
        test_proj_path = f"{path}tensors/layer_{curr_layer}_test_embedding_projections_targets.pt"

    projections, targets = torch.load(train_proj_path, map_location= "cpu")
    test_projections, test_targets = torch.load(test_proj_path, map_location= "cpu")

    clf_scores = []
    for k in possible_ks:
        print(k)
        relevant_projections = [projections[cls][:,:k].cpu().detach().numpy() for cls in projections]
        relevant_projections = np.concatenate(relevant_projections, axis = 0)

        clf = LinearSVC(random_state=0).fit(relevant_projections, targets)
        
        relevant_test_projections = test_projections[:, :k].cpu().detach().numpy()
        clf_scores.append(clf.score(relevant_projections, targets))

    max_accuracies.append(np.max(clf_scores))

    fig = plt.figure()
    ax = fig.add_subplot()   
    ax.plot(possible_ks, clf_scores)
    ax.set_xscale('log')
    ax.set_xlabel("k")
    ax.set_ylabel("Test Set Classification Accuracy")

    plt.savefig(f"{path}{fig_path}test_set_accuracy_vs_k.png")

    cls_averages = []

    for cls in range(num_classes):
        cls_projections = torch.mean(projections[cls], dim = 0)
        cls_averages.append(cls_projections)

    cls_averages = torch.stack(cls_averages)
    cls_averages = cls_averages.T

    classes = np.arange(num_classes)
    mkdir(f"{path}{fig_path}activation_barplots_by_evec")


    sns.set_theme(style="whitegrid")
    for i in range(evecs.shape[1]):
        plt.clf()
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()   
        ax = sns.barplot(classes, cls_averages[i].cpu().detach().numpy(), x = "Class", y = "Average Activation")
        ax.set_xlabel("Class")
        ax.set_ylabel("Average Activation")
        ax.set_title(f"Average Activations by Class (eigenvector: {i+1})")
        plt.savefig(f"{path}{fig_path}activation_barplots_by_evec/evec_{i+1}")


    evec_inds = np.arange(evecs.shape[1])

    plt.clf()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()   
    ax = sns.scatterplot(evec_inds, [np.max(row.cpu().detach().numpy()) - np.min(row.cpu().detach().numpy()) for row in cls_averages] , x = "Class", y = "Average Activation")
    ax.set_xlabel("Eigenvector")
    ax.set_ylabel("Max Inter-Class Activation Difference")

    plt.savefig(f"{path}{fig_path}max_activation_difference.png") 

if (projection_mode == "single_layer"):
    plt.clf()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()   
    ax.plot(np.arange(num_layers[model_name]), max_accuracies)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Accuracy")
    plt.savefig(f"{path}accuracy_by_layer.png")    
