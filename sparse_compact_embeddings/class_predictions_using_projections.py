import sys 
sys.path.append("./" )

from utils import *
from dataset import *
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model_name = "all_cnn"
dataset = "cifar10"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

(num_true, num_prior, num_random, num_approx, num_classes), args = get_args()    
path = create_path(model_name, args, num_true, num_random, dataset)

projections, targets = torch.load(f"{path}tensors/embedding_projections_targets.pt", map_location= "cpu")
test_projections, test_targets = torch.load(f"{path}tensors/test_embedding_projections_targets.pt", map_location= "cpu")

if model_name == "all_cnn":
    possible_ks = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]
else: 
    possible_ks = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100]
clf_scores = []

FIM_end, evals, evecs = torch.load(path + "tensors/FIM_true_end_full.pt", map_location= "cpu")
trace = sum(evals)


for k in possible_ks:
    print(k)
    relevant_projections = [projections[cls][:,:k].cpu().detach().numpy() for cls in projections]
    relevant_projections = np.concatenate(relevant_projections, axis = 0)

    clf = LinearSVC(random_state=0).fit(relevant_projections, targets)
    
    relevant_test_projections = test_projections[:, :k].cpu().detach().numpy()
    clf_scores.append(clf.score(relevant_projections, targets))


fig = plt.figure()
ax = fig.add_subplot()   
ax.plot(possible_ks, clf_scores)
ax.set_xscale('log')
ax.set_xlabel("k")
ax.set_ylabel("Test Set Classification Accuracy")

plt.savefig(f"{path}test set accuracy vs k.png")

cls_averages = []

for cls in range(num_classes):
    cls_projections = torch.mean(projections[cls], dim = 0)
    cls_averages.append(cls_projections)

cls_averages = torch.stack(cls_averages)
cls_averages = cls_averages.T

classes = np.arange(num_classes)
mkdir(path + "activation_barplots_by_evec")


sns.set_theme(style="whitegrid")
for i in range(evecs.shape[1]):
    plt.clf()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()   
    ax = sns.barplot(classes, cls_averages[i].cpu().detach().numpy(), x = "Class", y = "Average Activation")
    ax.set_xlabel("Class")
    ax.set_ylabel("Average Activation")
    ax.set_title(f"Average Activations by Class (eigenvector: {i+1})")
    plt.savefig(f"{path}activation_barplots_by_evec/evec_{i+1}")


evec_inds = np.arange(evecs.shape[1])

plt.clf()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()   
ax = sns.scatterplot(evec_inds, [np.max(row.cpu().detach().numpy()) - np.min(row.cpu().detach().numpy()) for row in cls_averages] , x = "Class", y = "Average Activation")
ax.set_xlabel("Eigenvector")
ax.set_ylabel("Max Inter-Class Activation Difference")

plt.savefig(f"{path}max_activation_difference.png")    