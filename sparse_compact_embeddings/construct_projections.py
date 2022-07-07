import sys 
sys.path.append("./" )
from utils import *
from dataset import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

model_name = "all_cnn"
dataset = "cifar10"
batch_size = 100
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

(num_true, num_prior, num_random, num_approx, num_classes), args = get_args()    
path = create_path(model_name, args, num_true, num_random, dataset)

if __name__ == "__main__":
    FIM_end, evals, evecs = torch.load(path + "tensors/FIM_true_end_full.pt", map_location= "cpu")
    
    test_embeddings, test_targets = torch.load(path + "tensors/img_param_space_test_embeddings.pt", map_location= "cpu")

    
    projections = {}
    test_projections = (test_embeddings) @ evecs

    for cls in range(num_classes):
        print(f"Class: {cls}")
        _, embedding = torch.load(path + f"tensors/cls_{cls}_img_param_space_embeddings.pt", map_location= "cpu")
        projections[cls] = embedding @ evecs

    num_examples = sum(projections[cls].shape[0] for cls in projections)
  
    targets = torch.zeros(num_examples)
    
    curr = 0

    cmap = get_cmap(num_classes + 1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()   
    for cls in projections:     
        ax.scatter(projections[cls][:, 0], projections[cls][:, 1], label = f"Class: {cls}", color = cmap(cls))

        targets[curr:curr + projections[cls].shape[0]] += cls
        curr += projections[cls].shape[0]
    
    ax.set_xlabel("Proj Coeff 1")
    ax.set_ylabel("Proj Coeff 2")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig(f"{path}top_2_comp_class_projections.png") 
    torch.save((projections, targets), f"{path}tensors/embedding_projections_targets.pt")
    torch.save((test_projections, test_targets), f"{path}tensors/test_embedding_projections_targets.pt")

   
    

    
    



    

   





