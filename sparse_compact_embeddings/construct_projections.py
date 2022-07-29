import sys
from xml.parsers.expat import model 
sys.path.append("./" )
from utils import *
from dataset import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

model_name = "all_cnn"
dataset = "cifar10"
projection_mode = "single_layer"
plot_mode = "3d"
batch_size = 100
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

(num_true, num_prior, num_random, num_approx, num_classes), args = get_args(model_name, dataset = dataset)    
path = create_path(model_name, args, num_true, num_random, dataset)

def get_param_layers(model_name = "all_cnn"):
    if (model_name == "all_cnn"):
        return {0, 4, 8, 12, 16}
    elif (model_name == "lenet"):
        return {0, 2, 4, 6, 8}


def get_param_inds(model: nn.Module, param_layers: set):
    index_pairs = []
    prev_param, curr_param = 0, 0
    for j, param in enumerate(model.parameters()):
        prev_param = curr_param
        curr_param += param.numel()
        if j in param_layers:
            index_pairs.append((prev_param, curr_param))


    return index_pairs

if __name__ == "__main__":
    FIM_end, evals, evecs = torch.load(path + "tensors/FIM_true_end_full.pt", map_location= "cpu")

    test_embeddings, test_targets = torch.load(path + "tensors/img_param_space_test_embeddings.pt", map_location= "cpu")

    if (projection_mode == "single_layer"):
        mc = get_model_class(model_name)
        param_layers = get_param_layers(model_name)
        dummy_model = mc(*args)
        index_pairs = get_param_inds(dummy_model, param_layers)
    else:
        index_pairs = [(0, test_embeddings.shape[1])]

    projections = {}
    curr_layer = 0
    for start, end in index_pairs:
        print(f"Layer: {curr_layer}")
        test_projections = (test_embeddings[:, start:end]) @ evecs[start:end, :]

        for cls in range(num_classes):
            print(f"Class: {cls}")
            _, embedding = torch.load(path + f"tensors/cls_{cls}_img_param_space_embeddings.pt", map_location= "cpu")
            projections[cls] = (embedding[:, start:end] @ evecs[start:end, :])

        
        num_examples = sum(projections[cls].shape[0] for cls in projections)
    
        targets = torch.zeros(num_examples)
        
        curr = 0

        cmap = get_cmap(num_classes + 1)

        fig = plt.figure(figsize=(10, 10))
        if (plot_mode == "3d"):
            ax = fig.add_subplot(projection = "3d") 
        else:
            ax = fig.add_subplot()  
        
        for cls in projections:     
            if (plot_mode == "3d"):
                ax.scatter(projections[cls][:, 0], projections[cls][:, 1], projections[cls][:, 2], label = f"Class: {cls}", color = cmap(cls))
                ax.set_zlabel("Proj Coeff 3")
            else:
                ax.scatter(projections[cls][:, 0], projections[cls][:, 1], label = f"Class: {cls}", color = cmap(cls))

            targets[curr:curr + projections[cls].shape[0]] += cls
            curr += projections[cls].shape[0]
        
        ax.set_xlabel("Proj Coeff 1")
        ax.set_ylabel("Proj Coeff 2")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=3, fancybox=True, shadow=True)

        if projection_mode == "all":
            fig_path = f"{path}top_2_comp_class_projections.png"
            train_proj_path = f"{path}tensors/embedding_projections_targets.pt"
            test_proj_path = f"{path}tensors/test_embedding_projections_targets.pt"
        elif projection_mode == "single_layer":
            fig_path = f"{path}layer_{curr_layer}_{plot_mode}top_2_comp_class_projections.png"
            train_proj_path = f"{path}tensors/layer_{curr_layer}_embedding_projections_targets.pt"
            test_proj_path = f"{path}tensors/layer_{curr_layer}_test_embedding_projections_targets.pt"

        plt.savefig(fig_path) 
        torch.save((projections, targets), train_proj_path)
        torch.save((test_projections, test_targets), test_proj_path)

        curr_layer += 1

   
    

    
    



    

   





