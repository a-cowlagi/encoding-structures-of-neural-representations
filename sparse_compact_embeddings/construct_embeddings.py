import sys 
sys.path.append("./" )
import torch.nn as nn
import torch.nn.functional as f
from utils import *
from dataset import *
import sys

def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        if (name in activation):
            activation[name].append(output.detach())
        else:
            activation[name] = [output.detach()]
    return hook


def convolve_no_sum(input, kernels, kernel_size, stride=1, padding=0):
    # first dim = batch_size
    bs = input.shape[0]
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
    input = unfold(input)

    # first dim = batch size, second dim = locs of convolution, third dim = covered elts at loc
    input = input.transpose(1, 2)

    # first dim = # kernels
    kernels_aligned_along_rows = kernels.reshape(kernels.shape[0], -1,)

    # sum along input rows
    input = input.sum(1)

    embedding = torch.zeros(
        (bs, kernels.shape[0], kernels_aligned_along_rows.shape[1])).to(device)
    for i in range(bs):
        embedding[i, :, :] += input[i] * kernels_aligned_along_rows

    embedding = embedding.reshape(bs, -1,)
    return embedding.to("cpu")


def weight_scaled_activation(prev_activations, weights):
    bs = prev_activations.shape[0]
    embedding = torch.zeros(
        (bs, weights.shape[0], weights.shape[1])).to(device)

    for i in range(bs):
        embedding[i, :, :] += prev_activations[i] * weights

    embedding = embedding.reshape(bs, -1,)
    return embedding


hooks = []

def batch_embeddings(curr_batch_embeddings, conv_param_layers, linear_param_layers,
                     model_name="lenet", prev_param=0, curr_param=0):
    if model_name == "lenet":
        for j, param in enumerate(model.parameters()):
            # calc pre-sum output of convolutional layer (param contributions = #channels * (kernel_sz1 * kernel_sz2))
            prev_param = curr_param
            if (j in conv_param_layers):
                if (j == 0):
                    embedding = convolve_no_sum(X, param.data, (5,5))
                elif (j == 2):
                    embedding = convolve_no_sum(
                        activation["pool"][0], param.data, (5, 5))
                embedding = f.normalize(embedding, p=2, dim=1)

            # calc weight-scaled activations for linear layers (param contributions = 256 * 120)
            elif (j in linear_param_layers):
                if (j == 4):
                    prev_layer = activation["pool"][1]
                    embedding = weight_scaled_activation(
                        (prev_layer.view(prev_layer.size()[0], -1)), param.data)
                elif (j == 6):
                    embedding = weight_scaled_activation(
                        act(activation["L1_out"][0]), param.data)
                else:
                    embedding = weight_scaled_activation(
                        act(activation["L2_out"][0]), param.data)

                embedding = f.normalize(embedding, p=2, dim=1)
            
            # bias terms ignored
            else:
                embedding = torch.zeros(X.shape[0], param.numel())

            curr_param += param.numel()
            curr_batch_embeddings[:, prev_param: curr_param] = embedding
        return curr_batch_embeddings
    elif model_name == "all_cnn":
        for j, param in enumerate(model.parameters()):
            # calc pre-sum output of convolutional layer (param contributions = #channels * (kernel_sz1 * kernel_sz2))
            prev_param = curr_param
            if (j in conv_param_layers):
                if (j == 0):
                    embedding = convolve_no_sum(X, param.data, (3, 3))
                elif (j == 4):
                    embedding = convolve_no_sum(
                        activation["bn1"][0], param.data, (3, 3))
                elif (j == 8):
                    embedding = convolve_no_sum(
                        activation["bn2"][0], param.data, (3, 3))
                elif (j == 12):
                    embedding = convolve_no_sum(
                        activation["bn3"][0], param.data, (3, 3))
                elif (j == 16):
                    embedding = convolve_no_sum(
                        activation["bn4"][0], param.data, (1, 1))
                embedding = f.normalize(embedding, p=2, dim=1)

            # bias terms ignored
            else:
                embedding = torch.zeros(X.shape[0], param.numel())

            curr_param += param.numel()
            curr_batch_embeddings[:, prev_param: curr_param] = embedding
        return curr_batch_embeddings

    else:
        raise NotImplementedError



if __name__ == "__main__":
    model_name = sys.argv[1]
    dataset = sys.argv[2]
    batch_size = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_examples = 5000

    (num_true, num_prior, num_random, num_approx, num_classes), args = get_args(model_name, dataset = dataset) 
    path = create_path(model_name, args, num_true, num_random, dataset)

    mkdir(path)
    mkdir(path + "tensors")

    mc = get_model_class(model_name)
    model = mc(*args)
    print('Num parameters: ', sum([p.numel() for p in model.parameters()]))

    trained = True

    if (trained):
        model.load_state_dict(torch.load(
            path + "tensors/model.pt", map_location='cpu'))
    else:
        model.load_state_dict(torch.load(
            path + "tensors/model_init.pt", map_location='cpu'))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    num_params = sum(param.numel() for param in model.parameters())

    activation = {}

    if model_name == "lenet" and dataset == "mnist":
        hp = model.pool.register_forward_hook(getActivation("pool"))
        hl1 = model.L1.register_forward_hook(getActivation("L1_out"))
        hl2 = model.L2.register_forward_hook(getActivation("L2_out"))
        hl3 = model.L3.register_forward_hook(getActivation("L3_out"))
        act = nn.Tanh()

        hooks = [hp, hl1, hl2, hl3]
        conv_param_layers = {0, 2}
        linear_param_layers = {4, 6, 8}
        bias_layers = {1, 3, 5, 7, 9}

        train_set, _, _, test_set = create_mnist(
            num_classes, num_true, num_prior, num_approx)
        x_train, y_train = train_set.data, train_set.targets
        num_samples, loaders = per_class_loader(
            "mnist", x_train, y_train, batch_size=batch_size)
        embeddings = {}

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


    elif model_name == "all_cnn" and dataset == "cifar10":
        act = nn.ReLU(True)
        hbn1 = model.m[0][2].register_forward_hook(getActivation("bn1"))
        hbn2 = model.m[1][2].register_forward_hook(getActivation("bn2"))
        hbn3 = model.m[2][2].register_forward_hook(getActivation("bn3"))
        hbn4 = model.m[3][2].register_forward_hook(getActivation("bn4"))
        hbn5 = model.m[4][2].register_forward_hook(getActivation("bn5"))

        hooks = [hbn1, hbn2, hbn3, hbn4, hbn5]
        conv_param_layers = {0, 4, 8, 12, 16}
        linear_param_layers = {}

        train_set, _, _, test_set = create_cifar(
            num_classes, num_true, num_prior, num_approx)
        x_train, y_train = train_set.data, train_set.targets
        num_samples, loaders = per_class_loader(
            "cifar10", x_train, y_train, batch_size=batch_size)
        embeddings = {}

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    for cls, loader in enumerate(loaders):
        print(f"Currently Processing Class: {cls}")
        prev_param = 0
        curr_param = 0
        curr_cls_embeddings = torch.zeros((num_samples[cls], num_params))

        for i, (X, y) in enumerate(loader):
            print(f"Batch: {i}")
            # forward pass -- getting the outputs
            X = X.to(device)
            activation = {}
            out = model(X)
            prev_param = 0
            curr_param = 0
            curr_batch_embeddings = torch.zeros((X.shape[0], num_params))
            curr_batch_embeddings = batch_embeddings(
                curr_batch_embeddings, conv_param_layers, linear_param_layers, model_name = model_name, 
            prev_param = prev_param, curr_param = curr_param)

            curr_cls_embeddings[i*batch_size:(i+1)
                                * batch_size, :] += curr_batch_embeddings
        

        torch.save((cls, curr_cls_embeddings), f"{path}tensors/cls_{cls}_img_param_space_embeddings.pt")

    test_embeddings =  torch.zeros((len(test_set), num_params))
    test_targets = torch.zeros(len(test_set))

    for i, (X, y) in enumerate(test_loader):
        if ((i+2)*batch_size > max_examples):
            break
        model.eval()
        X = X.to(device)
        activation = {}
        out = model(X)
        prev_param = 0
        curr_param = 0
        test_batch_embeddings = torch.zeros((X.shape[0], num_params))
        test_batch_embeddings = batch_embeddings(
            test_batch_embeddings, conv_param_layers, linear_param_layers, model_name = model_name, 
            prev_param = prev_param, curr_param = curr_param)
        
        
        test_embeddings[i*batch_size:(i+1)*batch_size, :] += test_batch_embeddings
        test_targets[i*batch_size:(i+1)* batch_size] += y   

    test_embeddings = test_embeddings[:max_examples, :]
    test_targets =  test_targets[:max_examples]

    torch.save((test_embeddings, test_targets), f"{path}tensors/img_param_space_test_embeddings.pt")

    for h in hooks:
        h.remove()
