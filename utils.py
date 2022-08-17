import os
from models.fc import *
from models.wide_resnet import wide_resnet_t
from models.wide_resnet_1 import WideResNet
from models.all_cnn import allcnn_t
from models.lenet import lenet


def create_path(model_name, args, num_true, num_random, dataset):
    path = model_name
    for item in args:
        path += "_" + str(item)

    path += "_" + str(num_true) + "_" + str(num_random) + "_" + dataset

    path = os.path.join("./runtime_generated/", path, "")

    return path


def mkdir(path):
    isfolder = os.path.exists(path)
    if not isfolder:
        os.makedirs(path)



def get_model_class(model_name):
    if model_name == "fc":
        mc = Network
    if model_name == "wr":
        mc = WideResNet
    if model_name == "all_cnn":
        mc = allcnn_t
    if model_name == "lenet":
        mc = lenet

    return mc


def get_args(model_name = "lenet", num_true = 55000, num_prior = 5000, num_random = 0, num_approx = 10000, num_classes = 10, dataset = "mnist",
            num_neurons = 600, num_layers = 2, depth = 10, num_channels = 4, widen_factor = 8, drop_out = 0, c1 = 96, c2 = 144):
    # dataset
    num_inplanes = 1 if dataset == "mnist" else 3

    if model_name == "fc":
        args = (num_classes, num_layers, num_neurons)
    if model_name == "wr":
        num_classes = 10
        args = (depth, num_inplanes, num_classes, num_channels, widen_factor, drop_out)
    if model_name == "all_cnn":
        num_classes = 10
        args = (num_classes, c1, c2)
    if model_name == "lenet":
        args = ()
        num_classes = 10
    
    return (num_true, num_prior, num_random, num_approx, num_classes), args
