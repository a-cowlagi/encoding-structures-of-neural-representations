from dataset import create_dataset_approx, CustomTensorDataset
import numpy as np
import torch

class SuppressDataset:
    def __init__(self, dataset: str, num_approx: int = None, retain_var = 0.96):
        train_set, self.x_train_approx, self.y_train_approx, test_set = create_dataset_approx(dataset, num_approx=num_approx)
        self.x_train, self.y_train = train_set.data, train_set.targets
        self.x_test, self.y_test = test_set.data, test_set.targets

        self.orig_x_train_shape, self.orig_x_test_shape = self.x_train.shape, self.x_test.shape
        self.dataset = dataset

        self.x_train_approx = self.x_train_approx.reshape(self.x_train_approx.shape[0], -1)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)

        self._compute_svd(retain_var = retain_var)

    def _compute_svd(self, retain_var = 0.96):
        self.input_mean = np.mean(self.x_train_approx, axis = 0)

        U, S, VT = np.linalg.svd(self.x_train_approx - self.input_mean)

        self.VT = VT
        self.x_train_projs = (self.x_train - self.input_mean)@(self.VT.T)
        self.x_test_projs = (self.x_test - self.input_mean)@(self.VT.T)

        sum_sing_vals = S.sum()
        num_vals, curr_sum = 0, 0.0
        
        while (curr_sum < retain_var * sum_sing_vals):
            curr_sum += S[num_vals]
            num_vals += 1

        self.retained_vals = num_vals


    def generate_suppressed(self, start_ind: int = 0, window_size: int = 30, transform = None):
        drop_cols = np.arange(start_ind, start_ind + window_size, 1, dtype = int)
        
        relevant_x_train_projs = np.delete(self.x_train_projs, drop_cols, axis = 1)
        relevant_x_test_projs = np.delete(self.x_test_projs, drop_cols, axis = 1)

        relevant_vecs = np.delete(self.VT, drop_cols, axis = 0)

        suppressed_x_train = relevant_x_train_projs @ relevant_vecs
        suppressed_x_test = relevant_x_test_projs @ relevant_vecs

        suppressed_x_train = suppressed_x_train.reshape(self.orig_x_train_shape).astype('float32')
        suppressed_x_test = suppressed_x_test.reshape(self.orig_x_test_shape).astype('float32')

        suppressed_x_train, y_train = torch.from_numpy(suppressed_x_train), self.y_train
        suppressed_x_test, y_test = torch.from_numpy(suppressed_x_test), self.y_test

        if (self.dataset == "cifar10"):
            suppressed_x_train = torch.permute(suppressed_x_train, (0, 3, 1, 2))
            suppressed_x_test = torch.permute(suppressed_x_test, (0, 3, 1, 2))

        suppressed_trainset = CustomTensorDataset((suppressed_x_train, y_train), transform = transform)
        suppressed_testset = CustomTensorDataset((suppressed_x_test, y_test), transform = transform)

        return suppressed_trainset, suppressed_testset