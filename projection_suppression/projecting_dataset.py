from dataset import create_dataset_approx, CustomTensorDataset
import numpy as np
import torch
import pywt

class ProjectDataset:
    def __init__(self, dataset: str, num_approx: int = None, mode = "covariance", retain_var = 0.96):
        train_set, self.x_train_approx, self.y_train_approx, test_set = create_dataset_approx(dataset, num_approx=num_approx)
        self.x_train, self.y_train = train_set.data, train_set.targets
        self.x_test, self.y_test = test_set.data, test_set.targets

        self.orig_x_train_shape, self.orig_x_test_shape = self.x_train.shape, self.x_test.shape
        self.dataset = dataset
            
        
        self.mode = mode

        if mode == "covariance":
            self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
            self.x_train_approx = self.x_train_approx.reshape(self.x_train_approx.shape[0], -1)

            self._compute_svd(retain_var = retain_var)
        elif mode == "wavelet":
            self.x_train = self.x_train.astype("float32") / 255.
            self.x_test = self.x_test.astype("float32") / 255.
            self._compute_wavelet()
        

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


    def _compute_wavelet(self):
        self.wavelet_coeffs_train, self.wavelet_slice_train, self.wavelet_coeffs_train_argsort = self._wavelet_dec(self.x_train)
        self.wavelet_coeffs_test, self.wavelet_slice_test, self.wavelet_coeffs_test_argsort = self._wavelet_dec(self.x_test)
        self.retained_vals = self.wavelet_coeffs_train_argsort.shape[1]
        
    def generate_suppressed(self, start_ind: int = 0, window_size: int = 30, transform = None):
        if self.mode == "covariance":
            return self._generate_suppressed_covariance(start_ind, window_size, transform)
        elif self.mode == "wavelet":
            return self._generate_suppressed_wavelet(start_ind, window_size, transform)

    def generate_projected(self, start_ind: int = 0, window_size: int = 30, transform = None):
        if self.mode == "covariance":
            return self._generate_projected_covariance(start_ind, window_size, transform)
        if self.mode == "wavelet":
            return self._generate_projected_wavelet(start_ind, window_size, transform)
    

    def _generate_suppressed_covariance(self, start_ind: int = 0, window_size: int = 30, transform = None):
        drop_cols = np.arange(start_ind, start_ind + window_size, 1, dtype = int)
        
        relevant_x_train_projs = np.delete(self.x_train_projs, drop_cols, axis = 1)
        relevant_x_test_projs = np.delete(self.x_test_projs, drop_cols, axis = 1)

        relevant_vecs = np.delete(self.VT, drop_cols, axis = 0)

        suppressed_x_train = relevant_x_train_projs @ relevant_vecs
        suppressed_x_test = relevant_x_test_projs @ relevant_vecs

        suppressed_trainset, suppressed_testset = self._convert_to_dataset(suppressed_x_train, suppressed_x_test, transform)
        
        return suppressed_trainset, suppressed_testset

    def _generate_projected_covariance(self, start_ind: int = 0, window_size: int = 30, transform = None):
        relevant_x_train_projs = self.x_train_projs[:, start_ind: start_ind + window_size]
        relevant_x_test_projs = self.x_test_projs[:, start_ind: start_ind + window_size]

        relevant_vecs = self.VT[start_ind: start_ind + window_size, :]

        suppressed_x_train = relevant_x_train_projs @ relevant_vecs
        suppressed_x_test = relevant_x_test_projs @ relevant_vecs

        suppressed_trainset, suppressed_testset = self._convert_to_dataset(suppressed_x_train, suppressed_x_test, transform)
        
        return suppressed_trainset, suppressed_testset
    
    def _generate_suppressed_wavelet(self, start_ind: int = 0, window_size: int = 30, transform = None):
        suppressed_x_train = self._wavelet_supp(self.wavelet_coeffs_train, self.wavelet_slice_train, self.wavelet_coeffs_train_argsort, start_ind, start_ind + window_size)
        suppressed_x_test = self._wavelet_supp(self.wavelet_coeffs_test, self.wavelet_slice_test, self.wavelet_coeffs_test_argsort, start_ind, start_ind + window_size)
        
        suppressed_trainset, suppressed_testset = self._convert_to_dataset(suppressed_x_train, suppressed_x_test, transform)
    
        return suppressed_trainset, suppressed_testset

    def _generate_projected_wavelet(self, start_ind: int = 0, window_size: int = 30, transform = None):
        projected_x_train = self._wavelet_proj(self.wavelet_coeffs_train, self.wavelet_slice_train, self.wavelet_coeffs_train_argsort, start_ind, start_ind + window_size, project = True)
        projected_x_test = self._wavelet_proj(self.wavelet_coeffs_test, self.wavelet_slice_test, self.wavelet_coeffs_test_argsort, start_ind, start_ind + window_size, project = True)

        projected_trainset, projected_testset = self._convert_to_dataset(projected_x_train, projected_x_test, transform)

        return projected_trainset, projected_testset

    def _convert_to_dataset(self, train_to_convert, test_to_convert, transform):
        train_to_convert = train_to_convert.reshape(self.orig_x_train_shape).astype('float32')
        test_to_convert = test_to_convert.reshape(self.orig_x_test_shape).astype('float32')

        train_to_convert, y_train = torch.from_numpy(train_to_convert), self.y_train
        test_to_convert, y_test = torch.from_numpy(test_to_convert), self.y_test

        if (self.dataset == "cifar10"):
            train_to_convert = torch.permute(train_to_convert, (0, 3, 1, 2))
            test_to_convert = torch.permute(test_to_convert, (0, 3, 1, 2))

        trainset = CustomTensorDataset((train_to_convert, y_train), transform = transform)
        testset = CustomTensorDataset((test_to_convert, y_test), transform = transform)

        return trainset, testset
        
    def _wavelet_dec(self, x):
        wavelet_decomp = pywt.wavedecn(x, wavelet = "db1", axes = (1, 2))
        C, S = pywt.coeffs_to_array(wavelet_decomp, axes = (1, 2))
        C_flat = np.reshape(C, (C.shape[0], -1))
        indices_argsort = np.argsort(np.abs(C_flat), axis = 1)

        return C, S, indices_argsort

    def _wavelet_supp(self, C, S, indices_argsort, suppress_start, suppress_end):
        C_flat = np.reshape(C, (C.shape[0], -1))
        suppressed_x_C = np.zeros_like(C_flat)
        suppressed_x_C += C_flat
        
        suppressed_x_C[np.arange(C_flat.shape[0])[:,None], indices_argsort[:, suppress_start:suppress_end]] = 0
 
        suppressed_coeffs = pywt.array_to_coeffs(np.reshape(suppressed_x_C, C.shape), S)
       
        suppressed_x = pywt.waverecn(suppressed_coeffs, wavelet = "db1", axes = (1, 2))
        return suppressed_x

    def _wavelet_proj(self, C, S, indices_argsort, proj_start, proj_end):
        C_flat = np.reshape(C, (C.shape[0], -1))
        suppressed_x_C = np.zeros_like(C_flat)
        
        suppressed_x_C[np.arange(C_flat.shape[0])[:,None], indices_argsort[:, proj_start:proj_end]] = C_flat[np.arange(C_flat.shape[0])[:,None], indices_argsort[:, proj_start:proj_end]]
        suppressed_coeffs = pywt.array_to_coeffs(np.reshape(suppressed_x_C, C.shape), S)
       
        suppressed_x = pywt.waverecn(suppressed_coeffs, wavelet = "db1", axes = (1, 2))
        return suppressed_x
        
