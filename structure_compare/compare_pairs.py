import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler


class StructureComparePair():
    def __init__(self, x1: np.ndarray, x2: np.ndarray, 
                    x_other: Optional[np.ndarray] = None, retain_var = 0.96, num_divisions = 10, rank_mode = "projection", set_all = True) -> None:
        """A general-purpose implementation to compare the structure of sets of examples with
        the same dimensionality from geometric and information theoeretic perspectives. 
        
        Currently, the primary inteded comparison scheme is as follows: 
        1) Divide each set of examples into num_divisions subsets of roughly equal sizes,
        ranked by rank_mode. The goal is for the i-th set of examples to be the i-th
        most representative set of examples of the full class. For instance, if 
        x1 = airplanes, then self.x1_divided[0] is the set of most "airplane-like" images
        2) Compare Union(x1_divided[i], x2_divided[i]) to Union(x1_divided[j], x2_divided[j])
        for pairs (i, j).  

        Example use cases: Comparing 2 classes from CIFAR-10, MNIST, or random images

        Args:
            x1 (np.ndarray): 
                            A 2D numpy array (or array-like) object consisting of examples of the first set.
                            The array should be shape (n1, d) where d is the dimensionality of each example 
                            and n1 is the number of examples. An ```AssertionError``` will be thrown if
                            x1 is not 2-dimensional -- if passing in images, they must be flattened appropriately.
            
            x2 (np.ndarray):
                            A 2D numpy array (or array-like) object consisting of examples of the second set.
                            The array should be shape (n2, d) where d is the dimensionality of each example 
                            and n2 is the number of examples. An ```AssertionError``` will be thrown if
                            x2 is not 2-dimensional -- if passing in images, they must be flattened appropriately.
            
            x_other (Optional[np.ndarray], optional): 
                            Set of examples to use as the basis of comparisons, if provided. Otherwise, set to
                            np.vstack((x1, x2)). Defaults to None. 
            
            retain_var (float, optional): 
                            Proportion of variance that must be explained by the retained principal components 
                            of x_other. Defaults to 0.96.

            num_divisions (int, optional): 
                            Number of subsets to divide each set of examples into, ranked by ```rank_mode```. 
                            Defaults to 10.
            
            rank_mode (str, optional): 
                            Can be either "projection" or "reconstruction". If "projection", examples
                            of each ranked according to sum of projection coefficients along their largest
                            principal components (i.e. x1 ranked according to x1's PCs, 
                            x2 ranked according to x2's PCs). If "reconstruction" examples are ranked
                            according to reconstruction error from using the largest principal components,
                            this time computed across both sets of samples. 
            
                            Defaults to "projection".
            
            set_all (bool, optional): 
                            (Deprecated) If true, the projection coefficients of
                            each set of examples  is computed using the PCs of x_other if provided,
                            else along x1. Computed values are stored in x1_proj_coeffs, x2_proj_coeffs.
                            Defaults to True.

        Attributes:
            x1_divided (List[np.ndarray]):
                            x1 partitioned by scheme prescribed by rank_mode. A list with num_divisions
                            np.ndarrays of shape (n1 // num_divisions, d) (last entry may have fewer).
            x2_divided (List[np.ndarray]):
                            x2 partitioned by scheme prescribed by rank_mode. A list with num_divisions
                            np.ndarrays of shape (n2 // num_divisions, d) (last entry may have fewer).

            x1_div_coeffs (List[np.ndarray]): 
                            Projection coefficients for each ranked subset of x1 examples, computed 
                            relative to the PCs of divided_base (x_other if not None, else 
                            np.vstack(x1, x2)).
            x2_div_coeffs (Listp[np.ndarray]): 
                            Projection coefficients for each ranked subset of x2 examples, computed 
                            relative to the PCs of divided_base (x_other if not None, else 
                            np.vstack(x1, x2)).

            x1_proj_coeffs (np.ndarray):
                        Projection coefficients for full set of x1 examples, computed relative
                        to the PCs of base (x_other if not None, else x1), with coefficients scaled by
                        1/singular value for normalization.
            x2_proj_coeffs (np.ndarray): 
                        Projection coefficients for full set of x2 examples, computed relative
                        to the PCs of base (x_other if not None, else x1 -- yes, x1!), with coefficients 
                        scaled by 1/singular value.

        """       
        
        assert len(x1.shape) == 2 and len(x2.shape) == 2, "Inputs must be flattened with rows as examples"
        assert x1.shape[1] == x2.shape[1], "Inputs have different # of features"

        # in case torch tensors are passed in...
        x1, x2 = np.array(x1), np.array(x2)
        self.num_features = x1.shape[1]

        self.x1_proj_coeffs, self.x2_proj_coeffs = None, None
        self.s1_proj, self.v1_proj, self.s2_proj, self.v2_proj = None, None, None, None
        mode = "other"

        if x_other is not None:
            assert x_other.shape[1] == self.num_features, "Inputs have different # of features"
            self.other_proj_coeffs = None
            divided_base = x_other
        else:
            divided_base = np.vstack((x1, x2))
            x_other = divided_base
        
        # Separate each set of examples from "most to least characteristic"
        self.x1_divided, self.x2_divided, self.divided_coeffs = None, None, None

        if (num_divisions != None):
            if (rank_mode == "projection"):
                self.x1_divided = self.divide_examples(examples = x1, base = x1, num_divisions = num_divisions, rank_mode = rank_mode, prop_var = 0.5)
                self.x2_divided = self.divide_examples(examples = x2, base = x2, num_divisions = num_divisions, rank_mode = rank_mode, prop_var = 0.5)
            elif (rank_mode == "reconstruction"): 
                self.x1_divided = self.divide_examples(examples = x1, base = divided_base, num_divisions = num_divisions, rank_mode = rank_mode, prop_var = 0.5)
                self.x2_divided = self.divide_examples(examples = x2, base = divided_base, num_divisions = num_divisions, rank_mode = rank_mode, prop_var = 0.5)
            
            self.num_divisions = num_divisions
            self.divided_coeffs = self.__compute_divided_coeffs(divided_base, retain_var = retain_var)
        else:
            self.x1, self.x2 = x1, x2
            self.num_divisions = None
        
        if (set_all):
            self.set_all_coeffs(x1, x2, x_other, mode, retain_var=retain_var)
        else:
            self.all_retained_vals = self.num_features
            

    def set_all_coeffs(self, x1: np.ndarray, x2: np.ndarray, 
                    x_other: Optional[np.ndarray], mode: str = "first", retain_var = 0.96) -> None:
        log_other_coeff = False
        if (mode == "first"):
            base = x1
        elif (mode == "second"):
            base = x2
        elif (mode == "other"):
            assert x_other is not None, "External samples can't be None when mode is other"
            base = x_other
            log_other_coeff = True
        else: 
            raise NotImplementedError

        self.__compute_all_coeffs(base, x1, x2, retain_var, log_other_coeff)


    def __compute_all_coeffs(self, base: np.ndarray, x1: np.ndarray, 
                        x2: np.ndarray, retain_var = 0.96, log_other_coeff: bool = False):
        u_base, s_base, vt_base = np.linalg.svd(base - np.mean(base, axis = 0), full_matrices = False)
        
        if (log_other_coeff):
            self.other_proj_coeffs = base @ vt_base.T
        
        self.base_pcs_T = vt_base
        self.base_sing_vals = s_base

        sum_sing_vals = self.base_sing_vals.sum()
        num_vals, curr_sum = 0, 0.0
        
        while (curr_sum < retain_var * sum_sing_vals):
            curr_sum += self.base_sing_vals[num_vals]
            num_vals += 1

        self.x1_proj_coeffs = (x1 - np.mean(base, axis = 0)) @ (vt_base.T[:, :num_vals]) # @ (np.diag(1/s_base)[:num_vals, :])
        self.x2_proj_coeffs = (x2 - np.mean(base, axis = 0)) @ (vt_base.T[:, :num_vals]) # @ (np.diag(1/s_base)[:num_vals, :])
    
        self.all_retained_vals = num_vals


    def compare_all_coeffs(self, method: str = "covariance",
                threshold_start: int = 0, threshold_end: Optional[int] = None, variance_bound = None) -> float:
        assert (self.x1_proj_coeffs != None and self.x2_proj_coeffs != None), "Coefficients not set in constructor. Must call \
                                    ```set_all_coeffs``` first"

        if threshold_end is None and variance_bound is None:
            threshold_end = self.all_retained_vals
        elif threshold_end is None and variance_bound is not None:
            sum_sing_vals = self.sing_vals.sum()
            num_vals, curr_sum = 0, 0
            while (curr_sum < variance_bound * sum_sing_vals):
                curr_sum += self.sing_vals[num_vals]
                num_vals += 1
            threshold_end = min(threshold_start + num_vals, self.all_retained_vals)
        
        out = 0.0
        if (method == "covariance"):
            out = self.compare_proj_covs(threshold_start = threshold_start, 
                    threshold_end = threshold_end)
        elif (method == "projection"):
            out = self.compare_proj_overlap(threshold_start = threshold_start, threshold_end = threshold_end)
        else:
            raise NotImplementedError 

        return out

    def compare_proj_covs(self, threshold_start: int, threshold_end: int):
        if self.v1_proj is None or self.v2_proj is None or self.s1_proj is None or self.s2_proj is None:
            self.cov_x1 = np.cov(self.x1_proj_coeffs.T)
            self.cov_x2 = np.cov(self.x2_proj_coeffs.T)

        distance = np.linalg.norm(self.cov_x1[threshold_start:threshold_end,threshold_start:threshold_end] - self.cov_x2[threshold_start:threshold_end,threshold_start:threshold_end])

        return distance 

    def __compare_proj_overlap(self,  threshold_start: int, threshold_end: int):
        if self.v1_proj is None or self.v2_proj is None or self.s1_proj is None or self.s2_proj is None:
            if self.num_divisions != None:
                x1_base = np.vstack(tuple(self.x1_divided))
                x2_base = np.vstack(tuple(self.x2_divided))
            else:
                x1_base, x2_base = self.x1, self.x2

            x1_base = x1_base - np.mean(x1_base, axis = 0)
            x2_base = x2_base - np.mean(x2_base, axis = 0)
            _, self.s1_proj, self.v1_proj = np.linalg.svd(x1_base)
            _, self.s2_proj, self.v2_proj = np.linalg.svd(x2_base)

        overlap = np.linalg.norm(self.v1_proj[threshold_start:threshold_end] @ self.v2_proj[threshold_start:threshold_end].T, ord = 'fro') / np.sqrt(threshold_end - threshold_start)

        return overlap


    def divide_examples(self, examples: np.ndarray, base: np.ndarray, num_divisions = 10, prop_var = 0.5, rank_mode = "projection", num_evecs = None):
        u_base, s_base, v_base_T = np.linalg.svd(base - np.mean(base, axis = 0)) 
        num_vals = 0

        if (num_evecs is None):
            sum_s_examples = s_base.sum()
            curr_sum = 0.0
            
            while (curr_sum < prop_var * sum_s_examples):
                curr_sum += s_base[num_vals]
                num_vals += 1
        else:
            num_vals = num_evecs
        

        if rank_mode == "reconstruction":
            reconstruction_sample = (((examples - np.mean(base, axis = 0)) @ (v_base_T.T)[:, :num_vals])) @ v_base_T[:num_vals, :]
            reconstruction_diff = np.linalg.norm((examples - np.mean(base, axis = 0)) - reconstruction_sample, ord = 2, axis = 1)
            sorted_inds = np.argsort(reconstruction_diff)
        elif rank_mode == "projection":
            within_class_coeffs_sum = (np.abs(examples @ (v_base_T.T)[:, :num_vals])).sum(axis = 1).flatten()
            norms = np.linalg.norm(examples, axis = 1)
            sorted_inds = np.argsort(within_class_coeffs_sum / norms)[::-1]


        sorted_examples = examples[sorted_inds]
        divided_examples = []
        curr_ind, prev_ind = 0, 0
        increment = examples.shape[0] // num_divisions

        while(curr_ind < examples.shape[0]):
            curr_ind += increment
            curr_division = sorted_examples[prev_ind:curr_ind, :]
            divided_examples.append(curr_division)
            prev_ind = curr_ind

        return divided_examples

    def __compute_divided_coeffs(self, divided_base, retain_var = 0.96):
        divided_base_mean = np.mean(divided_base, axis = 0)
        divided_base = divided_base - divided_base_mean
        u_div_base, s_div_base, v_div_base_T = np.linalg.svd(divided_base)

        self.div_base_pcs_T = v_div_base_T
        self.div_base_mean = divided_base_mean
        self.div_sing_vals = s_div_base

        sum_div_sing_vals = self.div_sing_vals.sum()
        num_vals, curr_sum = 0, 0.0
        
        while (curr_sum < retain_var * sum_div_sing_vals):
            curr_sum += self.div_sing_vals[num_vals]
            num_vals += 1

        self.div_retained_vals = num_vals

        self.x1_div_coeffs = []
        self.x2_div_coeffs = []

        for i in range(self.num_divisions):
            curr_x1_div, curr_x2_div = self.x1_divided[i] - divided_base_mean, self.x2_divided[i] - divided_base_mean
            self.x1_div_coeffs.append(curr_x1_div @ v_div_base_T.T[:, :num_vals])
            self.x2_div_coeffs.append(curr_x2_div @ v_div_base_T.T[:, :num_vals])

    def compute_div_overlap(self, window_size: Optional[int] = None) -> np.ndarray:
        """ 
            Computes the overlap between all subset pairs (i, j) in self.x1_divided and self.x2_divided.
            In particular, for each pair (i, j), the principal components of 
            Union(x1_divided[i], x2_divided[i]) and Union(x1_divided[j], x2_divided[j]) are computed.
            
            Then, we compute the overlap between principal components 0:window_size, window_size:2*window_size,
            ... up until self.div_retained_vals (number of eigenvectors to explain retained_var proportion
            of variance). Overlap between matrices A, B whose columns are orthogonal is computed
            as ||A.T @ B||_F^2 / c, where c is the number of columns in A and B

            Returns a 3D tensor, overlaps, where overlaps[i, j, k] specifies overlap between subsets i,j
            for eigenvectors k*window_size : (k + 1)*window_size

        Args:
            window_size (Optional[int]): Desired window size to use for eigenvector computation. 
            If not provided, self.div_retained_vals // 20 is used. Defaults to None.

        Returns:
            overlaps (np.ndarray): 
                    A 3D tensor, overlaps, where overlaps[i, j, k] specifies overlap between subsets i,j
                    for eigenvectors k*window_size : (k + 1)*window_size. 
                    Shape = (self.num_divisions, self.num_divisions, self.div_retained_vals // window_size)
        """      
        assert self.num_divisions != None, "Constructor called without divisions intended. Aborting"

        if window_size is None:
            window_size = self.div_retained_vals // 20
        
        overlaps = np.zeros((self.num_divisions, self.num_divisions, self.div_retained_vals // window_size))
        for i in range(self.num_divisions):
            for j in range(i, self.num_divisions):
                x_combined_i = np.vstack((self.x1_divided[i], self.x2_divided[i]))
                x_combined_j = np.vstack((self.x1_divided[j], self.x2_divided[j]))

                _, _, v_i_T = np.linalg.svd(x_combined_i - np.mean(x_combined_i, axis = 0))
                _, _, v_j_T = np.linalg.svd(x_combined_j - np.mean(x_combined_j, axis = 0))

                for start in range(0, window_size*(self.div_retained_vals // window_size), window_size):
                    curr_overlap = np.linalg.norm(v_i_T[start:start + window_size] @ v_j_T[start:start + window_size].T, ord = 'fro')**2 / (window_size)
                    overlaps[i, j, start // window_size] = curr_overlap

        return overlaps


    def compute_class_overlap(self, window_size: int = 30, stride = 1):
        if self.all_retained_vals == None:
            len_overlaps = self.num_features
        else:
            len_overlaps = self.all_retained_vals
        
        inds = np.arange(0, len_overlaps - window_size, stride)
        overlaps = []
        
        for start in inds:
            curr_overlap = self.__compare_proj_overlap(start, start + window_size)  
            overlaps.append(curr_overlap)

        return np.array(overlaps)


