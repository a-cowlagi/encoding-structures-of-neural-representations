import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler


class StructureComparePair():
    def __init__(self, x1: np.ndarray, x2: np.ndarray, 
                    x_other: Optional[np.ndarray] = None, retain_var = 0.96) -> None:
        assert len(x1.shape) == 2 and len(x2.shape) == 2, "Inputs must be flattened with rows as examples"
        assert x1.shape[1] == x2.shape[1], "Inputs have different # of features"
        
        self.num_features = x1.shape[1]

        self.x1_proj_coeffs, self.x2_proj_coeffs = None, None
        self.s1_proj, self.v1_proj, self.s2_proj, self.v2_proj = None, None, None, None
        mode = "first"

        if x_other is not None:
            assert x_other.shape[1] == self.num_features, "Inputs have different # of features"
            self.other_proj_coeffs = None
            mode = "other"
         
        self.set_proj_coeffs(x1, x2, x_other, mode, retain_var=retain_var)


    def set_proj_coeffs(self, x1: np.ndarray, x2: np.ndarray, 
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

        self.__compute_coeffs(base, x1, x2, retain_var, log_other_coeff)


    def __compute_coeffs(self, base: np.ndarray, x1: np.ndarray, 
                        x2: np.ndarray, retain_var = 0.96, log_other_coeff: bool = False):
        u_base, s_base, vt_base = np.linalg.svd(base, full_matrices = False)
        
        if (log_other_coeff):
            self.other_proj_coeffs = base @ vt_base.T
        
        self.base_pcs_T = vt_base
        self.sing_vals = s_base

        sum_sing_vals = self.sing_vals.sum()
        num_vals, curr_sum = 0, 0.0
        
        while (curr_sum < retain_var * sum_sing_vals):
            curr_sum += self.sing_vals[num_vals]
            num_vals += 1

        # Should projection coefficients be scaled by 1/singular value?
        
        self.x1_proj_coeffs = x1 @ (vt_base.T[:, :num_vals]) @ (np.diag(1/s_base)[:num_vals, :])
        self.x2_proj_coeffs = x2 @ (vt_base.T[:, :num_vals]) @ (np.diag(1/s_base)[:num_vals, :])

    
        self.retained_vals = num_vals

        

    def compare_structure(self, method: str = "correlation",
                threshold_start: int = 0, threshold_end: Optional[int] = None, variance_bound = None) -> float:
        if threshold_end is None and variance_bound is None:
            threshold_end = self.retained_vals
        elif threshold_end is None and variance_bound is not None:
            sum_sing_vals = self.sing_vals.sum()
            num_vals, curr_sum = 0, 0
            while (curr_sum < variance_bound * sum_sing_vals):
                curr_sum += self.sing_vals[num_vals]
                num_vals += 1
            threshold_end = min(threshold_start + num_vals, self.retained_vals)
        
        out = 0.0
        if (method == "correlation"):
            out = self.compare_proj_covs(threshold_start = threshold_start, 
                    threshold_end = threshold_end)
        elif (method == "mutual_info"):
            out = self.compare_proj_mutual_info(threshold_start = threshold_start, 
                    threshold_end = threshold_end)
        elif (method == "sigma_points"):
            out = self.compare_proj_sig_points(threshold_start = threshold_start, 
                    threshold_end = threshold_end)
        else:
            raise NotImplementedError 

        return out

    def compare_proj_covs(self, threshold_start: int, threshold_end: int):
        if self.v1_proj is None or self.v2_proj is None or self.s1_proj is None or self.s2_proj is None:
            _, self.s1_proj, self.v1_proj = np.linalg.svd(self.x1_proj_coeffs)
            _, self.s2_proj, self.v2_proj = np.linalg.svd(self.x2_proj_coeffs)
            self.cov_x1 = np.cov(self.x1_proj_coeffs.T)
            self.cov_x2 = np.cov(self.x2_proj_coeffs.T)

        distance = np.linalg.norm(self.cov_x1[threshold_start:threshold_end,threshold_start:threshold_end] - self.cov_x2[threshold_start:threshold_end,threshold_start:threshold_end])

        # Which distance to use??
        # distance = np.linalg.norm((self.v1_proj.T[:, threshold_start:threshold_end]) 
        #                        -  (self.v2_proj.T[:, threshold_start:threshold_end]), ord = "fro")


        # distance = np.linalg.norm((np.diag(1/self.s1_proj)[threshold_start:threshold_end, :]) @ (self.v1_proj.T[:, threshold_start:threshold_end]) 
        #                        -  (np.diag(1/self.s2_proj)[threshold_start:threshold_end, :]) @ (self.v2_proj.T[:, threshold_start:threshold_end]), ord = "fro")
        
        return distance 

        
   

    def compare_proj_mutual_info(self, threshold_start: int, threshold_end: int):
        return float("inf")

    def compare_proj_sig_points(self, threshold_start: int, threshold_end: int):
        return float("inf")