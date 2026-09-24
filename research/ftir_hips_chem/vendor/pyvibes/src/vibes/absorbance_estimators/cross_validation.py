import numpy as np
from itertools import product
from vibes.absorbance_estimators.map import  MAPEstimator
from tqdm import tqdm


class BlockCV:
    """
    Blocked cross-validation procedure for hyper-parameter calibration. 
    
    An input spectrum is divided into a number of contiguous blocks and the interference inside each block is 
    reconstructed using the latent interference variables estimated from other blocks. The corresponding reconstructions
    for each block are stacked to form the cross-validated interference and used to extract the cross-validated 
    absorbance. The error is measured by applying the loss function provided to ebs or map to the 
    cross-validated absorbance.

    The above procedure is performed over user supplied grids and the hyperparameter configuration yielding 
    the lowest error is selected.

    Args:
    ----------
    total_wavenums: int
        Number of wavenumbers at which absorbances are estimated.
    loss : str, optional
        Loss function to use. Options are "PB" (default) or "ALS".
    tau_grid : list of float, optional
        Candidate values for the asymmetry parameter. Default is [0.1].
    c_grid : list of int, optional
        Candidate values for the number of PCA components. Default is [10].
    sigma_grid : list of float, optional
        Candidate values for the regularization/temperature parameter. Default is [0.0001].
    num_folds: int, optional
        Number of blocks used in cross validation. Default is 5.
    mit: int, optional
        Maximum number of iterations for the ebs, map procedures. Default is 100.

    Attributes
    ----------
    loss : str
        Provided loss function.
    num_folds : int
        Provided number of blocks.
    mit : int
        Provided maximum number of iterations.
    tau_grid : list of float
        Provided asymmetry parameter grid.
    c_grid : list of int
        Provided number of PCA components grid.
    sigma_grid :  list of float
        Provided regularization/temperature parameter grid.
    map_solver :
        Selected solver used to estimate interference and absorbance.
    cv_errs : numpy.ndarray of float
        Contains the cross-validation errors over the grids.
        
    Methods
    -------
    compute_cv_errors(self, y, mu, W, verbose=False)
        Compute the cross-validation errors over the supplied grids and initialize 
        the solver with the hyper parameters corresponding to the smallest error.
    """
    def __init__(
        self,
        total_wavenums,
        loss="PB",
        tau_grid=[0.1],
        c_grid=[10],
        sigma_grid=[0.0001],
        num_folds=5,
        mit=100
    ):
        self.loss = loss
        self.num_folds = num_folds
        self.mit = mit

        self.tau_grid = tau_grid
        self.c_grid = c_grid
        self.sigma_grid = sigma_grid
        self.folds = np.array_split(np.arange(total_wavenums), num_folds)
        
        self.c = c_grid[0]
        self.tau = tau_grid[0]
        self.sigma = sigma_grid[0]      
        self.map_solver = MAPEstimator(loss=self.loss, tau=self.tau, sigma=self.sigma)
        self.cv_errs = np.full(
            (len(self.tau_grid), len(self.c_grid), len(self.sigma_grid)), np.nan
        )
        
    def compute_cv_errors(self, y, mu, W, verbose=False):
        idx_pairs = list(
            product(
                range(self.cv_errs.shape[0]),
                range(self.cv_errs.shape[1]),
                range(self.cv_errs.shape[2]),
            )
        )

        for i, j, k in tqdm(idx_pairs, disable=not verbose):
            if not np.isnan(self.cv_errs[i, j, k]):
                continue
            
            a = np.zeros(y.shape[0])
            tau = self.tau_grid[i]
            c = self.c_grid[j]
            sigma = self.sigma_grid[k]

            self.map_solver.tau = tau
            self.map_solver.sigma = sigma

            for fold in self.folds:
                keep_idx = np.concatenate([f for f in self.folds if f is not fold])
                _, x = self.map_solver.solve(y[keep_idx], mu[keep_idx], W[keep_idx, :c], self.mit)
                a[fold] = y[fold] - (mu[fold] + W[fold, :c] @ x)

            self.cv_errs[i, j, k] = np.sum(self.map_solver.compute_loss(a))

        idx1, idx2, idx3 = np.unravel_index(np.argmin(self.cv_errs), self.cv_errs.shape)
        self.tau = self.tau_grid[idx1]
        self.c = self.c_grid[idx2]
        self.sigma = self.sigma_grid[idx3]
        
        self.map_solver = MAPEstimator(loss=self.loss, tau=self.tau, sigma=self.sigma)