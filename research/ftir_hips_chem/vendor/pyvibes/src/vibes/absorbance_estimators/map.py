import cvxpy as cp
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm
from vibes.utils.loss_functions import pb_loss, als_loss


def map_als(y, mu, W, tau, sigma, mit=100, verbose=False):
    """
    Computes the maximum a posteriori (MAP) solution for the latent interference variables according to the probabilistic model 
    under the asymmetrically weighted least squares loss and estimates the interference.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
        Leading scaled right singular vectors from the centered intereference examples.
    tau : float
        Asymmetry loss parameter in (0, 1).
    sigma: float
        Regularization/temperature parameter.
    mit: int, optional
        Maximum number of allowable iterations. Default is 100.
    verbose: bool, optional
        Should progress be printed? Default is False.

    Returns
    ----------
    z : numpy.ndarray of float
        Estimated interference.
    x : numpy.ndarray of float
        Estimated latent interference variables.
        
    Notes
    ----------
    1. The sigma = 0 case corresponds to EBS.
    2. Estimation is done by iterative applying linear regression or ridge with observational weightes.
    3. Observational weights are updated following each iteration.
    """
    if sigma == 0:
        reg_mod = LinearRegression(fit_intercept=False)

    else:
        reg_mod = Ridge(fit_intercept=False, alpha=sigma / 2, solver="svd")

    w = np.repeat(tau, y.shape[0])

    iterator = range(mit)
    iterator = tqdm(iterator) if verbose else iterator

    for i in iterator:
        reg_mod.fit(W, y - mu, sample_weight=w)
        z = mu + reg_mod.predict(W)
        wold = w
        w = np.repeat(tau, y.shape[0])
        w[y - z < 0] = 1 - tau
        if np.all(wold == w):
            break
        if verbose:
            print(i, end=" \r")

    x = reg_mod.coef_

    return z, x


def map_pb(y, mu, W, tau, sigma, mit=100, verbose=False):
    """
    Computes the maximum a posteriori (MAP) solution for the latent interference variables according to the probabilistic model 
    under the pinball loss and estimates the interference.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
        Leading scaled right singular vectors from the centered intereference examples.
    tau : float
        Asymmetry loss parameter in (0, 1).
    sigma: float
        Regularization/temperature parameter.
    mit: int 
        Ignored. Included for convenience and consistency with related function calls.
    verbose: bool, optional
        Should progress be printed? Default is False.

    Returns
    ----------
    z : numpy.ndarray of float
        Estimated interference.
    x : numpy.ndarray of float
        Estimated latent interference variables.

    Notes
    ----------
    1. The sigma = 0 case corresponds to EBS.
    2. Estimation is done using cvx.
    """
    x = cp.Variable(W.shape[1])

    if sigma == 0:
        prob = cp.Problem(
            cp.Minimize(
                cp.sum(0.5 * cp.abs(y - mu - W @ x) + (tau - 0.5) * (y - mu - W @ x))
            )
        )

    else:
        prob = cp.Problem(
            cp.Minimize(
                cp.sum(0.5 * cp.abs(y - mu - W @ x) + (tau - 0.5) * (y - mu - W @ x))
                + 0.5 * sigma * cp.sum_squares(x)
            )
        )

    prob.solve(solver=cp.CLARABEL, verbose=verbose)
    z = mu + W @ x.value

    return z, x.value


class MAPEstimator:
    """
    Convenience class for working with the ALS and PB solvers.

    Args:
    ----------
    loss : str, optional
        Loss function to use. Options are "PB" (default) or "ALS".
    tau : float, optional
        Asymmetry parameter. Default is 0.1.
    sigma : list of float, optional
        Regularization/temperature parameter. Default is 0 which corresponds to EBS.
    
    Attributes
    ----------
    loss : str
        Provided loss function.
    tau : float
        Provided asymmetry parameter.
    sigma : float
        Provided regularization/temperature parameter.

    Methods
    -------
    compute_loss(self, a)
        Computes the loss assocaited with the absorbances "a".

    solve(self, y, mu, W, mit=100, verbose=False)
        Applies the initialized solver in order to estimate the interference
        for a given set of inputs.
    """
    def __init__(
        self,
        loss ="PB",
        tau = 0.1,
        sigma = 0,
    ):
        self.loss = loss
        self.tau = tau
        self.sigma = sigma
    
    def compute_loss(self, a):
        if self.loss == "PB":
            return  pb_loss(a, tau=self.tau)
        elif self.loss == "ALS":
            return als_loss(a, tau=self.tau)
        
    def solve(self, y, mu, W, mit=100, verbose=False):
        if self.loss == "PB":
            return map_pb(y, mu, W, self.tau, self.sigma, mit=mit, verbose=verbose)
        elif self.loss == "ALS":
            return map_als(y, mu, W, self.tau, self.sigma, mit=mit, verbose=verbose)
        