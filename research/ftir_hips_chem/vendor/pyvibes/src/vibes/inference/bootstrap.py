import numpy as np
from vibes.absorbance_estimators.map import MAPEstimator
from vibes.interference_models.pca import pca
from tqdm import tqdm


def bootstrap_sample(Z):
    """
    Extracts a bootstrap sample from a matrix Z containing interference examples 
    as rows.

    Args:
    ----------
    Z : np.ndarray
        Input interference examples as rows of a matrix.
    
    Returns
    ----------
    np.ndarray
        Bootstrap sample
    """
    idx = np.random.choice(Z.shape[0], size=Z.shape[0], replace=True)
    return Z[idx]

def pca_bootstrap(y, Z, tau, sigma, c, B=100, loss="PB", verbose=False):
    """
    Extracts bootstrap replicates of the estimated interference produced by the 
    map solver for fixed tau, sigma and c by bootstrapping from the interference 
    examples.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    Z : np.ndarray
        Input interference examples as rows of a matrix.
    tau : float
        Asymmetry parameter.
    sigma : float
        Regularization/temperature parameter.
    c : int
        Number of PCA components.
    B : int, optional
        Number of bootstrap replicates. Default is 100.
    loss : str, optional
        Loss function. Can either be "PB" or "ALS". Default is "PB".
    
    Returns
    ----------
    boot_x : np.ndarray
        Bootstrap replicates of the scores.
    boot_z : np.ndarray
        Bootstrap replicates of the interference.
    """
    boot_x = []
    boot_z = []
    map_solver = MAPEstimator(loss= loss, tau = tau, sigma = sigma)

    for b in tqdm(range(B), disable=not verbose):
        Zb = bootstrap_sample(Z)
        mub, _, _, Wb = pca(Zb)
        Wb = Wb[:, :c]
    
        z , x = map_solver.solve(y, mub, Wb)

        boot_x.append(x)
        boot_z.append(z)

    boot_x = np.array(boot_x)
    boot_z = np.array(boot_z)

    return boot_x, boot_z

def compute_lod(boot_z, z, alpha=0.01):
    """
    Computes the limit of detection (LOD) using the method described in the paper.

    Args:
    ----------
    boot_z : np.ndarray
        Bootstrap replicates of the interference as rows of a matrix.
    z : np.ndarray
        Original estimate of the interference.
    alpha : float
        The coverage is 1 - alpha.
        
    Returns
    ----------
    xi : float
        Multiplier of the standard deviations to compute the lod.
    stdev : np.ndarray
        Standard deviations of the bootstrap replicates computed for each wavenumber.
    """
    stdev = np.std(boot_z, axis=0)
    t = (boot_z - z) / stdev
    xi = np.quantile(np.max(t, axis=1), 1 - alpha)

    return xi, stdev
