import time
import numpy as np
from vibes.absorbance_estimators.vibes import VibeSpec
from vibes.absorbance_estimators.cross_validation import BlockCV

def vibes_help_fun(args):
    """
    Help function to process vibes corrections in the Python Scripts.

    Parameters
    ----------
    args : tuple
        consists of the following in order:
        y : np.ndarray
            Input spectrum.
        mu : np.ndarray
            Mean interference spectrum.
        W : np.ndarray
            Right singular vectors obtained from a SVD of the centered interference examples.
        tau : float
            Asymmetry parameter.
        c : int 
            Number of pca components. If c is None it is estimated by maximization of the ELBO over 
            a grid.
        loss : str, optional
            Loss function to use. Options are "PB" (default) or "ALS".
        mit : int, optional
            Maximum number of iterations allowed for L-BFGS-B when maximizing the ELBO.
        sample_id : str 
            Identifyer for the sample being analyzed.

    Returns
    ----------
    res : dict
        Dictionary containing the relevant information following the correction procedure.
    """
    y, mu, W, tau, c, loss, mit, sample_id = args
    
    if tau is not None:
        tau_min, tau_max = tau , tau
        tau_init = tau
        
    else:
        tau_min, tau_max = 1e-5, 0.5 + 1e-5
        tau_init = 0.1
    
    elbos_grid = []
    
    if c is None:
        nu_init = np.array([0])
        d_init = np.array([1])
          
        checkpoint = {"nu": nu_init, "d": d_init, "tau": tau_init}
        
        for i in range(W.shape[1]):
            mod = VibeSpec(c = i+1,loss=loss,nu_init=nu_init,d_init=d_init,tau_init=tau_init) 
            mod.fit(y, mu, W[:, : (i+1)], tau_min=tau_min, tau_max=tau_max,mit=500)
            elbos_grid.append(mod.elbo)
            
            if elbos_grid[-1] == np.max(elbos_grid):
                checkpoint = {"nu": mod.nu, "d": mod.d, "tau": mod.tau}
            
            nu_init = np.append(mod.nu,0)
            d_init = np.append(mod.d,1)
            tau_init = mod.tau
            
        elbos_grid = np.array(elbos_grid)
        c = np.argmax(elbos_grid) + 1        
        
        nu_init = checkpoint["nu"]
        d_init = checkpoint["d"]
        tau_init = checkpoint["tau"]
            
    else:
        nu_init = np.zeros(c)
        d_init = np.ones(c)        
            
    W = W[:, :c]
    mod = VibeSpec(c = c,loss=loss,nu_init=nu_init,d_init=d_init,tau_init=tau_init)
    start = time.time()
    mod.fit(y, mu, W, tau_min=tau_min, tau_max=tau_max, mit=mit)
    z,x = mod.map_solver.solve(y,mu,W)
    end = time.time()
    
    res = {
        "sample_id": sample_id,
        "absorbance": y-z,
        "time": end - start,
        "sigma": mod.sigma_hat,
        "tau":mod.tau,
        "c": c,
        "x": x,
        "d": mod.d,
        "nu": mod.nu,
        "elbos_grid": elbos_grid,
        "elbo": mod.elbo,
        "mit": mit,
    }

    return res

def ebs_help_fun(args):
    """
    Help function to process EBS corrections in the Python Scripts.

    Parameters
    ----------
    args : tuple
        consists of the following in order:
        y : np.ndarray
            Input spectrum.
        mu : np.ndarray
            Mean interference spectrum.
        W : np.ndarray
            Right singular vectors obtained from a SVD of the centered interference examples.
        tau : float
            Asymmetry parameter.
        c : int 
            Number of pca components. If c is None it is estimated by the blocked cross-validation
            procedure over a grid.
        loss : str, optional
            Loss function to use. Options are "PB" (default) or "ALS".
        sample_id : str 
            Identifyer for the sample being analyzed.

    Returns
    ----------
    res : dict
        Dictionary containing the relevant information following the correction procedure.
    """
    y, mu, W, tau, c, loss, sample_id = args
    
    sigma_grid = [0]
    
    if tau is None:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))    
    
    else:
        tau_grid = [tau]
        
    if c is None:
        c_grid = np.arange(1, W.shape[1], 1)
    
    else:
        c_grid = [c]    
    
    mod = BlockCV(total_wavenums=y.shape[0], tau_grid=tau_grid,c_grid=c_grid,sigma_grid=sigma_grid,loss = loss)
    start = time.time()
    
    if tau is None or c is None:   
        mod.compute_cv_errors(y, mu, W)
    
    z,x = mod.map_solver.solve(y, mu, W[:, :mod.c], mod.mit)
    
    end = time.time()

    res = {
        "sample_id": sample_id,
        "absorbance": y-z,
        "time": end - start,
        "sigma": 0,
        "tau":mod.tau,
        "c": mod.c,
        "x": x,
    }

    return res

def map_help_fun(args):
    """
    Help function to process MAP corrections in the Python Scripts.

    Parameters
    ----------
    args : tuple
        consists of the following in order:
        y : np.ndarray
            Input spectrum.
        mu : np.ndarray
            Mean interference spectrum.
        W : np.ndarray
            Right singular vectors obtained from a SVD of the centered interference examples.
        tau : float
            Asymmetry parameter.
        c : int 
            Number of pca components. If c is None it is estimated by the blocked cross-validation
            procedure over a grid.
        loss : str, optional
            Loss function to use. Options are "PB" (default) or "ALS".
        sample_id : str 
            Identifyer for the sample being analyzed.

    Returns
    ----------
    res : dict
        Dictionary containing the relevant information following the correction procedure.
    """
    y, mu, W, tau, c, loss, sample_id = args
    
    if tau is None:
        tau_grid = 0.1 * np.power(2.0, np.arange(-2, 3))    
    
    else:
        tau_grid = [tau]
        
    if c is None:
        c_grid = np.arange(1, W.shape[1], 1)
    
    else:
        c_grid = [c]    
        
    sigma_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    mod = BlockCV(total_wavenums=y.shape[0], tau_grid=tau_grid,c_grid=c_grid,sigma_grid=sigma_grid,loss = loss)
    start = time.time()
    mod.compute_cv_errors(y, mu, W)
            
    z,x = mod.map_solver.solve(y, mu, W[:, :mod.c], mod.mit)
    
    end = time.time()

    res = {
        "sample_id": sample_id,
        "absorbance": y-z,
        "time": end - start,
        "sigma": mod.sigma,
        "tau":mod.tau,
        "c": mod.c,
        "x": x,
    }

    return res