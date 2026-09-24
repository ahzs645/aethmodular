import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
from vibes.absorbance_estimators.map import MAPEstimator


def norm_cdf(z):
    """
    Calculates the cumulative distribution function (CDF) of the standard normal distribution for a given value.

    Args:
    ----------
    z : float or array-like
        The value(s) at which to evaluate the standard normal CDF.

    Returns
    ----------
    float or ndarray
        The CDF value(s) corresponding to the input z.

    Notes
    ----------
    This function uses the `erf` function for computation.
    """
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def norm_pdf(z):
    """
    Compute the value of the standard normal probability density function (PDF) at a given point.

    Args:
    ----------
    z : float or array-like
        The point(s) at which to evaluate the standard normal PDF.

    Returns
    ----------
    float or ndarray
        The value(s) of the standard normal PDF at the specified point(s).
    """
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def vibes_als_elbo(y, tau, nu, d, mu, W):
    """
    Computes the Evidence Lower Bound (ELBO) for the model y = mu + Wx + a, where "x" consists of iid standard normal random variables,
    and "a" of iid random variables distributed according the the Gibbs distribution corresponding to the asymmetrically weighted 
    least squares (ALS) loss. The variational approximation for x|y consists of independent, but not identical, normal random variables.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    tau : float
        Asymmetry parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation.
    d : np.ndarray
        Standard deviation of the components of the variational approximation.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
        Leading scaled right singular vectors obtained from a SVD of the centered intereference examples.

    Returns
    ----------
    float
        The computed ELBO value.
    """
    p, c = y.shape[0], W.shape[1]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    g = (theta**2 + delta**2) * (1 - tau) + (2 * tau - 1) * (
        delta * theta * norm_pdf(z) + (theta**2 + delta**2) * norm_cdf(z)
    )
    sigma_hat = 2 * np.mean(g)

    return (
        (c - p) / 2
        + p * np.log(4 / np.pi) / 2
        - p * np.log(sigma_hat) / 2
        + p * np.log(tau) / 2
        - p * np.log(1 + np.sqrt(tau / (1 - tau)))
        - 0.5 * np.sum(nu**2)
        - 0.5 * np.sum(d**2)
        + np.sum(np.log(d))
    )


def vibes_als_jac_elbo(y, tau, nu, d, mu, W):
    """
    Computes the gradients (Jacobian) of tau and the variational parameters in vibes_als_elbo.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    tau : float
        Asymmetry parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation.
    d : np.ndarray
        Standard deviation of the components of the variational approximation.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
        Leading scaled right singular vectors obtained from a SVD of the centered intereference examples.

    Returns
    ----------
    dtau : float
        Gradient of the ELBO with respect to tau.
    dnu : np.ndarray
        Gradient of the ELBO with respect to nu.
    dd : np.ndarray
        Gradient of the ELBO with respect to d.
    """
    p = y.shape[0]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)

    g = (theta**2 + delta**2) * (1 - tau) + (2 * tau - 1) * (
        delta * theta * phi + (theta**2 + delta**2) * Phi
    )
    sigma_hat = 2 * np.mean(g)

    a = np.sqrt(tau / (1 - tau))

    dtau = (
        -np.sum(2 * delta * theta * phi + (theta**2 + delta**2) * (2 * Phi - 1))
        / sigma_hat
        + p / tau / 2
        - p / (a * (1 + a) * (1 - tau) ** 2) / 2
    )
    dnu = (
        W.T @ (2 * theta * (1 - tau) + 2 * (2 * tau - 1) * (delta * phi + theta * Phi))
    ) / sigma_hat - nu

    dd = 1 / d - d - 2 * ((W**2).T @ (1 - tau + (2 * tau - 1) * Phi)) * (d / sigma_hat)

    return dtau, dnu, dd


def vibes_als_optim_prep(
    y, tau_init, nu_init, d_init, mu, W, tau_min=1e-5, tau_max=0.25, sd_min=1e-10
):
    """
    Prepares the objective function, its Jacobian, initial parameters, and bounds for optimization
    of the vibes_als_elbo function w.r.t. tau and the variational parameters.

    Args:
    ----------
        y : np.ndarray
            Input spectrum.
        tau_init : float
            Starting value for the asymmetry parameter in (0, 1).
        nu_init : np.ndarray
            Intitial values for the means of the components of the variational approximation.
        d_init : np.ndarray
            Intitial values for the standard deviations of the components of the variational approximation.
        mu : np.ndarray
            Mean interference spectrum.
        W : np.ndarray
            Leading scaled right singular vectors obtained from a SVD of the centered intereference examples.
        tau_min : float
            Minimum bound for tau. Default is 1e-5.
        tau_max : float
            Maximum bound for tau. Default is 0.25.
        sd_min : float
            Minimum bound for d parameters. Default is 1e-10.

    Returns:
    ----------
        fun (callable): Objective function to be minimized (negative ELBO).
        jac_fun (callable): Jacobian of the objective function.
        init (np.ndarray): Initial parameter vector for optimization.
        bnds (tuple): Bounds for each parameter in the optimization.
    """
    c = nu_init.shape[0]
    nu_slice = slice(1, 1 + c)
    d_slice = slice(1 + c, None)

    def fun(pars):
        return -vibes_als_elbo(y, pars[0], pars[nu_slice], pars[d_slice], mu, W)

    def jac_fun(pars):
        dtau, dnu, dd = vibes_als_jac_elbo(
            y, pars[0], pars[nu_slice], pars[d_slice], mu, W
        )
        return -np.concatenate(([dtau], dnu, dd))

    init = np.concatenate(([tau_init], nu_init, d_init))
    bnds = ((tau_min, tau_max),) + ((-np.inf, np.inf),) * c + ((sd_min, np.inf),) * c

    return fun, jac_fun, init, bnds


def als_sigma_hat(y, tau, nu, d, mu, W):
    """
    Computes the estimated regularization/temperature parameter corresponding to the ALS loss.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    tau : float
        Asymmetry loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation.
    d : np.ndarray
        Standard deviation of the components of the variational approximation.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
       Leading scaled right singular vectors from the centered intereference examples.

    Returns
    ----------
    sigma_hat : float
        Computed sigma_hat value.
    """
    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    g = (theta**2 + delta**2) * (1 - tau) + (2 * tau - 1) * (
        delta * theta * norm_pdf(z) + (theta**2 + delta**2) * norm_cdf(z)
    )
    sigma_hat = 2 * np.mean(g)

    return sigma_hat

def vibes_pb_elbo(y, tau, nu, d, mu, W):
    """
    Computes the Evidence Lower Bound (ELBO) for the model y = mu + Wx + a, where "x" consists of iid standard normal random variables,
    and "a" of iid random variables distributed according the the Gibbs distribution corresponding to the pinball (PB) loss. The variational
    approximation for x|y consists of independent, but not identical, normal random variables.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    tau : float
        Asymmetry parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation.
    d : np.ndarray
        Standard deviation of the components of the variational approximation.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
        Leading scaled right singular vectors obtained from a SVD of the centered intereference examples.

    Returns
    ----------
    float
        The computed ELBO value.
    """
    p, c = y.shape[0], W.shape[1]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    g = theta * norm_cdf(z) + delta * norm_pdf(z) + (tau - 1) * theta
    sigma_hat = np.mean(g)

    return (
        p * np.log(tau)
        + p * np.log(1 - tau)
        - p * np.log(sigma_hat)
        - 0.5 * np.sum(nu**2)
        - 0.5 * np.sum(d**2)
        + np.sum(np.log(d))
        + (c / 2 - p)
    )


def vibes_pb_jac_elbo(y, tau, nu, d, mu, W):
    """
    Computes the gradients (Jacobian) of tau and the variational parameters in vibes_pb_elbo.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    tau : float
        Asymmetry parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation.
    d : np.ndarray
        Standard deviation of the components of the variational approximation.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
        Leading scaled right singular vectors obtained from a SVD of the centered intereference examples.

    Returns
    ----------
    dtau : float
        Gradient of the ELBO with respect to tau.
    dnu : np.ndarray
        Gradient of the ELBO with respect to nu.
    dd : np.ndarray
        Gradient of the ELBO with respect to d.
    """
    p = y.shape[0]

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)
    g = theta * Phi + delta * phi + (tau - 1) * theta
    sigma_hat = np.mean(g)

    dtau = p / tau - p / (1 - tau) - np.sum(theta) / sigma_hat
    dnu = -(W.T @ ((1 - tau) - Phi)) / sigma_hat - nu
    dd = 1 / d - d - ((W**2).T @ (phi / delta)) * (d / sigma_hat)

    return dtau, dnu, dd


def vibes_pb_optim_prep(
    y, tau_init, nu_init, d_init, mu, W, tau_min=1e-5, tau_max=0.25, sd_min=1e-10
):
    """
    Prepares the objective function, its Jacobian, initial parameters, and bounds for optimization
    of the vibes_pb_elbo function w.r.t. tau and the variational parameters.

    Args:
    ----------
        y : np.ndarray
            Input spectrum.
        tau_init : float
            Starting value for the asymmetry parameter in (0, 1).
        nu_init : np.ndarray
            Intitial values for the means of the components of the variational approximation.
        d_init : np.ndarray
            Intitial values for the standard deviation of the components of the variational approximation.
        mu : np.ndarray
            Mean interference spectrum.
        W : np.ndarray
            Leading scaled right singular vectors obtained from a SVD of the centered intereference examples.
        tau_min : float
            Minimum bound for tau. Default is 1e-5.
        tau_max : float
            Maximum bound for tau. Default is 0.25.
        sd_min : float
            Minimum bound for d parameters. Default is 1e-10.

    Returns:
    ----------
        fun (callable): Objective function to be minimized (negative ELBO).
        jac_fun (callable): Jacobian of the objective function.
        init (np.ndarray): Initial parameter vector for optimization.
        bnds (tuple): Bounds for each parameter in the optimization.
    """
    c = nu_init.shape[0]
    nu_slice = slice(1, 1 + c)
    d_slice = slice(1 + c, None)

    def fun(pars):
        return -vibes_pb_elbo(y, pars[0], pars[nu_slice], pars[d_slice], mu, W)

    def jac_fun(pars):
        dtau, dnu, dd = vibes_pb_jac_elbo(
            y, pars[0], pars[nu_slice], pars[d_slice], mu, W
        )
        return -np.concatenate(([dtau], dnu, dd))

    init = np.concatenate(([tau_init], nu_init, d_init))
    bnds = ((tau_min, tau_max),) + ((-np.inf, np.inf),) * c + ((sd_min, np.inf),) * c

    return fun, jac_fun, init, bnds


def pb_sigma_hat(y, tau, nu, d, mu, W):
    """
    Computes the estimated regularization/temperature parameter corresponding to the PB loss.

    Args:
    ----------
    y : np.ndarray
        Input spectrum.
    tau : float
        Asymmetry loss parameter in (0, 1).
    nu : np.ndarray
        Means of the components of the variational approximation.
    d : np.ndarray
        Standard deviation of the components of the variational approximation.
    mu : np.ndarray
        Mean interference spectrum.
    W : np.ndarray
        Leading scaled right singular vectors obtained from a SVD of the centered intereference examples.

    Returns
    ----------
    sigma_hat : float
        Computed sigma_hat value.
    """

    theta = y - mu - W @ nu
    delta = np.sqrt(np.sum((W * d) ** 2, axis=1))
    z = theta / delta

    Phi = norm_cdf(z)
    phi = norm_pdf(z)
    g = theta * Phi + delta * phi + (tau - 1) * theta
    sigma_hat = np.mean(g)

    return sigma_hat


class VibeSpec:
    """
    Variational-Inference-based Background Elimination in Spectroscopy (VIBES) 
    
    Calibrates the hyperparameters of the probalistic model by maximizing the 
    ELBO and initializes the corresponding MAP solver.

    Args:
    ----------
    c : int
        Number of components in the interference PCA model.
    tau_init : float, optional
        Initial value for asymmetry parameter. Defaults to 0.1.
    nu_init : array-like, optional
        Initial values for nu parameter. Defaults to zeros of length `c`.
    d_init : array-like, optional
        Initial values for d parameter. Defaults to ones of length `c`.
    loss : str, optional
        Loss function to use. Options are "PB" (default) or "ALS".

    Attributes
    ----------
    tau : float
        Estimated asymmetry parameter.
    nu : array-like
        Estimated means of the variational approximation.
    d : array-like
        Estimated standard deviations of the variational approximation.
    sigma_hat : array-like
        Estimated regularization/temperature parameter.
    map_solver : MAPEstimator
        Solver initialized with the calibrated tau and sigma_hat parameters.

    Methods
    -------
    fit(self, y, mu, W, mit=10000, tau_min=1e-5, tau_max=0.5 + 1e-5, sd_min=1e-10)
        Calibrate hyper parameters by maximization of the ELBO.
    """

    def __init__(
        self,
        c,
        tau_init=None,
        nu_init=None,
        d_init=None,
        loss="PB",
    ):
        self.loss = loss
        self.c = c

        if tau_init is None:
            self.tau_init = 0.1
        else:
            self.tau_init = tau_init

        if nu_init is None:
            self.nu_init = np.zeros(self.c)
        else:
            self.nu_init = nu_init

        if d_init is None:
            self.d_init = np.ones(self.c)
        else:
            self.d_init = d_init

        if loss == "ALS":
            self.optim_prep = vibes_als_optim_prep
            self.comp_sigma_hat = als_sigma_hat

        if loss == "PB":
            self.optim_prep = vibes_pb_optim_prep
            self.comp_sigma_hat = pb_sigma_hat
    
    def fit(self, y, mu, W, mit=10000, tau_min=1e-5, tau_max=0.5 + 1e-5, sd_min=1e-10):
        fun, jac_fun, init, bnds = self.optim_prep(
            y, self.tau_init, self.nu_init, self.d_init, mu, W, tau_min, tau_max, sd_min
        )
        opt = minimize(
            fun,
            init,
            method="L-BFGS-B",
            jac=jac_fun,
            bounds=bnds,
            options={"maxiter": mit},
        )

        self.tau, self.nu, t0 = opt.x[0], opt.x[1 : (1 + self.c)], self.c + 1
        self.d = opt.x[t0:]
        self.elbo = -opt.fun
        self.sigma_hat = self.comp_sigma_hat(y, self.tau, self.nu, self.d, mu, W)
        
        self.map_solver = MAPEstimator(loss=self.loss, tau=self.tau, sigma=self.sigma_hat)