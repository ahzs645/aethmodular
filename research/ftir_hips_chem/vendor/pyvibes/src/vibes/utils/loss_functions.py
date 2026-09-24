import numpy as np


def pb_loss(a, tau=0.1):
    """
    Compute the component-wise pinball loss values.

    Args:
    ----------
    a : np.ndarray
        The input array for which to compute the loss values.

    Returns
    ----------
    np.ndarray
        The computed pinball loss values.
    """
    return 0.5 * np.abs(a) + (tau - 0.5) * a


def als_loss(a, tau=0.1):
    """
    Compute the component-wise asymmetrically weighted squared loss values.

    Args:
    ----------
    a : np.ndarray
        The input array for which to compute the loss values.

    Returns
    ----------
    np.ndarray
        The computed asymmetrically weighted squared loss values.
    """
    loss = a**2
    loss[a > 0] = tau * loss[a > 0]
    loss[a < 0] = (1 - tau) * loss[a < 0]

    return loss
