"""Which instrument bends? Response-exponent, triple-collocation and saturation helpers.

Used by ``workflows/run_addis_reference_closure_20261110.py``. Everything here is plain NumPy/SciPy on paired
per-filter values, with no project data access, so it can be tested on synthetic data.

The question these answer: several instruments measure the same filters, none of them is a reference, and
two of them disagree with a large intercept. A straight-line fit of one on the other cannot say which of the
two is non-proportional. A third (and fourth) measurement with independent errors can.

* ``response_exponent``  b in ``Y ∝ X^b``, estimated with an instrument Z so that noise in X does not flatten
  it (a log-log OLS slope is biased toward zero by error in X; this is not).
* ``triple_collocation``  for three measurements of one quantity with mutually independent errors: each one's
  random error and its response exponent relative to the first, with no reference (Stoffelen 1998; McColl et
  al. 2014), here in log space so a power-law response is a straight line.
* ``fit_response`` / ``linearise``  a saturating response ``tau = aL / (1 + aL / tau_inf)`` and its inverse.

None of these establishes an absolute level. They establish shape (who is proportional to whom) and scatter.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

__all__ = [
    "response_exponent", "triple_collocation", "hyperbolic", "patterned", "power_law",
    "fit_response", "linearise", "RESPONSE_MODELS",
]


def _logs(*arrays):
    """Natural logs of the rows where every array is finite and positive."""
    stack = np.vstack([np.asarray(a, dtype=float).reshape(-1) for a in arrays])
    keep = np.all(np.isfinite(stack) & (stack > 0), axis=0)
    return np.log(stack[:, keep])


def _percentiles(values, level):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def response_exponent(y, x, z, *, n_boot=2000, level=0.95, seed=20261110) -> dict[str, float]:
    """Exponent b of ``y ∝ x^b`` with ``z`` as the instrument: ``cov(ln y, ln z) / cov(ln x, ln z)``.

    ``z`` must track the same underlying quantity and have errors independent of the errors in ``x`` and
    ``y``. Then random error in ``x`` cancels, which it does not in a log-log regression of y on x. b = 1
    means y is proportional to x; b < 1 means y is compressed relative to x. Rows where any of the three is
    missing or non-positive are dropped. The interval is a percentile bootstrap over filters.
    """
    ly, lx, lz = _logs(y, x, z)
    n = ly.size
    if n < 10:
        raise ValueError("at least ten complete positive triplets are required")

    def estimate(index):
        c = np.cov(np.vstack([ly[index], lx[index], lz[index]]))
        return c[0, 2] / c[1, 2]

    everything = np.arange(n)
    rng = np.random.default_rng(seed)
    boot = [estimate(rng.integers(0, n, n)) for _ in range(int(n_boot))]
    lo, hi = _percentiles(boot, level)
    return {"n": int(n), "exponent": float(estimate(everything)), "lo": lo, "hi": hi}


def triple_collocation(a, b, c, *, n_boot=2000, level=0.95, seed=20261110) -> dict[str, object]:
    """Log-space triple collocation of three measurements of one quantity.

    Model: ``ln m_i = alpha_i + beta_i ln T + e_i`` with the three errors mutually independent and independent
    of the truth T. Returns, for each measurement, ``exponent`` (beta_i / beta_a, so the first is 1 by
    construction) and ``error_sd``, the random error as a fractional (log) standard deviation rescaled to the
    first measurement's response so the three are comparable. Also ``signal_sd``, the spread of the truth in
    those units. A negative estimated error variance (possible in small samples or when an error is shared)
    is reported as ``nan`` rather than clipped.

    Shared error breaks the method: two optical instruments that both respond to the same non-EC absorber are
    not independent in that respect. Read the result as scatter about a common signal, not as accuracy.
    """
    logs = _logs(a, b, c)
    n = logs.shape[1]
    if n < 10:
        raise ValueError("at least ten complete positive triplets are required")

    def estimate(index):
        cov = np.cov(logs[:, index])
        c11, c22, c33 = cov[0, 0], cov[1, 1], cov[2, 2]
        c12, c13, c23 = cov[0, 1], cov[0, 2], cov[1, 2]
        beta2, beta3 = c23 / c13, c23 / c12          # relative to measurement 1
        signal = c12 * c13 / c23                      # variance of beta_1 ln T
        errors = np.array([c11 - signal, c22 - c12 * c23 / c13, c33 - c13 * c23 / c12])
        scaled = errors / np.array([1.0, beta2 ** 2, beta3 ** 2])
        return np.array([1.0, beta2, beta3]), scaled, signal

    rng = np.random.default_rng(seed)
    exponents, scaled, signal = estimate(np.arange(n))
    boot_exp, boot_err = [], []
    for _ in range(int(n_boot)):
        e, s, _signal = estimate(rng.integers(0, n, n))
        boot_exp.append(e)
        boot_err.append(np.sqrt(np.where(s > 0, s, np.nan)))
    boot_exp, boot_err = np.array(boot_exp), np.array(boot_err)
    rows = []
    for i in range(3):
        e_lo, e_hi = _percentiles(boot_exp[:, i], level)
        s_lo, s_hi = _percentiles(boot_err[:, i], level)
        rows.append({
            "exponent": float(exponents[i]), "exponent_lo": e_lo, "exponent_hi": e_hi,
            "error_sd": float(np.sqrt(scaled[i])) if scaled[i] > 0 else float("nan"),
            "error_sd_lo": s_lo, "error_sd_hi": s_hi,
        })
    return {"n": int(n), "signal_sd": float(np.sqrt(signal)) if signal > 0 else float("nan"), "measurements": rows}


def hyperbolic(loading, a, tau_inf):
    """``tau = aL / (1 + aL / tau_inf)``: proportional at low loading, levelling off at ``tau_inf``."""
    x = a * np.asarray(loading, dtype=float)
    return x / (1.0 + x / tau_inf)


def patterned(loading, a, covered):
    """Patterned deposit: a fraction ``covered`` of the area holds all of it; the rest transmits freely."""
    x = a * np.asarray(loading, dtype=float)
    return -np.log((1.0 - covered) + covered * np.exp(-x / covered))


def power_law(loading, c, b):
    """``tau = c L^b``; a description of a bend inside the observed range, with no proportional limit."""
    return c * np.asarray(loading, dtype=float) ** b


def _proportional(loading, a):
    return a * np.asarray(loading, dtype=float)


RESPONSE_MODELS = {
    # name: (function, lower bounds, upper bounds, start as a function of (mean tau / mean loading))
    "proportional": (_proportional, [0.0], [np.inf], lambda r: [r]),
    "hyperbolic": (hyperbolic, [0.0, 0.3], [np.inf, 50.0], lambda r: [2.0 * r, 2.0]),
    "patterned": (patterned, [0.0, 0.05], [np.inf, 0.9999], lambda r: [2.0 * r, 0.8]),
    "power": (power_law, [0.0, 0.0], [np.inf, 3.0], lambda r: [r, 0.5]),
}


def fit_response(loading, tau, model="hyperbolic", *, n_boot=300, level=0.90, seed=20261110) -> dict[str, object]:
    """Least-squares fit of a response model in log space (multiplicative errors).

    Returns the parameters, bootstrap intervals, the RMS log residual and an AIC for comparing the models on
    the same rows. ``loading`` can be any linear measure of what is on the filter (µg per filter from another
    instrument, or gravimetric mass); only its shape matters, so the fitted ``a`` carries that measure's units.
    """
    function, lower, upper, start = RESPONSE_MODELS[model]
    loading = np.asarray(loading, dtype=float).reshape(-1)
    tau = np.asarray(tau, dtype=float).reshape(-1)
    keep = np.isfinite(loading) & np.isfinite(tau) & (loading > 0) & (tau > 0)
    loading, tau = loading[keep], tau[keep]
    if loading.size < 10:
        raise ValueError("at least ten positive pairs are required")

    def solve(index, guess):
        residual = lambda p: np.log(function(loading[index], *p)) - np.log(tau[index])  # noqa: E731
        result = optimize.least_squares(residual, guess, bounds=(lower, upper))
        return result.x, float(np.sqrt(np.mean(result.fun ** 2)))

    params, rms = solve(np.arange(loading.size), start(tau.mean() / loading.mean()))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        try:
            boot.append(solve(rng.integers(0, loading.size, loading.size), params)[0])
        except ValueError:
            continue
    boot = np.array(boot) if boot else np.empty((0, len(params)))
    intervals = [_percentiles(boot[:, i], level) if boot.size else (float("nan"), float("nan")) for i in range(len(params))]
    n = int(loading.size)
    return {
        "model": model, "n": n, "params": [float(p) for p in params], "intervals": intervals,
        "rms_log": rms, "aic": float(n * np.log(rms ** 2) + 2 * len(params)),
    }


def linearise(tau, tau_inf):
    """Invert the hyperbolic response: the optical depth a proportional instrument would have reported.

    ``tau / (1 - tau / tau_inf)``. Values at or above ``tau_inf`` have no finite inverse and return ``nan``.
    The correction grows without bound as tau approaches tau_inf, and so does its noise: a filter at 70 % of
    tau_inf has its measurement error multiplied by about 11. Carry the tau_inf interval through.
    """
    tau = np.asarray(tau, dtype=float)
    ratio = tau / float(tau_inf)
    return np.where(ratio < 1.0, tau / np.where(ratio < 1.0, 1.0 - ratio, np.nan), np.nan)
