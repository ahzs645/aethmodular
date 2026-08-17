"""Trial: scipy.odr as the numerical engine for Deming-λ, vs the repo's
closed-form ``calibration_explorer/app.py::deming`` and methcomp's Deming.

λ convention used throughout the repo (and here): λ = σy²/σx², the ratio of
*error* variances, y relative to x. The repo's closed form is the standard
Deming estimator:

    slope = (syy − λ·sxx + sqrt((syy − λ·sxx)² + 4·λ·sxy²)) / (2·sxy)

scipy.odr expresses the same thing through per-axis error standard deviations:
setting sx = 1 and sy = sqrt(λ) makes ODR minimize Σ(dx² + dy²/λ), which is
Deming with error-variance ratio λ.  (Equivalently λ = (sy/sx)².)

methcomp's ``vr`` parameter documents the same convention ("variance of the ys
relative to that of the xs") — but see the cross-check below: methcomp 1.0.0's
internal formula is wrong, so it is compared and reported, not trusted.

Run:  python validate_deming.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import odr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent

LAMBDAS = (0.5, 1.0, 2.96)   # 2.96 = DEMING_LAMBDA_MAC10 in calibration_explorer


# --------------------------------------------------------------------------- #
# the repo's closed form, copied verbatim from calibration_explorer/app.py
# (imported by copy: app.py loads Drive-backed data at import time)
# --------------------------------------------------------------------------- #
def deming_app(x, y, lam):
    """Deming slope/intercept with λ = σy²/σx² (error-variance ratio)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    sxx = np.var(x, ddof=1); syy = np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    slope = ((syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2))
             / (2 * sxy))
    return float(slope), float(y.mean() - slope * x.mean())


# --------------------------------------------------------------------------- #
# candidate scipy.odr wrapper
# --------------------------------------------------------------------------- #
def deming_odr(x, y, lam, beta0=None):
    """Deming regression via scipy.odr with explicit λ = σy²/σx².

    ODR weights each axis by 1/σ²; sx=1, sy=sqrt(λ) fixes the error-variance
    *ratio* without needing absolute uncertainties. ODR iterates from an OLS
    start, so this is a numerical (not closed-form) solution.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if beta0 is None:
        slope0, intercept0 = np.polyfit(x, y, 1)
        beta0 = [slope0, intercept0]
    data = odr.RealData(x, y, sx=1.0, sy=np.sqrt(lam))
    model = odr.Model(lambda beta, xv: beta[0] * xv + beta[1])
    # Default sstol/partol stop ~1e-5 short of the closed form; tighten them.
    output = odr.ODR(data, model, beta0=beta0,
                     sstol=1e-14, partol=1e-14, maxit=200).run()
    return float(output.beta[0]), float(output.beta[1])


def deming_methcomp(x, y, lam):
    """methcomp 1.0.0's Deming point estimate (bootstrap off, no plotting)."""
    from methcomp.regression import _Deming
    fit = _Deming(np.asarray(x, float), np.asarray(y, float),
                  vr=lam, sdr=None, bootstrap=None,
                  x_label="x", y_label="y", title=None, CI=0.95,
                  line_reference=False, line_CI=False, legend=False,
                  color_points="k", color_deming="b")
    return float(fit.beta), float(fit.alpha)


def synthetic(true_slope=2.0, true_intercept=0.5, lam=1.0, n=4000, seed=7):
    """EIV data with a *known* generating slope and error-variance ratio λ."""
    rng = np.random.default_rng(seed)
    xi = rng.uniform(0, 10, n)                    # latent truth
    sigma_x = 0.6
    sigma_y = np.sqrt(lam) * sigma_x              # so σy²/σx² = λ exactly
    x = xi + rng.normal(0, sigma_x, n)
    y = true_slope * xi + true_intercept + rng.normal(0, sigma_y, n)
    return x, y


def main():
    import scipy
    print(f"scipy {scipy.__version__}")
    try:
        import methcomp
        have_methcomp = True
        print(f"methcomp {methcomp.__version__}")
    except ImportError:
        have_methcomp = False
        print("methcomp not installed")

    print("\ntrue slope = 2.0, intercept = 0.5, n = 4000; "
          "data generated with matching error-variance ratio λ")
    header = (f"{'λ':>6} {'app slope':>12} {'odr slope':>12} {'|Δslope|':>10} "
              f"{'|Δintcpt|':>10} {'methcomp slope':>15} {'|Δ| vs app':>11}")
    print(header)
    worst = 0.0
    for lam in LAMBDAS:
        x, y = synthetic(lam=lam)
        s_app, i_app = deming_app(x, y, lam)
        s_odr, i_odr = deming_odr(x, y, lam)
        ds, di = abs(s_odr - s_app), abs(i_odr - i_app)
        worst = max(worst, ds, di)
        if have_methcomp:
            s_mc, _ = deming_methcomp(x, y, lam)
            mc = f"{s_mc:15.6f} {abs(s_mc - s_app):11.4f}"
        else:
            mc = f"{'—':>15} {'—':>11}"
        print(f"{lam:6.2f} {s_app:12.8f} {s_odr:12.8f} {ds:10.2e} {di:10.2e} {mc}")
    print(f"\nscipy.odr vs app closed form: worst |Δ| = {worst:.2e} "
          f"({'PASS' if worst < 1e-6 else 'FAIL'} at 1e-6)")

    # λ-convention sanity: with λ=1 both must equal orthogonal regression,
    # and swapping axes must invert the slope under λ -> 1/λ.
    x, y = synthetic(lam=1.0)
    s_xy, _ = deming_app(x, y, 1.0)
    s_yx, _ = deming_app(y, x, 1.0)
    print(f"axis-swap check at λ=1: slope(x,y)·slope(y,x) = {s_xy * s_yx:.8f} "
          f"(should be 1)")

    if have_methcomp:
        print(
            "\nmethcomp 1.0.0 note: _Deming._derive_params computes\n"
            "  spdxy = np.cov(x, y)[1][1] * (n-1)   # <- [1][1] is var(y), not cov(x,y)\n"
            "  ... + 4*lamb*(ssdy**2)               # <- should be the cross-product spdxy**2\n"
            "so its point estimate does not solve the Deming problem; the\n"
            "columns above quantify the resulting slope error."
        )


if __name__ == "__main__":
    main()
