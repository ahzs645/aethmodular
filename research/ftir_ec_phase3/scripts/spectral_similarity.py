"""Spectral comparison metrics and nearest-analog search over the full library.

The explorer's Analogs tab already scores a pool spectrum against the Addis
*median* with five metrics (cosine/SAM, Pearson, normalized Euclidean,
nearest-neighbour cosine, Mahalanobis in PCA-10 scores). That answers "rank the
library by Addis-likeness". This module answers three questions it cannot:

1. **Where** do two spectra differ, not just how much (band-resolved and
   moving-window correlation).
2. **Which kind** of outlier is a spectrum: inside the model but far from the
   centre (Hotelling T2), or off the model plane entirely (Q residual / SPE).
   The pair is the standard chemometric decomposition; a single Mahalanobis
   number conflates them.
3. **Which library spectra are analogs of a given target filter**, symmetrically
   (mutual k-nearest neighbours) rather than by distance to a pooled median.

Everything is matmul- or PCA-based on row-standardized spectra, so the whole
13.6k-spectrum library runs in seconds and the same call works on Colab against
the full database.

Conventions (shared with the explorer so numbers are comparable):
- similarity always computed on L2-normalized, mean-centred rows unless a metric
  says otherwise, which makes Pearson == cosine == (1 - SAM/pi) monotonically;
- every DISTANCE is oriented so that LOWER = more similar;
- spectra spaces: "raw", "deriv2" (Savitzky-Golay w=11 p=2 d=2, the LOCAL
  representation), or any pre-corrected matrix the caller passes in.

References: Shenk & Westerhaus 1997 (LOCAL, Pearson on derivatives); Kruse 1993
(SAM); Chang 2000 (spectral information divergence); Jackson & Mudholkar 1979
(Q residual / SPE); Reggente et al. 2016 (score-space distance as an FTIR
extrapolation diagnostic); Haghpanah 2018 / Haghighat 2016 (mutual NN matching).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

SAVGOL = dict(window_length=11, polyorder=2, deriv=2)

# Interpretable mid-IR windows (cm-1). Names match the phase-3 usage.
BANDS = {
    "OH_broad": (3100, 3600),
    "aliphatic_CH": (2800, 3000),
    "carbonyl": (1650, 1800),
    "band_1617": (1560, 1680),
    "COO_arom": (1350, 1500),
    "fingerprint": (900, 1300),
}


def to_space(X: np.ndarray, space: str) -> np.ndarray:
    """Represent spectra in `raw` or `deriv2` space."""
    X = np.asarray(X, float)
    if space == "deriv2":
        return savgol_filter(X, axis=1, **SAVGOL)
    if space == "raw":
        return X
    raise ValueError(f"unknown space {space!r} (use raw/deriv2, or pass a matrix)")


def standardize(X: np.ndarray) -> np.ndarray:
    """Mean-centre each row and scale to unit norm: Pearson becomes a matmul."""
    Xc = X - X.mean(axis=1, keepdims=True)
    return Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)


def unit_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------- #
# pairwise similarity
# --------------------------------------------------------------------------- #
def correlation_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n_A, n_B) Pearson r. Rows are standardized here, so this is one matmul."""
    return standardize(A) @ standardize(B).T


def spectral_angle(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """SAM in radians (Kruse 1993): angle between non-centred unit vectors.

    Distinct from Pearson: SAM keeps the offset, so a raised baseline changes
    the angle. On deriv2 spectra the two nearly coincide, which is itself a
    useful check that a ranking is not baseline-driven.
    """
    cos = np.clip(unit_rows(A) @ unit_rows(B).T, -1.0, 1.0)
    return np.arccos(cos)


def spectral_information_divergence(A: np.ndarray, B: np.ndarray,
                                    floor: float = 1e-9) -> np.ndarray:
    """Symmetric KL divergence between spectra read as distributions (Chang 2000).

    Shape-sensitive in a different way from correlation: it weights *relative*
    differences, so a small band that is twice as strong matters as much as a
    large band that is 10% stronger. Absorbances are shifted to be positive and
    normalized to sum 1 before the divergence.
    """
    def as_dist(M):
        M = np.asarray(M, float)
        M = M - M.min(axis=1, keepdims=True) + floor
        return M / M.sum(axis=1, keepdims=True)

    P, Q = as_dist(A), as_dist(B)
    logP, logQ = np.log(P), np.log(Q)
    # sum_j p_j (log p_j - log q_j) + q_j (log q_j - log p_j), vectorized
    return ((P * logP).sum(1)[:, None] + (Q * logQ).sum(1)[None, :]
            - P @ logQ.T - (Q @ logP.T).T)


def band_profile(A: np.ndarray, B: np.ndarray, wn: np.ndarray,
                 bands: dict | None = None) -> dict:
    """Per-band Pearson r between each row of A and each row of B.

    Returns {band_name: (n_A, n_B) r}. This is the "where do they differ"
    answer: two spectra can agree at r=0.97 overall while disagreeing at the
    carbonyl window, and only the band profile shows it.
    """
    bands = bands or BANDS
    wn = np.asarray(wn, float)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (wn >= lo) & (wn <= hi)
        # a band only counts if the grid really covers it: the AIRSpec-corrected
        # grid starts at 1425 cm-1, so fingerprint bands silently truncate to a
        # sliver and return meaningless correlations unless this is enforced
        covered = (min(hi, wn.max()) - max(lo, wn.min())) / (hi - lo)
        if m.sum() < 5 or covered < 0.6:
            continue
        out[name] = correlation_matrix(A[:, m], B[:, m])
    return out


def moving_window_correlation(a: np.ndarray, b: np.ndarray, wn: np.ndarray,
                              width: int = 80, step: int = 20) -> tuple:
    """Sliding-window Pearson r between two single spectra.

    Returns (centres_cm1, r). Localizes disagreement without pre-chosen bands
    (the forensic-IR review's method); use it to *discover* the windows that
    then go into `bands`.
    """
    a, b, wn = np.asarray(a, float), np.asarray(b, float), np.asarray(wn, float)
    centres, rs = [], []
    for start in range(0, len(wn) - width + 1, step):
        sl = slice(start, start + width)
        av, bv = a[sl] - a[sl].mean(), b[sl] - b[sl].mean()
        denom = np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12
        centres.append(float(wn[sl].mean()))
        rs.append(float(av @ bv / denom))
    return np.array(centres), np.array(rs)


# --------------------------------------------------------------------------- #
# domain diagnostics: the T2 / Q decomposition
# --------------------------------------------------------------------------- #
def pca_domain_diagnostics(library: np.ndarray, target: np.ndarray,
                           n_components: int = 10) -> dict:
    """Hotelling T2 and Q residual (SPE) for target rows against a library PCA.

    T2 = distance from the library centre WITHIN the model plane (the quantity
    Mahalanobis-in-scores measures). Q = distance OFF the plane: spectral
    structure the library has never seen. A target can sit at low T2 and high Q
    -- ordinary loading, unfamiliar chemistry -- which is exactly the case a
    single "in-domain" number misses, and the case the Addis Mahalanobis result
    left open.

    Both are returned raw and as a fraction of the library's own 95th
    percentile, so >1 means "beyond where the library lives".
    """
    Ls, Ts = standardize(library), standardize(target)
    pca = PCA(n_components=n_components).fit(Ls)
    SL, ST = pca.transform(Ls), pca.transform(Ts)
    var = SL.var(axis=0) + 1e-12

    t2_lib = (SL ** 2 / var).sum(1)
    t2_tgt = (ST ** 2 / var).sum(1)
    q_lib = ((Ls - pca.inverse_transform(SL)) ** 2).sum(1)
    q_tgt = ((Ts - pca.inverse_transform(ST)) ** 2).sum(1)

    t2_lim = float(np.percentile(t2_lib, 95))
    q_lim = float(np.percentile(q_lib, 95))
    return {
        "T2": t2_tgt, "Q": q_tgt,
        "T2_ratio": t2_tgt / t2_lim, "Q_ratio": q_tgt / q_lim,
        "T2_limit95": t2_lim, "Q_limit95": q_lim,
        "library_T2": t2_lib, "library_Q": q_lib,
        "explained": pca.explained_variance_ratio_,
    }


# --------------------------------------------------------------------------- #
# nearest-analog search
# --------------------------------------------------------------------------- #
def nearest_analogs(target: np.ndarray, library: np.ndarray, k: int = 50,
                    metric: str = "pearson", wn: np.ndarray | None = None,
                    block: int = 512) -> tuple:
    """Top-k library neighbours for every target row.

    Returns (idx, score) arrays of shape (n_target, k), score sorted best-first
    and always oriented so HIGHER = more similar. Blocked over targets so the
    full 13.6k library stays memory-flat on Colab.
    """
    T, L = np.asarray(target, float), np.asarray(library, float)
    idx_out, sc_out = [], []
    for start in range(0, len(T), block):
        chunk = T[start:start + block]
        if metric == "pearson":
            S = correlation_matrix(chunk, L)
        elif metric == "sam":
            S = -spectral_angle(chunk, L)
        elif metric == "sid":
            S = -spectral_information_divergence(chunk, L)
        elif metric == "euclidean":
            S = -np.linalg.norm(unit_rows(chunk)[:, None, :] - unit_rows(L)[None, :, :],
                                axis=2) if len(chunk) * len(L) < 4e7 else None
            if S is None:                      # memory-safe fallback via identity
                A2 = (unit_rows(chunk) ** 2).sum(1)[:, None]
                B2 = (unit_rows(L) ** 2).sum(1)[None, :]
                S = -np.sqrt(np.maximum(A2 + B2 - 2 * unit_rows(chunk) @ unit_rows(L).T, 0))
        else:
            raise ValueError(f"unknown metric {metric!r}")
        kk = min(k, S.shape[1])
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(chunk))[:, None]
        order = np.argsort(-S[rows, part], axis=1)
        idx_out.append(part[rows, order])
        sc_out.append(S[rows, part][rows, order])
    return np.vstack(idx_out), np.vstack(sc_out)


def mutual_nearest(target: np.ndarray, library: np.ndarray, k: int = 50,
                   metric: str = "pearson", k_library: int | None = None) -> list:
    """Mutual k-NN: library spectra that pick the target back within their own list.

    One-directional "nearest to Addis" lists are dominated by library spectra
    near *everything* (hubs); requiring the match to be mutual is the standard
    fix. The subtlety is **selectivity**: a target picking 50 of 13,634 library
    spectra is a 0.4% cut, while a library row picking 50 of ~250 targets is a
    20% cut, so a naive mutual test with equal k is nearly vacuous -- it passes
    almost everything and looks like a null result. `k_library` therefore
    defaults to the count that matches the target-side selectivity,
    ``k * n_target / n_library`` (at least 1). Pass an explicit `k_library` to
    loosen it deliberately.

    Returns a list of arrays, one per target row, holding library indices.
    """
    n_t, n_l = len(target), len(library)
    if k_library is None:
        k_library = max(1, int(round(k * n_t / n_l)))
    t_idx, _ = nearest_analogs(target, library, k=k, metric=metric)
    l_idx, _ = nearest_analogs(library, target, k=k_library, metric=metric)
    back = [set(row) for row in l_idx]
    return [np.array([j for j in row if i in back[j]], dtype=int)
            for i, row in enumerate(t_idx)]


def hubness(target: np.ndarray, library: np.ndarray, k: int = 50,
            metric: str = "pearson") -> np.ndarray:
    """How many target filters each library spectrum is a top-k neighbour of.

    A skewed distribution means a handful of library spectra dominate every
    analog list -- a known high-dimensional pathology, and a direct check on
    whether a "spectral analog cohort" is really 500 distinct analogs.
    """
    idx, _ = nearest_analogs(target, library, k=k, metric=metric)
    counts = np.zeros(len(library), dtype=int)
    u, c = np.unique(idx.ravel(), return_counts=True)
    counts[u] = c
    return counts


def redundancy(target: np.ndarray, library: np.ndarray, k: int = 50,
               metric: str = "pearson") -> dict:
    """How much a target's analog neighbourhoods repeat themselves.

    `hubness` counts alone are not comparable between targets: a site with 26
    filters can fill at most 26*k neighbour slots, so its share of the library
    is bounded by its filter count. The comparable quantity is the ratio of
    DISTINCT neighbours to slots -- 1.0 means every filter draws its own
    analogs, low values mean the same spectra serve everyone.
    """
    counts = hubness(target, library, k=k, metric=metric)
    slots = len(target) * k
    distinct = int((counts > 0).sum())
    order = np.argsort(-counts)
    top1 = order[:max(1, len(counts) // 100)]
    return {
        "slots": slots, "distinct": distinct,
        "unique_ratio": distinct / slots,
        "share_top_1pct": float(counts[top1].sum() / counts.sum()),
        "max_times_chosen": int(counts.max()),
        "counts": counts,
    }


def rank_fusion(rankings: dict) -> np.ndarray:
    """Borda-count fusion of several distance rankings (lower = more similar).

    Metrics disagree (the repo already found agreement collapsing to rho ~ 0 on
    corrected spectra). Fusion gives one ordering that no single metric's quirk
    dominates, and the spread across metrics is itself the uncertainty.
    """
    mats = [np.argsort(np.argsort(np.asarray(v, float))) for v in rankings.values()]
    return np.mean(mats, axis=0)
