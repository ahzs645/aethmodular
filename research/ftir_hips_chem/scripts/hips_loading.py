"""HIPS on loaded filters: identify the bend, trace what it does, correct it, and check the correction.

HIPS reads low as a filter darkens. On IMPROVE filters, against thermal (TOR) EC, its optical depth follows
``tau = L / (1 + L / ceiling)``, so a filter whose optical depth is tau reports a share ``tau / ceiling`` less
absorption than it carries. This module turns that into four steps that can be run on any set of filters
with HIPS Fabs and a sampled volume:

    screen     each filter: optical depth, expected under-read, a status, the corrected value and its range
    identify   one site: do its own data show the bend (no curve assumed)?
    propagate  one site: what the bend does to the numbers people read off HIPS
    check      one site: after correcting, are the signs of the bend gone? (check_by: the same within each season)

The ceiling is IMPROVE's. HIPS is optical and FTIR EC is FTIR-derived; neither is thermal EC. Nothing here
uses aethalometer data. Protocol: docs/hips-loading-protocol.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import reference_closure as rc
from pls_transfer import deming_regression

DEPOSIT_AREA_CM2 = 3.53
MAC_VALUE = 10.0
# Fractional under-read up to which a filter is corrected and used, and from which it is not used for level.
USE_UP_TO = 0.20
SHAPE_ONLY_FROM = 0.30
STATUSES = ("use, linearised", "caution", "shape only")
# λ of the Deming fits (FTIR EC on HIPS BC), the Addis value used at every site so sites are comparable.
ERROR_RATIO = 2.96
MIN_TEST = 40          # filters a site needs for its own tests
WIDE = 0.6             # an exponent whose 95 % interval is wider than this says nothing
MIN_GROUP = 4          # filters in a calendar month or a volume group


@dataclass(frozen=True)
class Ceiling:
    """The optical depth HIPS approaches: central value, 90 % interval, and how dark the reference filters get."""

    central: float
    lo: float
    hi: float
    reach: float            # darkest group of filters the curve was fitted on
    source: str = ""

    def values(self) -> dict[str, float]:
        return {"central": self.central, "lo": self.lo, "hi": self.hi}


# IMPROVE, all sites, 2005-2025, against thermal EC (run_addis_reference_closure_20261110.py, stage improve;
# docs/addis-reference-closure-2026-11-10.json). tests/test_hips_loading.py fails if the summary moves.
IMPROVE_CEILING = Ceiling(2.2596, 1.8212, 2.7014, 1.179, "IMPROVE 2005-2025, 347,936 samples at 210 sites, against thermal EC")


def optical_depth(fabs, volume, area=DEPOSIT_AREA_CM2):
    """HIPS optical depth of the filter from Fabs (Mm-1) and sampled volume (m3). Equals the laboratory's HIPS tau."""
    return np.asarray(fabs, dtype=float) * np.asarray(volume, dtype=float) / (100.0 * area)


def under_read(tau, top):
    """Share of the absorption on the filter that HIPS does not report: tau / ceiling, at most 1."""
    return np.clip(np.asarray(tau, dtype=float) / top, 0.0, 1.0)


def factor(tau, top):
    """What a reported value is multiplied by to linearise it; infinite at or past the ceiling."""
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(tau < top, 1.0 / (1.0 - np.minimum(tau, top) / top), np.inf)


def at_volume(tau, volume, new_volume, top):
    """The optical depth the same air would give at another sampled volume, through the same curve."""
    tau = np.asarray(tau, dtype=float)
    true = rc.linearise(tau, top) * new_volume / np.asarray(volume, dtype=float)
    # a filter at or past the ceiling has no finite loading under the curve; it stays at the ceiling
    return np.where(np.isfinite(true), true / (1.0 + np.where(np.isfinite(true), true, 0.0) / top), top)


def status_of(share):
    """Three statuses by expected under-read: corrected and used, used with caution, or shape only."""
    share = np.asarray(share, dtype=float)
    return np.where(share <= USE_UP_TO, STATUSES[0], np.where(share <= SHAPE_ONLY_FROM, STATUSES[1], STATUSES[2]))


def screen(frame: pd.DataFrame, *, fabs="fabs", volume="volume", tau=None, ceiling: Ceiling = IMPROVE_CEILING) -> pd.DataFrame:
    """Step 1 and 3 for each filter: optical depth, expected under-read, status, and the linearised value with its range.

    ``tau`` names a column holding the laboratory's own HIPS optical depth, when there is one; otherwise it is
    computed from ``fabs`` and ``volume``. The range of the corrected value comes from the ceiling's interval:
    a low ceiling gives the high end. ``beyond_reference`` marks filters darker than any the curve was fitted on.
    """
    out = frame.copy()
    out["tau"] = frame[tau].astype(float) if tau else optical_depth(frame[fabs], frame[volume])
    out["under_read"] = under_read(out["tau"], ceiling.central)
    out["under_read_lo"] = under_read(out["tau"], ceiling.hi)
    out["under_read_hi"] = under_read(out["tau"], ceiling.lo)
    out["status"] = status_of(out["under_read"])
    out["beyond_reference"] = out["tau"] > ceiling.reach
    for name, top in (("", ceiling.central), ("_lo", ceiling.hi), ("_hi", ceiling.lo)):
        out[f"factor{name}"] = factor(out["tau"], top)
        out[f"fabs_linearised{name}"] = frame[fabs].astype(float) * out[f"factor{name}"]
    return out


def _exponent(y, x, z, n_boot):
    try:
        return rc.response_exponent(y, x, z, n_boot=n_boot)
    except ValueError:
        return {"n": 0, "exponent": np.nan, "lo": np.nan, "hi": np.nan}


def _point(y, x, z):
    """The exponent of y against x with z as the instrument, without an interval; rows with a non-finite log are dropped."""
    logs = np.log(np.vstack([np.asarray(v, dtype=float) for v in (y, x, z)]))
    c = np.cov(logs[:, np.isfinite(logs).all(axis=0)])
    return float(c[0, 2] / c[1, 2])


def _ratio_of_thirds(part, value, by, n_boot, rng):
    """Median of ``value`` in the top third of ``by`` over the bottom third, with a 90 % bootstrap interval."""
    low, high = part[by].quantile([1 / 3, 2 / 3])
    a, b = part.loc[part[by] >= high, value].to_numpy(), part.loc[part[by] <= low, value].to_numpy()
    if min(len(a), len(b)) < MIN_GROUP:
        return np.nan, np.nan, np.nan
    boot = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n_boot)]
    return float(np.median(a) / np.median(b)), float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def identify(screened: pd.DataFrame, *, fabs="fabs", reference="ftir_ec", instrument="pm25", volume="volume", n_boot=500,
             seed=20261111, ceiling: Ceiling = IMPROVE_CEILING) -> dict[str, object]:
    """Step 1 for a site: the signs of the bend in its own data, none of which assumes the curve.

    exposure   how dark its filters are, by status
    exponent   b in Fabs ∝ reference^b with ``instrument`` as the instrument: below 1 means HIPS is compressed
    ratio      Fabs ÷ reference on the most loaded third of filters over the least loaded third, the thirds chosen by
               the instrument's loading (not by HIPS or the reference, which would select on their own errors)
    fit        Deming line of the reference on HIPS BC: a slope above 1 with a negative intercept is the signature
               a bent HIPS leaves (the published Addis fit is 2.31x - 6.17)
    Beside each sits what the curve predicts, so "seen" and "expected" can be told apart.
    """
    rng = np.random.default_rng(seed)
    tau = screened["tau"]
    out: dict[str, object] = {
        "filters": int(len(screened)), "median_tau": float(tau.median()), "expected_under_read": float(under_read(tau.median(), ceiling.central)),
        **{f"share_{name.split(',')[0].replace(' ', '_')}": float((screened["status"] == name).mean()) for name in STATUSES},
        "share_beyond_reference": float(screened["beyond_reference"].mean()),
    }
    have = [c for c in (reference, instrument) if c in screened]
    part = screened[(screened[have] > 0).all(axis=1)] if len(have) == 2 else screened.iloc[0:0]
    out["tested_filters"] = int(len(part))
    if len(part) < MIN_TEST:
        return out | {"reading": "expected, cannot be tested" if out["expected_under_read"] > 0.10 else "not expected"}

    e = _exponent(part[fabs], part[reference], part[instrument], n_boot)
    predicted = float(np.mean(1.0 - under_read(part["tau"], ceiling.central)))
    out |= {"exponent": e["exponent"], "exponent_lo": e["lo"], "exponent_hi": e["hi"], "exponent_predicted": predicted}

    part = part.assign(per_reference=part[fabs] / part[reference], loading=part[instrument] * part[volume])
    ratio, lo, hi = _ratio_of_thirds(part, "per_reference", "loading", n_boot, rng)
    low, high = part["loading"].quantile([1 / 3, 2 / 3])
    expected = (1.0 - under_read(part.loc[part["loading"] >= high, "tau"].median(), ceiling.central)) / (
        1.0 - under_read(part.loc[part["loading"] <= low, "tau"].median(), ceiling.central))
    out |= {"ratio_heavy_over_light": ratio, "ratio_lo": lo, "ratio_hi": hi, "ratio_predicted": float(expected)}

    line = deming_regression(part[fabs] / MAC_VALUE, part[reference], ERROR_RATIO)
    out |= {"slope": float(line["slope"]), "intercept": float(line["intercept"])}

    # two tests, either of which can show it: the exponent (when its interval is narrow) and the ratio of thirds
    narrow = np.isfinite(e["exponent"]) and (e["hi"] - e["lo"]) <= WIDE
    seen = (narrow and e["hi"] < 1.0) or (np.isfinite(hi) and hi < 1.0)
    other_way = (narrow and e["lo"] > 1.0) or (np.isfinite(lo) and lo > 1.0)
    blind = not narrow and not (np.isfinite(lo) and np.isfinite(hi))
    expected_here = out["expected_under_read"] > 0.10
    out["reading"] = ("cannot tell" if blind else "HIPS rises faster than the reference" if other_way and not seen
                      else "expected and seen" if expected_here and seen else "expected, not seen" if expected_here
                      else "not expected, but compressed" if seen else "not expected, not seen")
    return out


def _numbers(part, column, reference, pm, month):
    """The numbers people read off HIPS at a site, from one Fabs column."""
    bc = part[column] / MAC_VALUE
    out = {"median_bc": float(bc.median()), "mean_bc": float(bc.mean())}
    monthly = bc.groupby(part[month]).agg(["median", "size"])
    monthly = monthly.loc[monthly["size"] >= MIN_GROUP, "median"]
    ordered = np.sort(monthly.to_numpy())
    out["seasonal_contrast"] = float(ordered[-3:].mean() / ordered[:3].mean()) if len(ordered) >= 9 and ordered[:3].mean() > 0 else np.nan
    if pm in part:
        ok = part[pm] > 1
        out["bc_share_of_pm25"] = float((bc[ok] / part.loc[ok, pm]).median()) if ok.sum() >= MIN_GROUP else np.nan
    if reference in part:
        ok = (part[reference] > 0) & np.isfinite(bc)
        if ok.sum() >= MIN_TEST:
            line = deming_regression(bc[ok], part.loc[ok, reference], ERROR_RATIO)
            out |= {"slope": float(line["slope"]), "intercept": float(line["intercept"]),
                    "fabs_per_reference": float((part.loc[ok, column] / part.loc[ok, reference]).median())}
    return out


def propagate(screened: pd.DataFrame, *, fabs="fabs", reference="ftir_ec", pm="pm25", month="month") -> pd.DataFrame:
    """Step 2 for a site: each number read off HIPS, as reported and linearised, with the range the ceiling allows.

    Rows: median and mean BC (Fabs ÷ 10), the seasonal contrast (the three highest calendar-month medians over the three lowest),
    BC's share of PM2.5, Fabs per unit of the reference, and the Deming slope and intercept of the reference on
    HIPS BC. Filters at or past the ceiling have no finite corrected value and are left out of the linearised
    columns; ``dropped`` counts them.
    """
    finite = np.isfinite(screened["fabs_linearised_hi"])
    part = screened[finite]
    before = _numbers(part, fabs, reference, pm, month)
    after = {name: _numbers(part, f"fabs_linearised{name}", reference, pm, month) for name in ("", "_lo", "_hi")}
    rows = []
    for key, label in (("median_bc", "Median BC (µg/m³)"), ("mean_bc", "Mean BC (µg/m³)"), ("seasonal_contrast", "Three highest months ÷ three lowest"),
                       ("bc_share_of_pm25", "BC share of PM2.5"), ("fabs_per_reference", "Fabs ÷ FTIR EC (m²/g)"),
                       ("slope", "Slope, FTIR EC on HIPS BC"), ("intercept", "Intercept (µg/m³)")):
        if key not in before:
            continue
        ends = sorted([after["_lo"].get(key, np.nan), after["_hi"].get(key, np.nan)])
        rows.append({"quantity": label, "key": key, "reported": before[key], "linearised": after[""].get(key, np.nan), "lo": ends[0], "hi": ends[1],
                     "change": after[""].get(key, np.nan) / before[key] - 1.0 if key != "intercept" and before[key] else np.nan})
    table = pd.DataFrame(rows)
    table["dropped"] = int((~finite).sum())
    return table


def check(screened: pd.DataFrame, *, fabs="fabs", reference="ftir_ec", instrument="pm25", n_boot=500, min_filters=MIN_TEST) -> dict[str, float]:
    """Step 4 for a site: after linearising, is HIPS proportional to the reference, and is the intercept gone?

    The correction has done its job when the exponent's interval includes 1. Below 1, the site is more compressed than
    the curve allows for (under-corrected); above 1, less (over-corrected). ``exponent_predicted`` is the exponent the
    curve alone gives these filters, so before ÷ predicted = after. The same caveat as in ``identify``: against
    FTIR EC this cannot separate HIPS from FTIR.
    """
    have = [c for c in (reference, instrument) if c in screened]
    part = screened[(screened[have] > 0).all(axis=1) & np.isfinite(screened["fabs_linearised"])] if len(have) == 2 else screened.iloc[0:0]
    if len(part) < min_filters:
        return {"tested_filters": int(len(part)), "verdict": "cannot be tested"}
    before = _exponent(part[fabs], part[reference], part[instrument], n_boot)
    after = _exponent(part["fabs_linearised"], part[reference], part[instrument], n_boot)
    # what the curve alone does to this site's filters: reported against linearised HIPS, same instrument, no reference.
    # It allows for filters sampled at different volumes, which an average of 1 - tau / ceiling does not.
    predicted = {name: _point(part[fabs], part[f"fabs_linearised{name}"], part[instrument]) for name in ("", "_lo", "_hi")}
    line = deming_regression(part["fabs_linearised"] / MAC_VALUE, part[reference], ERROR_RATIO)
    blind = not np.isfinite(before["exponent"]) or (before["hi"] - before["lo"]) > WIDE or not np.isfinite(after["exponent"])
    verdict = ("cannot tell" if blind else "corrected" if after["lo"] <= 1.0 <= after["hi"]
               else "under-corrected" if after["hi"] < 1.0 else "over-corrected")
    return {"tested_filters": int(len(part)), "exponent_before": before["exponent"], "before_lo": before["lo"], "before_hi": before["hi"],
            "exponent_predicted": predicted[""], "predicted_lo": min(predicted["_lo"], predicted["_hi"]), "predicted_hi": max(predicted["_lo"], predicted["_hi"]),
            "exponent_after": after["exponent"], "after_lo": after["lo"], "after_hi": after["hi"],
            "slope_after": float(line["slope"]), "intercept_after": float(line["intercept"]), "verdict": verdict}


def check_by(screened: pd.DataFrame, by="season", *, min_filters=30, n_boot=500, **columns) -> pd.DataFrame:
    """Step 4 within each group of ``by`` (a season): the check, run where a group has enough filters.

    The correction is per filter, but a site's seasons can sit at different levels of Fabs per unit of the reference
    for reasons that have nothing to do with loading. Pooled, that reads as a failed correction; within seasons it
    does not. Run both.
    """
    rows = []
    for name, part in screened.groupby(by, sort=False):
        rows.append({by: name, "filters": int(len(part)), "median_tau": float(part["tau"].median())}
                    | check(part, n_boot=n_boot, min_filters=min_filters, **columns))
    return pd.DataFrame(rows)
