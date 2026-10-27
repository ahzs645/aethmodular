"""Does IMPROVE HIPS read a nonzero Fabs at EC = 0, and is its Fabs-TOR response curved? (ftir_26, ftir_37)

Run where the Davis data folder is available (AETHMODULAR_DAVIS_DIR, or the Drive mount data_paths resolves):
    uv run python gallery/data/export_research_improve_origin.py

Inputs (not in the repo):
  <Davis>/FTIR/local_db/tables/{results_hips,results_tor,ftir_catalog}.csv   the offline IMPROVE mirror
  <Davis>/Spartan/SPARTAN_HIPS_Batch1-51.v2.csv                                the shipped SPARTAN HIPS batch

What it writes, from the notebooks' own code where the repo has it:
  intercepts   research/ftir_ec_phase3/ftir_26_improve_hips_origin.ipynb Part 1: the IMPROVE HIPS x TOR join
               (run_ftir_26.py, verbatim), the OLS Fabs-at-EC-0 intercept over five fit ranges with a site-cluster
               bootstrap 95 % interval (sufficient statistics per site, as the notebook resamples)
  lambda       the pooled Deming intercept and slope across the error-variance ratio λ (monotone: OLS is the bound)
  zero_ec      filters with TOR EC <= 0: what HIPS reads next to them
  scale        how many IMPROVE filters carry total Fabs at or above the Addis intercept at all
  curvature    ftir_37: held-out-site linear / quadratic / spline comparison, fitted curves at the median OC/EC,
               per-site curvature and the high-minus-low loading intercept shift
               (research/ftir_ec_phase3/scripts/theory_test_suite.curvature_analysis)
  blank_domain ftir_37: SPARTAN samples against the R1 range of their lot's field blanks, by Fabs quintile
               (theory_test_suite.hips_blank_domain), over the whole shipped batch

The IMPROVE tables carry a data-use condition (notify the IMPROVE/NPS contact before a journal publication;
<Davis>/Improve/MANIFEST.md), so this stays in the private research section.

Writes gallery/app/public/data/research/improve_origin/origin.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("research/ftir_hips_chem/scripts", "research/ftir_ec_phase3/scripts"):
    sys.path.insert(0, str(ROOT / p))

OUT = ROOT / "gallery/app/public/data/research/improve_origin"
ADDIS_C = 21.5            # Mm-1, the Addis Fabs-vs-EC intercept ftir_26 measures against (2.15 µg/m³ x MAC 10)
OCEC_ADDIS_LIKE = 2.27    # ftir_11 lowest-OC/EC cohort
BOOT_REPS = 1000
BOOT_SEED = 20260812
LAMBDAS = np.logspace(-1, 6, 29)


def r(v, digits=4):
    if v is None:
        return None
    v = float(v)
    return float(f"{v:.{digits}g}") if np.isfinite(v) else None


def ftir26_bridge(tables: Path) -> pd.DataFrame:
    """The IMPROVE HIPS x TOR join exactly as run_ftir_26.py builds it (EC <= 0 rows kept)."""
    hips_raw = pd.read_csv(tables / "results_hips.csv", usecols=["MatchedFilterId", "Parameter", "Value"], low_memory=False)
    fabs = (hips_raw[hips_raw["Parameter"].str.casefold().eq("fabs")].drop_duplicates("MatchedFilterId")
            .rename(columns={"MatchedFilterId": "FilterId", "Value": "Fabs"}))
    tor = pd.read_csv(tables / "results_tor.csv", usecols=["Site", "SampleDate", "Parameter", "Value"], low_memory=False)
    cat = pd.read_csv(tables / "ftir_catalog.csv", usecols=["FilterId", "SampleDate", "Site"]).drop_duplicates("FilterId")
    cat["date"] = pd.to_datetime(cat["SampleDate"], format="mixed", errors="coerce").dt.normalize()
    tor["date"] = pd.to_datetime(tor["SampleDate"], format="mixed", errors="coerce").dt.normalize()
    tor_wide = (tor[tor["Parameter"].isin(["EC", "OC"])].drop_duplicates(["Site", "date", "Parameter"])
                .pivot_table(index=["Site", "date"], columns="Parameter", values="Value", aggfunc="first").reset_index())
    bridge = fabs.merge(cat[["FilterId", "Site", "date"]], on="FilterId", how="inner").merge(tor_wide, on=["Site", "date"], how="inner")
    bridge["EC_ugm3"] = bridge["EC"] / 1000.0           # TOR in ng/m³
    bridge["OC_EC"] = np.where((bridge["EC"] > 0) & (bridge["OC"] > 0), bridge["OC"] / bridge["EC"], np.nan)
    return bridge


def site_stats(frame: pd.DataFrame) -> np.ndarray:
    """Per-site sufficient statistics (n, Σx, Σy, Σx², Σxy) of Fabs on EC, so a site resample refits exactly."""
    g = frame.assign(xx=frame["EC_ugm3"] ** 2, xy=frame["EC_ugm3"] * frame["Fabs"]).groupby("Site")
    return np.column_stack([g.size(), g["EC_ugm3"].sum(), g["Fabs"].sum(), g["xx"].sum(), g["xy"].sum()]).astype(float)


def ols_from_stats(block: np.ndarray) -> tuple[float, float]:
    n, sx, sy, sxx, sxy = block.sum(axis=0)
    slope = (sxy - sx * sy / n) / (sxx - sx * sx / n)
    return slope, sy / n - slope * sx / n


def cluster_ci(frame: pd.DataFrame, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> tuple[float, float]:
    blocks = site_stats(frame)
    rng = np.random.default_rng(seed)
    draws = np.array([ols_from_stats(blocks[rng.integers(0, len(blocks), len(blocks))])[1] for _ in range(reps)])
    return tuple(np.nanpercentile(draws, [2.5, 97.5]))


def deming(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, float]:
    """Deming with λ = Var(y error) / Var(x error)."""
    mx, my = x.mean(), y.mean()
    sxx, syy, sxy = ((x - mx) ** 2).sum(), ((y - my) ** 2).sum(), ((x - mx) * (y - my)).sum()
    a = syy - lam * sxx
    slope = (a + np.sqrt(a * a + 4 * lam * sxy * sxy)) / (2 * sxy)
    return slope, my - slope * mx


def intercepts(bridge: pd.DataFrame) -> list[dict]:
    pos = bridge[bridge["EC_ugm3"] > 0]
    p95, p10 = pos["EC_ugm3"].quantile(0.95), pos["EC_ugm3"].quantile(0.10)
    like = bridge[bridge["OC_EC"].le(OCEC_ADDIS_LIKE)]
    like_p95 = like["EC_ugm3"].quantile(0.95)
    ranges = {"Pooled, full range": bridge, f"Pooled, EC ≤ p95 ({p95:.2f} µg/m³)": bridge[bridge["EC_ugm3"] <= p95],
              f"Addis-like OC/EC ≤ {OCEC_ADDIS_LIKE}": like, f"Addis-like, EC ≤ its p95 ({like_p95:.2f})": like[like["EC_ugm3"] <= like_p95],
              f"Lowest EC decile (≤ {p10:.3f})": bridge[bridge["EC_ugm3"] <= p10]}
    out = []
    for label, f in ranges.items():
        slope, icpt = ols_from_stats(site_stats(f))
        lo, hi = cluster_ci(f)
        out.append({"range": label, "n": int(len(f)), "sites": int(f["Site"].nunique()), "slope": r(slope), "intercept": r(icpt),
                    "ci_lo": r(lo), "ci_hi": r(hi), "share_of_addis_c": r(icpt / ADDIS_C)})
    per_site = [ols_from_stats(site_stats(g))[1] for _, g in bridge.groupby("Site") if len(g) >= 100 and g["EC_ugm3"].nunique() > 10]
    out.append({"range": "Per-site median (sites with ≥ 100 filters)", "n": len(per_site), "sites": len(per_site), "slope": None,
                "intercept": r(np.median(per_site)), "ci_lo": None, "ci_hi": None, "share_of_addis_c": r(np.median(per_site) / ADDIS_C)})
    return out


def lambda_sweep(bridge: pd.DataFrame) -> list[dict]:
    x, y = bridge["EC_ugm3"].to_numpy(float), bridge["Fabs"].to_numpy(float)
    return [{"lambda": r(lam), "slope": r(s), "intercept": r(i)} for lam in LAMBDAS for s, i in [deming(x, y, lam)]]


def zero_ec(bridge: pd.DataFrame) -> dict:
    z = bridge[bridge["EC"] <= 0]["Fabs"]
    return {"n": int(len(z)), "median": r(z.median()), "mean": r(z.mean()), "p25": r(z.quantile(0.25)), "p75": r(z.quantile(0.75)),
            "negative_share": r((z < 0).mean())}


def scale(bridge: pd.DataFrame, addis_intercept: float | None) -> dict:
    out = {"n": int(len(bridge)), "median_fabs": r(bridge["Fabs"].median()), "addis_c": ADDIS_C,
           "n_at_or_above_c": int((bridge["Fabs"] >= ADDIS_C).sum())}
    if addis_intercept is not None:
        out.update({"addis_ols_intercept": r(addis_intercept), "n_at_or_above_addis_ols": int((bridge["Fabs"] >= addis_intercept).sum())})
    return out


def addis_ols_intercept() -> float | None:
    """The gallery's own Addis Fabs-on-FTIR-EC OLS intercept (the focus-site rows, exclusions applied)."""
    from export_research_improve_envelope import spartan_reference

    etad = spartan_reference().query("code == 'ETAD'")
    return float(np.polyfit(etad["ec_conc"], etad["fabs"], 1)[1]) if len(etad) > 3 else None


def records(frame: pd.DataFrame, digits=4) -> list[dict]:
    return [{k: (r(v, digits) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v)) for k, v in row.items()}
            for row in frame.to_dict(orient="records")]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from theory_test_suite import curvature_analysis, davis_root, hips_blank_domain, load_improve_hips_tor_bridge

    root = davis_root()
    bridge = ftir26_bridge(root / "FTIR/local_db/tables")
    bridge = bridge[np.isfinite(bridge["Fabs"]) & np.isfinite(bridge["EC_ugm3"])]
    payload: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"bridge": "<Davis>/FTIR/local_db/tables/{results_hips,results_tor,ftir_catalog}.csv (run_ftir_26.py join)",
                   "spartan_hips": "<Davis>/Spartan/SPARTAN_HIPS_Batch1-51.v2.csv"},
        "notebooks": ["research/ftir_ec_phase3/ftir_26_improve_hips_origin.ipynb", "research/ftir_ec_phase3/ftir_37_hips_loading_and_curvature.ipynb"],
        "bridge_n": int(len(bridge)), "bridge_sites": int(bridge["Site"].nunique()),
        "intercepts": intercepts(bridge), "lambda": lambda_sweep(bridge[bridge["EC_ugm3"] > 0]), "zero_ec": zero_ec(bridge),
        "scale": scale(bridge, addis_ols_intercept()),
    }
    clean, ledger = load_improve_hips_tor_bridge()
    cv, curves, sites, tests = curvature_analysis(clean)
    payload["curvature"] = {"ledger": records(ledger), "cv": records(cv), "curves": records(curves[["model", "EC_ugm3", "predicted_Fabs"]]),
                            "sites": records(sites), "tests": records(tests)}
    _, loading, quality = hips_blank_domain()
    payload["blank_domain"] = {"bins": records(loading), "quality": records(quality)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "origin.json").write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT / 'origin.json'}: bridge {payload['bridge_n']:,} filters at {payload['bridge_sites']} sites")
    for row in payload["intercepts"]:
        print(f"  {row['range']}: intercept {row['intercept']} [{row['ci_lo']}, {row['ci_hi']}] n {row['n']}")


if __name__ == "__main__":
    main()
