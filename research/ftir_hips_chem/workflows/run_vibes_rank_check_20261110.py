"""Does the rank-70 VIBES background absorb aerosol signal? Two checks on the network bundle.

    python run_vibes_rank_check_20261110.py --bundle vibes_bundle --out OUT --summary docs/vibes-rank-check-2026-11-10.json

`--bundle` holds background_model.npz, X_model_grid.npy and cases.csv (the 19 Oct network bundle).

1. noise: fit every spectrum to convergence (Newton ELBO, certified MAP) and look at the scans whose
   fitted noise level is as low as a field blank's. If the baseline had absorbed their aerosol, their
   raw spectra would still show bands. Band heights here are measured above a straight line between
   the band's two edges, so they use no baseline model.
2. inject: add a Gaussian band of known height and width to real spectra, refit at ranks 70, 40 and
   20, and report the share of the band left in the corrected spectrum.

Fits are cached in OUT, so a rerun only recomputes the tables.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vibes_batch  # noqa: E402

TAU = 0.1
# Band: (left edge, right edge, peak window) in cm-1.
BANDS = {
    "C-H 2920": (2800, 3000, (2900, 2940)),
    "Ammonium 1420": (1350, 1500, (1400, 1450)),
    "Carbonyl 1720": (1660, 1800, (1700, 1740)),
    "N-H and O-H 3200": (2600, 3700, (3000, 3300)),
}
FLAT_RANGE = 0.05          # a scan whose absorbance spans less than this carries no filter at all
RANKS = (70, 40, 20)
CENTRES = (1720.0, 2920.0)
WIDTHS = (15.0, 50.0, 150.0, 500.0, 1500.0)
HEIGHTS = (0.001, 0.01)
PER_GROUP = 40
SEED = 20261110


def load(bundle):
    bundle = Path(bundle)
    model = np.load(bundle / "background_model.npz", allow_pickle=False)
    spectra = np.load(bundle / "X_model_grid.npy").astype(float)
    cases = pd.read_csv(bundle / "cases.csv", dtype=str)
    return model, spectra, cases


def fit(spectra, mean, components):
    """Noise level, baseline and success for each spectrum, solved to convergence."""
    elbo = vibes_batch.fit_elbo_newton(spectra, mean, components, tau=TAU, chunk=512)
    solved = vibes_batch.solve_map_batch(spectra, mean, components, sigma=elbo.sigma, tau=TAU)
    return elbo.sigma, solved.baseline, elbo.success & solved.converged


def band_height(spectra, wn, left, right, peak):
    """Peak height above the straight line through the band's two edges."""
    a, b = np.abs(wn - left) < 15, np.abs(wn - right) < 15
    inside = (wn > peak[0]) & (wn < peak[1])
    ya, yb = spectra[:, a].mean(1), spectra[:, b].mean(1)
    line = ya[:, None] + (yb - ya)[:, None] * ((wn[inside] - left) / (right - left))[None, :]
    return (spectra[:, inside] - line).max(1)


def network_fit(model, spectra, out):
    cache = Path(out) / "network_fit.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["sigma"], saved["baseline"].astype(float), saved["ok"]
    sigma, baseline, ok = fit(spectra, model["mean"], model["components"])
    np.savez(cache, sigma=sigma, baseline=baseline.astype(np.float32), ok=ok)
    return sigma, baseline, ok


def stage_noise(model, spectra, cases, out):
    wn, components = model["wn"], model["components"]
    sigma, baseline, ok = network_fit(model, spectra, out)
    corrected = spectra - baseline
    table = cases[["analysis_id", "code", "filter_id", "filter_type", "qc"]].copy()
    table["sigma"] = sigma
    table["corrected_rms"] = np.sqrt((corrected ** 2).mean(1))
    table["raw_range"] = np.ptp(spectra, axis=1)
    latent = np.linalg.lstsq(components, (baseline - model["mean"]).T, rcond=None)[0].T
    table["high_order_share"] = np.sqrt(((latent[:, 40:] @ components[:, 40:].T) ** 2).mean(1)) / table["corrected_rms"]
    for name, (left, right, peak) in BANDS.items():
        table[f"raw {name}"] = band_height(spectra, wn, left, right, peak)
        table[f"corrected {name}"] = band_height(corrected, wn, left, right, peak)
    table.to_csv(Path(out) / "noise_by_scan.csv", index=False)

    blank = table["filter_type"] == "FB"
    sample = ~blank
    raw = [f"raw {name}" for name in BANDS]
    limit = table.loc[blank, "sigma"].quantile(0.95)
    blank_like = sample & (table["sigma"] < limit)
    ceiling = table.loc[blank, raw].quantile(0.99)
    no_bands = (table[raw] <= ceiling).all(axis=1)
    flat = table["raw_range"] < FLAT_RANGE
    group = np.where(blank, "Field blank", np.where(blank_like, "Sample scan, noise as low as a blank", "Other sample scan"))

    by_type = []
    for name, rows in table.groupby("filter_type"):
        by_type.append({"filter_type": name, "n": int(len(rows)),
                        "sigma_median": float(rows["sigma"].median()),
                        "corrected_rms_median": float(rows["corrected_rms"].median()),
                        "high_order_share_median": float(rows["high_order_share"].median())})
    raw_bands = []
    for name, rows in table.groupby(group):
        raw_bands.append({"group": name, "n": int(len(rows)), **{b: float(rows[f"raw {b}"].median()) for b in BANDS}})
    retention = []
    for name in ("Ammonium 1420", "C-H 2920"):
        clear = sample & (table[f"raw {name}"] > 5 * ceiling[f"raw {name}"])
        kept = table.loc[clear, f"corrected {name}"] / table.loc[clear, f"raw {name}"]
        retention.append({"band": name, "n": int(clear.sum()), **{f"p{q:02d}": float(np.percentile(kept, q)) for q in (5, 50, 95)}})
    flat_rows = table.loc[flat, ["filter_id", "analysis_id", "filter_type", "qc", "raw_range"]].sort_values("filter_id")
    components_rms = np.linalg.norm(components, axis=0) / np.sqrt(len(wn))
    return {
        "scans": int(len(table)), "fitted": int(ok.sum()),
        "background_blanks": int(len(model["blank_ids"])),
        "background_blanks_in_this_network": int(cases["filter_id"].isin(model["blank_ids"]).sum()),
        "background_blank_sources": sorted({str(b).split(":")[0] for b in model["blank_ids"]}),
        "component_rms": {str(k + 1): float(components_rms[k]) for k in (0, 9, 19, 39, 69)},
        "by_type": by_type,
        "blank_sigma_p95": float(limit),
        "blank_like_sample_scans": int(blank_like.sum()),
        "blank_like_with_no_raw_bands": int((blank_like & no_bands).sum()),
        "blank_like_flat": int((blank_like & flat).sum()),
        "blank_like_with_raw_bands": int((blank_like & ~no_bands).sum()),
        "raw_bands": raw_bands,
        "retention": retention,
        "flat_scans": flat_rows.to_dict("records"),
    }


def stage_inject(model, spectra, cases, out):
    wn, mean, components = model["wn"], model["mean"], model["components"]
    sigma, baseline, _ = network_fit(model, spectra, out)
    rms = np.sqrt(((spectra - baseline) ** 2).mean(1))
    rng = np.random.default_rng(SEED)
    pm = np.flatnonzero((cases["filter_type"] == "PM2.5").to_numpy())
    middle = pm[(rms[pm] > np.quantile(rms[pm], 0.4)) & (rms[pm] < np.quantile(rms[pm], 0.6))]
    groups = {
        "Field blank": rng.choice(np.flatnonzero((cases["filter_type"] == "FB").to_numpy()), PER_GROUP, replace=False),
        "PM2.5 scan with no signal": pm[np.argsort(rms[pm])[:PER_GROUP]],
        "Typical PM2.5": rng.choice(middle, PER_GROUP, replace=False),
        "Addis PM2.5": rng.choice(pm[(cases["code"].to_numpy()[pm] == "ETAD")], PER_GROUP, replace=False),
    }
    rows = np.concatenate(list(groups.values()))
    names = np.concatenate([[name] * len(index) for name, index in groups.items()])
    cache = Path(out) / "injection.csv"
    if cache.exists():
        long = pd.read_csv(cache)
    else:
        records = []
        for rank in RANKS:
            basis = components[:, :rank]
            _, base0, ok0 = fit(spectra[rows], mean, basis)
            corrected0 = spectra[rows] - base0
            for centre in CENTRES:
                for width in WIDTHS:
                    band = np.exp(-4 * np.log(2) * ((wn - centre) / width) ** 2)
                    for height in HEIGHTS:
                        spiked = spectra[rows] + height * band
                        _, base1, ok1 = fit(spiked, mean, basis)
                        left = ((spiked - base1 - corrected0) @ band) / (height * (band @ band))
                        records += [{"rank": rank, "group": names[i], "case_row": int(rows[i]), "centre": centre, "fwhm": width,
                                     "height": height, "share_left": float(left[i]), "ok": bool(ok0[i] & ok1[i])}
                                    for i in range(len(rows))]
            print(f"rank {rank} done", flush=True)
        long = pd.DataFrame(records)
        long.to_csv(cache, index=False)
    long = long[long["ok"]]
    table = (long.groupby(["group", "rank", "fwhm", "height"])["share_left"]
             .agg(median="median", p10=lambda v: v.quantile(0.1), p90=lambda v: v.quantile(0.9), n="count").reset_index())
    return {"per_group": PER_GROUP, "centres": list(CENTRES), "table": table.to_dict("records")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True, help="folder for cached fits and per-scan tables")
    ap.add_argument("--stages", nargs="+", default=["noise", "inject"], choices=["noise", "inject"])
    ap.add_argument("--summary", help="write the tables to this JSON file")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model, spectra, cases = load(a.bundle)
    result = {"bundle": Path(a.bundle).name, "tau": TAU}
    if a.summary and Path(a.summary).exists():
        result.update(json.loads(Path(a.summary).read_text()))
    for stage in a.stages:
        result[stage] = {"noise": stage_noise, "inject": stage_inject}[stage](model, spectra, cases, a.out)
    text = json.dumps(result, indent=1)
    if a.summary:
        Path(a.summary).write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
