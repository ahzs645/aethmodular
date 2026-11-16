"""Which VIBES tau gives the best EC once the calibration matches it? IMPROVE refitted at each tau.

    uv run python run_vibes_tau_calibration_20261116.py --bundle DECISION_BUNDLE --area research/ftir_hips_chem --out OUT

`--bundle` is the 22 Oct decision bundle (improve_raw.npy, addis_raw.npy and their metadata). For each tau the
IMPROVE and Addis raw spectra are corrected with the 87-blank rank-70 background, solved to convergence, and two
scorings are repeated on the result:

1. calibration: the frozen procedure of vibes_newton_refit_20261103.calibration (site-grouped 5-fold curve on the
   train split, first major minimum, fit on train, score on the held-out test split), scored at every k 1-25 so
   that tau values can be compared at one k. It is run twice: on the whole train split, and without the train
   spectra the bundle marks as failing raw QC.
2. transfer: the out-of-site scoring of run_transfer_diagnosis_20261102 (site-grouped 5-fold over all eligible
   IMPROVE filters, k 1-25) and the Addis prediction of a model fitted on all of them, against HIPS.

qc_spectrum_*.csv compare the k 4 model with and without the QC-failed train spectrum on every network scan.
domain_*.csv ask whether a model trained without the heaviest filters is stable on the network scans, and
(domain_addis_by_k.csv) whether the Addis slope against HIPS still follows k at a fixed training range.
bootstrap.csv compares each tau with tau 0.1 on the same filters, with sites resampled 2,000 times.

Corrected spectra and predictions are cached in OUT, so a rerun only recomputes what is missing.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "scripts"), str(HERE.parents[3])]
import continuum_anchor as ca  # noqa: E402
import vibes_batch as vb  # noqa: E402
from candidate_metrics import recorded_source_qc  # noqa: E402
from pls_transfer import component_cv_curve, select_first_major_minimum  # noqa: E402

TAUS = (0.1, 0.05, 0.2)
KMAX = 25


def corrected(raw, mean, components, tau, cache):
    """Baseline-corrected spectra at one tau, stored as float32 like the 3 Nov arrays."""
    if not cache.exists():
        X = np.asarray(raw, dtype=np.float64)
        elbo = vb.fit_elbo_newton(X, mean, components, tau=tau, chunk=512)
        solved = vb.solve_map_batch(X, mean, components, sigma=elbo.sigma, tau=tau, chunk=256)
        ok = elbo.success & solved.converged
        if not ok.all():
            raise RuntimeError(f"{cache.name}: {int((~ok).sum())} fits not certified")
        np.save(cache, (X - solved.baseline).astype(np.float32))
        np.save(cache.with_name(cache.stem + "_sigma.npy"), elbo.sigma)
    return np.asarray(np.load(cache), dtype=np.float64)


def bootstrap(out, n=2000, seed=20261116):
    """Each tau against tau 0.1 on the same filters, with sites resampled: the difference in a score and its 90 % interval."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))  # noqa: E731
    medabs = lambda e: float(np.median(np.abs(e)))  # noqa: E731

    def compare(sites, a, b, stat):
        codes, index = np.unique(sites, return_inverse=True)
        members = [np.flatnonzero(index == i) for i in range(len(codes))]
        draws = np.empty(n)
        for j in range(n):
            rows = np.concatenate([members[i] for i in rng.integers(0, len(codes), len(codes))])
            draws[j] = stat(a[rows]) - stat(b[rows])
        return stat(a) - stat(b), *np.quantile(draws, [0.05, 0.95])

    held = {tau: np.load(out / f"heldout_tau{tau}.npz", allow_pickle=True) for tau in TAUS}
    rows_ = pd.read_csv(out / "improve_rows.csv")
    oof = {tau: np.load(out / f"transfer_tau{tau}.npz")["improve_oof"] for tau in TAUS}
    domain = (np.isfinite(rows_.y) & (rows_.y < 150) & rows_.raw_qc_ok).to_numpy()
    y = rows_.y.to_numpy()
    records = []
    for tau in TAUS[1:]:
        for k in (4, 8, 11, 18):
            e, e0 = held[tau]["qc"][:, k - 1] - held[tau]["y"], held[TAUS[0]]["qc"][:, k - 1] - held[TAUS[0]]["y"]
            for name, stat in (("RMSE", rmse), ("median |error|", medabs)):
                d, lo, hi = compare(held[tau]["site"], e, e0, stat)
                records.append({"score": f"Held-out test, {name}", "tau": tau, "k": k, "difference": d, "lo": lo, "hi": hi})
        for k in (5, 8, 12, 18):
            d, lo, hi = compare(rows_.site.to_numpy()[domain], (oof[tau][:, k - 1] - y)[domain], (oof[TAUS[0]][:, k - 1] - y)[domain], medabs)
            records.append({"score": "Out-of-site, median |error|", "tau": tau, "k": k, "difference": d, "lo": lo, "hi": hi})
    table = pd.DataFrame(records)
    table["excludes_zero"] = (table.lo > 0) | (table.hi < 0)
    table.to_csv(out / "bootstrap.csv", index=False)
    return table


def qc_spectrum_effect(area, out, k=4):
    """What leaving the QC-failed train spectrum out of the calibration changes in EC, on every network scan.

    Both models are fitted on the train split of the converged tau 0.1 spectra and applied to the converged network
    run (network_baselines_v2_20261103, scans saved under the 87-blank background), the Addis set and the held-out
    IMPROVE split. Writes, all prefixed qc_spectrum_: scans.csv (one row per scan, at k), groups.csv, sites.csv,
    by_k.csv (the same comparison at every k), checks.csv (scores of the two models at k) and leverage.csv (the train
    spectra farthest from the mean, and what dropping each one alone does).
    """
    import re

    area, out = Path(area), Path(out)
    rows_ = pd.read_csv(out / "improve_rows.csv")
    cases = pd.read_csv(area / "output/tables/vibes_fullrange/interpretation_20261002/cases.csv")
    y = cases[cases.kind.eq("calibration")].y.to_numpy(float)
    train, test, qc = rows_.split.eq("train").to_numpy(), rows_.split.eq("test").to_numpy(), rows_.raw_qc_ok.to_numpy(bool)
    clean = train & qc
    X = np.asarray(np.load(out / "corrected_tau0.1_improve.npy"), dtype=np.float64)
    whole, without = ca.pls_fit(X[train], y[train], KMAX), ca.pls_fit(X[clean], y[clean], KMAX)

    v2 = area / "output/tables/network_baselines_v2_20261103"
    diagnostics = pd.read_csv(v2 / "vibes_fit_diagnostics.csv")
    keep = diagnostics.vibes_model.eq("b87").to_numpy()
    spectra = np.asarray(np.load(area / "output/tables/network_baselines_20261019/vibes_bundle/X_model_grid.npy"), dtype=np.float64)[keep]
    network = spectra - np.asarray(np.load(v2 / "vibes_baselines_model_grid.npy"), dtype=np.float64)[keep]
    net_with, net_without = ca.pls_predict(whole, network), ca.pls_predict(without, network)
    test_with, test_without = ca.pls_predict(whole, X[test]), ca.pls_predict(without, X[test])
    scans = diagnostics.loc[keep, ["analysis_id", "filter_id", "filter_key", "code", "filter_type", "premature_stop"]].reset_index(drop=True)
    scans["ec_with"], scans["ec_without"] = net_with[:, k - 1], net_without[:, k - 1]
    # the saved run's own rebuilt k 4 model, as a check that "with" is that model
    scans["ec_v2_rebuilt_model"] = diagnostics.loc[keep, "ec_newton_baseline_rebuilt_model_ug"].to_numpy()
    scans.to_csv(out / "qc_spectrum_scans.csv", index=False)
    pm, blank = scans.filter_type.eq("PM2.5").to_numpy(), scans.filter_type.eq("FB").to_numpy()

    def moved(a, b):
        """Median |change| in percent and the share moved by more than 10 %, over scans reading at least 1 µg."""
        loaded = a >= 1
        relative = np.abs(b - a)[loaded] / a[loaded]
        return 100 * float(np.median(relative)), 100 * float(np.mean(relative > 0.1))

    groups = []
    for name, m in (("PM2.5 scans", pm), ("Addis PM2.5 scans", pm & scans.code.eq("ETAD").to_numpy()),
                    ("PM10 scans", scans.filter_type.eq("PM10").to_numpy()), ("Field blanks", blank)):
        a, b = scans.ec_with.to_numpy()[m], scans.ec_without.to_numpy()[m]
        pct, share = moved(a, b)
        groups.append({"group": name, "scans": int(m.sum()), "median_with": np.median(a), "median_without": np.median(b),
                       "median_abs_ug": np.median(np.abs(b - a)), "median_abs_pct": pct, "moved_gt10_pct": share, "largest_ug": np.abs(b - a).max()})
    pd.DataFrame(groups).to_csv(out / "qc_spectrum_groups.csv", index=False)

    sites = []
    for code, g in scans[pm].groupby("code"):
        sites.append({"site": code, "scans": len(g), "median_with": g.ec_with.median(), "median_without": g.ec_without.median(),
                      "change_pct": 100 * (g.ec_without.median() / g.ec_with.median() - 1)})
    pd.DataFrame(sites).sort_values("change_pct").to_csv(out / "qc_spectrum_sites.csv", index=False)

    by_k = []
    for kk in range(1, KMAX + 1):
        pct, share = moved(net_with[pm, kk - 1], net_without[pm, kk - 1])
        t_pct, _ = moved(test_with[:, kk - 1], test_without[:, kk - 1])
        rmse = lambda p: float(np.sqrt(np.mean((p[:, kk - 1] - y[test]) ** 2)))  # noqa: E731
        by_k.append({"k": kk, "network_median_abs_pct": pct, "network_moved_gt10_pct": share, "improve_test_median_abs_pct": t_pct,
                     "coefficient_correlation": float(np.corrcoef(whole[2][:, kk - 1], without[2][:, kk - 1])[0, 1]),
                     "heldout_rmse_with": rmse(test_with), "heldout_rmse_without": rmse(test_without),
                     "blank_median_with": float(np.median(net_with[blank, kk - 1])), "blank_median_without": float(np.median(net_without[blank, kk - 1]))})
    pd.DataFrame(by_k).to_csv(out / "qc_spectrum_by_k.csv", index=False)

    # scores of the two models at k: IMPROVE held-out, Addis against HIPS, blanks, repeat scans of one filter
    addis = np.asarray(np.load(out / "corrected_tau0.1_addis.npy"), dtype=np.float64)
    hips = np.load(out / "transfer_tau0.1.npz")["hips_ug"]
    reviewed = set(re.findall(r"'([A-Z]{4}-\d{4}-\d)':", (HERE.parents[3] / "gallery/app/src/lib/nativeScanReview.ts").read_text()))
    kept = scans[pm & ~scans.filter_id.isin(reviewed).to_numpy()]
    pairs = np.array([kept.index.to_numpy()[v[:2]] for v in kept.groupby("filter_id").indices.values() if len(v) >= 2])
    checks = []
    for name, model, net, held in (("With the spectrum", whole, net_with, test_with), ("Without it", without, net_without, test_without)):
        error = held[:, k - 1] - y[test]
        on_addis = ca.pls_predict(model, addis)[:, k - 1]
        slope, intercept, r2, n = ca.ols(hips, on_addis)
        ec = net[:, k - 1]
        checks.append({"model": name, "heldout_rmse": float(np.sqrt(np.mean(error ** 2))), "heldout_median_abs": float(np.median(np.abs(error))),
                       "heldout_bias": float(error.mean()), "addis_slope": slope, "addis_intercept": intercept, "addis_r2": r2,
                       "blank_median": float(np.median(ec[blank])), "blank_iqr": float(np.subtract(*np.quantile(ec[blank], [0.75, 0.25]))),
                       "repeat_gap": float(np.median(np.abs(ec[pairs[:, 0]] - ec[pairs[:, 1]]))), "negative_share_pct": 100 * float(np.mean(ec[pm] < 0)),
                       "failed_spectrum_ec": float(ca.pls_predict(model, X[train & ~qc])[0, k - 1])})
    pd.DataFrame(checks).to_csv(out / "qc_spectrum_checks.csv", index=False)

    # is it only this spectrum? the train spectra farthest from the mean, each dropped alone from the cleaned split
    distance = np.linalg.norm(X - X[train].mean(0), axis=1)
    typical = float(np.median(distance[train]))
    reference = net_without[pm, k - 1]
    leverage = []
    for i in np.argsort(-np.where(train, distance, 0))[:8]:
        if qc[i]:
            rows = clean.copy()
            rows[i] = False
            effect = moved(reference, ca.pls_predict(ca.pls_fit(X[rows], y[rows], k), network)[pm, k - 1])[0]
        else:
            effect = moved(net_with[pm, k - 1], reference)[0]
        leverage.append({"filter": str(rows_.filter_id[i]), "site": rows_.site[i], "raw_qc": "passed" if qc[i] else "failed", "tor_ec_ug": y[i],
                         "distance": distance[i] / typical, "span": float(np.ptp(X[i])), "network_median_abs_pct": effect})
    rng = np.random.default_rng(20261116)
    random = []
    for _ in range(20):
        rows = clean.copy()
        rows[rng.choice(np.flatnonzero(clean), 100, replace=False)] = False
        random.append(moved(reference, ca.pls_predict(ca.pls_fit(X[rows], y[rows], k), network)[pm, k - 1])[0])
    leverage.append({"filter": "100 random spectra (median of 20 draws)", "site": "", "raw_qc": "passed", "tor_ec_ug": np.nan, "distance": np.nan, "span": np.nan,
                     "network_median_abs_pct": float(np.median(random))})
    leverage.append({"filter": "100 random spectra (largest of 20 draws)", "site": "", "raw_qc": "passed", "tor_ec_ug": np.nan, "distance": np.nan, "span": np.nan,
                     "network_median_abs_pct": float(np.max(random))})
    pd.DataFrame(leverage).to_csv(out / "qc_spectrum_leverage.csv", index=False)
    return scans


def training_sets(X, y, clean):
    """The six training sets of the training-range checks, as row masks over the IMPROVE spectra."""
    distance = np.linalg.norm(X - X[clean].mean(0), axis=1)
    return {"All QC-passed": clean, "EC below 150 µg": clean & (y < 150), "EC below 100 µg": clean & (y < 100),
            "EC below 50 µg": clean & (y < 50), "EC below 25 µg": clean & (y < 25),
            "Nearest 99 % of spectra": clean & (distance <= np.quantile(distance[clean], 0.99))}


def addis_by_training_set(area, out, kmax=25, draws=30, seed=20261116):
    """Does the Addis slope against HIPS still follow k once the training range is held fixed?

    For each of the six training sets, one model per k 1-kmax applied to the Addis spectra: the slope, intercept and R2
    of FTIR EC on HIPS BC, and the median Addis EC. The 5th to 95th percentile of the slope and intercept over
    `draws` refits on IMPROVE sites drawn with replacement says how far the training sites alone move them.
    Writes domain_addis_by_k.csv. Needs only the cached corrected spectra and the calibration cases.
    """
    area, out = Path(area), Path(out)
    rng = np.random.default_rng(seed)
    rows_ = pd.read_csv(out / "improve_rows.csv")
    cases = pd.read_csv(area / "output/tables/vibes_fullrange/interpretation_20261002/cases.csv")
    y = cases[cases.kind.eq("calibration")].y.to_numpy(float)
    site = rows_.site.to_numpy()
    clean = rows_.split.eq("train").to_numpy() & rows_.raw_qc_ok.to_numpy(bool)
    X = np.asarray(np.load(out / "corrected_tau0.1_improve.npy"), dtype=np.float64)
    addis = np.asarray(np.load(out / "corrected_tau0.1_addis.npy"), dtype=np.float64)
    hips = np.load(out / "transfer_tau0.1.npz")["hips_ug"]
    lines = lambda ec: np.array([ca.ols(hips, ec[:, k])[:3] for k in range(kmax)])  # noqa: E731  (k, [slope, intercept, r2])
    records = []
    for name, rows in training_sets(X, y, clean).items():
        ec = ca.pls_predict(ca.pls_fit(X[rows], y[rows], kmax), addis)
        fit = lines(ec)
        members = {s_: np.flatnonzero(rows & (site == s_)) for s_ in np.unique(site[rows])}
        names = list(members)
        boot = np.empty((draws, kmax, 3))
        for j in range(draws):
            chosen = np.concatenate([members[names[i]] for i in rng.integers(0, len(names), len(names))])
            boot[j] = lines(ca.pls_predict(ca.pls_fit(X[chosen], y[chosen], kmax), addis))
        lo, hi = np.percentile(boot, [5, 95], axis=0)
        records += [{"training": name, "k": k + 1, "addis_slope": fit[k, 0], "slope_lo": lo[k, 0], "slope_hi": hi[k, 0],
                     "addis_intercept": fit[k, 1], "intercept_lo": lo[k, 1], "intercept_hi": hi[k, 1], "addis_r2": fit[k, 2],
                     "addis_median_ec": float(np.median(ec[:, k])), "addis_n": int(np.isfinite(hips).sum())} for k in range(kmax)]
        print(name, "done", flush=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "domain_addis_by_k.csv", index=False)
    return table


def training_domain_stability(area, out, ks=(4, 8, 12), draws=30, seed=20261116):
    """Is a model trained without the heaviest filters stable on SPARTAN? One model per training set, each tested three ways.

    Training sets: every QC-passed train spectrum, then only those with thermal EC below 150, 100, 50 and 25 µg, and
    the 99 % of spectra nearest the mean spectrum. For each, at each k:
    - one spectrum: the largest change in network PM2.5 EC from dropping any one of its six most extreme spectra;
    - resampled sites: IMPROVE sites drawn with replacement `draws` times, and the spread of each scan's EC and of
      each SPARTAN site's median EC across the refits;
    - scores: held-out IMPROVE error on filters below 25 µg and below 150 µg, blanks, repeat scans, Addis against HIPS.
    Writes domain_policies.csv, domain_pairs.csv (how far two training sets' network EC differ) and domain_sites.csv.
    """
    import re

    area, out = Path(area), Path(out)
    rng = np.random.default_rng(seed)
    rows_ = pd.read_csv(out / "improve_rows.csv")
    cases = pd.read_csv(area / "output/tables/vibes_fullrange/interpretation_20261002/cases.csv")
    y = cases[cases.kind.eq("calibration")].y.to_numpy(float)
    train, test, qc = rows_.split.eq("train").to_numpy(), rows_.split.eq("test").to_numpy(), rows_.raw_qc_ok.to_numpy(bool)
    site = rows_.site.to_numpy()
    clean = train & qc
    X = np.asarray(np.load(out / "corrected_tau0.1_improve.npy"), dtype=np.float64)
    v2 = area / "output/tables/network_baselines_v2_20261103"
    diagnostics = pd.read_csv(v2 / "vibes_fit_diagnostics.csv")
    keep = diagnostics.vibes_model.eq("b87").to_numpy()
    network = (np.asarray(np.load(area / "output/tables/network_baselines_20261019/vibes_bundle/X_model_grid.npy"), dtype=np.float64)[keep]
               - np.asarray(np.load(v2 / "vibes_baselines_model_grid.npy"), dtype=np.float64)[keep])
    scans = diagnostics.loc[keep, ["filter_id", "code", "filter_type"]].reset_index(drop=True)
    pm, blank = scans.filter_type.eq("PM2.5").to_numpy(), scans.filter_type.eq("FB").to_numpy()
    codes = scans.code.to_numpy()
    spartan_sites = sorted(set(codes[pm]))
    addis = np.asarray(np.load(out / "corrected_tau0.1_addis.npy"), dtype=np.float64)
    hips = np.load(out / "transfer_tau0.1.npz")["hips_ug"]
    reviewed = set(re.findall(r"'([A-Z]{4}-\d{4}-\d)':", (HERE.parents[3] / "gallery/app/src/lib/nativeScanReview.ts").read_text()))
    kept = scans[pm & ~scans.filter_id.isin(reviewed).to_numpy()]
    pairs = np.array([kept.index.to_numpy()[v[:2]] for v in kept.groupby("filter_id").indices.values() if len(v) >= 2])
    policies = training_sets(X, y, clean)
    kmax = max(ks)

    def moved(a, b):
        loaded = a >= 1
        return 100 * float(np.median(np.abs(b - a)[loaded] / a[loaded]))

    site_median = lambda ec: np.array([np.median(ec[pm & (codes == c)]) for c in spartan_sites])  # noqa: E731
    full, records, site_rows = {}, [], []
    for name, rows in policies.items():
        model = ca.pls_fit(X[rows], y[rows], kmax)
        net, held, on_addis = ca.pls_predict(model, network), ca.pls_predict(model, X[test]), ca.pls_predict(model, addis)
        full[name] = net
        # one spectrum: drop each of the six spectra farthest from this training set's own mean
        far = np.linalg.norm(X - X[rows].mean(0), axis=1)
        extreme = np.argsort(-np.where(rows, far, 0))[:6]
        one = {k: [] for k in ks}
        for i in extreme:
            fewer = rows.copy()
            fewer[i] = False
            dropped = ca.pls_predict(ca.pls_fit(X[fewer], y[fewer], kmax), network)
            for k in ks:
                one[k].append(moved(net[pm, k - 1], dropped[pm, k - 1]))
        # resampled sites: each IMPROVE site's train rows taken as many times as the site is drawn
        members = {s_: np.flatnonzero(rows & (site == s_)) for s_ in np.unique(site[rows])}
        names = list(members)
        boot = np.empty((draws, len(network), len(ks)))
        for j in range(draws):
            chosen = np.concatenate([members[names[i]] for i in rng.integers(0, len(names), len(names))])
            boot[j] = ca.pls_predict(ca.pls_fit(X[chosen], y[chosen], kmax), network)[:, [k - 1 for k in ks]]
        for n_k, k in enumerate(ks):
            ec = net[:, k - 1]
            loaded = pm & (ec >= 1)
            scan_spread = 100 * np.median(boot[:, loaded, n_k].std(0) / ec[loaded])
            medians = np.array([site_median(boot[j, :, n_k]) for j in range(draws)])
            site_spread = 100 * medians.std(0) / site_median(ec)
            error = held[:, k - 1] - y[test]
            slope, intercept, r2, _ = ca.ols(hips, on_addis[:, k - 1])
            records.append({
                "training": name, "k": k, "train_spectra": int(rows.sum()), "heaviest_ug": float(y[rows].max()),
                "one_spectrum_max_pct": max(one[k]), "scan_spread_pct": float(scan_spread),
                "site_spread_median_pct": float(np.median(site_spread)), "site_spread_max_pct": float(site_spread.max()),
                "site_spread_max_site": spartan_sites[int(site_spread.argmax())],
                "heldout_rmse_below25": float(np.sqrt(np.mean(error[y[test] < 25] ** 2))), "heldout_rmse_below150": float(np.sqrt(np.mean(error[y[test] < 150] ** 2))),
                "heldout_median_abs_below25": float(np.median(np.abs(error[y[test] < 25]))),
                "blank_median": float(np.median(ec[blank])), "repeat_gap": float(np.median(np.abs(ec[pairs[:, 0]] - ec[pairs[:, 1]]))),
                "addis_slope": slope, "addis_r2": r2, "network_median_ec": float(np.median(ec[pm])),
                "change_from_all_pct": moved(full["All QC-passed"][pm, k - 1], ec[pm])})
            if k == ks[0]:
                site_rows += [{"training": name, "site": c, "median_ec": m_, "spread_pct": sp} for c, m_, sp in zip(spartan_sites, site_median(ec), site_spread)]
        print(name, "done", flush=True)
    pd.DataFrame(records).to_csv(out / "domain_policies.csv", index=False)
    pd.DataFrame(site_rows).to_csv(out / "domain_sites.csv", index=False)
    pd.DataFrame([{"training": a, "against": b, "k": k, "median_abs_pct": 0.0 if a == b else moved(full[a][pm, k - 1], full[b][pm, k - 1])}
                  for k in ks for a in policies for b in policies]).to_csv(out / "domain_pairs.csv", index=False)
    return pd.DataFrame(records)


def main():
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--area", required=True, help="research/ftir_hips_chem")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bundle, area, out = Path(a.bundle), Path(a.area), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = np.load(area / "output/tables/network_baselines_20261019/vibes_bundle/background_model.npz")
    mean, components = model["mean"].astype(float), model["components"].astype(float)
    im, ad = pd.read_csv(bundle / "improve_meta.csv"), pd.read_csv(bundle / "addis_meta.csv")
    cases = pd.read_csv(area / "output/tables/vibes_fullrange/interpretation_20261002/cases.csv")
    cases = cases[cases.kind.eq("calibration")].reset_index(drop=True)
    if len(cases) != len(im) or not np.array_equal(cases.filter_id.astype(str), im.filter_id.astype(str)):
        raise ValueError("calibration cases and the bundle's IMPROVE rows are not the same filters in the same order")
    train, test = cases.split.eq("train").to_numpy(), cases.split.eq("test").to_numpy()
    y_cal = cases.y.to_numpy(float)

    # transfer scoring, as run_transfer_diagnosis_20261102.fit
    y, sites = im.ec_ug.to_numpy(float), im.site.to_numpy()
    splits = list(GroupKFold(5).split(np.zeros((len(y), 1)), y, sites))
    qc = recorded_source_qc(im)[0]
    eligible = np.isfinite(y) & (y < 150) & qc
    im[["filter_id", "site"]].assign(y=y, eligible=eligible, split=cases.split.to_numpy(), raw_qc_ok=qc).to_csv(out / "improve_rows.csv", index=False)
    hips_ug = (ad.fabs_Mm1 * ad.volume_m3 / 10).to_numpy()

    raw = {g: np.load(bundle / f"{g}_raw.npy", mmap_mode="r") for g in ("improve", "addis")}
    scores, curves, transfer = [], [], []
    for tau in TAUS:
        t0 = time.perf_counter()
        X = corrected(raw["improve"], mean, components, tau, out / f"corrected_tau{tau}_improve.npy")
        XA = corrected(raw["addis"], mean, components, tau, out / f"corrected_tau{tau}_addis.npy")
        print(f"tau {tau}: corrected in {time.perf_counter() - t0:.0f} s", flush=True)

        t0 = time.perf_counter()
        held = {"filter_id": cases.filter_id[test].to_numpy().astype(str), "site": cases.Site[test].to_numpy().astype(str), "y": y_cal[test]}
        # "all" is the frozen procedure; "qc" leaves the bundle's QC-failed raw spectra out of the train split
        for training, rows in (("all", train), ("qc", train & qc)):
            curve = component_cv_curve(X[rows], y_cal[rows], range(1, 31), groups=cases.Site[rows], n_splits=5, random_state=42)
            k_selected = int(select_first_major_minimum(curve)[0])
            curves.append(curve.assign(tau=tau, training=training, selected_k=k_selected))
            pred = ca.pls_predict(ca.pls_fit(X[rows], y_cal[rows], KMAX), X[test])
            held[training] = pred
            error = pred - y_cal[test][:, None]
            for k in range(1, KMAX + 1):
                e = error[:, k - 1]
                scores.append({"tau": tau, "training": training, "k": k, "selected": k == k_selected,
                               "R2": 1 - (e ** 2).sum() / ((y_cal[test] - y_cal[test].mean()) ** 2).sum(),
                               "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
                               "median |error|": float(np.median(np.abs(e))), "bias": float(e.mean()),
                               "n_train": int(rows.sum()), "n_test": int(test.sum())})
            print(f"tau {tau}, train split {training}: selected k {k_selected}", flush=True)
        np.savez_compressed(out / f"heldout_tau{tau}.npz", **held)

        oof = np.full((len(y), KMAX), np.nan)
        for tr, te in splits:
            oof[te] = ca.pls_predict(ca.pls_fit(X[tr[eligible[tr]]], y[tr[eligible[tr]]], KMAX), X[te])
        addis = ca.pls_predict(ca.pls_fit(X[eligible], y[eligible], KMAX), XA)
        np.savez_compressed(out / f"transfer_tau{tau}.npz", improve_oof=oof, addis=addis, hips_ug=hips_ug)
        domain = np.isfinite(y) & (y < 150)
        for k in range(1, KMAX + 1):
            slope, intercept, r2, n = ca.ols(hips_ug, addis[:, k - 1])
            error = oof[domain, k - 1] - y[domain]
            transfer.append({"tau": tau, "k": k, "IMPROVE out-of-site median |error| (µg)": float(np.nanmedian(np.abs(error))),
                             "IMPROVE out-of-site RMSE (µg)": float(np.sqrt(np.nanmean(error ** 2))),
                             "Addis slope vs HIPS": slope, "Addis intercept vs HIPS (µg)": intercept, "Addis R2": r2, "Addis n": n})
        print(f"tau {tau}: scored in {time.perf_counter() - t0:.0f} s", flush=True)
        pd.DataFrame(scores).to_csv(out / "calibration_scores.csv", index=False)
        pd.concat(curves).to_csv(out / "calibration_cv_curves.csv", index=False)
        pd.DataFrame(transfer).to_csv(out / "transfer_scores.csv", index=False)
    s = pd.DataFrame(scores)
    print(s[s.selected | s.k.isin([4, 11])].round(4).to_string(index=False))
    t = pd.DataFrame(transfer)
    print(t[t.k.isin([5, 8, 12, 18])].round(3).to_string(index=False))
    print(bootstrap(out).round(3).to_string(index=False))
    qc_spectrum_effect(area, out)
    training_domain_stability(area, out)
    addis_by_training_set(area, out)


if __name__ == "__main__":
    main()
