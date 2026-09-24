"""Portable, checkpointed VIBES/AIRSpec experiment used by the Colab notebook.

Only staged arrays and metadata are read: no local Drive path discovery. Raw
replicates, exclusion flags, labels and outer site split are frozen by the builder.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import time
import sys

# Same relative layout in the repository and the portable bundle.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ftir_ec_phase3/scripts"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from airspec_baseline import airspec_baseline_matrix
from pls_transfer import component_cv_curve, select_first_major_minimum, ftir_source_band_features
from vibes_baseline import fit_vibes_background, vibes_baseline_matrix
from config import MAC_VALUE


@dataclass(frozen=True)
class RunConfig:
    profile: str = "full"  # smoke, pilot, full
    seed: int = 20260920
    workers: int = 2
    batch_size: int = 64
    max_blank_per_lot: int = 50
    max_background_components: int = 30
    tau: float = 0.1
    loss: str = "PB"
    maxiter: int = 10000
    max_pls_components: int = 30
    bootstrap_repeats: int = 2000
    background_source: str = "combined"  # combined, etad_only

    def validate(self):
        if self.profile not in ("smoke", "pilot", "full"):
            raise ValueError("profile must be smoke, pilot or full")
        if self.background_source not in ("combined", "etad_only"):
            raise ValueError("background_source must be combined or etad_only")
        if self.loss not in ("PB", "ALS") or not 0 < self.tau < 0.5:
            raise ValueError("Invalid VIBES loss/tau")
        for name in (
            "workers",
            "batch_size",
            "max_blank_per_lot",
            "max_background_components",
            "maxiter",
            "max_pls_components",
            "bootstrap_repeats",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "BUNDLE_MANIFEST.json").read_text())
    files = manifest["files_sha256"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != manifest["content_hash"]:
        raise ValueError("Bundle manifest hash mismatch")
    for name, expected in files.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or file_sha(path) != expected:
            raise ValueError(f"Bundle integrity failure: {name}")
    return manifest


def _sample(frame, n, rng):
    return (
        frame
        if n is None or len(frame) <= n
        else frame.loc[np.sort(rng.choice(frame.index, n, replace=False))]
    )


def prepare_experiment(bundle, config):
    """Select rows and independent blanks, without using any held-out outcomes."""
    config.validate()
    data = Path(bundle) / "data"
    wn = np.load(data / "wn.npy")
    pool = pd.read_csv(data / "pool_metadata.csv")
    etad = pd.read_csv(data / "etad_metadata.csv")
    px = np.load(data / "pool_raw.npy", mmap_mode="r")
    ex = np.load(data / "etad_raw.npy", mmap_mode="r")
    rng = np.random.default_rng(config.seed)
    # Outer-test-site blanks are never used to fit the VIBES background.
    blank_candidates = pool[
        pool.FilterPurposeId.eq(2) & pool.finite_spectrum & pool.all_scans_have_purpose
    ].copy()
    train_bank, hold_bank = [], []
    for _, g in blank_candidates.groupby("LotNumber", dropna=False):
        outer_hold = g[~g.split.eq("train")]
        candidates = g[g.split.eq("train")]
        n_hold = max(1, round(0.25 * len(candidates))) if len(candidates) >= 4 else 0
        hold = _sample(candidates, n_hold, rng)
        train = _sample(candidates.drop(hold.index), config.max_blank_per_lot, rng)
        train_bank.extend(train.index)
        hold_bank.extend([*outer_hold.index, *hold.index])
    if config.background_source == "etad_only":
        train_bank = []
        hold_bank = list(blank_candidates.index)
    et = etad.index[etad.role.eq("blank_train")].to_numpy()
    bt = np.concatenate([px[train_bank], ex[et]], axis=0)
    bids = [f"improve:{int(pool.loc[i, 'FilterId'])}" for i in train_bank]
    bids += [f"etad:{int(etad.loc[i, 'MediaId'])}" for i in et]
    bmeta = pd.concat(
        [
            pool.loc[train_bank, ["FilterId", "Site", "LotNumber", "split"]].assign(
                source="IMPROVE"
            ),
            etad.loc[et, ["MediaId", "LotId"]]
            .rename(columns={"MediaId": "FilterId", "LotId": "LotNumber"})
            .assign(source="ETAD", Site="ETAD", split="independent_blank_train"),
        ],
        ignore_index=True,
    )
    bmeta["sample_id"] = bids
    limits = {
        "smoke": (32, 32, 12, 6),
        "pilot": (800, 400, 253, 30),
        "full": (None, None, None, None),
    }
    n_locked, n_other, n_etad, n_blank = limits[config.profile]
    eligible = pool[pool.eligible]
    selected = pd.concat(
        [
            _sample(eligible[eligible.locked800], n_locked, rng),
            _sample(eligible[~eligible.locked800], n_other, rng),
        ]
    ).sort_index()
    aerosol = _sample(etad[etad.role.eq("sample")], n_etad, rng)
    # Both source types are represented in pilot diagnostics; full uses all held-out blanks.
    hb_pool = _sample(pool.loc[hold_bank], n_blank, rng)
    hb_etad = _sample(etad[etad.role.eq("blank_test")], n_blank, rng)
    matrices, records = [], []

    def append_rows(frame, matrix, source, kind):
        for i, row in frame.iterrows():
            sid = f"{source}:{int(row.FilterId if source == 'improve' else row.MediaId)}"
            record = {
                "sample_id": sid,
                "kind": kind,
                "source": source,
                "source_row": int(i),
                "Site": row.get("Site", "ETAD"),
                "lot": row.get("LotNumber", row.get("LotId", np.nan)),
                "split": row.get("split", "external"),
                "locked800": bool(row.get("locked800", False)),
                "y": row.get("TOR_EC_loading_ug", np.nan),
                "volume": row.get("SampleVolume_m3", np.nan),
                "Fabs": row.get("Fabs", np.nan),
                "date": row.get("date", ""),
                "filter_id": row.get("ExternalFilterId", row.get("FilterId", "")),
            }
            matrices.append(np.asarray(matrix[i], float))
            records.append(record)

    append_rows(selected, px, "improve", "calibration")
    append_rows(aerosol, ex, "etad", "target")
    append_rows(hb_pool, px, "improve", "blank")
    append_rows(hb_etad, ex, "etad", "blank")
    base_count = len(records)
    # Fixed additions to up to nine ETAD held-out blanks (all nine in full mode).
    injection_rows = [
        i for i, r in enumerate(records) if r["kind"] == "blank" and r["source"] == "etad"
    ]
    shape = sum(
        a * np.exp(-0.5 * ((wn - c) / w) ** 2)
        for a, c, w in zip(
            [0.7, 0.5, 1.0, 0.8], [1610.0, 1720.0, 2920.0, 3400.0], [30.0, 22.0, 35.0, 130.0]
        )
    )
    for amplitude in (0.01, 0.05, 0.15):
        for parent in injection_rows:
            rec = dict(
                records[parent],
                sample_id=f"spike:{records[parent]['sample_id']}:{amplitude}",
                kind="injection",
                parent=parent,
                amplitude=amplitude,
            )
            matrices.append(matrices[parent] + amplitude * shape)
            records.append(rec)
    cases = pd.DataFrame(records)
    if not cases.sample_id.is_unique or set(cases.sample_id) & set(bids):
        raise ValueError("Physical blank leakage or duplicated evaluation IDs")
    cal = cases[cases.kind.eq("calibration")]
    if set(cal.loc[cal.split.eq("train"), "Site"]) & set(cal.loc[cal.split.eq("test"), "Site"]):
        raise ValueError("Outer site split overlaps")
    if set(bmeta.loc[bmeta.source.eq("IMPROVE"), "Site"]) & set(
        pool.loc[pool.split.eq("test"), "Site"]
    ):
        raise ValueError("Outer-test-site blank leaked into background model")
    return wn, np.stack(matrices), cases, bt, bids, bmeta, shape


def _correct_one(i, y, wn, bg, config):
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        _, air = airspec_baseline_matrix(wn, y[None], df1=6, df2=4)
        air_seconds = time.perf_counter() - start
        _, vib, diag = vibes_baseline_matrix(
            wn,
            y[None],
            bg,
            sample_ids=[str(i)],
            tau=config.tau,
            loss=config.loss,
            maxiter=config.maxiter,
        )
    record = diag.iloc[0].to_dict()
    record.update(case_row=i, airspec_seconds=air_seconds)
    return air[0], vib[0], record


def atomic_checkpoint(path, signature, indices, airspec, vibes, diagnostics):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.array(signature),
            indices=indices,
            airspec=np.asarray(airspec, np.float32),
            vibes=np.asarray(vibes, np.float32),
            diagnostics=np.array(json.dumps(diagnostics)),
        )
    os.replace(temporary, path)


def read_checkpoint(path, signature, indices, channels):
    with np.load(path) as z:
        if str(z["signature"]) != signature or not np.array_equal(z["indices"], indices):
            raise ValueError(f"Stale/misaligned checkpoint: {path}")
        air, vib = z["airspec"], z["vibes"]
        if air.shape != (len(indices), channels) or vib.shape != air.shape:
            raise ValueError(f"Invalid checkpoint shape: {path}")
        diagnostics = json.loads(str(z["diagnostics"]))
        if [r["case_row"] for r in diagnostics] != indices.tolist():
            raise ValueError(f"Misaligned checkpoint diagnostics: {path}")
        return air, vib, diagnostics


def paired_site_bootstrap(y, air, vib, sites, *, repeats=2000, seed=20260920):
    """Paired VIBES−AIRSpec RMSE CI, resampling held-out sites as clusters."""
    d = pd.DataFrame({"site": sites, "a": (air - y) ** 2, "v": (vib - y) ** 2, "n": 1})
    sums = d.groupby("site")[["a", "v", "n"]].sum().to_numpy()
    if len(sums) < 2:
        return {
            "delta_rmse": float(
                np.sqrt(np.mean((vib - y) ** 2)) - np.sqrt(np.mean((air - y) ** 2))
            ),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_sites": len(sums),
        }
    draw = np.random.default_rng(seed).integers(0, len(sums), (repeats, len(sums)))
    total = sums[draw].sum(axis=1)
    delta = np.sqrt(total[:, 1] / total[:, 2]) - np.sqrt(total[:, 0] / total[:, 2])
    return {
        "delta_rmse": float(np.sqrt(d.v.mean()) - np.sqrt(d.a.mean())),
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
        "n_sites": len(sums),
    }


def compare_calibrations(cases, arrays, output, config):
    output = Path(output)
    paired = np.isfinite(arrays["AIRSpec"]).all(axis=1) & np.isfinite(arrays["VIBES"]).all(axis=1)
    case_audit = cases.copy()
    case_audit["paired_valid"] = paired
    case_audit.to_csv(output / "case_audit.csv", index=False)
    scores, predictions, external, curves, intervals = [], [], [], [], []
    for cohort in ("locked800", "full_pool"):
        mask = cases.kind.eq("calibration").to_numpy() & paired
        if cohort == "locked800":
            mask &= cases.locked800.to_numpy(bool)
        train = mask & cases.split.eq("train").to_numpy()
        test = mask & cases.split.eq("test").to_numpy()
        target = cases.kind.eq("target").to_numpy() & paired
        n_sites = cases.loc[train, "Site"].nunique()
        if n_sites < 5 or test.sum() < 2:
            raise ValueError(f"{cohort}: insufficient sites/rows for grouped CV; enlarge profile")
        y = cases.loc[train, "y"].to_numpy(float)
        truth = cases.loc[test, "y"].to_numpy(float)
        paired_predictions = {}
        for method, X in arrays.items():
            curve = component_cv_curve(
                X[train],
                y,
                range(1, config.max_pls_components + 1),
                groups=cases.loc[train, "Site"],
                n_splits=5,
                random_state=42,
            )
            k, _ = select_first_major_minimum(curve)
            model = PLSRegression(n_components=int(k), scale=False, max_iter=1000, tol=1e-10).fit(
                X[train], y
            )
            pred = model.predict(X[test]).ravel()
            paired_predictions[method] = pred
            scores.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "k": int(k),
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "predictive_R2": r2_score(truth, pred),
                    "RMSE": np.sqrt(np.mean((pred - truth) ** 2)),
                    "MAE": np.mean(np.abs(pred - truth)),
                    "bias": np.mean(pred - truth),
                    "n_train_sites": n_sites,
                }
            )
            frame = cases.loc[test, ["sample_id", "Site", "y"]].copy()
            frame["prediction"] = pred
            frame["cohort"] = cohort
            frame["method"] = method
            predictions.append(frame)
            ext = cases.loc[target, ["sample_id", "filter_id", "date", "volume", "Fabs"]].copy()
            ext["prediction_ugm3"] = model.predict(X[target]).ravel() / ext.volume.to_numpy(float)
            ext["HIPS_EC_equivalent"] = ext.Fabs / MAC_VALUE
            ext["method"] = method
            ext["cohort"] = cohort
            external.append(ext)
            curves.append(curve.assign(method=method, cohort=cohort, selected_k=int(k)))
            # Portable coefficient/intercept representation; no pickle required.
            np.savez_compressed(
                output / f"pls_{cohort}_{method}.npz",
                coefficient=model.coef_,
                x_mean=model._x_mean,
                y_mean=model._y_mean,
                intercept=model.intercept_,
                components=np.array(int(k)),
            )
        intervals.append(
            dict(
                cohort=cohort,
                **paired_site_bootstrap(
                    truth,
                    paired_predictions["AIRSpec"],
                    paired_predictions["VIBES"],
                    cases.loc[test, "Site"].to_numpy(),
                    repeats=config.bootstrap_repeats,
                    seed=config.seed,
                ),
            )
        )
    for name, value in [
        ("calibration_scores", pd.DataFrame(scores)),
        ("heldout_predictions", pd.concat(predictions)),
        ("addis_predictions", pd.concat(external)),
        ("cv_curves", pd.concat(curves)),
        ("paired_site_bootstrap", pd.DataFrame(intervals)),
    ]:
        value.to_csv(output / (name + ".csv"), index=False)
    return pd.DataFrame(scores)


def run_large_comparison(bundle, results_root, config=RunConfig(), *, progress=print):
    config.validate()
    bundle = Path(bundle)
    manifest = verify_bundle(bundle)
    versions = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "scipy", "pandas", "scikit-learn", "cvxpy", "clarabel"]
    }
    scientific_config = asdict(config)
    scientific_config.pop("workers")
    signature = hashlib.sha256(
        json.dumps(
            {"bundle": manifest["content_hash"], "config": scientific_config, "versions": versions},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    out = Path(results_root) / f"{config.profile}-{signature[:16]}"
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    wn, X, cases, blanks, blank_ids, bmeta, shape = prepare_experiment(bundle, config)
    cases.to_csv(out / "cases.csv", index=False)
    bmeta.to_csv(out / "background_training_blanks.csv", index=False)
    progress(
        f"{len(cases)} evaluation spectra; {len(blanks)} independent background blanks; output {out}"
    )
    with threadpool_limits(limits=1):
        bg = fit_vibes_background(
            wn, blanks, blank_ids=blank_ids, max_components=config.max_background_components
        )
    bg.cv_errors.to_csv(out / "background_rank_cv.csv", index=False)
    np.savez_compressed(
        out / "background_model.npz",
        wn=wn,
        mean=bg.mean,
        components=bg.components,
        blank_ids=np.asarray(blank_ids, dtype=str),
    )
    progress(
        f"Selected blank PCA rank {bg.components.shape[1]} (cap {config.max_background_components})"
    )
    arrays = {m: np.full(X.shape, np.nan, dtype=np.float32) for m in ("AIRSpec", "VIBES")}
    records = []
    resumed = 0
    start = time.perf_counter()
    for first in range(0, len(X), config.batch_size):
        idx = np.arange(first, min(first + config.batch_size, len(X)))
        path = checkpoints / f"batch_{first:06d}.npz"
        if path.exists():
            air, vib, diag = read_checkpoint(path, signature, idx, len(wn))
            resumed += len(idx)
        else:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                rows = Parallel(n_jobs=config.workers)(
                    delayed(_correct_one)(int(i), X[i], wn, bg, config) for i in idx
                )
            air = np.stack([r[0] for r in rows])
            vib = np.stack([r[1] for r in rows])
            diag = [r[2] for r in rows]
            atomic_checkpoint(path, signature, idx, air, vib, diag)
        arrays["AIRSpec"][idx] = air
        arrays["VIBES"][idx] = vib
        records.extend(diag)
        progress(
            f"{idx[-1] + 1}/{len(X)} saved; {resumed} resumed; elapsed {(time.perf_counter() - start) / 60:.1f} min"
        )
    correction_wall_seconds = time.perf_counter() - start
    diagnostics = (
        pd.DataFrame(records)
        .drop(columns=["sample_id"])
        .merge(cases.reset_index(names="case_row"), on="case_row", validate="one_to_one")
    )
    diagnostics.to_csv(out / "fit_diagnostics.csv", index=False)
    for method, values in arrays.items():
        np.save(out / f"corrected_{method}.npy", values)
    np.save(out / "wn.npy", wn)
    finite_pair = np.isfinite(arrays["AIRSpec"]).all(axis=1) & np.isfinite(arrays["VIBES"]).all(
        axis=1
    )
    target_rows = np.flatnonzero(cases.kind.eq("target").to_numpy() & finite_pair)
    if len(target_rows):
        delta = np.sqrt(
            np.mean((arrays["VIBES"][target_rows] - arrays["AIRSpec"][target_rows]) ** 2, axis=1)
        )
        ordered = target_rows[np.argsort(delta)]
        chosen = ordered[np.rint(np.array([0.1, 0.5, 0.9]) * (len(ordered) - 1)).astype(int)]
        np.savez_compressed(
            out / "spectral_examples.npz",
            wn=wn,
            raw=X[chosen],
            airspec=arrays["AIRSpec"][chosen],
            vibes=arrays["VIBES"][chosen],
            ids=cases.loc[chosen, "filter_id"].to_numpy(dtype=str),
        )
    # Diagnostics keep failures. Feature and calibration comparisons pair valid rows.
    features = []
    blank_records = []
    injection_records = []
    for method, values in arrays.items():
        f = ftir_source_band_features(values, wn).assign(
            sample_id=cases.sample_id, kind=cases.kind, method=method
        )
        features.append(f)
        for i in cases.index[cases.kind.eq("blank")]:
            blank_records.append(
                {
                    "sample_id": cases.loc[i, "sample_id"],
                    "source": cases.loc[i, "source"],
                    "method": method,
                    "rms_from_zero": np.sqrt(np.mean(values[i] ** 2)),
                }
            )
        for i in cases.index[cases.kind.eq("injection")]:
            parent = int(cases.loc[i, "parent"])
            amplitude = cases.loc[i, "amplitude"]
            delta = values[i] - values[parent] - amplitude * shape
            injection_records.append(
                {
                    "sample_id": cases.loc[i, "sample_id"],
                    "parent_id": cases.loc[parent, "sample_id"],
                    "amplitude": amplitude,
                    "method": method,
                    "recovery_rmse": np.sqrt(np.mean(delta**2)),
                }
            )
    pd.concat(features).to_csv(out / "spectral_features.csv", index=False)
    pd.DataFrame(blank_records).to_csv(out / "blank_metrics.csv", index=False)
    pd.DataFrame(injection_records).to_csv(out / "injection_metrics.csv", index=False)
    with threadpool_limits(limits=1):
        scores = compare_calibrations(cases, arrays, out, config)
    run = {
        "signature": signature,
        "config": asdict(config),
        "bundle_hash": manifest["content_hash"],
        "versions": versions,
        "n_cases": len(cases),
        "n_background_blanks": len(blanks),
        "background_rank": bg.components.shape[1],
        "background_rank_hits_cap": bg.components.shape[1] == int(bg.cv_errors.components.max()),
        "background_fit_seconds": bg.fit_seconds,
        "n_failed": int((~diagnostics.success).sum()),
        "n_retried": int(diagnostics.retry_count.sum()),
        "rows_resumed_this_invocation": resumed,
        "correction_wall_seconds_this_invocation": correction_wall_seconds,
    }
    (out / "RUN_MANIFEST.json").write_text(json.dumps(run, indent=2) + "\n")
    progress(scores.to_string(index=False))
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    args = parser.parse_args()
    config = RunConfig(**json.loads(args.config.read_text()))
    out = run_large_comparison(args.bundle, args.results, config)
    from plotting import PlotConfig
    from plotting.vibes_large import large_run_figures
    import matplotlib.pyplot as plt

    PlotConfig.set(sites="Addis_Ababa", layout="individual", show_stats=True, show_1to1=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    for name, fig in large_run_figures(out):
        fig.savefig(plots / (name + ".png"), bbox_inches="tight")
        plt.close(fig)
    args.pointer.write_text(json.dumps({"result_path": str(out)}, indent=2) + "\n")
    print("RESULT_PATH=" + str(out), flush=True)


if __name__ == "__main__":
    main()
