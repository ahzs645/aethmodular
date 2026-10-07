#!/usr/bin/env python3
"""Export the phase-3 calibration grid into JSON the React gallery can chart.

Source: `calibration_explorer/cache/batch_results.jsonl` — one row per
(configuration × evaluation target × k) produced by the explorer's server
batches (dense cutoff sweeps 2026-08-20, five-site grid 2026-08-23, and later).
Scores are the explorer's cached readouts (same shared-script estimators as
the notebooks); no model or cross-validation curve is refitted here.
Optional enrichment resolves the cohort membership to recover actual split
counts, outer IMPROVE holdout RMSE and inner factor-selection RMSECV.

Writes gallery/app/public/data/calibration.json:
  grid     one row per configuration × target at the *rule* k (k == auto_k),
           the number the explorer would report without a manual k
  sweeps   every k for the Addis site-held-out rows, so the k-selection
           ("optimization") curve can be drawn per configuration
  docs     the explorer's dated result write-ups, title + first paragraph

Run:  uv run python gallery/data/export_calibration.py
      uv run python gallery/data/export_calibration.py --enrich-cache
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPLORER = REPO / "calibration_explorer"
SRC = EXPLORER / "cache" / "batch_results.jsonl"
OUT = REPO / "gallery" / "app" / "public" / "data" / "calibration.json"
METADATA = EXPLORER / "cache" / "gallery_calibration_metadata.json"

SCRIPTS = REPO / "research" / "ftir_hips_chem" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import config  # noqa: E402

# Site colours come from config.SITES so the calibration tab matches every
# other tab. Bishoftu (ETBI) is not a config site; it gets a fixed colour here.
SITE_COLOR_BY_CODE = {v["code"]: v["color"] for v in config.SITES.values()}
SITE_COLOR_BY_CODE.setdefault("ETBI", "#8e44ad")


def literal_dict(source: str, name: str) -> dict:
    """Pull a top-level `NAME = {...}` literal out of app.py without importing it
    (importing loads Flask and the spectra)."""
    # single-line literal first, then a multi-line one closed by a line that is just "}"
    m = re.search(rf"^{name}\s*=\s*(\{{[^\n]*\}})\s*$", source, re.M)
    if not m:
        m = re.search(rf"^{name}\s*=\s*(\{{.*?^\}})", source, re.S | re.M)
    if not m:
        return {}
    return ast.literal_eval(m.group(1))


def compact(r: dict) -> dict:
    """Short keys: 80k rows × long keys is most of the file."""
    g = r.get
    return {
        "co": r["cohort"], "cut": r.get("cutoff"), "sel": r.get("selection_space"),
        "sp": r["spectra"], "mo": r["mode"], "tg": r["target"],
        "lot": r.get("lot", "all"), "el": r.get("eval_lot", "all"),
        "k": r.get("k"), "ak": r.get("auto_k"),
        # fixed evaluation set (Addis fixed-190 or the target's default subset)
        "om": g("ols_slope"), "ob": g("ols_intercept"),
        "dm": g("deming_slope"), "db": g("deming_intercept"),
        "ob6": g("ols_intercept_mac6"), "db6": g("deming_intercept_mac6"),
        "r2": g("R2"), "rmse": g("RMSE"),
        # all pairs
        "aom": g("all_ols_slope"), "aob": g("all_ols_intercept"),
        "adm": g("all_deming_slope"), "adb": g("all_deming_intercept"),
        "ar2": g("all_R2"),
        # held-out IMPROVE TOR test of the calibration itself
        "ho": g("heldout_R2"),
        # IMPROVE scores are TOR loadings (µg/filter), unlike target RMSE
        # above (µg/m³). The outer site holdout and inner factor-selection
        # folds are different populations and deliberately separate fields.
        "cv_rmse": g("heldout_RMSE"), "rmsecv": g("rmsecv_floor"),
        "n_cohort": g("n_cohort"), "n_fit": g("n_train"),
        "n_heldout": g("n_heldout"),
        "n": g("n_eval") or g("n"),
    }


def metadata_key(row: dict) -> str:
    fields = ("cohort", "cutoff", "selection_space", "spectra", "mode",
              "lot", "target", "eval_lot", "k", "auto_k", "heldout_R2")
    return json.dumps([row.get(f) for f in fields]
                      + [row.get("provenance")], sort_keys=True, separators=(",", ":"))


def cache_key(parts, schema=None, source=None):
    """Reproduce the explorer's legacy and content-fingerprinted cache keys."""
    prefix = [] if schema is None else [schema, source]
    value = "".join(f"{p}|" for p in [*prefix, *parts])
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def cache_readout(row, ids_hash, n_cohort, n_fit, cache_dir, namespaces,
                  target_fingerprints, max_components=30):
    """Read exact config/k caches, never estimate errors or fit another model.

    Reject a cache if its fitted count, selected-k rule, or outer holdout R²
    disagrees with the batch row. Cohort membership is in the curve key; a
    nominal cutoff is never substituted for the resolved count.
    """
    p = row.get("provenance") or {}
    if p.get("resolved_cohort_hash") not in (None, ids_hash):
        return {}
    candidates = ([(p["cache_schema"], p["source_fingerprint"])]
                  if p.get("cache_schema") and p.get("source_fingerprint")
                  else [(None, None), *namespaces])
    for schema, source in candidates:
        parts = ("curve", row["cohort"], row.get("selection_space", "raw"),
                 row["spectra"], row["mode"], max_components,
                 row.get("lot", "all"), ids_hash)
        curve_key = cache_key(parts, schema, source)
        path = cache_dir / f"{curve_key}.json"
        if not path.exists():
            continue
        curve = json.loads(path.read_text())
        selected = next((c for c in curve if c["n_components"] == row["k"]), None)
        if selected is None:
            continue
        # Different CV candidate ranges can change the rule-k choice. The
        # stored curve must reproduce the batch's rule, not only contain k.
        import pandas as pd
        from calibration_modes import protocol_select_k
        if protocol_select_k(row["mode"], pd.DataFrame(curve)) != row["auto_k"]:
            continue
        fps = ([p["target_fingerprint"]] if p.get("target_fingerprint")
               else target_fingerprints.get(row["target"], []))
        fit_parts = [("fit", curve_key, row["k"], row["target"], fp) for fp in fps]
        if schema is None:
            fit_parts.insert(0, ("fit", curve_key, row["k"], row["target"]))
        for parts in fit_parts:
            fit_path = cache_dir / f"{cache_key(parts, schema, source)}.json"
            if not fit_path.exists():
                continue
            fit = json.loads(fit_path.read_text())
            heldout = fit.get("heldout")
            if fit.get("n_train") != n_fit:
                continue
            if (heldout or {}).get("R2") != row.get("heldout_R2"):
                continue
            if row["mode"] == "site_heldout" and not heldout:
                continue
            return {"n_cohort": n_cohort, "n_train": fit["n_train"],
                    "n_heldout": n_cohort - fit["n_train"],
                    "heldout_RMSE": (heldout or {}).get("RMSE"),
                    "rmsecv_floor": round(float(selected["rmsecv"]), 4),
                    "metadata_source": {"curve": path.name, "fit": fit_path.name,
                                        "resolved_cohort_hash": ids_hash}}
    return {}


def staged_cohort_resolver(metadata_path):
    """Recover memberships from a local prepared TOR table, without Drive I/O.

    This table is only a candidate membership source. cache_readout still
    requires the explorer's exact resolved-membership cache key and count.
    Corrected-space Ethiopia ranking requires the accompanying local Addis
    metadata; its distance is the explorer's shared selection helper.
    """
    import numpy as np
    import pandas as pd
    tables = SCRIPTS.parent / "output" / "tables"
    meta = pd.read_csv(metadata_path)
    tor_columns = ["Site", "date", "TOR_EC_loading_ug", "TOR_EC_ugm3", "TOR_OC_ugm3", "OC_EC_ratio"]
    tor = meta[tor_columns].drop_duplicates()
    if tor.duplicated(["Site", "date"]).any():
        raise ValueError("staged TOR metadata has conflicting site/date references")
    similarity = pd.read_csv(tables / "pls_transfer/improve_full_pool_addis_similarity.csv")
    similarity["date"] = pd.to_datetime(similarity["SampleDate"], format="mixed").dt.strftime("%Y-%m-%d")
    joined = similarity.merge(tor, on=["Site", "date"], how="left", validate="many_to_one")
    pool = joined[joined.TOR_EC_loading_ug.gt(0)].drop_duplicates("FilterId").drop_duplicates("AnalysisId").set_index("AnalysisId")
    smoke = pd.read_csv(tables / "pls_calibration_phase2/smoke_cohort_spectral_selection.csv")
    locked = set(pd.read_csv(tables / "pls_calibration_phase2/locked_analog_train_test_split.csv").AnalysisId)
    analog = similarity.drop_duplicates("AnalysisId").sort_values("analog_rank_score")
    ranked = analog.AnalysisId.to_numpy(int)
    if len(set(ranked[::-1][:len(locked)]) & locked) > len(set(ranked[:len(locked)]) & locked):
        ranked = ranked[::-1]
    ocec = joined[joined.TOR_EC_loading_ug.gt(0) & joined.TOR_EC_ugm3.gt(0) & joined.TOR_OC_ugm3.gt(0)].drop_duplicates("FilterId").sort_values("OC_EC_ratio")
    rankings = {("ocec", "raw"): ocec.AnalysisId.to_numpy(int),
                ("eth_shaped", "raw"): smoke.sort_values("Addis_band_feature_distance").AnalysisId.to_numpy(int),
                ("analogs", "raw"): ranked}
    corrected = EXPLORER / "cache/analog_corrected_ranking.npz"
    if corrected.exists():
        rankings["analogs", "airspec"] = np.load(corrected)["ids"].astype(int)
    corr = np.load(REPO / "research/ftir_ec_phase3/output/corrected/improve_pool_corrected_df6.npz")
    corr_ids = corr["analysis_id"].astype(int)
    etad_metadata = Path(metadata_path).with_name("etad_metadata.csv")
    if etad_metadata.exists():
        from pls_transfer import band_feature_distance
        etad = pd.read_csv(etad_metadata)
        view = etad[etad.has_complete_spectrum & etad.Fabs.notna() & etad.SampleVolume_m3.gt(0)]
        etad_corr = np.load(REPO / "research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz", allow_pickle=True)
        values = pd.DataFrame(etad_corr["corrected"].astype(float))
        values["MediaId"] = etad_corr["media_id"].astype(int)
        reference = values.groupby("MediaId").mean().loc[view.MediaId.astype(int)].to_numpy(float)
        ids = smoke.AnalysisId.to_numpy(int)
        ids = ids[np.isin(ids, corr_ids)]
        positions = {int(a): i for i, a in enumerate(corr_ids)}
        distance = band_feature_distance(corr["corrected"][[positions[int(i)] for i in ids]].astype(float), corr["wn"].astype(float), reference)
        rankings["eth_shaped", "airspec"] = ids[np.argsort(distance)]
    neutral_ids = np.load(REPO / "research/ftir_ec_phase3/output/corrected/improve_pool_neutral_pspline_arpls_lam1e6.npz", allow_pickle=True)["analysis_id"].astype(int)
    lots = meta.set_index("AnalysisId").LotNumber.to_dict()

    def resolve(cohort, cutoff, selection, spectra, lot):
        if cohort == "pool":
            ids = pool.index.to_numpy(int)
        elif cohort == "smoke":
            ids = smoke.AnalysisId.to_numpy(int)
        else:
            ids = rankings[cohort, selection]
            ids = ids[np.isin(ids, pool.index)]
            ids = ids[:cutoff]
        ids = np.array(list(dict.fromkeys(int(i) for i in ids if i in pool.index)), dtype=int)
        if spectra == "airspec":
            ids = ids[np.isin(ids, corr_ids)]
        if spectra == "neutral":
            ids = ids[np.isin(ids, neutral_ids)]
        if lot not in (None, "all"):
            ids = np.array([i for i in ids if pd.notna(lots.get(i)) and str(int(lots[i])) == str(lot)], dtype=int)
        return ids, pool.loc[ids]
    return resolve


def enrich_cached_metadata(rows, staged_metadata=None):
    """Load cohort membership once, then recover existing caches only.

    A sidecar lets ordinary gallery exports stay lightweight. No call to
    run_config/protocol_cv_curve is made: a missing cache remains missing.
    """
    import numpy as np
    sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))
    from calibration_modes import protocol_train_mask
    if staged_metadata:
        resolve = staged_cohort_resolver(staged_metadata)
        explorer = None
    else:
        sys.path.insert(0, str(EXPLORER))
        import app as explorer
        start = time.monotonic()
        last = None
        while not explorer.STATE["ready"]:
            if explorer.STATE.get("error"):
                raise RuntimeError(explorer.STATE["error"])
            if time.monotonic() - start > 1800:
                raise TimeoutError("explorer data load exceeded 30 minutes")
            message = explorer.STATE.get("message")
            if message != last:
                print(message, flush=True)
                last = message
            time.sleep(2)
        def resolve(*co):
            ids, _ = explorer.resolve_cohort(*co)
            return ids, explorer.D["pool"].loc[ids]
    namespaces = sorted({(r["provenance"]["cache_schema"],
                          r["provenance"]["source_fingerprint"])
                         for r in rows if r.get("provenance")})
    fingerprints = {}
    for row in rows:
        fp = (row.get("provenance") or {}).get("target_fingerprint")
        if fp:
            fingerprints.setdefault(row["target"], set()).add(fp)
    # Current target fingerprints cover legacy rows without provenance.
    for target in ({r["target"] for r in rows} if explorer else []):
        try:
            fingerprints.setdefault(target, set()).add(explorer.get_target(target)["fingerprint"])
        except (ValueError, FileNotFoundError):
            pass
    cohorts, recovered = {}, {}
    eligible = [r for r in rows if not r.get("error") and
                (r.get("k") == r.get("auto_k") or
                 (r.get("target") == "addis" and r.get("mode") == "site_heldout"
                  and r.get("lot", "all") == "all" and r.get("eval_lot", "all") == "all"))]
    for i, row in enumerate(eligible):
        key = metadata_key(row)
        if key in recovered:
            continue
        co = (row["cohort"], row.get("cutoff"), row.get("selection_space", "raw"),
              row["spectra"], row.get("lot", "all"), row["mode"])
        if co not in cohorts:
            try:
                ids, sub = resolve(*co[:5])
            except KeyError:
                continue  # this local metadata does not cover that ranking
            train = protocol_train_mask(co[5], np.zeros((len(ids), 1)),
                                        sub["TOR_EC_loading_ug"], sub["Site"])
            cohorts[co] = (hashlib.sha1(np.sort(ids).tobytes()).hexdigest()[:12],
                           len(ids), int(train.sum()))
        recovered[key] = cache_readout(row, *cohorts[co], EXPLORER / "cache",
                                       namespaces, fingerprints)
        if i and i % 2000 == 0:
            print(f"checked {i:,}/{len(eligible):,} cached rows", flush=True)
    metadata = {k: v for k, v in recovered.items() if v}
    METADATA.write_text(json.dumps(metadata, separators=(",", ":")))
    print(f"recovered cache metadata: {len(metadata):,}/{len(recovered):,} rows", flush=True)
    return metadata


def first_paragraph(md: str) -> str:
    body = md.split("\n", 1)[1] if md.startswith("#") else md
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if p and not p.startswith(("#", "|", "```", ">", "-", "*")):
            return re.sub(r"\s+", " ", p)[:600]
    return ""


def main(enrich_cache=False, staged_metadata=None):
    if not SRC.exists():
        print(f"! {SRC} not found — run a server batch in the calibration explorer first")
        return
    app_src = (EXPLORER / "app.py").read_text()
    cohorts = literal_dict(app_src, "COHORTS")
    spectra_label = literal_dict(app_src, "SPECTRA_LABEL")
    default_cutoff = literal_dict(app_src, "DEFAULT_CUTOFF")
    registry = json.loads((EXPLORER / "target_registry.json").read_text()) if (EXPLORER / "target_registry.json").exists() else {}

    source_rows = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    metadata = (enrich_cached_metadata(source_rows, staged_metadata) if enrich_cache else
                json.loads(METADATA.read_text()) if METADATA.exists() else {})
    grid, sweeps = [], []
    seen_grid = set()
    n_rows = 0
    for r in source_rows:
        n_rows += 1
        if r.get("error"):
            continue
        c = compact({**r, **metadata.get(metadata_key(r), {})})
        key = (c["co"], c["cut"], c["sel"], c["sp"], c["mo"], c["tg"], c["lot"], c["el"])
        # Rule-k row: later batches can re-run a configuration; keep the last.
        if c["k"] == c["ak"]:
            if key in seen_grid:
                grid = [x for x in grid if (x["co"], x["cut"], x["sel"], x["sp"], x["mo"], x["tg"], x["lot"], x["el"]) != key]
            seen_grid.add(key)
            grid.append(c)
        if c["tg"] == "addis" and c["mo"] == "site_heldout" and c["lot"] == "all" and c["el"] == "all":
            sweeps.append({k: c[k] for k in ("co", "cut", "sel", "sp", "k", "ak", "om", "ob", "dm", "db", "r2", "ho", "cv_rmse", "rmsecv", "n_cohort", "n_fit", "n_heldout")})

    # Dedupe sweeps on (config, k), last wins.
    dd = {}
    for s in sweeps:
        dd[(s["co"], s["cut"], s["sel"], s["sp"], s["k"])] = s
    sweeps = sorted(dd.values(), key=lambda s: (s["co"], s["cut"] or 0, s["sel"], s["sp"], s["k"]))

    targets = {}
    for tid, t in registry.items():
        code = t.get("site_code", "")
        targets[tid] = {
            "id": tid, "label": t.get("display_name", tid), "site": t.get("physical_site", tid),
            "code": code, "color": SITE_COLOR_BY_CODE.get(code, "#7b8794"),
            "role": t.get("role"), "provisional": bool(t.get("provisional")),
        }
    for tg in {c["tg"] for c in grid}:
        targets.setdefault(tg, {"id": tg, "label": tg, "site": tg, "code": tg.upper(), "color": "#7b8794", "role": None, "provisional": False})

    docs = []
    for md in sorted(EXPLORER.glob("*_20??-??-??.md")):
        text = md.read_text()
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        m = re.search(r"(\d{4}-\d{2}-\d{2})", md.name)
        docs.append({"file": f"calibration_explorer/{md.name}", "title": title, "date": m.group(1) if m else None, "summary": first_paragraph(text)})
    docs.sort(key=lambda d: d["date"] or "", reverse=True)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "calibration_explorer/cache/batch_results.jsonl",
        "n_source_rows": n_rows,
        "cohorts": cohorts,
        "spectra": spectra_label,
        "modes": {"site_heldout": "A · site-grouped 5-fold, first major minimum",
                  "app": "B · interleaved 10-fold, within 5 %",
                  "app_fmm": "B2 · interleaved 10-fold, first major minimum"},
        "default_cutoff": default_cutoff,
        "mac_value": config.MAC_VALUE,
        "deming_lambda_mac10": 2.96,
        "slope_box": [0.85, 1.18],
        "heldout_floor": 0.85,
        "calibration_rmse_units": "µg/filter",
        "targets": targets,
        "grid": grid,
        "sweeps": sweeps,
        "docs": docs,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    by_target = Counter(c["tg"] for c in grid)
    by_mode = Counter(c["mo"] for c in grid)
    print(f"read {n_rows:,} rows")
    print(f"grid: {len(grid):,} rule-k rows  by target {dict(by_target)}  by mode {dict(by_mode)}")
    print(f"sweeps: {len(sweeps):,} Addis site-held-out rows across {len({(s['co'], s['cut'], s['sel'], s['sp']) for s in sweeps})} configurations")
    print(f"docs: {len(docs)}")
    print("grid metadata coverage:", {field: sum(c[field] is not None for c in grid)
                                      for field in ("cv_rmse", "rmsecv", "n_fit", "n_heldout", "n_cohort")})
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrich-cache", action="store_true",
                        help="recover actual counts and IMPROVE errors from existing content-keyed caches (no refits)")
    parser.add_argument("--staged-metadata", type=Path,
                        help="prepared local TOR pool metadata used as candidate memberships; exact cache hashes must still match")
    args = parser.parse_args()
    main(enrich_cache=args.enrich_cache, staged_metadata=args.staged_metadata)
