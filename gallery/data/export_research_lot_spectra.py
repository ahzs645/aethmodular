"""Export Research → PTFE lot comparison: per-filter FTIR spectra with their PTFE lot, for any site.

    AETHMODULAR_MAIA_DATA_ROOT=<Davis Data> uv run --locked --no-sync python gallery/data/export_research_lot_spectra.py

char_12 §4 (research/ftir_etad_char/scripts/run_char_12.py) compares the two ETAD PTFE lots (248, 251): dry-season
filters by lot in the aliphatic C–H region of the second derivative (04_lot_effect, left), the envelope-escape score
by lot × season (04, right) and the lots in the AIRSpec-baselined, area-normalised view, all filters against the dry
season only (04b). It draws only Addis. This export ships each filter's spectrum in three views so the browser can
draw the same comparison for any site or sites in view, with the reader's seasons, exclusions and dates:

* raw    raw absorbance as measured, means of 6 channels (~7.7 cm⁻¹), 500–3998 cm⁻¹ (char_12 §1);
* deriv  char_12's processing: raw resampled to its 2 cm⁻¹ grid, Savitzky–Golay 2nd derivative (13 points,
         order 2), kept at full resolution in its two pertinent regions, 2800–3000 and 1450–1850 cm⁻¹ (§2, §4);
* airspec the AIRSpec baseline (research/ftir_ec_phase3/scripts/airspec_baseline.py, the Python port of the APRLssb
         segmented spline, df1 = 6, df2 = 4) on the ETAD grid, clipped to ≥ 0 and scaled to unit positive area
         (char_10's convention, char_12 §4b), means of 3 channels (~3.9 cm⁻¹), 1425–3998 cm⁻¹.

Each view is a binary file: float32 [F] per-filter scale, then int16 [F × P] (as export_similarity.py stores its
traces). Each filter also carries char_12's envelope-escape score, computed here exactly as char_12 §3 does: the
share of the pertinent wavenumbers where its 2nd derivative leaves the IMPROVE 5–95 % envelope of the 916 IMPROVE
spectra in ftir-spectra-2026-06-24.csv. char_12 saves no per-filter table, so this is a recomputation; `check`
holds char_12's lot × season numbers rebuilt from it on char_12's own 239-filter cohort, beside its saved tables.

Sources (Davis Data, through AETHMODULAR_MAIA_DATA_ROOT; the IMPROVE run through AETHMODULAR_FTIR_SPECTRA_DIR):
Addis from DAVIS/ETAD FTIR (ETAD_FTIR_spectra.csv, ETAD_metadata.csv, etad_spectra_lotmap.csv; replicate scans
averaged per MediaId as phase3_common.load_addis_evaluation does); every other site from DAVIS/SPARTAN FTIR pulls/
<SITE> (the scan blobs decoded with build_spartan_target.decode_scans, absorbance = −log10(sample/background) on
the ETAD grid; PM2.5 and PM10 sample filters, QC-1 analyses where a filter has one, lot from ExternalLotId). The AIRSpec port
is checked against the R output for Addis (DAVIS/ETAD FTIR/baseline_plots_AIRSPEC/spectra_baselined_AIRSPEC.csv).

Writes gallery/app/public/data/research/lot_spectra/{lot_spectra.json, raw.bin, deriv.bin, airspec.bin}.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
for _p in ("research/ftir_hips_chem/scripts", "research/ftir_ec_phase3/scripts", "research/ftir_etad_char/scripts"):
    sys.path.insert(0, str(ROOT / _p))
from airspec_baseline import SEG1, SEG2, airspec_baseline_matrix, make_mask  # noqa: E402
from build_spartan_target import decode_scans  # noqa: E402
from charcoal_spectra import BANDS, band_area, resample  # noqa: E402
from config import season_for_month  # noqa: E402
from data_paths import ftir_spectra_dir, maia_data_root  # noqa: E402

OUT = ROOT / "gallery/app/public/data/research/lot_spectra"
CHAR = ROOT / "research/ftir_etad_char"
TABLES = CHAR / "output/tables/char12"
NB = "research/ftir_etad_char/char_12_weakley_transferability.ipynb"

ETAD_REL = Path("DAVIS/ETAD FTIR")
PULLS_REL = Path("DAVIS/SPARTAN FTIR pulls")
HIPS_REL = Path("Spartan/SPARTAN_HIPS_Batch1-51.v2.csv")
IMPROVE_NAME = "ftir-spectra-2026-06-24.csv"
SITES = ("ETAD", "CHTS", "INDH", "USPA", "ETBI")

# char_12's processing (run_char_12.py): the 2 cm⁻¹ grid, SG window and the two pertinent regions
SG_WINDOW, SG_ORDER = 13, 2
REGION_CH = (2800.0, 3000.0)
REGION_CO = (1450.0, 1850.0)
RAW_BIN, AIRSPEC_BIN = 6, 3
TYPES = ("PM2.5", "PM10")  # sample filters; field blanks (FB) left out


def improve_path() -> Path:
    base = ftir_spectra_dir()
    for cand in (base / IMPROVE_NAME, base / "runs/2026-06-24" / IMPROVE_NAME):
        if cand.exists():
            return cand
    raise SystemExit(f"{IMPROVE_NAME} not found under the FTIR spectra folder (set AETHMODULAR_FTIR_SPECTRA_DIR)")


def gallery_id(external: str) -> str:
    """ETAD-0017-1 → ETAD-0017: the gallery's row id for a SPARTAN ExternalFilterId (scan position dropped)."""
    parts = str(external).strip().split("-")
    return "-".join(parts[:2]) if len(parts) >= 3 else str(external).strip()


def read_pull(site: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(maia_data_root() / PULLS_REL / site / f"{site}_{kind}.csv", encoding="utf-8-sig")
    df.columns = [c.strip('﻿"') for c in df.columns]
    return df


def load_etad(wcols: list[str]):
    """Addis: one spectrum per MediaId (replicates averaged), its lot, date and char_12's cohort flag; and the
    replicate scans with the row each belongs to (the AIRSpec baseline is fitted per scan, then averaged, as the
    phase-3 cache char_12 reads was built)."""
    folder = maia_data_root() / ETAD_REL
    raw = pd.read_csv(folder / "ETAD_FTIR_spectra.csv")
    spectra = raw.groupby("MediaId", as_index=False)[wcols].mean()
    n_scans = raw.groupby("MediaId").size().rename("analyses").reset_index()
    meta = pd.read_csv(folder / "ETAD_metadata.csv")
    lots = pd.read_csv(folder / "etad_spectra_lotmap.csv")[["MediaId", "LotId"]].drop_duplicates("MediaId")
    hips = pd.read_csv(maia_data_root() / HIPS_REL, encoding="cp1252", usecols=["Site", "FilterId", "Fabs"])
    hips = hips[hips["Site"].eq("ETAD")].drop_duplicates("FilterId").rename(columns={"FilterId": "ExternalFilterId"})
    df = (spectra.merge(meta, on="MediaId", how="left", validate="one_to_one")
          .merge(lots, on="MediaId", how="left").merge(n_scans, on="MediaId", how="left")
          .merge(hips[["ExternalFilterId", "Fabs"]], on="ExternalFilterId", how="left", validate="many_to_one"))
    # sample filters (char_12's cohort includes the six PM10 filters; the field blanks never have a Fabs)
    df = df[df["ExternalFilterType"].isin(TYPES) & df[wcols].notna().all(axis=1)].reset_index(drop=True)
    # char_12's 239: phase3_common.load_addis_evaluation keeps complete spectra with a HIPS Fabs and a positive volume
    df["char12"] = df["Fabs"].notna() & df["SampleVolume_m3"].gt(0)
    out = pd.DataFrame({
        "filter_id": df["ExternalFilterId"].astype(str), "code": "ETAD", "type": df["ExternalFilterType"],
        "date": pd.to_datetime(df["SamplingStartDate"]).dt.strftime("%Y-%m-%d"),
        "lot": pd.to_numeric(df["LotId"], errors="coerce"), "char12": df["char12"],
        "instrument": None, "qc": None, "analyses": df["analyses"].astype(int),
    })
    row = pd.Series(np.arange(len(df)), index=df["MediaId"])
    reps = raw[raw["MediaId"].isin(row.index)]
    return out, df[wcols].to_numpy(float), (reps[wcols].to_numpy(float), row.loc[reps["MediaId"]].to_numpy())


def load_spartan(site: str, wcols: list[str], wn: np.ndarray):
    """A SPARTAN pull: sample filters, decoded with the repo's decode_scans, replicate analyses averaged per filter
    (and kept, with their row, for the per-scan AIRSpec baseline)."""
    if not (maia_data_root() / PULLS_REL / site).is_dir():
        return None
    filters, analyses, scans = read_pull(site, "filters"), read_pull(site, "ftir_analysis"), read_pull(site, "scans_base64")
    filters = filters[filters["ExternalFilterType"].isin(TYPES)]
    analyses = analyses[analyses["MediaId"].isin(filters["MediaId"])].copy()
    # QC-1 analyses where a filter has one; otherwise whatever it has (its codes are kept)
    has_ok = analyses.groupby("MediaId")["AnalysisQcCode"].transform(lambda q: (q == 1).any())
    analyses = analyses[~has_ok | analyses["AnalysisQcCode"].eq(1)].copy()
    analyses["rep"] = analyses.groupby("MediaId").cumcount()
    # decode_scans keeps one spectrum per MediaId, so call it once per replicate rank and average
    parts = [decode_scans(scans, analyses[analyses["rep"] == k], wcols, wn) for k in sorted(analyses["rep"].unique())]
    spectra = pd.concat(parts).groupby(level=0).mean()
    inst = scans.set_index("Id")["InstrumentSerialNumber"]
    info = analyses.assign(inst=analyses["SampleScanId"].map(inst)).groupby("MediaId").agg(
        instrument=("inst", lambda s: "/".join(sorted({str(v) for v in s.dropna()})) or None),
        qc=("AnalysisQcCode", lambda s: "/".join(sorted({str(int(v)) for v in s}))),
        analyses=("AnalysisQcCode", "size"))
    f = filters.set_index("MediaId").loc[spectra.index]
    out = pd.DataFrame({
        "filter_id": f["ExternalFilterId"].astype(str).to_numpy(), "code": site, "type": f["ExternalFilterType"].to_numpy(),
        "date": pd.to_datetime(f["SamplingStartDate"], format="%m/%d/%Y %I:%M:%S %p").dt.strftime("%Y-%m-%d").to_numpy(),
        "lot": pd.to_numeric(f["ExternalLotId"], errors="coerce").to_numpy(), "char12": False,
        "instrument": info.loc[spectra.index, "instrument"].to_numpy(),
        "qc": info.loc[spectra.index, "qc"].to_numpy(), "analyses": info.loc[spectra.index, "analyses"].to_numpy(),
    })
    X = spectra[wcols].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    row = pd.Series(np.arange(ok.sum()), index=spectra.index[ok])
    reps = pd.concat(parts)
    reps = reps[reps.index.isin(row.index)]
    return out[ok].reset_index(drop=True), X[ok], (reps[wcols].to_numpy(float), row.loc[reps.index].to_numpy())


def site_names() -> dict:
    out = {}
    for site in SITES[1:]:
        path = maia_data_root() / PULLS_REL / site / f"{site}_site.csv"
        if path.exists():
            d = read_pull(site, "site")["Description"].dropna()
            if len(d):
                out[site] = str(d.iloc[0]).split(",")[0].strip()
    return out


def bin_means(X: np.ndarray, wn: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    bins = [np.arange(i, min(i + width, len(wn))) for i in range(0, len(wn), width)]
    return np.stack([X[:, b].mean(axis=1) for b in bins], axis=1), np.array([wn[b].mean() for b in bins])


def write_view(name: str, X: np.ndarray) -> dict:
    """float32 [F] per-filter scale, then int16 [F × P]."""
    scale = np.abs(X).max(axis=1) / 32767
    scale[~(scale > 0)] = 1
    q = np.round(X / scale[:, None]).astype("<i2")
    path = OUT / f"{name}.bin"
    with path.open("wb") as out:
        out.write(scale.astype("<f4").tobytes())
        out.write(q.tobytes())
    back = q * scale.astype("<f4")[:, None]
    return {"file": path.name, "points": int(X.shape[1]), "bytes": path.stat().st_size,
            "max_abs_error": float(np.max(np.abs(back - X)))}


def sig(values, digits=5) -> list:
    return [float(f"{v:.{digits}g}") for v in np.asarray(values, float)]


def mw(a, b) -> float | None:
    return float(stats.mannwhitneyu(a, b).pvalue) if len(a) and len(b) else None


def char12_check(meta: pd.DataFrame, esc: np.ndarray, esc_imp: np.ndarray) -> dict:
    """char_12's §3/§4 numbers rebuilt on its 239-filter Addis cohort, beside the tables it saved."""
    c = meta["char12"].to_numpy()
    e, lot = esc[c], meta.loc[c, "lot"].to_numpy()
    season = meta.loc[c, "date"].str.slice(5, 7).astype(int).map(season_for_month).to_numpy()
    dry, has = season == "Dry (Oct-Feb)", np.isfinite(lot)
    rows = []
    for label, base in [("all filters", has), ("dry season only", has & dry)]:
        a, b = e[base & (lot == 248)], e[base & (lot == 251)]
        rows.append({"subset": label, "n248": int(len(a)), "n251": int(len(b)), "median248": float(np.median(a)),
                     "median251": float(np.median(b)), "p": mw(a, b)})
    groups = [{"group": f"lot {int(l)} {d}", "n": int(m.sum()), "median": float(np.median(e[m])),
               "q1": float(np.percentile(e[m], 25)), "q3": float(np.percentile(e[m], 75))}
              for l in (248, 251) for d, dm in (("dry", dry), ("non-dry", ~dry)) for m in [has & (lot == l) & dm] if m.sum()]
    saved = {}
    if (TABLES / "lot_effect.csv").exists():
        t = pd.read_csv(TABLES / "lot_effect.csv")
        saved["lot_effect"] = [{"subset": r["subset"], "n248": int(r["n lot 248"]), "n251": int(r["n lot 251"]),
                                "median248": float(r["median escape 248"]), "median251": float(r["median escape 251"]),
                                "p": float(r["Mann–Whitney p"])} for _, r in t.iterrows()]
    if (TABLES / "escape_by_group.csv").exists():
        t = pd.read_csv(TABLES / "escape_by_group.csv", index_col=0)
        saved["escape_by_group"] = [{"group": k, "n": int(r["n"]), "median": float(r["median escape"])} for k, r in t.iterrows()]
    if (TABLES / "lot_bands_within_dry.csv").exists():
        t = pd.read_csv(TABLES / "lot_bands_within_dry.csv")
        saved["lot_bands"] = [{"band": r["band"], "median248": float(r["lot 248 median (×10⁻⁴)"]), "median251": float(r["lot 251 median (×10⁻⁴)"]),
                               "p": float(r["Mann–Whitney p (within dry)"])} for _, r in t.iterrows()]
    by_season = {s: e[season == s] for s in ("Dry (Oct-Feb)", "Belg (Mar-May)", "Kiremt (Jun-Sep)")}
    return {"cohort": int(c.sum()), "lot_effect": rows, "lot_season": groups,
            "escape": [{"group": "IMPROVE (null ≈ 0.10)", "n": int(len(esc_imp)), "median": float(np.median(esc_imp))},
                       {"group": "ETAD — all", "n": int(len(e)), "median": float(np.median(e))},
                       *[{"group": f"ETAD — {k}", "n": int(len(v)), "median": float(np.median(v))} for k, v in by_season.items()]],
            "saved": saved, "saved_from": "research/ftir_etad_char/output/tables/char12/{lot_effect,escape_by_group,lot_bands_within_dry}.csv"}


def airspec_check(meta: pd.DataFrame, corrected: np.ndarray, wn_a: np.ndarray) -> dict | None:
    """The Python AIRSpec port against the R output for Addis (per MediaId, replicate scans averaged)."""
    path = maia_data_root() / ETAD_REL / "baseline_plots_AIRSPEC/spectra_baselined_AIRSPEC.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path)
    cols = [c for c in r.columns if c != "id"]
    wr = np.array([float(c) for c in cols])
    lm = pd.read_csv(maia_data_root() / ETAD_REL / "etad_spectra_lotmap.csv")
    r = r.merge(lm[["SampleAnalysisId", "ExternalFilterId"]], left_on="id", right_on="SampleAnalysisId")
    by = r.groupby("ExternalFilterId")[cols].mean()
    etad = meta.index[meta["code"].eq("ETAD")]
    ids = meta.loc[etad, "filter_id"].to_numpy()
    keep = np.isin(ids, by.index)
    R = by.loc[ids[keep]].to_numpy(float)
    R = resample(R, wr, wn_a)
    P = corrected[etad[keep]]
    d = np.abs(P - R)
    span = np.nanmax(np.abs(R), axis=1)
    return {"filters": int(keep.sum()), "median_abs_diff": float(np.nanmedian(d)), "p99_abs_diff": float(np.nanpercentile(d, 99)),
            "median_rel_to_peak": float(np.nanmedian(np.nanmax(d, axis=1) / span))}


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # the ETAD grid (the IMPROVE app grid too): every site is decoded onto it
    head = pd.read_csv(maia_data_root() / ETAD_REL / "ETAD_FTIR_spectra.csv", nrows=0)
    wcols = sorted([c for c in head.columns if c not in ("SampleAnalysisId", "MediaId")], key=lambda v: -float(v))
    wn = np.array([float(c) for c in wcols])

    metas, mats, scans, owners, offset = [], [], [], [], 0
    for got in [load_etad(wcols), *(load_spartan(site, wcols, wn) for site in SITES[1:])]:
        if got is None:
            continue
        metas.append(got[0]), mats.append(got[1]), scans.append(got[2][0]), owners.append(got[2][1] + offset)
        offset += len(got[0])
    meta = pd.concat(metas, ignore_index=True)
    X, S, owner = np.vstack(mats), np.vstack(scans), np.concatenate(owners)
    # a filter with no sampling date has no season and no gallery row (none is in char_12's cohort)
    dated = meta["date"].notna().to_numpy()
    new_row = np.cumsum(dated) - 1
    keep = dated[owner]
    meta, X, S, owner = meta[dated].reset_index(drop=True), X[dated], S[keep], new_row[owner[keep]]
    meta["id"] = meta["filter_id"].map(gallery_id)
    dup = meta["id"].duplicated(keep=False)
    if dup.any():
        raise SystemExit(f"two spectra for one gallery id: {sorted(meta.loc[dup, 'id'])[:6]}")
    print(meta.groupby("code").agg(n=("id", "size"), lots=("lot", lambda s: s.value_counts(dropna=False).to_dict())).to_string())

    # ---------------------------------------------------------------- char_12's 2nd derivative and escape score
    imp = pd.read_csv(improve_path())
    icols = [c for c in imp.columns if c not in ("AnalysisId", "FilterId", "SampleDate", "Site")]
    wn_imp = np.array([float(c) for c in icols])
    lo = np.ceil(max(wn_imp.min(), wn.min())) + 2.0
    hi = np.floor(min(wn_imp.max(), wn.max())) - 2.0
    grid = np.arange(lo, hi + 1e-9, 2.0)
    X_imp = resample(imp[icols].to_numpy(float), wn_imp, grid)
    X_imp = X_imp[np.isfinite(X_imp).all(axis=1)]
    D_imp = savgol_filter(X_imp, SG_WINDOW, SG_ORDER, deriv=2, axis=1)
    D = savgol_filter(resample(X, wn, grid), SG_WINDOW, SG_ORDER, deriv=2, axis=1)
    pert = ((grid >= REGION_CH[0]) & (grid <= REGION_CH[1])) | ((grid >= REGION_CO[0]) & (grid <= REGION_CO[1]))
    env_lo, env_hi = np.percentile(D_imp[:, pert], [5, 95], axis=0)
    escape = lambda Dm: ((Dm[:, pert] < env_lo) | (Dm[:, pert] > env_hi)).mean(axis=1)  # noqa: E731
    esc, esc_imp = escape(D), escape(D_imp)

    # ---------------------------------------------------------------- AIRSpec, clipped, unit positive area
    # fitted per scan, then averaged per filter (etad_corrected_df6.npz holds one row per scan; load_etad averages)
    _, corr = airspec_baseline_matrix(wn, S, df1=6, df2=4, n_jobs=4, chunksize=8)
    analysed = make_mask(wn, SEG1) | make_mask(wn, SEG2)
    wn_a = wn[analysed]
    C = pd.DataFrame(corr[:, analysed]).groupby(owner).mean().sort_index().to_numpy()
    assert C.shape[0] == len(meta)
    pos = np.clip(C, 0, None)
    order = np.argsort(wn_a)
    area = np.trapezoid(pos[:, order], wn_a[order], axis=1)[:, None]
    area[area == 0] = np.nan
    shape = pos / np.abs(area)

    # ---------------------------------------------------------------- the three views
    raw_b, raw_wn = bin_means(X, wn, RAW_BIN)
    air_b, air_wn = bin_means(shape, wn_a, AIRSPEC_BIN)
    views = {
        "raw": {"label": "Raw absorbance", "unit": "absorbance (AU)", "wn": [round(float(w), 2) for w in raw_wn],
                "binning": f"means of {RAW_BIN} channels (~{RAW_BIN * 1.286:.1f} cm⁻¹)", **write_view("raw", raw_b)},
        "deriv": {"label": "2nd derivative (char_12)", "unit": "2nd-derivative absorbance (arb.)", "wn": [float(w) for w in grid[pert]],
                  "binning": "char_12's 2 cm⁻¹ grid, unbinned; only its two pertinent regions", **write_view("deriv", D[:, pert])},
        "airspec": {"label": "AIRSpec-baselined, area-normalised", "unit": "area-normalised absorbance", "wn": [round(float(w), 2) for w in air_wn],
                    "binning": f"means of {AIRSPEC_BIN} channels (~{AIRSPEC_BIN * 1.286:.1f} cm⁻¹)", **write_view("airspec", air_b)},
    }

    check = char12_check(meta, esc, esc_imp)
    check["airspec_vs_r"] = airspec_check(meta, C, wn_a)
    lot = meta["lot"]
    bundle = {
        "schema_version": 1,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_from": "gallery/data/export_research_lot_spectra.py",
        "notebook": NB,
        "inputs": [f"Davis Data/{ETAD_REL}/ETAD_FTIR_spectra.csv, ETAD_metadata.csv, etad_spectra_lotmap.csv",
                   f"Davis Data/{PULLS_REL}/<SITE>/<SITE>_filters.csv, _ftir_analysis.csv, _scans_base64.csv ({', '.join(SITES[1:])})",
                   f"Davis Data/{HIPS_REL} (char_12's cohort: a HIPS Fabs)",
                   f"Davis Data/FTIR/runs/2026-06-24/{IMPROVE_NAME} (the IMPROVE envelope)"],
        "filters": {
            "id": meta["id"].tolist(), "filter_id": meta["filter_id"].tolist(), "code": meta["code"].tolist(),
            "type": meta["type"].tolist(), "date": meta["date"].tolist(), "lot": [None if pd.isna(v) else int(v) for v in lot],
            "char12": [int(bool(v)) for v in meta["char12"]], "escape": [round(float(v), 6) for v in esc],
            "instrument": [None if v is None or pd.isna(v) else str(v) for v in meta["instrument"]],
            "qc": [None if v is None or pd.isna(v) else str(v) for v in meta["qc"]],
            "analyses": [int(v) for v in meta["analyses"]],
        },
        # the pulls' own site names, for sites the gallery has no metadata for (ETBI_site.csv: "Bishoftu, Ethiopia")
        "site_names": site_names(),
        "views": views,
        "regions": {"ch": list(REGION_CH), "co": list(REGION_CO)},
        "bands": {k: list(v) for k, v in BANDS.items() if v[0] >= wn_a.min()},
        # char_12 §4b's band table: each filter's mean area-normalised value in each band (×10⁻⁴), unbinned
        "band_means": {k: sig(band_area(shape, wn_a, v) * 1e4, 6) for k, v in BANDS.items() if v[0] >= wn_a.min()},
        "improve": {"n": int(len(X_imp)), "deriv_p05": sig(env_lo), "deriv_p95": sig(env_hi),
                    "deriv_mean": sig(D_imp[:, pert].mean(axis=0)), "escape_median": float(np.median(esc_imp)),
                    "escape_p90": float(np.percentile(esc_imp, 90))},
        "check": check,
    }
    (OUT / "lot_spectra.json").write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {len(meta)} filters to {OUT.relative_to(ROOT)} ({total / 1e6:.2f} MB)")
    print(json.dumps({k: check[k] for k in ("cohort", "lot_effect", "airspec_vs_r")}, indent=1))
    print(json.dumps(check["saved"], indent=1))


if __name__ == "__main__":
    main()
