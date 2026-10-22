"""Data bundle for the Colab decision workbook (22 Oct 2026).

One folder on Drive holds everything ``decision_workbook_20261022.ipynb`` needs, so the workbook runs on a
Colab VM with nothing but the Drive mount: no repo checkout, no local caches.

Three populations, each in three preprocessings on fixed grids:

* IMPROVE: the 12,393 calibration filters of the frozen rank-70 run (``vibes_current.load_context``), with
  TOR EC/OC/OP, sampled volume, HIPS Fabs, the frozen site split, lot, and non-soil K for the smoke tracer.
  Gravimetric PM2.5 (IMPROVE MF) is a side file, ``improve_mass.csv``, so it can be added to an existing bundle.
* Addis: the 233 evaluation filters of that run, with HIPS Fabs, volume, season and SPARTAN's published EC.
* SPARTAN network: every native PM2.5 analysis baselined on 19 Oct (``network_baselines_20261019.py``),
  repeats averaged per filter, with the public chemistry (EC/OC (FTIR), HIPS, PM2.5, volume) and each
  site's season calendar (``gallery/app/src/siteSeasons.network.json``).

Preprocessings: ``raw`` (4000-500 cm-1, 2,722 channels, what SPARTAN's deployed model uses), ``spline``
(AIRSpec DF1=6, 1425-4000 cm-1, 2,002 channels) and ``vibes`` (rank-70 VIBES, 4000-500 cm-1). Spectra are
float32. Nothing is fitted here.

    ../../.venv/bin/python workflows/build_decision_bundle_20261022.py [--out DIR] [--mass-only]

``--mass-only`` writes just ``improve_mass.csv`` into an existing bundle and adds it to the manifest;
``--adama-only`` likewise adds the Adama TOR pilot (``adama_meta.csv``, ``adama_spline.npy``).

The default output is ``<Drive>/Aethmodular VIBES/decision_workbook_20261022/bundle``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(AREA / "workflows")]
import analog_bias as ab  # noqa: E402
import data_paths  # noqa: E402
import vibes_current as vc  # noqa: E402
from raw_vs_spline_20261020 import BUNDLE_FULL, raw_rows  # noqa: E402
import network_baselines_20261019 as nb  # noqa: E402

KFE = AREA / "output/tables/ksweep_20261014/improve_k_fe.csv"
SEASONS = ROOT / "gallery/app/src/siteSeasons.network.json"
REFERENCE = ROOT / "gallery/app/public/data/spartan_reference.json"
# Ann's urban IMPROVE sites (28 Oct meeting): the nearest US analogue to Addis
URBAN = ["ATLA1", "BIRM1", "FRES1", "PHOE1", "PITT1", "PUSO1"]
NETWORK_FIELDS = {"EC (FTIR)": "ec_ftir_ugm3", "OC (FTIR)": "oc_ftir_ugm3", "HIPS Fabs": "fabs_Mm1",
                  "HIPS BC": "hips_bc_ugm3", "BC (public HIPS)": "public_bc_ugm3", "PM2.5 mass": "pm25_ugm3",
                  "Sampled volume": "volume_m3", "Silicon": "si_ugm3", "Aluminum": "al_ugm3",
                  "Potassium (element)": "k_xrf_ugm3", "Iron": "fe_ugm3", "Sulfate": "so4_ugm3"}


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def improve_and_addis():
    ctx, _ = vc.load_context()
    cases = pd.read_csv(ab.RUN / "cases.csv")
    if not (cases.sample_id.to_numpy()[ctx.lib.row] == ctx.lib.sample_id.to_numpy()).all():
        raise ValueError("frozen cases and the rank-70 library are not aligned")
    wn_full = np.load(BUNDLE_FULL / "data/wn.npy")
    if not np.allclose(ctx.wn_by["VIBES-full-r70"], wn_full):
        raise ValueError("VIBES and raw grids differ")
    X = {"raw": raw_rows(cases, ctx.lib.row, BUNDLE_FULL), "spline": ctx.X["AIRSpec"], "vibes": ctx.X["VIBES-full-r70"]}
    XA = {"raw": raw_rows(cases, ctx.addis.row, BUNDLE_FULL), "spline": ctx.XA["AIRSpec"], "vibes": ctx.XA["VIBES-full-r70"]}

    lib = ctx.lib
    kfe = pd.read_csv(KFE, parse_dates=["date"])
    meta = pd.DataFrame({
        "filter_id": lib.filter_id, "site": lib.Site, "date": lib.date.dt.date, "lot": lib.lot, "split": lib.split,
        "ec_ug": lib.y,                                   # TOR EC µg per filter, the calibration target
        "volume_m3": lib.tor_volume_m3, "tor_ec_ugm3": lib.tor_ec_ugm3, "tor_oc_ugm3": lib.tor_oc_ugm3,
        "tor_ec1_ugm3": lib.tor_ec1_ugm3, "tor_optr_ugm3": lib.tor_optr_ugm3, "fabs_Mm1": lib.fabs_Mm1,
        "locked800": lib.locked800.astype(bool),
    })
    meta = meta.merge(kfe.assign(date=kfe.date.dt.date).rename(columns={"Site": "site", "K_ugm3": "k_xrf_ugm3",
                      "Fe_ugm3": "fe_ugm3", "nsK_ugm3": "nsk_ugm3"}), on=["site", "date"], how="left", validate="many_to_one")
    meta["urban"] = meta.site.isin(URBAN)
    # one lot-251 filter (TONT1 2022-11-19) has a corrupt raw spectrum (~3 absorbance high): raw fits must drop it
    meta["raw_qc_ok"] = np.abs(X["raw"][:, wn_full >= 1425]).max(axis=1) <= 1.5
    if meta.raw_qc_ok.sum() != len(meta) - 1:
        raise ValueError(f"raw QC drops {len(meta) - meta.raw_qc_ok.sum()} filters, expected 1")

    a = ctx.addis
    ameta = pd.DataFrame({
        "filter_id": a.ExternalFilterId.str.replace(r"-\d+$", "", regex=True), "analysis_id": a.ExternalFilterId,
        "media_id": a.MediaId, "date": a.date.dt.date, "season": a.season, "lot": a.LotId,
        "fabs_Mm1": a.Fabs, "volume_m3": a.Volume_m3, "ec_published_ugm3": a.EC_deployed_ugm3,
        "excluded": a.is_excluded.astype(bool),
    })
    return meta, X, ameta, XA, wn_full, np.asarray(ctx.wn_by["AIRSpec"])


def network(wn_full, wn_spline):
    bundle = nb.OUT / "vibes_bundle"
    cases = pd.read_csv(bundle / "cases.csv", dtype={"analysis_id": str})
    if not np.allclose(np.load(bundle / "background_model.npz")["wn"], wn_full):
        raise ValueError("network raw grid differs from the IMPROVE raw grid")
    raw = np.load(bundle / "X_model_grid.npy").astype(float)
    vib = raw - np.load(nb.OUT / "vibes_baselines_model_grid.npy").astype(float)
    air = np.load(nb.OUT / "airspec_baselines.npz")
    air_by = {k.split("_")[0][1:]: (int(k.split("_")[1]), air[k]) for k in air.files}
    native = {str(a["analysis_id"]): (w, y) for a, w, y in nb.native_analyses()}

    pm = cases[cases.filter_type.eq("PM2.5")]
    spl = np.full((len(pm), len(wn_spline)), np.nan)
    for i, c in enumerate(pm.itertuples()):
        if c.analysis_id not in air_by:
            continue
        w, y = native[c.analysis_id]
        i0, b = air_by[c.analysis_id]
        spl[i], _ = nb.resample(w[i0:i0 + len(b)], y[i0:i0 + len(b)] - b, wn_spline)
    keys = pm.filter_key.to_numpy()
    order = pd.unique(keys)
    idx = {k: np.flatnonzero(keys == k) for k in order}
    avg = lambda A: np.stack([np.nanmean(A[r], axis=0) for r in idx.values()]).astype(np.float32)  # noqa: E731
    rows = pm.case_row.to_numpy()
    X = {"raw": avg(raw[rows]), "spline": avg(spl), "vibes": avg(vib[rows])}

    first = pm.groupby("filter_key").first()
    meta = pd.DataFrame({"filter_id": order, "code": first.loc[order, "code"].to_numpy(),
                         "date": first.loc[order, "date"].to_numpy(), "analyses": [len(r) for r in idx.values()]})
    ref = {r["id"]: r for r in json.loads(REFERENCE.read_text())["rows"]}
    for src, col in NETWORK_FIELDS.items():
        meta[col] = [ref.get(k, {}).get(src, np.nan) for k in order]
    meta["excluded"] = [bool(ref.get(k, {}).get("excluded", False)) for k in order]
    for p in X:
        meta[f"{p}_ok"] = np.isfinite(X[p]).all(axis=1)
    return meta, X


def improve_mass(meta: pd.DataFrame) -> pd.DataFrame:
    """Gravimetric PM2.5 (IMPROVE MF, µg/m³) for each library filter: validated rows, lowest POC per site and day."""
    import pls_ksweep as pk
    years = sorted(pd.to_datetime(meta.date).dt.year.unique())
    raw = pd.concat([pk._read_year(pk.IMPROVE_RAW / f"IMPAER_{y}.txt.zip", {"MF"}) for y in years], ignore_index=True)
    raw = raw.loc[raw.Status.astype(str).str.startswith("V") & raw.FactValue.gt(-900)]
    raw = raw.sort_values("POC").drop_duplicates(["SiteCode", "FactDate"])
    mf = pd.DataFrame({"site": raw.SiteCode, "date": pd.to_datetime(raw.FactDate).dt.date, "pm25_ugm3": raw.FactValue})
    keys = pd.DataFrame({"filter_id": meta.filter_id, "site": meta.site, "date": pd.to_datetime(meta.date).dt.date})
    return keys.merge(mf, on=["site", "date"], how="left", validate="many_to_one")[["filter_id", "pm25_ugm3"]]


ADAMA = AREA / "output/tables/adama_bishoftu_frozen_pilot"


def adama():
    """The five Adama quartz/PTFE date-candidate pairs (CSU AMOD sampler, lower flow): AIRSpec spectra on the
    spline grid (from run_adama_bishoftu_frozen_pilot.py, same pipeline as the bundle's spline) and their TOR EC."""
    z = np.load(ADAMA / "external_corrected.npz", allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    pairs = pd.read_csv(ADAMA / "adama_candidate_pairs.csv")
    pairs["sample_id"] = "adama:" + pairs.SampleAnalysisId_provisional.astype(str)
    rows = [ids.index(s) for s in pairs.sample_id]
    meta = pd.DataFrame({"sample_id": pairs.sample_id, "date": pairs.date, "ptfe_id": pairs.FilterId_ptfe, "quartz_id": pairs.FilterId_quartz,
                         "volume_m3": pairs.Volume_m3, "volume_ratio": pairs.volume_ratio, "flag": pairs.flag.fillna(""),
                         "ec_tr_ugm3": pairs.ECTR, "ec_tt_ugm3": pairs.ECTT, "airspec_frozen_ugm3": pairs.AIRSpec_EC_ugm3,
                         "vibes_frozen_ugm3": pairs.VIBES_EC_ugm3, "pairing_status": pairs.pairing_status})
    # the quartz side from the lab file: TOR OC, quartz volume, deposit area and the sampler's position
    sys.path.insert(0, str(ROOT / "research/ftir_ec_phase3/scripts"))
    from theory_test_suite import davis_root
    carbon = pd.read_csv(davis_root() / "DAVIS/Adama TOR/Carbon_concs_Batch54.csv")
    wide = carbon.pivot_table(index="FilterId", columns="Parameter", values="Concentration_ug_m3", aggfunc="first")
    first = carbon.groupby("FilterId").first()
    q = pd.DataFrame({"quartz_id": wide.index, "oc_tr_ugm3": wide.OCTR.to_numpy(), "ec_tr_quartz_ugm3": wide.ECTR.to_numpy(),
                      "quartz_volume_m3": first.Volume_liters.reindex(wide.index).to_numpy() / 1000,
                      "deposit_area_cm2": first.DepositArea_cm2.reindex(wide.index).to_numpy(),
                      "lat": first.Latitude.reindex(wide.index).to_numpy(), "lon": first.Longitude.reindex(wide.index).to_numpy(),
                      "device": first.DeviceId.reindex(wide.index).to_numpy()})
    meta = meta.merge(q, on="quartz_id", how="left", validate="one_to_one")
    if not np.allclose(meta.ec_tr_quartz_ugm3, meta.ec_tr_ugm3, rtol=1e-6):
        raise ValueError("Adama ECTR differs between the pairing table and the lab carbon file")
    return meta.drop(columns="ec_tr_quartz_ugm3"), np.asarray(z["AIRSpec"])[rows]


def seasons():
    s = json.loads(SEASONS.read_text())
    out = [{"code": code, "site_name": v["site"], "month": m, "season": z["name"]}
           for code, v in s.items() for z in v["seasons"] for m in z["months"]]
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=data_paths.drive_root() / "Aethmodular VIBES/decision_workbook_20261022/bundle")
    p.add_argument("--mass-only", action="store_true", help="add improve_mass.csv to an existing bundle")
    p.add_argument("--adama-only", action="store_true", help="add the Adama TOR pilot to an existing bundle")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.mass_only:
        manifest = json.loads((out / "MANIFEST.json").read_text())
        mass = improve_mass(pd.read_csv(out / "improve_meta.csv"))
        mass.to_csv(out / "improve_mass.csv", index=False)
        manifest["files"]["improve_mass.csv"] = {"shape": list(mass.shape), "sha256": sha(out / "improve_mass.csv")}
        manifest["sources"]["improve_mass"] = "IMPROVE MF from research/ftir_hips_chem/output/tables/improve_validated_archive"
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"improve_mass.csv: {mass.pm25_ugm3.notna().sum():,} of {len(mass):,} filters have gravimetric PM2.5")
        return
    if args.adama_only:
        manifest = json.loads((out / "MANIFEST.json").read_text())
        meta, XS = adama()
        if XS.shape[1] != len(np.load(out / "wn_spline.npy")):
            raise ValueError("Adama spectra are not on the bundle's spline grid")
        meta.to_csv(out / "adama_meta.csv", index=False)
        np.save(out / "adama_spline.npy", XS.astype(np.float32))
        for k, shape in (("adama_meta.csv", list(meta.shape)), ("adama_spline.npy", list(XS.shape))):
            manifest["files"][k] = {"shape": shape, "sha256": sha(out / k)}
        manifest["sources"]["adama"] = str(ADAMA.relative_to(ROOT))
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"adama: {len(meta)} pairs")
        return

    meta, X, ameta, XA, wn_full, wn_spline = improve_and_addis()
    nmeta, NX = network(wn_full, wn_spline)
    files = {}

    def save_npy(name, arr):
        np.save(out / name, np.asarray(arr, np.float32))
        files[name] = list(np.shape(arr))

    def save_csv(name, df):
        df.to_csv(out / name, index=False)
        files[name] = list(df.shape)

    save_npy("wn_full.npy", wn_full)
    save_npy("wn_spline.npy", wn_spline)
    for pop, arrays in (("improve", X), ("addis", XA), ("network", NX)):
        for prep, arr in arrays.items():
            save_npy(f"{pop}_{prep}.npy", arr)
    save_csv("improve_meta.csv", meta)
    save_csv("improve_mass.csv", improve_mass(meta))
    save_csv("addis_meta.csv", ameta)
    save_csv("network_meta.csv", nmeta)
    save_csv("site_seasons.csv", seasons())

    manifest = {"created": str(date.today()), "builder": "research/ftir_hips_chem/workflows/build_decision_bundle_20261022.py",
                "grids": {"raw": "wn_full.npy", "vibes": "wn_full.npy", "spline": "wn_spline.npy"},
                "units": {"ec_ug": "TOR EC, µg per filter", "*_ugm3": "µg/m³", "fabs_Mm1": "HIPS absorption, Mm⁻¹",
                          "spectra": "absorbance (raw) or baseline-corrected absorbance"},
                "files": {k: {"shape": v, "sha256": sha(out / k)} for k, v in files.items()},
                "sources": {"rank70_context": "research/ftir_hips_chem/scripts/vibes_current.py load_context()",
                            "raw_spectra": str(BUNDLE_FULL.relative_to(ROOT)),
                            "network_baselines": str(nb.OUT.relative_to(ROOT)),
                            "network_chemistry": str(REFERENCE.relative_to(ROOT)),
                            "site_seasons": str(SEASONS.relative_to(ROOT)),
                            "improve_k_fe": str(KFE.relative_to(ROOT)),
                            "improve_mass": "IMPROVE MF from research/ftir_hips_chem/output/tables/improve_validated_archive"}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum((out / k).stat().st_size for k in files) / 1e6
    print(f"{out}\n  IMPROVE {len(meta)} · Addis {len(ameta)} · network {len(nmeta)} filters at {nmeta.code.nunique()} sites · {total:.0f} MB")


if __name__ == "__main__":
    main()
