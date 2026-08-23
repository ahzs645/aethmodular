"""Turn a SPARTAN site's raw SQL export into a calibration_explorer target.

Generalized from the ETBI build (2026-08-22). Takes the three CSVs written by
`get_spartan_spectra.ps1` (<SITE>_filters.csv, <SITE>_ftir_analysis.csv,
<SITE>_scans_base64.csv), decodes the FTIR scan blobs into absorbance on the
app's IMPROVE wavenumber grid, joins the SPARTAN HIPS Fabs reference, and
writes calibration_explorer/targets/<name>/{spectra,reference,spectra_corrected}.csv
so the site appears in the explorer's "Evaluate on" dropdown.

Usage:
    python build_spartan_target.py INDH --src ~/Downloads --name delhi
    python build_spartan_target.py ETBI --src ~/Downloads/etbi_site --name etbi

Blob format (verified on ETBI, mirrors IMPROVE's ftir.Scan): little-endian
float32 single-beam intensity, DataPointFormat 1, YScalingFactor 1, N points
spanning FrequencyOfFirstPoint -> FrequencyOfLastPoint (descending, ~4003->416).
Absorbance = -log10(sample / background) using each analysis's own background
scan, interpolated onto the app grid.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
sys.path.insert(0, str(HERE))
from pls_transfer import FTIRTransferPaths  # noqa: E402

PATHS = FTIRTransferPaths.defaults()
TARGETS = REPO / "calibration_explorer/targets"

SEASON = {  # Ethiopian seasons; harmless elsewhere (rename via --season-scheme)
    **{m: "Dry (Oct-Feb)" for m in (10, 11, 12, 1, 2)},
    **{m: "Belg (Mar-May)" for m in (3, 4, 5)},
    **{m: "Kiremt (Jun-Sep)" for m in (6, 7, 8, 9)},
}
QUARTER = {m: f"Q{(m - 1) // 3 + 1}" for m in range(1, 13)}


def app_grid() -> tuple[list[str], np.ndarray]:
    """The explorer's raw wavenumber columns, as strings and floats."""
    hdr = pd.read_csv(PATHS.ftir_dir / "local_db/spectra_248_251.csv", nrows=0)

    def numeric(c: str) -> bool:
        try:
            float(c)
        except ValueError:
            return False
        return True

    wcols = [c for c in hdr.columns if numeric(c)]
    return wcols, np.array([float(c) for c in wcols])


def decode_scans(scans: pd.DataFrame, analyses: pd.DataFrame,
                 wcols: list[str], wn_app: np.ndarray) -> pd.DataFrame:
    """One absorbance spectrum per MediaId, on the app grid."""
    sc = scans.set_index("Id")

    def one(scan_id):
        r = sc.loc[scan_id]
        v = np.frombuffer(base64.b64decode(r["Data"]), dtype="<f4").astype(float)
        wn = np.linspace(r["FrequencyOfFirstPoint"], r["FrequencyOfLastPoint"],
                         int(r["NumberOfDataPoints"]))
        return wn, v

    rows: dict[int, np.ndarray] = {}
    for _, a in analyses.iterrows():
        try:
            wn_s, S = one(a["SampleScanId"])
            wn_b, B = one(a["BackgroundScanId"])
        except KeyError:                      # scan row absent from the export
            continue
        if len(S) != len(B) or not np.allclose(wn_s, wn_b):
            o = np.argsort(wn_b)
            B = np.interp(wn_s, wn_b[o], B[o])
        A = -np.log10(np.clip(S, 1e-9, None) / np.clip(B, 1e-9, None))
        o = np.argsort(wn_s)
        rows[int(a["MediaId"])] = np.interp(wn_app, wn_s[o], A[o])
    out = pd.DataFrame.from_dict(rows, orient="index", columns=wcols)
    out.index.name = "MediaId"
    return out


def hips_reference(site: str) -> pd.DataFrame:
    h = pd.read_csv(PATHS.spartan_hips_primary, encoding="cp1252",
                    usecols=["Site", "FilterId", "LotId", "Fabs", "Volume"])
    return (h[h["Site"].eq(site)].drop_duplicates("FilterId")
            .rename(columns={"FilterId": "ExternalFilterId"}))


def corrected(spec: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """AIRSpec df1=6 baseline correction on the analyzed sub-grid."""
    from airspec_baseline import SEG1, SEG2, airspec_baseline_matrix, make_mask

    wn = np.array([float(c) for c in spec.columns])
    desc = np.argsort(-wn)
    x = wn[desc]
    Y = spec.to_numpy(float)[:, desc]
    _, corr = airspec_baseline_matrix(x, Y, df1=6, df2=4)
    analyzed = make_mask(x, SEG1) | make_mask(x, SEG2)
    xa = x[analyzed]
    out = pd.DataFrame(corr[:, analyzed], columns=[str(v) for v in xa],
                       index=spec.index)
    return out, xa


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site", help="SPARTAN site code, e.g. INDH / CHTS / ETBI")
    ap.add_argument("--src", default="~/Downloads",
                    help="directory holding <SITE>_*.csv from the PowerShell pull")
    ap.add_argument("--name", default=None,
                    help="target folder name (default: lowercased site code)")
    ap.add_argument("--seasons", choices=("ethiopian", "quarter"),
                    default="ethiopian", help="Group column scheme")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    site = args.site.upper()
    name = args.name or site.lower()

    def read(kind: str) -> pd.DataFrame:
        # accept both <SITE>_x.csv and the lowercase etbi_x.csv from the first pull
        for cand in (src / f"{site}_{kind}.csv", src / f"{site.lower()}_{kind}.csv"):
            if cand.exists():
                df = pd.read_csv(cand, encoding="utf-8-sig")
                df.columns = [c.strip('﻿"') for c in df.columns]
                return df
        raise SystemExit(f"missing {site}_{kind}.csv in {src}")

    filters, analyses, scans = read("filters"), read("ftir_analysis"), read("scans_base64")
    wcols, wn_app = app_grid()
    spec = decode_scans(scans, analyses, wcols, wn_app)
    print(f"{site}: {len(filters)} filters, {len(analyses)} analyses, "
          f"{len(spec)} spectra decoded")

    ref = filters.merge(hips_reference(site), on="ExternalFilterId", how="left")
    ref["SamplingStartDate"] = pd.to_datetime(ref["SamplingStartDate"],
                                              errors="coerce")
    ref = ref[ref["MediaId"].isin(spec.index)].copy()
    scheme = SEASON if args.seasons == "ethiopian" else QUARTER
    out = pd.DataFrame({
        "MediaId": ref["MediaId"].astype(int),
        "Fabs": ref["Fabs"],
        "Volume_m3": ref["SampleVolume_m3"],
        "Date": ref["SamplingStartDate"].dt.date.astype(str)
                   .where(ref["SamplingStartDate"].notna(), ""),
        "Group": ref["SamplingStartDate"].dt.month
                    .map(lambda m: scheme.get(int(m), "unknown")
                         if pd.notna(m) else "unknown"),
    })
    usable = out[out["Fabs"].notna() & (out["Volume_m3"] > 0)].copy()
    print(f"  usable (Fabs + volume): {len(usable)}"
          f" | Fabs {usable['Fabs'].min():.1f}-{usable['Fabs'].max():.1f} Mm-1"
          f" | groups {usable['Group'].value_counts().to_dict()}"
          f" | lots {ref['LotId'].value_counts(dropna=False).to_dict()}")
    if usable.empty:
        raise SystemExit("no filters have both a HIPS Fabs and a positive volume — "
                         "nothing to write")

    d = TARGETS / name
    d.mkdir(parents=True, exist_ok=True)
    spec.loc[usable["MediaId"]].reset_index().to_csv(d / "spectra.csv", index=False)
    usable.to_csv(d / "reference.csv", index=False)
    corr, _ = corrected(spec.loc[usable["MediaId"]])
    corr.reset_index().to_csv(d / "spectra_corrected.csv", index=False)
    print(f"  wrote {d}/ (spectra, reference, spectra_corrected)"
          f" -> appears in the explorer as '{name} (custom)'")


if __name__ == "__main__":
    main()
