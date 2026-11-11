"""Screen filters for the HIPS bend: identify it, trace it, correct it, check the correction.

    aeth hips screen                                   # every SPARTAN site in the gallery's table
    aeth hips screen --site ETAD INDH --out OUT        # two sites, tables written to OUT
    aeth hips screen --input batch.csv --fabs Fabs --volume Volume --site-column Site \\
        --reference EC --instrument PM25 --date SampleDate

(or  python research/ftir_hips_chem/workflows/hips_screen.py ...)

For each filter: the HIPS optical depth, the expected under-read, a status (use linearised, caution, shape
only), and the linearised Fabs with its range. For each site: whether its own data show the bend, what the
bend does to the numbers read off HIPS, and whether the correction removes it. A file needs Fabs (Mm-1) and
the sampled volume (m3), or the laboratory's own optical depth with --tau; the site tests also need a
reference (FTIR EC), an instrument (weighed PM2.5) and a date. Protocol: docs/hips-loading-protocol.md.
The ceiling is IMPROVE's; no aethalometer data are used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(Path(__file__).resolve().parent)]
import hips_loading as hl  # noqa: E402

FILTER_COLUMNS = ["id", "code", "site", "date", "fabs", "volume", "tau", "under_read", "under_read_lo", "under_read_hi", "status",
                  "beyond_reference", "factor", "fabs_linearised", "fabs_linearised_lo", "fabs_linearised_hi"]


def from_gallery(codes):
    import run_hips_network_20261111 as network

    f = network.load(network.ceiling())
    if codes:
        missing = sorted(set(codes) - set(f["code"]))
        if missing:
            raise SystemExit(f"no filters with HIPS and a volume for: {', '.join(missing)}")
        f = f[f["code"].isin(codes)]
    return f


def from_file(a):
    raw = pd.read_csv(a.input)
    names = {"fabs": a.fabs, "volume": a.volume, "tau": a.tau, "ftir_ec": a.reference, "pm25": a.instrument, "date": a.date,
             "code": a.site_column, "id": a.id}
    missing = [f"--{k} {v}" for k, v in names.items() if v and v not in raw]
    if missing:
        raise SystemExit(f"{a.input} has no column for: {', '.join(missing)}")
    f = pd.DataFrame({k: raw[v] for k, v in names.items() if v})
    if "code" not in f:
        f["code"] = Path(a.input).stem
    f["site"] = f["code"]
    if "id" not in f:
        f["id"] = np.arange(len(f)).astype(str)
    if "date" in f:
        f["date"] = pd.to_datetime(f["date"])
        f["month"] = f["date"].dt.month
    need = ["tau"] if a.tau and not a.fabs else ["fabs", "volume"]
    return f[(f[need] > 0).all(axis=1)].reset_index(drop=True)


def report(code, site, found, traced, checked):
    pct = lambda v: f"{100 * v:.0f} %"  # noqa: E731
    lines = [f"\n{site} ({code}): {found['filters']} filters, median optical depth {found['median_tau']:.2f}",
             f"  1 identify   expected under-read {pct(found['expected_under_read'])}; use {pct(found['share_use'])}, "
             f"caution {pct(found['share_caution'])}, shape only {pct(found['share_shape_only'])}; reading: {found['reading']}"]
    if "exponent" in found:
        lines.append(f"               exponent {found['exponent']:.2f} [{found['exponent_lo']:.2f}, {found['exponent_hi']:.2f}]; "
                     f"heavy ÷ light {found['ratio_heavy_over_light']:.2f} [{found['ratio_lo']:.2f}, {found['ratio_hi']:.2f}], "
                     f"curve predicts {found['ratio_predicted']:.2f}")
    for row in traced.itertuples():
        lines.append(f"  2 propagate  {row.quantity:<34} {row.reported:8.2f} -> {row.linearised:8.2f}  [{row.lo:.2f}, {row.hi:.2f}]")
    usable = found["share_use"] * found["filters"]
    lines.append(f"  3 correct    {usable:.0f} of {found['filters']} filters corrected and usable for level; "
                 f"{found['share_shape_only'] * found['filters']:.0f} shape only; {found['share_beyond_reference'] * found['filters']:.0f} darker than the curve's own filters")
    if "exponent_after" in checked:
        lines.append(f"  4 check      exponent after correcting {checked['exponent_after']:.2f} [{checked['after_lo']:.2f}, {checked['after_hi']:.2f}]: "
                     f"{checked['verdict']}")
    else:
        lines.append(f"  4 check      {checked['verdict']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs="+", default=[], help="site codes in the gallery's SPARTAN table (default: all)")
    ap.add_argument("--input", help="a CSV of filters to screen instead of the gallery's table")
    ap.add_argument("--fabs", help="column: HIPS Fabs, Mm-1")
    ap.add_argument("--volume", help="column: sampled volume, m3")
    ap.add_argument("--tau", help="column: the laboratory's HIPS optical depth, used in place of Fabs x volume / 353")
    ap.add_argument("--reference", help="column: FTIR EC, µg/m3 (for the site tests)")
    ap.add_argument("--instrument", help="column: weighed PM2.5, µg/m3 (for the site tests)")
    ap.add_argument("--date", help="column: sampling date (for the seasonal contrast)")
    ap.add_argument("--site-column", help="column: site code (default: one site, named after the file)")
    ap.add_argument("--id", help="column: filter id")
    ap.add_argument("--out", help="folder for filters.csv, sites.csv and propagation.csv")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap draws for the intervals")
    a = ap.parse_args(argv)
    if a.input and not ((a.fabs and a.volume) or a.tau):
        ap.error("--input needs --fabs and --volume, or --tau")
    if a.input and a.tau and not a.fabs:
        ap.error("--tau alone gives the status but no corrected Fabs; pass --fabs and --volume as well")

    f = from_file(a) if a.input else from_gallery(a.site)
    screened = hl.screen(f, tau=a.tau if a.input else None)
    sites, traced_all = [], []
    for (code, site), part in screened.groupby(["code", "site"], sort=False):
        found, checked = hl.identify(part, n_boot=a.boot), hl.check(part, n_boot=a.boot)
        traced = hl.propagate(part) if "month" in part else pd.DataFrame(columns=["quantity", "reported", "linearised", "lo", "hi"])
        print(report(code, site, found, traced, checked))
        sites.append({"code": code, "site": site} | found | {f"check_{k}": v for k, v in checked.items()})
        traced_all.append(traced.assign(code=code, site=site))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        screened[[c for c in FILTER_COLUMNS if c in screened]].to_csv(out / "filters.csv", index=False)
        pd.DataFrame(sites).to_csv(out / "sites.csv", index=False)
        pd.concat(traced_all, ignore_index=True).to_csv(out / "propagation.csv", index=False)
        print(f"\nwrote filters.csv ({len(screened)} filters), sites.csv and propagation.csv to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
