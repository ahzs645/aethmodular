"""The whole IMPROVE network as a reference population for the filter drawer ("All IMPROVE sites").

Run with:  uv run python gallery/data/export_improve_reference.py   (after export_improve_samples.py)

The drawer places a filter in a reference: its percentile, and the range, quartiles and median of
each measurement. SPARTAN's reference ships every row (spartan_reference.json, ~5,000 filters), but
IMPROVE has ~388,000 samples in 217 site files (~48 MB), too much for a browser to fetch. This file
holds what the drawer needs instead:

    quantiles   per measurement, 1,001 quantiles (every 0.1 %) of all its finite values, computed as
                the drawer computes them from rows (linear between order statistics), so min, the
                quartiles, the median and max are exact and a percentile is good to 0.1 %
    sample      an even sample of whole filters across sites (the same number from each site, or all
                of a smaller site's), for the drawer's crossplot, which needs pairs, not quantiles;
                columnar like the site files (code, date, poc, then one list per measurement)

Read from the gallery's own IMPROVE site files (gallery/app/public/data/improve_samples/), so the
reference is exactly the network the gallery charts: every physical filter, collocated samplers
(POC > 1) included, as in those files; IMPROVE rows carry no study exclusions. Field names and units
follow the site files (HIPS BC = HIPS Fabs / MAC 10, as SPARTAN's).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "gallery/app/public/data"
SITES = DATA / "improve_samples"
OUT = DATA / "improve_reference.json"
STEPS = 1000          # quantiles at every 1 / STEPS
SAMPLE_PER_SITE = 24  # filters per site in the crossplot sample (~5,000 in all)
SEED = 20261027


def main() -> None:
    index = json.loads((SITES / "index.json").read_text())
    fields: list[str] = index["fields"]
    rng = np.random.default_rng(SEED)
    values: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    sample: dict[str, list] = {"code": [], "date": [], "poc": [], **{k: [] for k in fields}}
    dates: list[str] = []
    for site in index["sites"]:
        f = json.loads((SITES / f"{site['code']}.json").read_text())
        n = len(f["date"])
        dates += [f["date"][0], f["date"][-1]] if n else []
        cols = {k: np.array([np.nan if v is None else v for v in f[k]], dtype=float) for k in fields}
        for k in fields:
            values[k].append(cols[k][np.isfinite(cols[k])])
        # whole filters for the crossplot: the same number from every site, so no one site dominates it
        for i in sorted(rng.choice(n, size=min(n, SAMPLE_PER_SITE), replace=False).tolist()):
            sample["code"].append(f["code"]); sample["date"].append(f["date"][i]); sample["poc"].append(f["poc"][i])
            for k in fields:
                sample[k].append(float(f"{cols[k][i]:.6g}") if np.isfinite(cols[k][i]) else None)
    quantiles = {}
    for k in fields:
        v = np.sort(np.concatenate(values[k])) if values[k] else np.array([])
        if len(v) < 4:
            continue
        # numpy's default (linear) matches lib/referencePopulation quantile(): (n - 1) * p between order statistics
        q = np.quantile(v, np.linspace(0, 1, STEPS + 1))
        quantiles[k] = dict(n=int(len(v)), q=[float(f"{x:.6g}") for x in q])
    n = int(index["n"])
    OUT.write_text(json.dumps(dict(
        schema_version=1, generated=datetime.now(timezone.utc).isoformat(),
        source="gallery/app/public/data/improve_samples (export_improve_samples.py: the validated IMPROVE cleaned table)",
        note=(f"Every IMPROVE sample in the gallery's site files ({n:,} filters at {len(index['sites'])} sites, "
              f"{min(dates)} to {max(dates)}), collocated samplers included. Percentiles and ranges come from "
              f"{STEPS + 1} quantiles of each measurement over all its finite values; the crossplot uses an even sample "
              f"of {SAMPLE_PER_SITE} filters per site. EC/OC are TOR; HIPS BC = HIPS Fabs / MAC 10."),
        n=n, n_sites=len(index["sites"]), date_min=min(dates), date_max=max(dates),
        fields=fields, field_units=index["field_units"], steps=STEPS, quantiles=quantiles,
        site_names={x["code"]: x["name"] for x in index["sites"]},
        sample_per_site=SAMPLE_PER_SITE, sample=sample), separators=(",", ":"), allow_nan=False))
    print(f"Wrote {OUT.relative_to(ROOT)}: {n:,} filters, {len(quantiles)} measurements, "
          f"{len(sample['code']):,} sampled filters, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
