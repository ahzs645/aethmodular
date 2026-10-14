"""Fill Open-Meteo's ERA5 boundary-layer-height gap (Jan-Jun 2024) from ARCO-ERA5, on a Colab VM.

Open-Meteo returns no boundary_layer_height for 2024-01 to 2024-06 at any site.
The public ARCO-ERA5 store on Google Cloud (anonymous) has it, but stores each hour
as one global 721 x 1440 field (~3.2 MB compressed), so six months is ~14 GB.
Run on a Colab VM so the download stays in Google's network and only the site
values (a few MB) come back. Values were checked against Open-Meteo at Addis
(2023-01-10, 04-15 UTC): identical to within Open-Meteo's 5 m rounding.

This file runs ON THE VM. From the repo root:
    uv run python research/ftir_hips_chem/workflows/colab_arco_blh_gapfill.py --sites-arg   # prints the site argument
    colab run --keep -s blhfill --timeout 3600 research/ftir_hips_chem/workflows/colab_arco_blh_gapfill.py \
        2024-01-01T00 2024-06-30T23 "$(uv run python research/ftir_hips_chem/workflows/colab_arco_blh_gapfill.py --sites-arg)"
    colab download -s blhfill /content/arco_blh_fill.npz research/ftir_hips_chem/output/tables/boundary_layer_era5/arco_blh_fill_2024H1.npz
    colab stop -s blhfill
then merge locally with boundary_layer.import_arco_fill().
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
EPOCH = datetime(1900, 1, 1)   # the store's time axis is hours since 1900-01-01
OUT = os.environ.get("ARCO_FILL_OUT", "/content/arco_blh_fill.npz")   # override to test off Colab


def sites_arg(networks=("spartan", "improve")) -> str:
    """network/code:lat:lon;... for every site with coordinates in these networks (run locally)."""
    root = Path(__file__).resolve().parents[3]
    parts = []
    for network in networks:
        for s in json.loads((root / f"gallery/app/public/data/{network}_map.json").read_text())["sites"]:
            if s.get("lat") is not None and s.get("lon") is not None:
                parts.append(f"{network}/{s['code']}:{s['lat']:.4f}:{s['lon']:.4f}")
    return ";".join(parts)


def parse_hours(spec: str) -> list[int]:
    """"START..END" (ISO hours, inclusive) or "r:a-b,c-d" (store hour indices, inclusive runs)."""
    if spec.startswith("r:"):
        out = []
        for run in spec[2:].split(","):
            a, b = run.split("-")
            out.extend(range(int(a), int(b) + 1))
        return out
    start, end = spec.split("..")
    t0, t1 = datetime.fromisoformat(start), datetime.fromisoformat(end)
    return [int((t0 - EPOCH).total_seconds() // 3600) + k for k in range(int((t1 - t0).total_seconds() // 3600) + 1)]


def main(spec: str, sites: str):
    import numpy as np
    try:
        import numcodecs
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numcodecs"], check=True)
        import numcodecs

    codes, rows, cols = [], [], []
    for item in sites.split(";"):
        code, lat, lon = item.rsplit(":", 2)
        codes.append(code)
        rows.append(int(round((90 - float(lat)) / 0.25)))            # latitude runs 90 -> -90
        cols.append(int(round((float(lon) % 360) / 0.25)) % 1440)     # longitude runs 0 -> 359.75
    rows, cols = np.array(rows), np.array(cols)
    hours = parse_hours(spec)

    blosc = numcodecs.Blosc()
    nbytes = [0] * len(hours)

    def one(k):
        for attempt in range(5):
            try:
                raw = urllib.request.urlopen(f"{STORE}/boundary_layer_height/{hours[k]}.0.0", timeout=120).read()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"hour index {hours[k]} failed")
        nbytes[k] = len(raw)
        field = np.frombuffer(blosc.decode(raw), dtype="<f4").reshape(721, 1440)
        return field[rows, cols]

    tic = time.time()
    with ThreadPoolExecutor(96) as pool:
        values = np.stack(list(pool.map(one, range(len(hours)))))
    seconds = time.time() - tic
    times = np.array([np.datetime64(EPOCH + timedelta(hours=h), "h") for h in hours])
    np.savez_compressed(OUT, time=times, code=np.array(codes), blh_m=values.astype("float32"),
                        grid_row=rows, grid_col=cols)
    summary = {"hours": len(hours), "sites": len(codes), "downloaded_GB": round(sum(nbytes) / 1e9, 2),
               "mean_MB_per_hour": round(sum(nbytes) / len(hours) / 1e6, 2), "seconds": round(seconds),
               "throughput_MB_s": round(sum(nbytes) / 1e6 / max(seconds, 1e-9)), "nan_values": int(np.isnan(values).sum()),
               "output_MB": round(Path(OUT).stat().st_size / 1e6, 2), "output": OUT}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sites-arg"]:
        print(sites_arg(sys.argv[2:] or ("spartan", "improve")))
    elif len(sys.argv) == 4:   # START END SITES, the original Jan-Jun 2024 gap-fill call
        main(f"{sys.argv[1]}..{sys.argv[2]}", sys.argv[3])
    else:                      # HOURS SITES, e.g. "r:906000-906043,906072-906115" (workflows/improve_blh_colab.py)
        main(sys.argv[1], sys.argv[2])
