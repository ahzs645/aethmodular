#!/usr/bin/env python3
"""Fetch + shrink a world basemap so the map chart works offline.

Natural Earth 110m country outlines, stripped to a name property and rounded
to 2 decimal degrees (~1 km, far finer than a 4-site overview needs). Writes
gallery/app/public/data/world.geojson. Only needs re-running if the file is
lost; the result is committed.

Run:  python gallery/data/fetch_basemap.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
       "master/geojson/ne_110m_admin_0_countries.geojson")
OUT = Path(__file__).resolve().parents[1] / "app" / "public" / "data" / "world.geojson"
PRECISION = 2


def round_coords(node):
    if isinstance(node, list):
        if node and isinstance(node[0], (int, float)):
            return [round(float(v), PRECISION) for v in node]
        return [round_coords(n) for n in node]
    return node


def main():
    print(f"fetching {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as fh:
        gj = json.load(fh)

    features = []
    for f in gj["features"]:
        props = f.get("properties", {})
        name = props.get("NAME") or props.get("SOVEREIGNT") or ""
        features.append({
            "type": "Feature",
            "properties": {"name": name},
            "geometry": {
                "type": f["geometry"]["type"],
                "coordinates": round_coords(f["geometry"]["coordinates"]),
            },
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                              separators=(",", ":")))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} kB, {len(features)} countries)")


if __name__ == "__main__":
    main()
