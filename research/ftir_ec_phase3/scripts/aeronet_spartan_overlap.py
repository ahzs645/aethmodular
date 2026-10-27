"""For each SPARTAN site: AERONET stations within 50 km, and the filter days with daily L1.5 AOD / AAOD.

Usage: python aeronet_spartan_overlap.py <repo root> <aeronet_locations_v3.txt> <out.json>
The site list is https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt. Filter days come from the
gallery exports (spartan_reference.json, filters.json). Results: docs/notebook-gallery-coverage.md §E.
"""

import collections
import csv
import datetime as dt
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

AERONET = "https://aeronet.gsfc.nasa.gov/cgi-bin"
MAX_KM = 50


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(a))


def aeronet_days(station, first, last, inversion):
    """Days with a daily Level 1.5 value (direct-sun AOD, or the almucantar absorption AOD)."""
    a, b = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    query = (f"site={station}&year={a.year}&month={a.month}&day={a.day}"
             f"&year2={b.year}&month2={b.month}&day2={b.day}&AVG=20&if_no_html=1")
    url = (f"{AERONET}/print_web_data_inv_v3?product=TAB&ALM15=1&{query}" if inversion
           else f"{AERONET}/print_web_data_v3?AOD15=1&{query}")
    try:
        text = urllib.request.urlopen(url, timeout=180).read().decode("latin-1")
    except OSError:
        return None
    days = set()
    for line in text.splitlines():
        fields = line.split(",")
        if len(fields) > 2 and fields[0] == station:
            d, m, y = fields[1].split(":")
            days.add(f"{y}-{m}-{d}")
    return days


def widen(days):
    one = dt.timedelta(days=1)
    return {(dt.date.fromisoformat(d) + k * one).isoformat() for d in days for k in (-1, 0, 1)}


def main(repo, locations, out):
    data = f"{repo}/gallery/app/public/data"
    rows = json.load(open(f"{data}/spartan_reference.json"))["rows"] + json.load(open(f"{data}/filters.json"))["rows"]
    filter_days = collections.defaultdict(set)
    for row in rows:
        if row.get("date"):
            filter_days[row["code"]].add(row["date"])
    sites = {s["code"]: s for s in json.load(open(f"{data}/spartan_map.json"))["sites"]}
    stations = [(r[0], float(r[1]), float(r[2])) for r in list(csv.reader(open(locations)))[2:] if len(r) >= 3]

    jobs = []
    for code, site in sites.items():
        days = sorted(filter_days.get(code, []))
        near = sorted((km(site["lat"], site["lon"], lat, lon), name) for name, lon, lat in stations)
        for dist, name in [n for n in near if n[0] <= MAX_KM][:3]:
            if days:
                jobs.append((code, name, round(dist, 1), days[0], days[-1]))

    def run(job):
        code, name, _, first, last = job
        return job, aeronet_days(name, first, last, False), aeronet_days(name, first, last, True)

    results = []
    for (code, name, dist, first, last), aod, inv in ThreadPoolExecutor(8).map(run, jobs):
        days = filter_days[code]
        results.append({
            "code": code, "name": sites[code]["name"], "n_filter_days": len(days), "first": first, "last": last,
            "station": name, "km": dist,
            "aod_days": None if aod is None else len(aod),
            "aod_match": None if aod is None else len(days & aod),
            "aod_match_pm1": None if aod is None else len(days & widen(aod)),
            "aaod_days": None if inv is None else len(inv),
            "aaod_match": None if inv is None else len(days & inv),
        })
    json.dump(results, open(out, "w"), indent=1)
    for row in results:
        print(row)


if __name__ == "__main__":
    main(*sys.argv[1:4])
