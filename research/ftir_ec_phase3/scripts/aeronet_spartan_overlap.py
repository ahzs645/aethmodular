"""For each SPARTAN site: AERONET stations within 50 km, and the filter days with daily L1.5 AOD / AAOD.

Usage: python aeronet_spartan_overlap.py <repo root> <aeronet_locations_v3.txt> <out.json>
The site list is https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt. Filter days come from the
gallery exports (spartan_reference.json, filters.json). Results: docs/notebook-gallery-coverage.md §E.
"""
import json, math, csv, sys, io, urllib.request, datetime as dt, collections
from concurrent.futures import ThreadPoolExecutor
repo, locs, out = sys.argv[1], sys.argv[2], sys.argv[3]
ref = json.load(open(f'{repo}/gallery/app/public/data/spartan_reference.json'))['rows']
foc = json.load(open(f'{repo}/gallery/app/public/data/filters.json'))['rows']
days = collections.defaultdict(set)
for r in ref + foc:
    if r.get('date'): days[r['code']].add(r['date'])
sites = {s['code']: s for s in json.load(open(f'{repo}/gallery/app/public/data/spartan_map.json'))['sites']}
A = [(r[0], float(r[1]), float(r[2])) for r in list(csv.reader(open(locs)))[2:] if len(r) >= 3]
def km(a, b, c, d):
    p = math.pi/180
    return 6371*2*math.asin(math.sqrt(math.sin((c-a)*p/2)**2+math.cos(a*p)*math.cos(c*p)*math.sin((d-b)*p/2)**2))
def fetch(url):
    try:
        return urllib.request.urlopen(url, timeout=180).read().decode('latin-1')
    except Exception as e:
        return f'ERR {e}'
def aeronet_days(station, d0, d1, inv):
    a, b = dt.date.fromisoformat(d0), dt.date.fromisoformat(d1)
    q = f'site={station}&year={a.year}&month={a.month}&day={a.day}&year2={b.year}&month2={b.month}&day2={b.day}&AVG=20&if_no_html=1'
    url = ('https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3?product=TAB&ALM15=1&' if inv
           else 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?AOD15=1&') + q
    txt = fetch(url)
    if txt.startswith('ERR'): return None
    s = set()
    for line in txt.splitlines():
        p = line.split(',')
        if len(p) > 2 and p[0] == station or (len(p) > 2 and p[1].count(':') == 2 and p[1][2] == ':'):
            for f in p[:3]:
                if f.count(':') == 2 and len(f) == 10:
                    d, m, y = f.split(':'); s.add(f'{y}-{m}-{d}'); break
    return s
def near(c):
    return [(round(d, 1), n) for d, n in sorted((km(sites[c]['lat'], sites[c]['lon'], la, lo), n) for n, lo, la in A) if d <= 50][:3]
jobs = []
for c, s in sites.items():
    fd = sorted(days.get(c, []))
    for d, st in near(c):
        if fd: jobs.append((c, st, d, fd[0], fd[-1]))
def run(j):
    c, st, d, d0, d1 = j
    return j, aeronet_days(st, d0, d1, False), aeronet_days(st, d0, d1, True)
res = list(ThreadPoolExecutor(8).map(run, jobs))
def plus1(s): return s | {(dt.date.fromisoformat(x)+dt.timedelta(1)).isoformat() for x in s} | {(dt.date.fromisoformat(x)-dt.timedelta(1)).isoformat() for x in s}
rows = []
for (c, st, d, d0, d1), aod, inv in res:
    fd = days[c]
    rows.append(dict(code=c, name=sites[c]['name'], n_filter_days=len(fd), first=d0, last=d1, station=st, km=d,
                     aod_days=None if aod is None else len(aod), aod_match=None if aod is None else len(fd & aod),
                     aod_match_pm1=None if aod is None else len(fd & plus1(aod)),
                     aaod_days=None if inv is None else len(inv), aaod_match=None if inv is None else len(fd & inv)))
json.dump(rows, open(out, 'w'), indent=1)
for r in rows: print(r)
