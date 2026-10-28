"""Rebuild the gallery data in the right order, checking each step's inputs first.

The gallery's newer data (AERONET, the Instruments section's aethalometer pages, the Research
diagnostics pages, the MAIA catalogue) is made by about twenty scripts that read each other's
outputs, local caches and the "Davis Data" Drive folder. This file lists them once, in dependency
order, with what each needs, so a rebuild runs the same way every time. The full guide is
docs/gallery-data-pipeline.md; its step table is generated from STEPS (`--markdown`).

    uv run aeth gallery plan                  # every step, and whether its inputs are present
    uv run aeth gallery rebuild               # run every step whose inputs are present
    uv run aeth gallery rebuild --fetch       # also the network steps (AERONET pulls, MAIA catalogue)
    uv run aeth gallery rebuild --only aeronet aeronet-sites availability
    uv run aeth gallery rebuild --from aethalometer
    uv run aeth gallery check                 # chart index, typecheck, tests, public-build check, sizes
    uv run aeth gallery check-public          # only: private sections stay out of a public build
    uv run aeth gallery sizes                 # data per folder, with and without the private sections

A step whose inputs are missing is skipped, and its committed output is left as it is (most outputs
under gallery/app/public/data are committed). A step after a failed one is skipped. Every run writes
research/ftir_hips_chem/output/tables/gallery_rebuild/last_run.json (gitignored): what ran, how long,
which data roots resolved, and the commit.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "gallery/app"
DATA = APP / "public/data"
AREA = ROOT / "research/ftir_hips_chem"
AERONET_CACHE = AREA / "output/tables/aeronet_spartan"
RUN_LOG = AREA / "output/tables/gallery_rebuild/last_run.json"
SECTIONS = APP / "src/sections/sections.json"

sys.path.insert(0, str(ROOT))
try:  # the repo .env (Drive mirror, data roots), as every resolver loads it
    from aethmodular_cli.env import load_repo_env
    load_repo_env()
except ImportError:
    pass


# ----------------------------------------------------------------------------- inputs

def _data_paths():
    """research/ftir_hips_chem/scripts/data_paths, or None where it cannot be imported."""
    sys.path.insert(0, str(AREA / "scripts"))
    try:
        import data_paths
        return data_paths
    except Exception:  # noqa: BLE001 - a missing optional dependency must not stop the plan
        return None


def drive_root() -> Path | None:
    """The "Davis Data" root (AETHMODULAR_MAIA_DATA_ROOT, else data_paths' probing), when it exists."""
    env = os.environ.get("AETHMODULAR_MAIA_DATA_ROOT")
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    dp = _data_paths()
    try:
        p = dp.maia_data_root() if dp else None
    except Exception:  # noqa: BLE001
        p = None
    return p if p and p.is_dir() else None


@dataclass(frozen=True)
class Need:
    """One input a step reads: a label for the plan and a check that it is there."""
    label: str
    check: Callable[[], bool]

    def ok(self) -> bool:
        try:
            return bool(self.check())
        except Exception:  # noqa: BLE001
            return False


def repo(rel: str) -> Need:
    return Need(rel, lambda: (ROOT / rel).exists())


def drive(rel: str) -> Need:
    return Need(f"Drive: {rel}", lambda: (r := drive_root()) is not None and (r / rel).exists())


def env_root(var: str, rel: str = "") -> Need:
    """A path below the directory an environment variable names (a layout some loaders require)."""
    return Need(f"${var}/{rel}".rstrip("/"), lambda: bool(os.environ.get(var)) and (Path(os.environ[var]).expanduser() / rel).exists())


AERONET_L15 = repo("research/ftir_hips_chem/output/tables/aeronet_spartan/inv_points_lev15.parquet")
WINDOWS = repo("gallery/app/public/data/weather/filters.json")
PICKLES = repo("research/ftir_hips_chem/processed_sites/df_Addis_Ababa_9am_resampled.pkl")
MINUTE = drive("Aethalometry Data/Raw/Jacros_MA350_1-min_2022-2024_Cleaned.csv")
HOURLY = drive("Aethalometry Data/Kyan Data/MAIA data/JPL_cleaned_corrected_small_hourly.csv")
NODE = repo("gallery/app/node_modules")


# ----------------------------------------------------------------------------- steps

@dataclass(frozen=True)
class Step:
    key: str
    group: str            # fetch (network; only with --fetch), export, research, check
    run: tuple[str, ...]  # script path relative to the repo, then its arguments; or a command in gallery/app
    what: str
    writes: str
    needs: tuple[Need, ...] = ()
    after: tuple[str, ...] = ()
    tz: bool = False       # needs timezonefinder (not in the lock; added with `uv run --with`)
    node: bool = False     # an npm/npx command run in gallery/app
    note: str = ""
    order: int = field(default=0, compare=False)


STEPS: tuple[Step, ...] = (
    # --- network pulls: only with --fetch -------------------------------------------------------------
    Step("aeronet-fetch", "fetch", ("research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py",),
         "AERONET Level 1.5 for every SPARTAN site's best station: raw downloads by year, every retrieval parsed",
         "research/ftir_hips_chem/output/tables/aeronet_spartan/ (gitignored, ~1 GB)", (WINDOWS,),
         note="resumable; add --refresh to re-download reprocessed years"),
    Step("aeronet-fetch-l20", "fetch", ("research/ftir_hips_chem/workflows/fetch_aeronet_spartan.py", "--level", "20"),
         "AERONET Level 2.0 for the same stations", "aeronet_spartan/*_lev20.parquet", (WINDOWS,), ("aeronet-fetch",)),
    Step("maia-catalog", "fetch", ("scripts/pipelines/maia_catalog.py", "catalog"),
         "MAIA portal and NASA CMR: target areas, layers, monitors, granule coverage", "research/maia/inventory/"),
    # --- gallery exports ----------------------------------------------------------------------------
    Step("weather", "export", ("gallery/data/export_weather.py",),
         "ERA5 (and the Addis station) over each filter's sampling window; the filter windows every later step uses",
         "gallery/app/public/data/weather/",
         (repo("research/ftir_hips_chem/output/tables/weather_era5/spartan_hourly.parquet"), repo("data/spartan/raw/FilterBased/ChemSpecPM25")),
         tz=True, note="its ERA5 cache and the public ChemSpec files are local only; the committed weather/ is kept otherwise"),
    Step("aeronet", "export", ("gallery/data/export_aeronet.py",),
         "AERONET means over each filter's window (both levels whose parquet exists), daily series, stations",
         "gallery/app/public/data/aeronet/ and aeronet/lev20/", (AERONET_L15, WINDOWS), ("weather", "aeronet-fetch", "aeronet-fetch-l20"), tz=True),
    Step("aeronet-sites", "export", ("gallery/data/export_aeronet_sites.py",),
         "Per-site and per-season medians (AOD, AAOD, AAE, SSA, PM2.5/AOD, scale height) and MAIA targets, for the Map",
         "gallery/app/public/data/aeronet/sites_summary.json", (repo("gallery/app/public/data/aeronet/filters.json"),), ("aeronet",)),
    Step("aethalometer", "export", ("gallery/data/export_aethalometer.py",),
         "MA350 daily bins per site and each filter's overlap-weighted values (the Instruments section's base data)",
         "gallery/app/public/data/aethalometer/{index,sites}.json, filters/, daily/", (PICKLES, WINDOWS), ("weather",), tz=True),
    Step("aeth-filters", "export", ("gallery/data/export_aeth_filters.py",),
         "Smoothed BCc and ATN per filter for the Against the filters page", "aethalometer/filters_extra/", (PICKLES,), ("aethalometer",)),
    Step("aeth-quality", "export", ("gallery/data/export_aeth_quality.py",),
         "Daily QC series (flow, ATN, dual spot, smoothing) for the Data quality page", "aethalometer/quality/", (PICKLES,), ("aethalometer",)),
    Step("aeth-quality-minute", "export", ("gallery/data/export_aeth_quality_minute.py",),
         "Minute-level flow QC, rolling medians, tape advances, clean mask (Jacros, Pasadena)", "aethalometer/quality/minute/",
         (MINUTE,), ("aeth-quality",)),
    Step("aeth-sources", "export", ("gallery/data/export_aeth_sources.py",),
         "The instruments' own AAE and source split against the gallery's, for the Sources page", "aethalometer/sources/instrument.json",
         (PICKLES,), ("aethalometer",), note="without the Drive's hourly files the hourly part is dropped"),
    Step("aeth-overview", "export", ("gallery/data/export_aeth_overview.py",),
         "Hourly record, high-BC episodes with minute traces, for the Overview page", "aethalometer/overview/", (MINUTE, HOURLY), ("aethalometer",)),
    Step("aeth-diurnal", "export", ("gallery/data/export_aeth_diurnal.py",),
         "Hour-of-day cycles (surface and column), the brown-carbon test, AERONET's daylight window", "aethalometer/diurnal/",
         (MINUTE, HOURLY, AERONET_L15), ("aethalometer", "aeronet"), tz=True),
    Step("research-hips-optics", "export", ("gallery/data/export_research_hips_optics.py",),
         "HIPS raw optics and mass per area per filter (HIPS operating envelope)", "research/hips_optics/"),
    Step("research-addis-met", "export", ("gallery/data/export_research_addis_met.py",),
         "The Addis station's day on each sample date (PMF sources and weather)", "research/addis_met/"),
    Step("research-spartan-improve", "export", ("gallery/data/export_research_spartan_improve.py",),
         "SPARTAN vs IMPROVE from the notebooks' tracked tables", "research/spartan_improve/"),
    Step("research-improve-envelope", "export", ("gallery/data/export_research_improve_envelope.py",),
         "The IMPROVE network in loading space (IMPROVE data-use condition applies)", "research/improve_envelope/",
         (drive("Improve/IMPROVE_chem-mass-fabs_2026-04-22.xlsx"),)),
    Step("research-improve-origin", "export", ("gallery/data/export_research_improve_origin.py",),
         "IMPROVE HIPS at EC = 0, curvature, blank domain (ftir_26, ftir_37)", "research/improve_origin/",
         (drive("FTIR/local_db/tables/results_tor.csv"),)),
    Step("research-spectral", "export", ("gallery/data/export_research_spectral.py",),
         "The spectral and calibration diagnostics (1617 band, MAC choice, applicability, network map, …)", "research/spectral/",
         note="recomputes spartan_ec 08 where $AETHMODULAR_DRIVE_ROOT holds University/Research/Grad/Data/Davis Data"),
    Step("availability", "export", ("gallery/data/export_availability.py",),
         "Sites → Data availability, including the AERONET photometer view", "gallery/app/public/data/availability.json",
         (repo("gallery/app/public/data/aeronet/stations.json"),), ("aeronet", "aeronet-sites")),
    # --- research outputs that are not gallery data ---------------------------------------------------
    Step("diurnal-coverage", "research", ("research/ftir_ec_phase3/scripts/ma350_diurnal_coverage.py",),
         "24 h vs AERONET-hours Red BCc ratio (AERONET_SPARTAN_2026-08-22.md, corrected 2026-10-27)",
         "research/ftir_ec_phase3/output/tables/aeronet/diurnal_coverage_correction.csv", (MINUTE,)),
    # --- checks ---------------------------------------------------------------------------------------
    Step("chart-index", "check", ("npm", "run", "index:charts"), "Regenerate the ⌘K chart index", "gallery/app/src/chartIndex.json", (NODE,), node=True),
    Step("typecheck", "check", ("npx", "tsc", "-b"), "Typecheck the gallery", "-", (NODE,), ("chart-index",), node=True),
    Step("node-tests", "check", ("npm", "test"), "The gallery's node tests (7 fail without gitignored exports)", "-", (NODE,), ("chart-index",), node=True,
         note="failures are reported, not fatal: the baseline, VIBES and similarity tests need exports absent from a fresh clone"),
    Step("python-tests", "check", ("-m", "pytest", "-q", "tests/test_gallery_rebuild.py", "tests/test_gallery_sections.py",
                                   "tests/test_aeronet_gallery.py", "tests/test_aeronet_sites_export.py", "tests/test_availability_aeronet.py",
                                   "tests/test_aeth_quality_export.py", "tests/test_aeth_quality_minute.py", "tests/test_aeth_sources_export.py",
                                   "tests/test_aeth_diurnal_export.py", "tests/test_aeth_overview_export.py", "tests/test_gallery_research_hips_met.py",
                                   "tests/test_gallery_research_spectral.py", "tests/test_maia_catalog.py", "tests/test_gallery_publication.py"),
         "The pipeline's pytest files", "-", tz=True),
)
STEPS = tuple(Step(**{**s.__dict__, "order": i}) for i, s in enumerate(STEPS))
BY_KEY = {s.key: s for s in STEPS}
NONFATAL = {"node-tests"}


def command(step: Step) -> tuple[list[str], Path]:
    if step.node:
        return list(step.run), APP
    uv = shutil.which("uv")
    py = (["uv", "run", "--no-sync"] + (["--with", "timezonefinder"] if step.tz else []) + (["--with", "pytest"] if step.key == "python-tests" else []) + ["python"]) \
        if uv else [sys.executable]
    first = list(step.run)
    return py + ([str(ROOT / first[0])] + first[1:] if not first[0].startswith("-") else first), ROOT


def status(step: Step) -> tuple[str, list[str]]:
    missing = [n.label for n in step.needs if not n.ok()]
    script = step.run[0]
    if not step.node and not script.startswith("-") and not (ROOT / script).exists():
        missing.insert(0, f"script {script}")
    return ("ready" if not missing else "blocked"), missing


# ----------------------------------------------------------------------------- run

def select(args) -> list[Step]:
    steps = [s for s in STEPS if args.fetch or s.group != "fetch"]
    if args.only:
        unknown = [k for k in args.only if k not in BY_KEY]
        if unknown:
            raise SystemExit(f"unknown step(s) {unknown}; see `aeth gallery plan`")
        steps = [BY_KEY[k] for k in args.only]
    if args.from_step:
        start = BY_KEY[args.from_step].order
        steps = [s for s in steps if s.order >= start]
    if args.skip:
        steps = [s for s in steps if s.key not in args.skip]
    if not args.checks and not args.only:
        steps = [s for s in steps if s.group != "check"]
    return sorted(steps, key=lambda s: s.order)


def roots() -> dict[str, str | None]:
    r = drive_root()
    return {"drive (Davis Data)": str(r) if r else None,
            "AETHMODULAR_DRIVE_ROOT": os.environ.get("AETHMODULAR_DRIVE_ROOT"),
            "aeronet cache": str(AERONET_CACHE) if AERONET_CACHE.is_dir() else None}


def plan(steps: list[Step]) -> None:
    print("data roots:")
    for k, v in roots().items():
        print(f"  {k:24} {v or '— not found'}")
    print()
    for s in steps:
        st, missing = status(s)
        mark = "✓" if st == "ready" else "·"
        print(f"{mark} {s.key:26} [{s.group}] {s.what}")
        for m in missing:
            print(f"      missing: {m}")


def run(steps: list[Step], dry: bool) -> int:
    log = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git("rev-parse", "HEAD"),
           "dirty": bool(git("status", "--porcelain", "--", "gallery", "research", "scripts")), "roots": roots(), "steps": []}
    failed: set[str] = set()
    for s in steps:
        st, missing = status(s)
        cmd, cwd = command(s)
        entry = {"step": s.key, "command": " ".join(cmd), "cwd": str(cwd.relative_to(ROOT)) or "."}
        if [d for d in s.after if d in failed]:
            entry["status"] = f"skipped: {', '.join(d for d in s.after if d in failed)} failed"
        elif st == "blocked":
            entry["status"] = "skipped: missing " + "; ".join(missing)
        elif dry:
            entry["status"] = "would run"
        else:
            print(f"\n=== {s.key}: {' '.join(cmd)}", flush=True)
            t0 = time.time()
            code = subprocess.run(cmd, cwd=cwd).returncode
            entry["seconds"] = round(time.time() - t0, 1)
            entry["status"] = "ok" if code == 0 else f"failed (exit {code})"
            if code and s.key not in NONFATAL:
                failed.add(s.key)
        log["steps"].append(entry)
    print("\nsummary:")
    for e in log["steps"]:
        print(f"  {e['step']:26} {e['status']}" + (f"  ({e['seconds']} s)" if "seconds" in e else ""))
    if not dry:
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        RUN_LOG.write_text(json.dumps(log, indent=1))
        print(f"\nrun log: {RUN_LOG.relative_to(ROOT)}")
    return 1 if failed else 0


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip()


# ----------------------------------------------------------------------------- the public build

def private_sections() -> dict[str, dict]:
    return {k: s for k, s in json.loads(SECTIONS.read_text())["sections"].items() if not s.get("public")}


def unlisted_sections() -> dict[str, dict]:
    """Sections a public build compiles and publishes but leaves out of its menu (sections.json "unlisted")."""
    return {k: s for k, s in json.loads(SECTIONS.read_text())["sections"].items() if s.get("public") == "unlisted"}


def section_markers(sections: dict[str, dict] | None = None) -> dict[str, set[str]]:
    """Per section (the private ones by default): the data paths its own code fetches (string literals
    under its data folders). None of a private section's may appear in a public build."""
    out: dict[str, set[str]] = {}
    for key, s in (private_sections() if sections is None else sections).items():
        folders = [d.strip("/") for d in s.get("data", [])]
        pat = re.compile(r"""['"`]((?:%s)/[A-Za-z0-9_./${}-]+?\.json)['"`]""" % "|".join(map(re.escape, folders)))
        found = set()
        for f in (APP / "src/sections" / key).rglob("*.ts*"):
            found |= {m for m in pat.findall(f.read_text()) if "${" not in m}
        out[key] = found
    return out


def check_public() -> int:
    """Build with VITE_PUBLIC_BUILD=1 and confirm no private section's data paths are in the bundle, then
    confirm the publish packer leaves the private data folders out of the archive. An unlisted section
    must be the other way round: its data paths in the bundle and its data in the archive."""
    markers = section_markers()
    shown = section_markers(unlisted_sections())
    problems = [f"section {k} has no data-path literals to check" for k, m in {**markers, **shown}.items() if not m]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "site"
        env = {**os.environ, "VITE_PUBLIC_BUILD": "1"}
        r = subprocess.run(["npx", "vite", "build", "--outDir", str(out), "--emptyOutDir"], cwd=APP, env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit("public build failed")
        bundle = "".join(p.read_text(errors="ignore") for p in (out / "assets").glob("*.js"))
        for key, ms in markers.items():
            leaked = sorted(m for m in ms if m in bundle)
            print(f"section {key}: {len(ms)} data paths checked, {len(leaked)} in the public bundle")
            problems += [f"{key}: {m} is in the public bundle" for m in leaked]
        for key, ms in shown.items():
            missing = sorted(m for m in ms if m not in bundle)
            print(f"unlisted section {key}: {len(ms)} data paths checked, {len(missing)} missing from the public bundle")
            problems += [f"{key}: {m} is not in the public bundle" for m in missing]
        sys.path.insert(0, str(ROOT / "gallery/share"))
        import publish_pages  # noqa: E402
        import tarfile
        archive = Path(tmp) / "data.tar.gz"
        publish_pages.pack_data(archive)
        with tarfile.open(archive) as tar:
            names = tar.getnames()
        dirs = [d.strip("/") for s in private_sections().values() for d in s.get("data", [])]
        leaked = [n for n in names if any(n == d or n.startswith(d + "/") for d in dirs)]
        print(f"publish archive: {len(names)} files, {len(leaked)} from private sections")
        problems += [f"archive carries {n}" for n in leaked[:10]]
        for key, s in unlisted_sections().items():
            n = sum(any(x.startswith(d.strip("/") + "/") for d in s.get("data", [])) for x in names)
            print(f"publish archive: {n} files from unlisted section {key}")
            if not n:
                problems.append(f"archive carries no data for unlisted section {key}")
    for p in problems:
        print("PROBLEM:", p)
    print("public build: OK" if not problems else "public build: NOT OK")
    return 1 if problems else 0


def sizes() -> int:
    priv = {d.strip("/") for s in private_sections().values() for d in s.get("data", [])}
    rows: dict[str, list[int]] = {}
    for f in DATA.rglob("*"):
        if f.is_file():
            rel = f.relative_to(DATA).parts
            top = rel[0] if len(rel) > 1 else "(top-level files)"
            r = rows.setdefault(top, [0, 0, 0])
            body = f.read_bytes()
            r[0] += len(body)
            r[1] += len(gzip.compress(body, 6))
            r[2] += 1
    print(f"{'folder':28} {'MB':>7} {'gz MB':>7} {'files':>6}")
    tot, pub = [0, 0], [0, 0]
    for k, (raw, gz, n) in sorted(rows.items(), key=lambda x: -x[1][0]):
        print(f"{k[:28]:28} {raw / 1e6:7.2f} {gz / 1e6:7.2f} {n:6d}" + ("  (private)" if k in priv else ""))
        tot = [tot[0] + raw, tot[1] + gz]
        if k not in priv:
            pub = [pub[0] + raw, pub[1] + gz]
    print(f"\nall data:             {tot[0] / 1e6:7.1f} MB ({tot[1] / 1e6:.1f} MB gzipped)")
    print(f"public build's data:  {pub[0] / 1e6:7.1f} MB ({pub[1] / 1e6:.1f} MB gzipped)")
    print(f"private sections:     {(tot[0] - pub[0]) / 1e6:7.1f} MB ({(tot[1] - pub[1]) / 1e6:.1f} MB gzipped)")
    return 0


DOC = ROOT / "docs/gallery-data-pipeline.md"
DOC_BEGIN = "<!-- STEPS:BEGIN (generated by `uv run aeth gallery markdown`; do not edit by hand) -->"
DOC_END = "<!-- STEPS:END -->"


def doc_table(text: str) -> str:
    """The table currently between the markers in docs/gallery-data-pipeline.md."""
    return text.split(DOC_BEGIN, 1)[1].split(DOC_END, 1)[0].strip("\n") + "\n"


def write_doc() -> None:
    text = DOC.read_text()
    head, rest = text.split(DOC_BEGIN, 1)
    DOC.write_text(head + DOC_BEGIN + "\n" + markdown() + DOC_END + rest.split(DOC_END, 1)[1])


def markdown() -> str:
    """The step table docs/gallery-data-pipeline.md carries between its STEPS markers."""
    lines = ["| # | step | group | what it makes | writes | needs | runs after |", "|---|---|---|---|---|---|---|"]
    for s in STEPS:
        needs = "<br>".join(n.label for n in s.needs) or "—"
        after = ", ".join(s.after) or "—"
        cmd = " ".join(s.run) if s.node else s.run[0] + (" " + " ".join(s.run[1:]) if len(s.run) > 1 else "")
        what = s.what + (f" ({s.note})" if s.note else "")
        lines.append(f"| {s.order + 1} | `{s.key}` | {s.group} | {what}<br>`{cmd}` | `{s.writes}` | {needs} | {after} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("plan", "rebuild"):
        p = sub.add_parser(name)
        p.add_argument("--fetch", action="store_true", help="include the network steps (AERONET pulls, MAIA catalogue)")
        p.add_argument("--checks", action="store_true", help="finish with the check steps (chart index, typecheck, tests)")
        p.add_argument("--only", nargs="+", metavar="STEP", help="run just these steps")
        p.add_argument("--from", dest="from_step", choices=sorted(BY_KEY), help="start at this step")
        p.add_argument("--skip", nargs="+", default=[], metavar="STEP")
        p.add_argument("--dry-run", action="store_true", help="show what would run")
    sub.add_parser("check", help="chart index, typecheck, node and python tests, public-build check, sizes")
    sub.add_parser("check-public", help="private sections stay out of a public build and the publish archive")
    sub.add_parser("sizes", help="data per folder, with and without the private sections")
    md = sub.add_parser("markdown", help="print the step table for docs/gallery-data-pipeline.md")
    md.add_argument("--write", action="store_true", help="write it into the doc between its markers")
    args = ap.parse_args(argv)
    if args.cmd == "markdown":
        if args.write:
            write_doc()
            print(f"wrote the step table into {DOC.relative_to(ROOT)}")
        else:
            print(markdown(), end="")
        return 0
    if args.cmd == "sizes":
        return sizes()
    if args.cmd == "check-public":
        return check_public()
    if args.cmd == "check":
        code = run([s for s in STEPS if s.group == "check"], dry=False)
        return max(code, check_public(), sizes())
    if args.cmd in (None, "plan"):
        if args.cmd is None:
            args = ap.parse_args(["plan"])
        plan(select(args))
        return 0
    return run(select(args), dry=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
