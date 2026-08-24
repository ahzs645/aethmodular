"""Stage a raw AQRC SPARTAN pull into the canonical Drive layout.

The PowerShell pull (`get_spartan_spectra.ps1`) drops CSVs into ~/Downloads with
inconsistent naming: sometimes lowercase (`etbi_filters.csv`), sometimes several
sites in one folder named after whichever site was pulled first. This normalises
them to one folder per site under

    Davis Data/DAVIS/SPARTAN FTIR pulls/<SITE>/
        <SITE>_filters.csv          one row per filter (MediaId, ExternalFilterId,
                                    sampling dates, ExternalLotId, volume)
        <SITE>_ftir_analysis.csv    one row per FTIR analysis (MediaId ->
                                    SampleScanId / BackgroundScanId)
        <SITE>_scans_base64.csv     interferogram blobs, little-endian float32
                                    single-beam intensity (see build_spartan_target.py)
        <SITE>_site.csv             site record (lat/lon/dates), when pulled

Files are copied **byte-for-byte** — no re-encoding — and a MANIFEST.md records
provenance, row counts and sha256 so a later pull can be diffed against this one.
Existing files are only overwritten with --force, and never silently.

Usage:
    python stage_spartan_pull.py ~/Downloads/CHTS_ftir_analysis ~/Downloads/etbi_site
    python stage_spartan_pull.py ~/Downloads/etbi_site --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
from data_paths import maia_data_root  # noqa: E402

KINDS = ("filters", "ftir_analysis", "scans_base64", "site")
NAME = re.compile(r"^(?P<site>[A-Za-z]{4})_(?P<kind>%s)\.csv$" % "|".join(KINDS))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rows(p: Path) -> int:
    """Data rows (total lines minus the header). Good enough for a manifest."""
    with p.open("rb") as fh:
        n = sum(1 for _ in fh)
    return max(0, n - 1)


def discover(src: Path) -> dict[str, dict[str, Path]]:
    """Group a pull directory's CSVs by site code, case-insensitively."""
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in sorted(src.glob("*.csv")):
        m = NAME.match(p.name)
        if m:
            found[m.group("site").upper()][m.group("kind")] = p
    return dict(found)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="pull directories from ~/Downloads")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a staged file whose checksum differs")
    args = ap.parse_args()

    dest_root = maia_data_root() / "DAVIS/SPARTAN FTIR pulls"
    print(f"destination: {dest_root}\n")

    staged: dict[str, dict] = {}
    for s in args.sources:
        src = Path(s).expanduser()
        if not src.is_dir():
            raise SystemExit(f"not a directory: {src}")
        sites = discover(src)
        if not sites:
            print(f"!! no <SITE>_<kind>.csv files in {src}")
            continue
        print(f"{src.name}: {', '.join(sorted(sites))}")
        for site, files in sorted(sites.items()):
            missing = [k for k in ("filters", "ftir_analysis", "scans_base64")
                       if k not in files]
            d = dest_root / site
            info = {"source": str(src), "files": {}, "missing": missing}
            for kind, p in sorted(files.items()):
                target = d / f"{site}_{kind}.csv"
                digest, n = sha256(p), rows(p)
                info["files"][f"{site}_{kind}.csv"] = {"rows": n, "sha256": digest}
                if target.exists() and sha256(target) == digest:
                    state = "already staged"
                elif target.exists() and not args.force:
                    state = "DIFFERS - kept existing (use --force)"
                else:
                    state = "copy"
                    if not args.dry_run:
                        d.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, target)
                print(f"  {site + '_' + kind + '.csv':<26} {n:>6,d} rows  {state}"
                      + (f"  <- {p.name}" if p.name != target.name else ""))
            if missing:
                print(f"  !! {site} incomplete, missing: {', '.join(missing)}")
            staged[site] = info

    if args.dry_run or not staged:
        print("\n(dry run - nothing written)" if args.dry_run else "")
        return

    lines = ["# SPARTAN FTIR pulls - staged from the AQRC database", "",
             "Raw exports from `Networks_1_0` via `scripts/get_spartan_spectra.ps1`,",
             "copied byte-for-byte from the original pull. Build a calibration_explorer",
             "target from one with:", "",
             "```", "python build_spartan_target.py CHTS --src '<this folder>/CHTS' --name beijing",
             "```", "",
             "| site | file | rows | sha256 (first 16) |", "|---|---|---|---|"]
    for site in sorted(staged):
        for fn, meta in sorted(staged[site]["files"].items()):
            lines.append(f"| {site} | `{fn}` | {meta['rows']:,} | `{meta['sha256'][:16]}` |")
    lines += ["", "Incomplete sites: "
              + (", ".join(f"{s} (missing {', '.join(v['missing'])})"
                           for s, v in sorted(staged.items()) if v["missing"]) or "none"), ""]
    man = dest_root / "MANIFEST.md"
    man.write_text("\n".join(lines))
    print(f"\nwrote {man}")


if __name__ == "__main__":
    main()
