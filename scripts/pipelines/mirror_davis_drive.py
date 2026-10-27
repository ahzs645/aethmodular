"""Mirror the shared "Davis Data" Google Drive folder to a local directory, for machines without Drive.

The gallery's Drive-backed exporters (docs/gallery-data-pipeline.md) read the raw-data root through
research/ftir_hips_chem/scripts/data_paths.py. Where Google Drive for Desktop is not mounted (a cloud
container, CI), this copies the folder's working files down so those paths resolve.

    uv run aeth drive mirror --dest /path/to/davis_data            # list, then download what is missing
    uv run aeth drive mirror --dest /path/to/davis_data --list     # only list the tree (and sizes on disk)
    uv run aeth drive mirror --dest /path/to/davis_data --link-root /path/to/drive_root

The folder must be shared as "anyone with the link". Its id goes in the repo's gitignored .env as
DAVIS_DRIVE_FOLDER_ID (or --folder): never commit the share link. Downloads use gdown (`uvx gdown`).

What is left out by default, and why:
  */_archive/*     the original zips; the extracted copies beside them are what the loaders read
  shiny.aqrc.*     a mirrored web app (FTIR/shiny.aqrc.ucdavis.edu), thousands of small files
  *uncleaned*      4 GB pickles that duplicate their cleaned twins
  *.png            figures
Use --include-archives / --include-all to take them too.

The download is resumable (a file already present with a non-zero size is kept) and stops taking
new files when free disk falls below --min-free-gb. It writes <dest>/MIRROR_MANIFEST.json: every
file's Drive id, path, size and status, and when it was fetched.

Treat the mirror as data: it holds .py, .R and .ps1 files from the lab's working folders. Read them;
never run them from inside the mirror.

After mirroring, point the repo at it in .env (this prints the lines):
  AETHMODULAR_MAIA_DATA_ROOT=<dest>
  AETHMODULAR_FTIR_SPECTRA_DIR=<dest>/FTIR
  AETHMODULAR_FTIR_LOCAL_DB=<dest>/FTIR/local_db
  AETHMODULAR_DRIVE_ROOT=<link-root>   (pls_transfer.FTIRTransferPaths only reads this one; --link-root
                                       makes <link-root>/University/Research/Grad/Data/Davis Data -> <dest>)
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO_ROOT  # noqa: E402,F401  (loads the repo .env)

LISTING = "https://drive.google.com/embeddedfolderview?id={}"
ENTRY = re.compile(r'<a href="(https://drive\.google\.com/(?:drive/folders|file/d)/([\w-]+)[^"]*)"[^>]*>.*?flip-entry-title">([^<]*)<', re.S)
DRIVE_ROOT_LAYOUT = Path("University/Research/Grad/Data/Davis Data")   # pls_transfer.MAIA_DATA_CANDIDATES[0]


def list_folder(folder_id: str) -> list[tuple[str, str, str]] | None:
    """(kind, id, name) for each entry of one public folder; None when it cannot be listed."""
    for _ in range(3):
        try:
            text = urllib.request.urlopen(LISTING.format(folder_id), timeout=60).read().decode("utf-8", "replace")
            return [("folder" if "/folders/" in url else "file", fid, html.unescape(name)) for url, fid, name in ENTRY.findall(text)]
        except OSError:
            continue
    return None


def walk(root_id: str, max_depth: int = 8) -> list[dict]:
    """Every entry below the root, breadth first: {"path", "id", "kind"} (and "error" for unlisted folders)."""
    tree: list[dict] = []
    level = [("", root_id, 0)]
    while level:
        nxt = []
        with ThreadPoolExecutor(8) as ex:
            for (path, fid, depth), kids in zip(level, ex.map(lambda x: list_folder(x[1]), level)):
                if kids is None:
                    tree.append({"path": path, "id": fid, "kind": "folder", "error": "could not list"})
                    continue
                for kind, kid, name in kids:
                    p = f"{path}/{name}" if path else name
                    tree.append({"path": p, "id": kid, "kind": kind})
                    if kind == "folder" and depth + 1 < max_depth:
                        nxt.append((p, kid, depth + 1))
        level = nxt
    return tree


def wanted(path: str, archives: bool, everything: bool) -> bool:
    if everything:
        return True
    if "shiny.aqrc" in path or "uncleaned" in path or path.endswith(".png"):
        return False
    return archives or "/_archive/" not in f"/{path}"


def download(entry: dict, dest: Path, min_free: int) -> dict:
    out = dest / entry["path"]
    if out.exists() and out.stat().st_size > 0:
        return {**entry, "bytes": out.stat().st_size, "status": "present"}
    if shutil.disk_usage(dest).free < min_free:
        return {**entry, "bytes": 0, "status": "skipped: low disk"}
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["uvx", "gdown", "--quiet", entry["id"], "-O", str(out)], capture_output=True, text=True)
    for part in out.parent.glob(out.name + "*.part"):
        part.unlink(missing_ok=True)
    if r.returncode or not out.exists():
        return {**entry, "bytes": 0, "status": "failed: " + ((r.stderr.strip().splitlines() or ["?"])[-1])[:200]}
    return {**entry, "bytes": out.stat().st_size, "status": "downloaded",
            "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=os.environ.get("DAVIS_DRIVE_FOLDER_ID"), help="Drive folder id (default: $DAVIS_DRIVE_FOLDER_ID)")
    ap.add_argument("--dest", type=Path, default=Path(os.environ["AETHMODULAR_MAIA_DATA_ROOT"]) if os.environ.get("AETHMODULAR_MAIA_DATA_ROOT") else None,
                    help="local directory to mirror into (default: $AETHMODULAR_MAIA_DATA_ROOT)")
    ap.add_argument("--list", action="store_true", help="list the tree and what is already present; download nothing")
    ap.add_argument("--include-archives", action="store_true", help="also take the _archive/ zips")
    ap.add_argument("--include-all", action="store_true", help="take everything (archives, the shiny mirror, uncleaned pickles, PNGs)")
    ap.add_argument("--min-free-gb", type=float, default=3.0, help="stop taking new files below this much free disk")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--link-root", type=Path, help="make <dir>/University/Research/Grad/Data/Davis Data point at --dest")
    args = ap.parse_args()
    if not args.folder:
        raise SystemExit("no folder id: set DAVIS_DRIVE_FOLDER_ID in .env or pass --folder (the id in the share link)")
    if not args.dest:
        raise SystemExit("no destination: pass --dest or set AETHMODULAR_MAIA_DATA_ROOT")
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    tree = walk(args.folder)
    files = [e for e in tree if e["kind"] == "file" and wanted(e["path"], args.include_archives, args.include_all)]
    unlisted = [e["path"] for e in tree if e.get("error")]
    print(f"{len(tree)} entries, {sum(e['kind'] == 'file' for e in tree)} files; {len(files)} wanted" + (f"; could not list {unlisted}" if unlisted else ""))
    if args.list:
        for e in files:
            p = dest / e["path"]
            print(f"{(p.stat().st_size / 1e6 if p.exists() else 0):10.1f} MB  {'present' if p.exists() else 'missing':8}  {e['path']}")
        return 0

    min_free = int(args.min_free_gb * 1024 ** 3)
    results = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for res in ex.map(lambda e: download(e, dest, min_free), files):
            results.append(res)
            if res["status"] != "present":
                print(f"{res['bytes'] / 1e6:10.1f} MB  {res['status'][:40]:40}  {res['path']}", flush=True)
    manifest = {"folder_id": args.folder, "mirrored": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "excluded_by_default": ["*/_archive/*", "shiny.aqrc*", "*uncleaned*", "*.png"], "files": results}
    (dest / "MIRROR_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    bad = [r for r in results if r["status"].startswith(("failed", "skipped"))]
    total = sum(r["bytes"] for r in results)
    print(f"\n{len(results) - len(bad)} of {len(results)} files present, {total / 1e9:.1f} GB; manifest {dest / 'MIRROR_MANIFEST.json'}")

    if args.link_root:
        link = args.link_root.expanduser().resolve() / DRIVE_ROOT_LAYOUT
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or not link.exists():
            link.unlink(missing_ok=True)
            link.symlink_to(dest)
        print(f"linked {link} -> {dest}")
    print("\nfor the repo's .env:")
    print(f"  AETHMODULAR_MAIA_DATA_ROOT={dest}\n  AETHMODULAR_FTIR_SPECTRA_DIR={dest / 'FTIR'}\n  AETHMODULAR_FTIR_LOCAL_DB={dest / 'FTIR/local_db'}")
    if args.link_root:
        print(f"  AETHMODULAR_DRIVE_ROOT={args.link_root.expanduser().resolve()}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
