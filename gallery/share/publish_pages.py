"""Publish the full gallery as a static GitHub Pages site in its own public repo.

Builds gallery/app with relative paths, copies every export in public/data, scrubs
machine paths and account names from the published copy (the sources are untouched),
refuses to continue if any remain, and pushes the result to a separate repo that
holds only the built site. Pages serves that repo's main branch.

Run from the repo root:
    python3 gallery/share/publish_pages.py                  # build, check, commit and push
    python3 gallery/share/publish_pages.py --dry-run        # build and check only

Defaults: repo ahzs645/aethmodular-gallery, local clone ../aethmodular-gallery,
page https://ahzs645.github.io/aethmodular-gallery/. The first run creates the repo
and turns on Pages.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "gallery/app"
BUILD = ROOT / "gallery/share/pages-dist"
REPO = "ahzs645/aethmodular-gallery"
CLONE = ROOT.parent / "aethmodular-gallery"
TEXT = {".json", ".ipynb", ".csv", ".html", ".js", ".css", ".md", ".txt", ".geojson"}

# machine-specific prefixes -> neutral placeholders (published copy only)
SCRUB = [
    (re.compile(r"~?/Users/[^/\"\\\s]+/Library/CloudStorage/GoogleDrive-[^/\"\\\s]+/My Drive/"), "<Drive>/"),
    (re.compile(r"~/Library/CloudStorage/GoogleDrive-[^/\"\\\s]+/My Drive/"), "<Drive>/"),
    (re.compile(r"/Users/[^/\"\\\s]+/github/aethmodular/"), ""),
    (re.compile(r"/Users/[^/\"\\\s]+/"), "~/"),
]
FORBIDDEN = re.compile(r"/Users/|GoogleDrive-|ahmadjalil|@gmail\.com|gh[op]_[A-Za-z0-9]{20,}")

README = """# Aethmodular graph gallery

Static build of the interactive gallery in [ahzs645/aethmodular](https://github.com/ahzs645/aethmodular)
(`gallery/`), published with `gallery/share/publish_pages.py`. This repo holds only the built
site; change the source repo, not these files.

**Live page:** https://ahzs645.github.io/aethmodular-gallery/

Built {built} from aethmodular `{sha}`.

## Data

Filter chemistry and FTIR/HIPS work on SPARTAN sites (Addis Ababa, Beijing, Delhi, Pasadena/JPL) and
the IMPROVE network. Several tabs (Meeting, Calibration, Baselines, Spectral similarity, Blanks) show
work in progress on FTIR elemental-carbon calibration for Addis Ababa; numbers there are preliminary.

- IMPROVE data: please cite Malm et al. (1994), J. Geophys. Res. 99, 1347-1370, and see the IMPROVE
  data advisories (http://vista.cira.colostate.edu/IMPROVE/Data/QA_QC/Advisory.htm).
- SPARTAN data: see https://www.spartan-network.org for data use and citation.
"""


def build():
    subprocess.run(["npx", "vite", "build", "--base", "./", "--outDir", str(BUILD), "--emptyOutDir"],
                   cwd=APP, env={k: v for k, v in os.environ.items() if k != "VITE_SHARE_TABS"}, check=True)
    changed = 0
    for path in BUILD.rglob("*"):
        if path.is_file() and path.suffix in TEXT:
            text = path.read_text(errors="surrogateescape")
            new = text
            for pattern, repl in SCRUB:
                new = pattern.sub(repl, new)
            if new != text:
                path.write_text(new, errors="surrogateescape")
                changed += 1
    left = [str(p.relative_to(BUILD)) for p in BUILD.rglob("*")
            if p.is_file() and p.suffix in TEXT and FORBIDDEN.search(p.read_text(errors="replace"))]
    if left:
        raise SystemExit(f"refusing to publish: machine paths or account names remain in {left}")
    (BUILD / ".nojekyll").write_text("")
    size = sum(p.stat().st_size for p in BUILD.rglob("*") if p.is_file())
    biggest = max((p for p in BUILD.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    if biggest.stat().st_size > 95e6:
        raise SystemExit(f"{biggest} is over GitHub's 100 MB file limit")
    print(f"built {size / 1e6:.0f} MB; scrubbed {changed} files; largest {biggest.name} {biggest.stat().st_size / 1e6:.1f} MB")


def publish():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "gallery/app/src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    exists = subprocess.run(["gh", "repo", "view", REPO], capture_output=True).returncode == 0
    if not exists:
        subprocess.run(["gh", "repo", "create", REPO, "--public", "--description",
                        "Static build of the aethmodular graph gallery (GitHub Pages)"], check=True)
    if not (CLONE / ".git").exists():
        CLONE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=CLONE, check=True)
        subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{REPO}.git"], cwd=CLONE, check=True)
        if exists:
            subprocess.run(["git", "pull", "origin", "main"], cwd=CLONE)
    for item in CLONE.iterdir():
        if item.name != ".git":
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    shutil.copytree(BUILD, CLONE, dirs_exist_ok=True)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (CLONE / "README.md").write_text(README.format(built=built, sha=sha + (" + uncommitted gallery changes" if dirty else "")))
    subprocess.run(["git", "add", "-A"], cwd=CLONE, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=CLONE).returncode == 0:
        print("nothing changed since the last publish")
        return
    subprocess.run(["git", "commit", "-q", "-m", f"Build from aethmodular {sha}"], cwd=CLONE, check=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=CLONE, check=True)
    pages = subprocess.run(["gh", "api", f"repos/{REPO}/pages"], capture_output=True)
    if pages.returncode != 0:
        subprocess.run(["gh", "api", "-X", "POST", f"repos/{REPO}/pages", "-f", "source[branch]=main", "-f", "source[path]=/"], check=True)
    print(f"pushed; page: https://{REPO.split('/')[0]}.github.io/{REPO.split('/')[1]}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build and check, do not commit or push")
    args = ap.parse_args()
    build()
    if not args.dry_run:
        publish()


if __name__ == "__main__":
    main()
