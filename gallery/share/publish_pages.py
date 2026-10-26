"""Publish the full gallery as a static GitHub Pages site from its own public repo.

Default (release): nothing is built or copied on this machine. The script packs
gallery/app/public/data into one scrubbed archive (machine paths and account names removed;
it refuses if any remain), attaches it to a release of the gallery repo (tag data-<hash>,
skipped when that exact data is already there), installs the deploy workflow
(gallery/share/pages-deploy.yml), and starts it: GitHub checks out this repo's gallery/app at
the pushed commit, unpacks the data, builds and deploys Pages. The gallery source must be
committed and pushed first, since GitHub builds from GitHub.

--branch (the earlier way, kept as a fallback): build here, copy into a local clone of the
gallery repo (CLONE) and push the built files; Pages then serves its main branch.

Run from the repo root:
    python3 gallery/share/publish_pages.py                  # data release + GitHub build and deploy
    python3 gallery/share/publish_pages.py --dry-run        # pack and check the data only
    python3 gallery/share/publish_pages.py --branch         # build here and push the built files

Defaults: repo ahzs645/aethmodular-gallery, local clone ../aethmodular-gallery,
page https://projects.ahmadjalil.com/aethmodular-gallery/ (the account's custom domain;
ahzs645.github.io/aethmodular-gallery redirects there). The first run creates the repo
and turns on Pages.
"""

from __future__ import annotations

import argparse
import base64
import calendar
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
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
def forbidden_pattern() -> re.Pattern:
    """Machine paths, Drive mounts, GitHub tokens, and this machine's own account names.

    The account names (login and git email) are read at run time so they never appear in
    this public script. A bare "@gmail.com" is not blocked: third-party licence headers in
    the bundle (e.g. lzma's author address) legitimately contain one.
    """
    import getpass
    login = getpass.getuser()
    # Build-service logins also occur in ordinary browser/library identifiers.
    terms = set() if login in {"agent", "root", "runner", "ubuntu", "codex"} else {login}
    email = subprocess.run(["git", "config", "user.email"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if email:
        terms |= {email, email.split("@")[0]}
    names = "|".join(re.escape(t) for t in sorted(terms) if len(t) >= 4)
    return re.compile(r"/Users/|GoogleDrive-|gh[op]_[A-Za-z0-9]{20,}" +
                      (rf"|(?<!\w)(?:{names})(?!\w)" if names else ""), re.IGNORECASE)


FORBIDDEN = forbidden_pattern()

README = """# Aethmodular graph gallery

Static build of the interactive gallery in [ahzs645/aethmodular](https://github.com/ahzs645/aethmodular)
(`gallery/`), published with `gallery/share/publish_pages.py`. This repo holds only the built
site; change the source repo, not these files.

**Live page:** https://projects.ahmadjalil.com/aethmodular-gallery/

Built {built} from aethmodular `{sha}`.

## Data

Filter chemistry and FTIR/HIPS work on SPARTAN sites (Addis Ababa, Beijing, Delhi, Pasadena/JPL) and
the IMPROVE network. Several tabs (Meeting, Calibration, Baselines, Spectral similarity, Blanks) show
work in progress on FTIR elemental-carbon calibration for Addis Ababa; numbers there are preliminary.

- IMPROVE data: please cite Malm et al. (1994), J. Geophys. Res. 99, 1347-1370, and see the IMPROVE
  data advisories (http://vista.cira.colostate.edu/IMPROVE/Data/QA_QC/Advisory.htm).
- SPARTAN data: see https://www.spartan-network.org for data use and citation.
"""


# ---------------------------------------------------------------- release (default)
WORKFLOW = Path(__file__).with_name("pages-deploy.yml")
WORKFLOW_PATH = ".github/workflows/deploy.yml"
DATA_ASSET = "gallery-data.tar.gz"
DATA = APP / "public/data"


def is_text_asset(path: Path) -> bool:
    return path.suffix in TEXT or (path.suffix == '.gz' and path.with_suffix('').suffix in TEXT)


def asset_text(path: Path, body: bytes) -> str:
    return (gzip.decompress(body) if path.suffix == '.gz' else body).decode('utf-8', errors='surrogateescape')


def encode_text_asset(path: Path, text: str) -> bytes:
    body = text.encode('utf-8', errors='surrogateescape')
    return gzip.compress(body, mtime=0) if path.suffix == '.gz' else body

README_RELEASE = """# Aethmodular graph gallery

GitHub Pages site for the interactive gallery in [ahzs645/aethmodular](https://github.com/ahzs645/aethmodular)
(`gallery/`). **Live page:** https://projects.ahmadjalil.com/aethmodular-gallery/

Latest successful publication: [aethmodular `{sha}`](https://github.com/ahzs645/aethmodular/commit/{sha})
with [data snapshot `{tag}`](https://github.com/ahzs645/aethmodular-gallery/releases/tag/{tag}).
[Deployment run](https://github.com/ahzs645/aethmodular-gallery/actions/runs/{run_id}).

Published with `gallery/share/publish_pages.py`: the data (`gallery/app/public/data`, machine paths
scrubbed) is a release of this repo (`data-<hash>`, asset `gallery-data.tar.gz`), and
`.github/workflows/deploy.yml` builds the site from the source repo at a pushed commit and deploys it.
The files on `main` from before October 2026 are an earlier static build, kept as a fallback
(`publish_pages.py --branch`); Pages no longer serves them.

## Data

Filter chemistry and FTIR/HIPS work on SPARTAN sites and the IMPROVE network. Several tabs (Meeting,
Calibration, Baselines, Spectral similarity, Blanks) show work in progress on FTIR elemental-carbon
calibration for Addis Ababa; numbers there are preliminary.

- IMPROVE data: please cite Malm et al. (1994), J. Geophys. Res. 99, 1347-1370, and see the IMPROVE
  data advisories (http://vista.cira.colostate.edu/IMPROVE/Data/QA_QC/Advisory.htm).
- SPARTAN data: see https://www.spartan-network.org for data use and citation.
"""


def run(*cmd, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, check=check)


def data_files(root: Path):
    """Include linked export folders; reject broken links and directory cycles."""
    def walk(folder, ancestors):
        resolved = folder.resolve()
        if resolved in ancestors:
            raise SystemExit(f"refusing to publish: directory cycle at {folder.relative_to(root)}")
        for path in sorted(folder.iterdir()):
            if path.name == '.DS_Store':
                continue
            if path.is_symlink() and not path.exists():
                raise SystemExit(f"refusing to publish: broken data link at {path.relative_to(root)}")
            if path.is_dir():
                yield from walk(path, ancestors | {resolved})
            elif path.is_file():
                yield path

    yield from walk(root, set())


def pack_data(dest: Path) -> tuple[str, int, int]:
    """public/data as a deterministic, scrubbed tar.gz (same data, same bytes); sha256, files, scrubbed."""
    files = sorted(data_files(DATA))
    left, scrubbed = [], 0
    with dest.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in files:
            rel = path.relative_to(DATA).as_posix()
            body = path.read_bytes()
            if is_text_asset(path):
                text = asset_text(path, body)
                new = text
                for pattern, repl in SCRUB:
                    new = pattern.sub(repl, new)
                if new != text:
                    scrubbed += 1
                    body = encode_text_asset(path, new)
                if FORBIDDEN.search(new):
                    left.append(rel)
            info = tarfile.TarInfo(rel)
            info.size, info.mtime, info.mode = len(body), 0, 0o644
            tar.addfile(info, io.BytesIO(body))
    if left:
        raise SystemExit(f"refusing to publish: machine paths or account names remain in {left}")
    return hashlib.sha256(dest.read_bytes()).hexdigest(), len(files), scrubbed


def put_file(path: str, content: str, message: str) -> bool:
    """Create or update one file on the gallery repo's main branch; True when it changed."""
    cur = run("gh", "api", f"repos/{REPO}/contents/{path}", check=False)
    sha = None
    if cur.returncode == 0:
        meta = json.loads(cur.stdout)
        if base64.b64decode(meta["content"]).decode() == content:
            return False
        sha = meta["sha"]
    args = ["gh", "api", "-X", "PUT", f"repos/{REPO}/contents/{path}", "-f", f"message={message}",
            "-f", f"content={base64.b64encode(content.encode()).decode()}", "-f", "branch=main"]
    run(*args, *(["-f", f"sha={sha}"] if sha else []))
    return True


def publish_release(dry_run: bool) -> None:
    run("git", "fetch", "-q", "origin", "main")
    sha = run("git", "rev-parse", "HEAD").stdout.strip()
    # GitHub builds the committed source, so what is drawn must be what is pushed
    dirty = run("git", "status", "--porcelain", "--", "gallery/app", ":!gallery/app/public/data").stdout.strip()
    if dirty:
        raise SystemExit(f"commit the gallery source first (GitHub builds the pushed commit):\n{dirty}")
    if run("git", "merge-base", "--is-ancestor", "HEAD", "origin/main", check=False).returncode != 0:
        raise SystemExit(f"push first: {sha[:7]} is not on origin/main, and GitHub builds from GitHub")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / DATA_ASSET
        digest, n, scrubbed = pack_data(archive)
        tag = f"data-{digest[:12]}"
        size = archive.stat().st_size
        print(f"data: {n} files, {scrubbed} scrubbed, {size / 1e6:.0f} MB packed, {tag}")
        if dry_run:
            return
        if run("gh", "release", "view", tag, "--repo", REPO, check=False).returncode == 0:
            print(f"{tag} is already a release: the data is unchanged, nothing uploaded")
        else:
            run("gh", "release", "create", tag, str(archive), "--repo", REPO, "--target", "main",
                "--title", f"Gallery data {digest[:12]}",
                "--notes", f"gallery/app/public/data from ahzs645/aethmodular {sha[:7]}, machine paths scrubbed. "
                           f"sha256 {digest}. Deployed by {WORKFLOW_PATH}.")
            print(f"uploaded {DATA_ASSET} to release {tag}")
    if put_file(WORKFLOW_PATH, WORKFLOW.read_text(), "Deploy workflow from aethmodular gallery/share/pages-deploy.yml"):
        print(f"installed {WORKFLOW_PATH}")
        time.sleep(5)  # let GitHub register the workflow before it is started
    pages = json.loads(run("gh", "api", f"repos/{REPO}/pages").stdout)
    switched = pages.get("build_type") != "workflow"
    if switched:
        run("gh", "api", "-X", "PUT", f"repos/{REPO}/pages", "-f", "build_type=workflow")
        print("Pages now deploys from the workflow (was: the main branch)")
    started = time.time()
    run("gh", "workflow", "run", "deploy.yml", "--repo", REPO, "--ref", "main", "-f", f"source_sha={sha}", "-f", f"data_tag={tag}")
    run_id = None
    for _ in range(30):  # the dispatched run appears after a moment
        time.sleep(3)
        runs = json.loads(run("gh", "run", "list", "--repo", REPO, "--workflow", "deploy.yml", "--limit", "5",
                              "--json", "databaseId,createdAt,event").stdout)
        fresh = [r for r in runs if r["event"] == "workflow_dispatch"
                 and calendar.timegm(time.strptime(r["createdAt"], "%Y-%m-%dT%H:%M:%SZ")) >= started - 60]
        if fresh:
            run_id = fresh[0]["databaseId"]
            break
    if run_id is None:
        raise SystemExit(f"started the deploy but could not find its run: see https://github.com/{REPO}/actions")
    print(f"building on GitHub: https://github.com/{REPO}/actions/runs/{run_id}")
    ok = subprocess.run(["gh", "run", "watch", str(run_id), "--repo", REPO, "--exit-status", "--interval", "10"],
                        stdout=subprocess.DEVNULL).returncode == 0
    if not ok:
        if switched:  # leave the site as it was: back to serving the branch
            run("gh", "api", "-X", "PUT", f"repos/{REPO}/pages", "-f", "build_type=legacy", "-f", "source[branch]=main", "-f", "source[path]=/")
            print("deploy failed: Pages is back on the main branch")
        raise SystemExit(f"deploy failed: https://github.com/{REPO}/actions/runs/{run_id}")
    put_file("README.md", README_RELEASE.format(sha=sha, tag=tag, run_id=run_id),
             f"Record successful gallery deployment from aethmodular {sha[:7]}")
    url = run("gh", "api", f"repos/{REPO}/pages", "-q", ".html_url").stdout.strip()
    print(f"deployed {sha[:7]} with {tag}: {url}")


# ---------------------------------------------------------------- branch (fallback)
def build():
    subprocess.run(["npx", "vite", "build", "--base", "./", "--outDir", str(BUILD), "--emptyOutDir"],
                   cwd=APP, env={**{k: v for k, v in os.environ.items() if k != "VITE_SHARE_TABS"}, "VITE_HIDE_EXPLAIN": "1"}, check=True)
    changed = 0
    for path in BUILD.rglob("*"):
        if path.is_file() and is_text_asset(path):
            text = asset_text(path, path.read_bytes())
            new = text
            for pattern, repl in SCRUB:
                new = pattern.sub(repl, new)
            if new != text:
                path.write_bytes(encode_text_asset(path, new))
                changed += 1
    left = [str(p.relative_to(BUILD)) for p in BUILD.rglob("*")
            if p.is_file() and is_text_asset(p) and FORBIDDEN.search(asset_text(p, p.read_bytes()))]
    if left:
        raise SystemExit(f"refusing to publish: machine paths or account names remain in {left}")
    (BUILD / ".nojekyll").write_text("")
    (BUILD / "build-info.json").write_text(json.dumps({
        "source_repository": "ahzs645/aethmodular",
        "source_sha": run("git", "rev-parse", "HEAD").stdout.strip(),
        "publication": "static branch",
    }, indent=2) + "\n")
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
    else:
        dirty_clone = run("git", "status", "--porcelain", cwd=CLONE).stdout.strip()
        if dirty_clone:
            raise SystemExit(f"gallery clone has uncommitted changes:\n{dirty_clone}")
        run("git", "fetch", "origin", "main", cwd=CLONE)
        run("git", "merge", "--ff-only", "origin/main", cwd=CLONE)
    for item in CLONE.iterdir():
        if item.name not in {".git", ".github"}:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    shutil.copytree(BUILD, CLONE, dirs_exist_ok=True)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (CLONE / "README.md").write_text(README.format(built=built, sha=sha + (" + uncommitted gallery changes" if dirty else "")))
    subprocess.run(["git", "add", "-A"], cwd=CLONE, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=CLONE).returncode == 0:
        print("nothing changed since the last publish")
        return
    subprocess.run(["git", "commit", "-q", "-m", f"Build from aethmodular {sha}"], cwd=CLONE, check=True)
    subprocess.run(["git", "push", "-u", "origin", "HEAD:main"], cwd=CLONE, check=True)
    pages = subprocess.run(["gh", "api", f"repos/{REPO}/pages"], capture_output=True)
    if pages.returncode != 0:
        subprocess.run(["gh", "api", "-X", "POST", f"repos/{REPO}/pages", "-f", "source[branch]=main", "-f", "source[path]=/"], check=True)
    else:
        settings = json.loads(pages.stdout)
        if settings.get("build_type") != "legacy" or settings.get("source") != {"branch": "main", "path": "/"}:
            run("gh", "api", "-X", "PUT", f"repos/{REPO}/pages", "-f", "build_type=legacy",
                "-f", "source[branch]=main", "-f", "source[path]=/")
    # an account-level custom domain (projects.ahmadjalil.com) replaces github.io; ask Pages for the real address
    url = subprocess.run(["gh", "api", f"repos/{REPO}/pages", "-q", ".html_url"], capture_output=True, text=True).stdout.strip()
    print(f"pushed; page: {url or 'https://ahzs645.github.io/aethmodular-gallery/'} (rebuilds in about a minute)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="pack (or with --branch, build) and check; do not upload or push")
    ap.add_argument("--branch", action="store_true", help="the earlier way: build here and push the built files to the gallery repo")
    args = ap.parse_args()
    if not args.branch:
        publish_release(args.dry_run)
        return
    build()
    if not args.dry_run:
        publish()


if __name__ == "__main__":
    main()
