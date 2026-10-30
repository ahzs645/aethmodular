"""Build the dashboard and its review pages as a static Sites bundle.

Run from any directory: python gallery/site/build.py [--out /absolute/output]
This prepares files only; publication uses the Sites workflow separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP = ROOT / 'gallery' / 'app'
EVIDENCE = ROOT / 'deliverables' / 'dashboard_superset'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'dist')
    args = parser.parse_args()
    out = args.out.resolve()
    if out in (ROOT, HERE, APP) or out in ROOT.parents:
        raise SystemExit('Choose a separate build-output directory.')
    vite = APP / 'node_modules' / 'vite' / 'bin' / 'vite.js'
    if not vite.is_file():
        raise SystemExit('Install gallery/app dependencies with npm ci first.')
    out.mkdir(parents=True, exist_ok=True)
    dashboard = out / 'dashboard'
    subprocess.run(
        ['node', str(vite), 'build', '--base', '/dashboard/', '--outDir', str(dashboard), '--emptyOutDir'],
        cwd=APP, env={**os.environ, 'VITE_PUBLIC_BUILD': '1'}, check=True,
    )

    # Apply the same data scrubbing and section exclusions as the existing publisher.
    sys.path.insert(0, str(ROOT / 'gallery' / 'share'))
    from publish_pages import pack_data
    with tempfile.TemporaryDirectory(prefix='aeth-site-') as temp:
        archive = Path(temp) / 'data.tar.gz'
        digest, file_count, scrubbed = pack_data(archive)
        shutil.rmtree(dashboard / 'data')
        (dashboard / 'data').mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(dashboard / 'data', filter='data')

    for name in ('index.html', 'site.css'):
        shutil.copyfile(HERE / name, out / name)
    review = out / 'review'
    review.mkdir(exist_ok=True)
    shutil.copytree(EVIDENCE / 'screenshots', review / 'screenshots', dirs_exist_ok=True)
    for name in (
        'README.md', 'browser-results.json', 'performance.json', 'browser-tests.txt',
        'public-build-check.txt', 'focused-tests.txt', 'full-unit-tests.txt', 'original-fixture-failures.txt',
    ):
        shutil.copyfile(EVIDENCE / name, review / name)
    shutil.copyfile(EVIDENCE.parent / 'dashboard_superset_review.zip', review / 'review-package.zip')
    shutil.copyfile(HERE / 'report.html', review / 'report.html')
    gallery = (EVIDENCE / 'index.html').read_text()
    gallery = gallery.replace(
        '<main><h1>',
        '<main><nav style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:24px">'
        '<a href="/">← Overview</a><a href="/dashboard/#tab=dashboard&amp;net=focus">Open dashboard →</a>'
        '<a href="report.html">Verification report</a>'
        '<a href="review-package.zip" download>Download package</a></nav><h1>',
    ).replace('href="README.md"', 'href="report.html"')
    (review / 'index.html').write_text(gallery)
    entry = dashboard / 'index.html'
    document = entry.read_text().replace(
        '<title>Air Gallery</title>', '<title>Aethmodular — Interactive Dashboard</title>',
    ).replace(
        '<div id="root"></div>',
        '<script>if (!location.hash) history.replaceState(null, "", location.pathname + location.search '
        '+ "#tab=dashboard&net=focus");</script>\n<div id="root"></div>',
    )
    entry.write_text(document)
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    (out / 'build-provenance.json').write_text(json.dumps({
        'source_repository': 'https://github.com/ahzs645/aethmodular', 'source_revision': sha,
        'public_data_sha256': digest, 'public_data_files': file_count, 'scrubbed_files': scrubbed,
    }, indent=2) + '\n')
    print(f'Static Site ready: {out}; {file_count} public data files; {scrubbed} scrubbed.')


if __name__ == '__main__':
    main()
