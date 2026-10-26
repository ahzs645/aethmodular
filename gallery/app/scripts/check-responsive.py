#!/usr/bin/env python3
"""Check gallery layout breakpoints against a running dev/preview server.

Requires Python Playwright and Chromium. Example:
  python scripts/check-responsive.py --base-url http://localhost:5178
  python scripts/check-responsive.py --screenshots /tmp/gallery-layout

Uses the shipped focus-site records, map summaries and calibration data. Dense
plots may scroll inside their own frames; the page and controls must fit.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
from playwright.sync_api import sync_playwright

WIDTHS = [320, 360, 375, 390, 414, 480, 640, 768, 820, 1024, 1280, 1440, 1920, 2560]
VIEWS = [
    ('distribution', '', 'distribution'),
    ('correlation', '', 'correlogram'),
    ('correlation', 'correlogram.encoding=Scatter+matrix', 'correlogram'),
    ('correlation', 'corrp=all', 'correlogram'),
    ('ranking', '', 'parallel'),
    ('evolution', 'clock.layout=Rings&clock.seasons=1', 'seasonal-clock'),
    ('partOfWhole', 'composition.bars=by+season', 'composition'),
    ('pmf', 'sourcestack.shape=Daily+bars', 'sourcestack'),
    ('map', 'mapscope=All+IMPROVE+sites', 'map'),
    ('availability', '', 'data-availability'),
    ('availability', 'availability.view=Study+overlap', 'availability-study'),
    ('availability', 'availability.view=Study+overlap&availability.completeness=At+least+80%25', 'availability-study'),
    ('availability', 'availability.view=Filter+explorer', 'availability-filter-inventory'),
    ('availability', 'availability.view=Filter+explorer&availability.network=IMPROVE', 'availability-filter-inventory'),
    ('availability', 'availability.view=Prior+comparison', 'availability-prior-comparison'),
    ('calibration', 'spec-curve.dots=custom', 'slope-trap'),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://localhost:5178')
    parser.add_argument('--screenshots', type=Path)
    args = parser.parse_args()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    with sync_playwright() as playwright:
        executable = shutil.which('chromium') or shutil.which('chromium-browser')
        browser = playwright.chromium.launch(**({'executable_path': executable} if executable else {}))
        page = browser.new_page(viewport={'width': 1280, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        for tab, query, anchor in VIEWS:
            print(f'Checking {tab}: {query or "default"}', file=sys.stderr, flush=True)
            page.goto('about:blank')
            page.goto(f'{args.base_url.rstrip("/")}/?{query}#tab={tab}&net=focus', wait_until='domcontentloaded')
            chart = page.locator(f'#{anchor}')
            chart.wait_for()
            page.wait_for_timeout(1200)
            if tab == 'distribution':
                chart.get_by_role('radio', name='Season', exact=True).click()
            for width in WIDTHS:
                page.set_viewport_size({'width': width, 'height': 1000})
                page.wait_for_timeout(180)
                result = page.evaluate('''() => ({
                  width: innerWidth, pageWidth: document.documentElement.scrollWidth,
                  overflowingControls: [...document.querySelectorAll('.frame-controls')]
                    .flatMap(e => [...e.children])
                    .filter(e => e.getBoundingClientRect().right > innerWidth + 2)
                    .map(e => e.textContent.slice(0, 80)),
                  invalidSvg: [...document.querySelectorAll('svg [x], svg [y], svg path')]
                    .filter(e => /NaN|Infinity/.test((e.getAttribute('d') || '') +
                      (e.getAttribute('x') || '') + (e.getAttribute('y') || ''))).length,
                  movingLegends: [...document.querySelectorAll('.chart-wrap')].flatMap(w => {
                    if (w.closest('.legend-right') || w.scrollWidth <= w.clientWidth + 1) return [];
                    const legends = [...w.querySelectorAll(':scope > .legend')];
                    const start = w.scrollLeft, before = legends.map(l => l.getBoundingClientRect().x);
                    w.scrollLeft = w.scrollWidth - w.clientWidth;
                    const moved = legends.filter((l, i) => Math.abs(l.getBoundingClientRect().x - before[i]) > 1)
                      .map(l => l.closest('.panel')?.id || l.textContent.slice(0, 60));
                    w.scrollLeft = start;
                    return moved;
                  })
                })''')
                checks.append({'view': tab, 'query': query, **result})
                if width in (320, 768, 2560):
                    print(f'  {width}px', file=sys.stderr, flush=True)
                if args.screenshots and width in (375, 1280):
                    tag = f'{tab}-{VIEWS.index((tab, query, anchor))}'
                    chart.screenshot(path=str((args.screenshots / f'{tag}-{width}.png').resolve()), animations='disabled')
        browser.close()
    failures = [c for c in checks if c['pageWidth'] > c['width'] + 1 or c['overflowingControls'] or c['invalidSvg'] or c['movingLegends']]
    print(json.dumps({'cases': len(checks), 'widths': WIDTHS, 'failures': failures, 'browserErrors': errors}, indent=2))
    return bool(failures or errors)

if __name__ == '__main__':
    raise SystemExit(main())
