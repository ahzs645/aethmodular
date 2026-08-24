"""Pre-populate the explorer's ./cache/ across the whole setup grid.

Walks every cohort x cutoff-ladder x selection-space x spectra x protocol x lot
combination (target=addis), computing each configuration's CV curve,
(Cohort x lot combos that resolve to <30 filters: e.g. smoke under the minority
lot: fail by design and are just logged.)
the rule-choice fit, and the sparse unattended-search k ladder (including 21 and 30)
- so that afterwards every click in the app resolves from cache in <1 s.

The cutoff axis is genuinely continuous in the UI (the rank-plot click), so we
warm a ladder around each locked cutoff rather than every integer; an off-ladder
cutoff still computes live on first click, then is cached.

Run:  uv run python calibration_explorer/warm_cache.py
(same data mounts as the app; safe to run while the app is up: same cache dir,
same file contents).
"""
from __future__ import annotations

import itertools
import time
import traceback

import app  # noqa: F401: importing app.py starts its background data loader

GRID_CUTOFFS = {
    "pool": [None],
    "smoke": [None],
    "eth_shaped": [200, 250, 300, 350, 400],
    "analogs": [400, 450, 500, 550, 600],
    "ocec": [600, 700, 800, 900, 1000],
}
SPECTRA = ["raw", "airspec", "deriv2"]
MODES = ["site_heldout", "app", "app_fmm"]   # options A, B, B2


def selection_spaces(cohort):
    return ["raw", "airspec"] if cohort in ("eth_shaped", "analogs") else ["raw"]


def sweep_ks(auto_k, curve_max):
    # Share the server batch ladder so prewarming cannot silently miss k=21.
    return app._batch_sweep_ks(auto_k, curve_max, k_min=1, k_max=30, dense=False)


def main():
    while not app.STATE["ready"]:
        if app.STATE["error"]:
            raise SystemExit(f"data load failed: {app.STATE['error']}")
        print(f"[loader] {app.STATE['message']}", flush=True)
        time.sleep(5)
    print("[loader] ready", flush=True)

    lots = ["all"] + list(app.D["lots_available"])
    configs = []
    for cohort, cutoffs in GRID_CUTOFFS.items():
        for cutoff, sel, spectra, mode, lot in itertools.product(
                cutoffs, selection_spaces(cohort), SPECTRA, MODES, lots):
            configs.append((cohort, cutoff, sel, spectra, mode, lot))
    # cheapest cohorts first so most of the grid is usable early; the 13k pool last
    order = {"eth_shaped": 0, "analogs": 1, "ocec": 2, "smoke": 3, "pool": 4}
    configs.sort(key=lambda c: order[c[0]])

    t0, failures = time.time(), []
    for i, (cohort, cutoff, sel, spectra, mode, lot) in enumerate(configs, 1):
        tag = f"{cohort}/{cutoff or '-'} sel={sel} cal={spectra} {mode} lot={lot}"
        try:
            t = time.time()
            out = app.run_config(cohort, cutoff, sel, spectra, mode,
                                 k_override=None,
                                 max_components=app.MAX_COMPONENTS, lot=lot)
            ks = sweep_ks(out["auto_k"], int(out["curve"][-1]["n_components"]))
            for k in ks:
                if k != out["k"]:
                    app.run_config(cohort, cutoff, sel, spectra, mode,
                                   k_override=k,
                                   max_components=app.MAX_COMPONENTS, lot=lot)
            cached = "cached" if out["curve_cached"] else "computed"
            print(f"[{i}/{len(configs)}] {tag}: curve {cached}, auto_k="
                  f"{out['auto_k']}, sweep {ks} done ({time.time() - t:.1f}s)",
                  flush=True)
        except Exception as exc:
            failures.append((tag, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(configs)}] {tag}: FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min; "
          f"{len(configs) - len(failures)}/{len(configs)} configurations warmed",
          flush=True)
    for tag, err in failures:
        print(f"  failed: {tag}: {err}", flush=True)


if __name__ == "__main__":
    main()
