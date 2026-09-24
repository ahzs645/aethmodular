"""The specification curve over the Addis calibration search, as data.

One definition of which readouts count, which pass the guardrails, and which
analytic choices the specification panel marks, shared by the matplotlib
figure (build_pathway_figures.fig_spec_curve) and the gallery export
(gallery/data/export_spec_curve.py), so the two cannot drift apart.
No plotting imports here.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "calibration_explorer/cache/batch_results.jsonl"

# Guardrails every reported configuration has to clear.
HELDOUT_FLOOR = 0.85
SLOPE_RANGE = (0.7, 1.3)
Q_RESIDUAL_MAX = 30
EXTRAP_MAX = 30
NEGATIVE_MAX = 10
GUARDRAIL_TEXT = (
    f"held-out TOR R² ≥ {HELDOUT_FLOOR}, Deming slope {SLOPE_RANGE[0]}–{SLOPE_RANGE[1]}, "
    f"Q-residual OOD ≤ {Q_RESIDUAL_MAX} %, score extrapolation ≤ {EXTRAP_MAX} %, "
    f"negative predictions ≤ {NEGATIVE_MAX} %"
)


def spec_rows(path: Path = RESULTS) -> list[dict]:
    """Every Addis readout with an intercept, sorted by that intercept.

    All-lots, unsplit readouts only, so the population matches every other
    number reported for this search; lot-specific readouts are a different view
    of the same configurations and would double-count them here.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("target") == "addis"
            and r.get("deming_intercept") is not None
            and str(r.get("eval_lot")) in ("all", "None")
            and r.get("eval_split") in (None, "all")]
    rows.sort(key=lambda r: r["deming_intercept"])
    return rows


def passes_guardrails(r: dict) -> bool:
    return (r.get("q_residual_pct") is not None
            and r.get("extrap_pct") is not None
            and r.get("negative_pct") is not None
            and r.get("heldout_R2") is not None and r["heldout_R2"] >= HELDOUT_FLOOR
            and SLOPE_RANGE[0] <= r["deming_slope"] <= SLOPE_RANGE[1]
            and r["q_residual_pct"] <= Q_RESIDUAL_MAX and r["extrap_pct"] <= EXTRAP_MAX
            and r["negative_pct"] <= NEGATIVE_MAX)


# The specification panel: one row per analytic choice, marked where used.
# Reader-facing names follow docs/naming-conventions.md (agreed 2026-09-23);
# rows are grouped under CHOICE_GROUPS headings in the order listed.
CHOICES = [
    ("Spline baseline", lambda r: r["spectra"] == "airspec"),
    ("Second derivative", lambda r: r["spectra"] == "deriv2"),
    ("Raw spectra", lambda r: r["spectra"] == "raw"),
    ("Lowest OC/EC", lambda r: r["cohort"] == "ocec"),
    ("Spectral analogs (VIP-weighted distance)", lambda r: r["cohort"] == "analogs"),
    ("Ethiopia-shaped smoke", lambda r: r["cohort"] == "eth_shaped"),
    ("Cohort selected on spline-baselined spectra", lambda r: r.get("selection_space") == "airspec"),
    ("Site-grouped cross-validation", lambda r: r["mode"] == "site_heldout"),
    ("Interleaved cross-validation", lambda r: r["mode"] in ("app", "app_fmm")),
    ("9 or fewer PLS factors", lambda r: (r.get("k") or 0) <= 9),
    ("15 or more PLS factors", lambda r: (r.get("k") or 0) >= 15),
    ("Cohort under 600 filters", lambda r: (r.get("cutoff") or 9999) < 600),
    ("Calibration set from lot 251 only", lambda r: str(r.get("lot")) == "251"),
]
CHOICE_GROUPS = [
    ("Spectral preprocessing", 0, 3),
    ("Calibration cohort", 3, 7),
    ("Calibration settings", 7, 13),
]


def uses(r: dict, test) -> bool:
    """A choice test that tolerates a missing field as 'not used'."""
    try:
        return bool(test(r))
    except Exception:                                          # noqa: BLE001
        return False
