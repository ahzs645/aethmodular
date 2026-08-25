"""Figures for the 2026-08-27 FTIR-group deck (house style: title-free,
white, flattened RGB; the slide title carries the claim).

Reuses the committed offset-story builders with TITLES=False, adds the
season x baseline panel and the cutoff-basin figure, and renders the status
table. Needs the explorer on :5058 (anchors validated beforehand).

Run: MPLBACKEND=Agg ~/anaconda3/bin/python build_figures.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))

INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
SEASON = {"Dry (Oct-Feb)": ACCENT, "Belg (Mar-May)": PURPLE,
          "Kiremt (Jun-Sep)": BLUE}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E8E6E1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white",
                     "savefig.dpi": 165})


def api_run(body):
    req = urllib.request.Request("http://127.0.0.1:5058/api/run",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def deming(x, y, lam=2.96):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    sxx, syy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean()
    sxy = ((x - mx) * (y - my)).mean()
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2
                                   + 4 * lam * sxy ** 2)) / (2 * sxy)
    return b, my - b * mx


def offset_story_title_free():
    import build_offset_story_figures as B
    B.TITLES = False
    B.OUT = FIG
    # hips_lab.blank_lines is now keyed per deployed line; the geometry figure
    # wants the lot-pooled view -> patch in a pooled version for this build.
    hl = B.hips_lab
    bt = hl.batch()

    def pooled_lines():
        out = {}
        for lot, g in bt[bt.FilterType.isin(["FB", "LB"])].dropna(
                subset=["R1", "T1"]).query("T1 > 0").groupby("LotId"):
            if len(g) < 5:
                continue
            R, T = g["R1"].to_numpy(float), g["T1"].to_numpy(float)
            out[lot] = {"lin": list(np.polyfit(R, T, 1)),
                        "quad": list(np.polyfit(R, T, 2)),
                        "r1_min": float(R.min()), "r1_max": float(R.max())}
        return out

    hl_blank_orig = hl.blank_lines
    hl.blank_lines = pooled_lines
    rows = B.york_variants()          # served from the committed JSON cache
    B.fig_intercept_ladder(rows)
    B.fig_slope_ladder(rows)
    B.fig_blank_geometry()
    B.fig_screening_cloud()
    hl.blank_lines = hl_blank_orig
    print("offset-story figures (title-free) done")


def fig_season_panels():
    """Two crossplots, same cohort, raw vs baseline-corrected (AIRSpec),
    colored by season with per-season Deming lines. Option A, MAC 10."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=False)
    for ax, (spectra, k, lab) in zip(axes, [("raw", 6, "raw spectra, k=6"),
                                            ("airspec", 9,
                                             "baseline-corrected (AIRSpec), k=9")]):
        d = api_run({"cohort": "ocec", "cutoff": 450, "spectra": spectra,
                     "mode": "site_heldout", "target": "addis", "k": k})
        pred = np.array(d["eval"]["pred"], float)
        x = np.array(d["eval"]["ref"], float) / 10.0
        g = np.array(d["eval"]["group"])
        lim = max(x.max(), pred.max()) * 1.06
        ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.2)
        lines = []
        for season, c in SEASON.items():
            m = g == season
            ax.scatter(x[m], pred[m], s=16, color=c, alpha=0.65,
                       label=f"{season.split(' ')[0]} (n={m.sum()})")
            b, a = deming(x[m], pred[m])
            xs = np.linspace(x[m].min(), x[m].max(), 20)
            ax.plot(xs, b * xs + a, color=c, lw=1.8)
            lines.append(f"{season.split(' ')[0]:7s} {b:4.2f}x{a:+5.2f}")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5, family="monospace",
                bbox=dict(fc="white", ec="#DDDAD2", alpha=0.9))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"HIPS Fabs / 10 (µg/m³) — {lab}")
        ax.set_ylabel("FTIR-predicted EC (µg/m³)")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "f_season_panels.png")
    plt.close(fig)
    print("season panels done")


def fig_cutoff_basin():
    """Intercept and held-out TOR R² vs OC/EC cutoff (AIRSpec, Option A) —
    the dense sweep basin and where the locked 800 sits. Stacked panels
    (one axis each; never dual-axis)."""
    rows = [json.loads(l) for l in
            (REPO / "calibration_explorer/cache/batch_results.jsonl").open()]
    sel = [r for r in rows
           if r.get("cohort") == "ocec" and r.get("spectra") == "airspec"
           and r.get("selection_space") == "raw"
           and r.get("mode") == "site_heldout"
           and (r.get("target") or "addis") == "addis"
           and (r.get("eval_lot") or "all") == "all"
           and r.get("cutoff") and r.get("k") == r.get("auto_k")]
    sel.sort(key=lambda r: r["cutoff"])
    cut = np.array([r["cutoff"] for r in sel], float)
    ic = np.array([r.get("all_deming_intercept") or r["deming_intercept"]
                   for r in sel], float)
    ho = np.array([r["heldout_R2"] for r in sel], float)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.0), sharex=True,
                                 gridspec_kw={"hspace": 0.12})
    a1.plot(cut, ic, color=BLUE, lw=1.4)
    a1.scatter(cut, ic, s=12, color=BLUE)
    a1.axvspan(440, 490, color=AMBER, alpha=0.18)
    a1.axvline(800, color=GREY, ls="--", lw=1.2)
    a1.axhline(0, color=INK, lw=1)
    a1.set_ylabel("Deming intercept (µg/m³)")
    a1.text(447, a1.get_ylim()[1] * 0.85, "basin", fontsize=9, color="#8A6A1F")
    a1.text(805, a1.get_ylim()[1] * 0.85, "locked 800", fontsize=9, color=GREY)
    a2.plot(cut, ho, color=ACCENT, lw=1.4)
    a2.scatter(cut, ho, s=12, color=ACCENT)
    a2.axvspan(440, 490, color=AMBER, alpha=0.18)
    a2.axvline(800, color=GREY, ls="--", lw=1.2)
    a2.axhline(0.85, color=INK, lw=1, ls=":")
    a2.set_ylabel("held-out TOR R²")
    a2.set_xlabel("lowest-OC/EC cohort cutoff N — baseline-corrected (AIRSpec), Option A, rule k")
    fig.tight_layout()
    fig.savefig(FIG / "f_cutoff_basin.png")
    plt.close(fig)
    print(f"cutoff basin done ({len(sel)} rule-k points)")


def fig_status():
    """The Aug-19 ask list, item by item — 'what do you have to show'."""
    items = [
        ("Select on corrected AND calibrate on corrected (“do baseline for both”)",
         "DONE", "led to the 440–490 basin"),
        ("Evaluate on one lot only (judge 251 on 251)", "DONE",
         "Eval-lot selector; winner holds on lot-251-only"),
        ("Chase the lot-248 pool anomaly at the DB", "RESOLVED",
         "DB holds 1,362 network-wide; a ~2-month lot (Jan–Feb 2021)"),
        ("Get Bishoftu (ETBI) spectra myself from the SPARTAN DB", "DONE",
         "26 filters pulled + evaluated; +3 more sites while at it"),
        ("Is baselining the lot or the aerosol?", "ANSWERED",
         "Bishoftu (same lots) has no offset; blank-line check → ~15% instrument"),
        ("Seasonal patterns stable across calibrations", "EXTENDED",
         "baselining relocates seasonality → dry-season slope deficit"),
        ("Adama 2–3 slides for Christian & Sina", "IN PROGRESS",
         "draft this week"),
    ]
    color = {"DONE": "#3D7A50", "RESOLVED": "#3D7A50", "ANSWERED": "#3D7A50",
             "EXTENDED": BLUE, "IN PROGRESS": AMBER}
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.axis("off")
    for i, (item, st, note) in enumerate(items):
        y = 1 - i * (1 / len(items)) - 0.06
        ax.text(0.00, y, item, fontsize=12.5, color=INK, va="top")
        ax.text(0.66, y, st, fontsize=12.5, color=color[st], va="top",
                fontweight="bold")
        ax.text(0.755, y, note, fontsize=10.5, color="#6E7178", va="top")
    fig.tight_layout()
    fig.savefig(FIG / "f_status.png")
    plt.close(fig)
    print("status figure done")


if __name__ == "__main__":
    offset_story_title_free()
    fig_season_panels()
    fig_cutoff_basin()
    fig_status()
    # flatten everything to RGB (RGBA breaks PowerPoint)
    from PIL import Image
    for p in FIG.glob("*.png"):
        im = Image.open(p)
        if im.mode != "RGB":
            Image.new("RGB", im.size, "white").paste(im, mask=im.split()[-1]) \
                if im.mode == "RGBA" else None
            im.convert("RGB").save(p)
    print("all figures flattened RGB:", sorted(p.name for p in FIG.glob("*.png")))
