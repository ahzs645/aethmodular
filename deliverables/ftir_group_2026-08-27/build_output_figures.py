"""Three figures for the app-OUTPUT slides (what the tool taught us, not the
tool): the per-site leaderboard, the analog-lab agreement result, and the
validate-top-5 stability output. House style: title-free, white, RGB.

Run: MPLBACKEND=Agg ~/anaconda3/bin/python build_output_figures.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
REPO = HERE.parents[1]
INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
SITE_COLOR = {"addis": "#F39C12", "indh": "#3498DB", "chts": "#E74C3C",
              "etbi": "#7A4FA3", "uspa": "#2ECC71"}
SITE_LABEL = {"addis": "Addis", "indh": "Delhi", "chts": "Beijing",
              "etbi": "Bishoftu", "uspa": "Pasadena"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E8E6E1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white",
                     "savefig.dpi": 165})


def fig_leaderboard_by_site():
    """Each site's best configuration (slope 0.85-1.18, Option A, held-out
    TOR R2 >= 0.85) from the 71k-row grid: the cutoff differs per city."""
    rows = [json.loads(l) for l in
            (REPO / "calibration_explorer/cache/batch_results.jsonl").open()]
    best = {}
    for r in rows:
        if (r.get("mode") != "site_heldout" or (r.get("heldout_R2") or 0) < 0.85
                or not r.get("cutoff")):
            continue
        sl = r.get("all_deming_slope") or r.get("deming_slope")
        ic = r.get("all_deming_intercept") or r.get("deming_intercept")
        if sl is None or ic is None or not (0.85 <= sl <= 1.18):
            continue
        t = r.get("target") or "addis"
        score = abs(ic) + 0.5 * abs(sl - 1)
        if t not in best or score < best[t][0]:
            best[t] = (score, r, sl, ic)
    order = ["addis", "indh", "chts", "etbi", "uspa"]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ys = np.arange(len(order))[::-1]
    for y, t in zip(ys, order):
        _, r, sl, ic = best[t]
        c = SITE_COLOR[t]
        ax.scatter([r["cutoff"]], [y], s=160, color=c, zorder=3)
        sel = "AIRSpec-sel × " if r.get("selection_space") == "airspec" else ""
        fam = {"ocec": "lowest-OC/EC", "analogs": "analogs"}.get(r["cohort"], r["cohort"])
        cal = {"airspec": "AIRSpec", "deriv2": "2nd deriv", "raw": "raw"}[r["spectra"]]
        note = f"{fam}-{r['cutoff']:.0f} × {sel}{cal}, k={r['k']:.0f}:  {sl:.2f}x{ic:+.2f}"
        if t == "indh":
            note += "   (96% extrapolated: fails holdout)"
        ax.annotate(note, (r["cutoff"], y), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 440, 800, 1850])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks(ys)
    ax.set_yticklabels([SITE_LABEL[t] for t in order])
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlim(90, 2600)
    ax.set_xlabel("best cohort cutoff N (log scale): slope-boxed 0.85–1.18, "
                  "Option A, held-out TOR R² ≥ 0.85")
    fig.tight_layout()
    fig.savefig(FIG / "f_out_leaderboard_by_site.png")
    plt.close(fig)
    print("leaderboard-by-site done:",
          {t: f"{v[1]['cohort']}-{v[1]['cutoff']:.0f}" for t, v in best.items()})


def fig_analog_agreement():
    """Analog-lab output: does the committed analog ranking agree with plain
    spectral similarity? Spearman agreement + top-500 overlap, per space."""
    spaces = ["raw", "airspec", "deriv2"]
    metrics = [("cosine_median", "cosine / SAM vs Addis median", BLUE),
               ("corr_median", "Pearson vs median (LOCAL)", PURPLE),
               ("mahalanobis_pca", "Mahalanobis PCA-10 (Reggente '16)", AMBER)]
    ag = {sp: {} for sp in spaces}
    ov = {}
    for sp in spaces:
        req = urllib.request.Request(
            "http://127.0.0.1:5058/api/analog_lab",
            json.dumps({"space": sp}).encode(),
            {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=900))
        for m, _, _ in metrics:
            ag[sp][m] = d["agreement"][m]
        com = np.argsort(d["ranks"]["committed"])[:500]
        cos = np.argsort(d["ranks"]["cosine_median"])[:500]
        ov[sp] = len(set(com) & set(cos)) / 5.0
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    x = np.arange(len(spaces))
    wd = 0.24
    for j, (m, lab, c) in enumerate(metrics):
        vals = [ag[sp][m] for sp in spaces]
        ax.bar(x + (j - 1) * wd, vals, wd - 0.03, color=c, label=lab)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(top-500 overlap {ov[s]:.0f}%)" for s in
                        ["raw", "airspec", "deriv2"]])
    ax.set_ylabel("Spearman ρ: committed analog score vs plain similarity")
    ax.set_ylim(-0.6, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "f_out_analog_agreement.png")
    plt.close(fig)
    print("analog agreement done:", {s: round(ag[s]["cosine_median"], 2) for s in spaces},
          "| overlap:", ov)


def fig_winner_stability():
    """Validate-top-5 output: selection frequency of the frozen finalists
    under bootstrap re-selection (VALIDATION_LAYER_2026-08-24)."""
    finalists = [("lowest-OC/EC 440\n× AIRSpec, k=5", 62.5, 63.0, BLUE),
                 ("analog-440 corrected-sel\n× 2nd deriv, k=15", 37.5, 31.5, PURPLE),
                 ("analog-530 corrected-sel\n× 2nd deriv, k=12", 0.0, 5.5, GREY)]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(finalists))
    ax.bar(x - 0.17, [f[1] for f in finalists], 0.3, color=[f[3] for f in finalists],
           label="target-filter bootstrap wins")
    ax.bar(x + 0.17, [f[2] for f in finalists], 0.3,
           color=[f[3] for f in finalists], alpha=0.45,
           label="IMPROVE source-site refit wins")
    for xi, f in zip(x, finalists):
        ax.text(xi - 0.17, f[1] + 1.5, f"{f[1]:.0f}%", ha="center", fontsize=10)
        ax.text(xi + 0.17, f[2] + 1.5, f"{f[2]:.0f}%", ha="center", fontsize=10)
    ax.axhline(50, color=INK, lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f[0] for f in finalists], fontsize=9.5)
    ax.set_ylabel("share of re-selection draws won (%)")
    ax.set_ylim(0, 78)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "f_out_winner_stability.png")
    plt.close(fig)
    print("winner stability done")


if __name__ == "__main__":
    fig_leaderboard_by_site()
    fig_analog_agreement()
    fig_winner_stability()
    from PIL import Image
    for n in ("f_out_leaderboard_by_site", "f_out_analog_agreement",
              "f_out_winner_stability"):
        im = Image.open(FIG / f"{n}.png")
        if im.mode != "RGB":
            im.convert("RGB").save(FIG / f"{n}.png")
    print("flattened RGB")
