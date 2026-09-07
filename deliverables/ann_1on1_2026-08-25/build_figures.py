"""Extra figures for the Ann 1:1 deck (house style: title-free, white, flattened RGB).

The six carried over from the group deck are already in figures/. This adds the
status table and the recovered-filter tally. Run:
    MPLBACKEND=Agg ~/anaconda3/bin/python build_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
plt.rcParams.update({"font.size": 11, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})

# --- her Aug-12 asks, item by item -------------------------------------------
ASKS = [
    ("Baseline-correct for BOTH selection and calibration", "DONE",
     "shipped 18 Aug; analogs -9.7 -> -4.3"),
    ("Analog cutoff sweep (where to cut the rank score)", "DONE",
     "dense sweep, step 10; basin located"),
    ("Focus lot 251; add evaluate-on-same-lot", "DONE",
     "eval_lot in the app"),
    ("Is baselining helping because of the LOT?", "ANSWERED",
     "yes - and it removes it (slide 4)"),
    ("Lot 248 count looks too small", "ANSWERED",
     "1,362 analyses, a 2-month lot"),
    ("Get the Bishoftu FTIR spectra yourself", "DONE",
     "pulled from SPARTAN DB; 26 -> 40 filters"),
    ("Adama: 2-3 slides for Christian and Sina", "PART",
     "seed figure; TOR-vs-FTIR panel still to build"),
]
STATUS_C = {"DONE": BLUE, "ANSWERED": PURPLE, "PART": AMBER, "BLOCKED": ACCENT}


def fig_status():
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    ax.axis("off")
    y = len(ASKS)
    for ask, st, note in ASKS:
        c = STATUS_C[st]
        ax.text(0.005, y, ask, fontsize=12, color=INK, va="center", ha="left")
        ax.text(0.605, y, st, fontsize=10.5, color="white", va="center", ha="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.34", facecolor=c, edgecolor="none"))
        ax.text(0.685, y, note, fontsize=10.5, color=GREY, va="center", ha="left")
        y -= 1
    ax.set_xlim(0, 1.32); ax.set_ylim(0.3, len(ASKS) + 0.7)
    fig.tight_layout()
    fig.savefig(FIG / "f_status_asks.png", dpi=168, bbox_inches="tight")
    print("wrote f_status_asks.png")


# --- filters recovered from raw hips.Results ---------------------------------
def fig_recovered():
    d = pd.DataFrame({"site": ["Bishoftu\n(ETBI)", "Delhi\n(INDH)", "Addis\n(ETAD)"],
                      "had": [26, 152, 239], "gained": [14, 26, 14],
                      "lot": ["251", "253", "253"]})
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = range(len(d))
    ax.bar(x, d.had, 0.56, color=GREY, label="already had (shipped HIPS CSV)")
    ax.bar(x, d.gained, 0.56, bottom=d.had, color=BLUE,
           label="recovered from raw hips.Results")
    for i, r in d.iterrows():
        ax.text(i, r.had + r.gained + 5, f"+{r.gained}", ha="center",
                fontsize=12, color=BLUE, fontweight="bold")
        ax.text(i, r.had + r.gained + 20, f"lot {r.lot}", ha="center",
                fontsize=9.5, color=GREY)
        pct = 100 * r.gained / r.had
        ax.text(i, r.had / 2, f"{r.had}", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        if pct > 40:
            ax.text(i, -22, f"+{pct:.0f}%", ha="center", fontsize=11,
                    color=BLUE, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(d.site, fontsize=11.5)
    ax.set_ylabel("filters with a usable Fabs", fontsize=11)
    ax.set_ylim(-40, 300)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.16)
    fig.tight_layout()
    fig.savefig(FIG / "f_recovered_filters.png", dpi=168, bbox_inches="tight")
    print("wrote f_recovered_filters.png")


if __name__ == "__main__":
    fig_status(); fig_recovered()
