"""The offset story in five figures — what the 2026-08-22/23 push established.

Builds deck-ready PNGs into output/plots/offset_story/. Slide titles carry the
claim (Ann's rule). Data: the hips_lab module (blank ledger + York fits, live
via the explorer on :5058 with a JSON cache so re-runs need no server),
cache/batch_results.jsonl (screening cloud), and the AERONET pull.

Site colors are the repo's canonical config.SITES palette (Addis orange,
Delhi blue, Beijing red, Pasadena green) + deck purple for Bishoftu —
CVD-validated in this display order; identity is always also carried by text.

Run: MPLBACKEND=Agg python scripts/build_offset_story_figures.py
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
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "calibration_explorer"))
import hips_lab  # noqa: E402

OUT = HERE.parent / "output/plots/offset_story"
TITLES = True   # deck reuse: set False (and repoint OUT) for title-free copies
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "york_variants_cache.json"

SITES = [("addis", "ETAD", "Addis"), ("indh", "INDH", "Delhi"),
         ("chts", "CHTS", "Beijing"), ("etbi", "ETBI", "Bishoftu"),
         ("uspa", "USPA", "Pasadena")]
COLOR = {"Addis": "#F39C12", "Delhi": "#3498DB", "Beijing": "#E74C3C",
         "Bishoftu": "#7A4FA3", "Pasadena": "#2ECC71"}
VARIANTS = [("deployed", "deployed blank line", "#8F8C84"),
            ("lot_lin", "lot-common linear", "#2C6E9E"),
            ("lot_quad", "lot quadratic", "#B23327")]
WINNER = {"cohort": "ocec", "cutoff": 450, "spectra": "airspec",
          "mode": "site_heldout", "k": 9}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E8E6E1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white"})


def york_variants() -> list[dict]:
    """Per-site York fits under the three blank lines (winner config).
    Fetched once from the live explorer, cached to JSON."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    rows = []
    for name, code, label in SITES:
        body = {**WINNER, "target": name}
        req = urllib.request.Request("http://127.0.0.1:5058/api/run",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=900))
        if name == "addis":
            sys.path.insert(0, str(HERE))
            import phase3_common as pc
            ev = next(x for x in pc.load_addis_evaluation()
                      if isinstance(x, pd.DataFrame) and "MediaId" in x.columns)
            fids = ev["ExternalFilterId"].astype(str).tolist()
        else:
            ref = pd.read_csv(REPO / f"calibration_explorer/targets/{name}/reference.csv")
            fids = ref["ExternalFilterId"].astype(str).tolist()
        r = hips_lab.site_rows(d["eval"]["pred"], d["eval"]["ref"], fids, code)
        r["label"] = label
        rows.append(r)
    CACHE.write_text(json.dumps(rows))
    return rows


def fig_intercept_ladder(rows):
    """F1 — the headline: the Addis offset survives the blank-line correction."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    order = [r for r in rows]
    ys = np.arange(len(order))[::-1]
    for j, (kind, vlab, c) in enumerate(VARIANTS):
        off = (j - 1) * 0.22
        for y, r in zip(ys, order):
            f = r["fits"].get(kind, {})
            if "intercept" not in f:
                continue
            ax.errorbar(f["intercept"], y + off, xerr=f["intercept_se"],
                        fmt="o", ms=7, color=c, ecolor=c, elinewidth=1.2,
                        capsize=3, label=vlab if y == ys[0] else None)
    ax.axvline(0, color="#22252A", lw=1.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in order])
    ax.set_xlabel("York intercept (µg/m³), x = Fabs/10; zero is the target")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    if TITLES:
        ax.set_title("The Addis offset survives the instrument-calibration check\n"
                 "(per-filter weighted York fits under three HIPS blank lines)",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "f1_intercept_blankline_ladder.png", dpi=200)
    plt.close(fig)


def fig_slope_ladder(rows):
    """F2 — Pasadena's slope anomaly was the blank line; Delhi's is real."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ys = np.arange(len(rows))[::-1]
    for j, (kind, vlab, c) in enumerate(VARIANTS):
        off = (j - 1) * 0.22
        for y, r in zip(ys, rows):
            f = r["fits"].get(kind, {})
            if "slope" not in f:
                continue
            ax.errorbar(f["slope"], y + off, xerr=f["slope_se"], fmt="o",
                        ms=7, color=c, ecolor=c, elinewidth=1.2, capsize=3,
                        label=vlab if y == ys[0] else None)
    ax.axvline(1, color="#22252A", lw=1.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("York slope; one is the target")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    if TITLES:
        ax.set_title("Pasadena's slope anomaly dissolves under a quadratic blank line;\n"
                 "Delhi's 1.8x is blank-line-robust", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "f2_slope_blankline_ladder.png", dpi=200)
    plt.close(fig)


def fig_blank_geometry():
    """F3 — why: loaded filters sit below the blanks that define the line."""
    b = hips_lab.batch()
    lot = "251"
    blanks = b[(b.LotId == lot) & b.FilterType.isin(["FB", "LB"])].dropna(
        subset=["R1", "T1"]).query("T1 > 0")
    # lot 251 carries more than one deployed calibration set; label the
    # blank clusters by their set instead of pooling them under one name
    grp = blanks.groupby([blanks.Intercept.round(1), blanks.Slope.round(3)])
    sets_ = sorted(grp, key=lambda kv: -len(kv[1]))
    R, T = blanks["R1"].to_numpy(float), blanks["T1"].to_numpy(float)
    lin = np.polyfit(R, T, 1)
    quad = np.polyfit(R, T, 2)
    r1_min = float(R.min())
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.1], "hspace": 0.08})
    for i, ((gi, gs), g) in enumerate(sets_[:2]):
        ax.scatter(g.R1, g.T1, s=14, alpha=0.6,
                   color=["#8F8C84", "#C49442"][i],
                   label=f"lot-251 blanks, calibration set {i + 1} (n={len(g)})")
    rest = blanks[~blanks.index.isin(
        sets_[0][1].index.union(sets_[1][1].index))]
    if len(rest):
        ax.scatter(rest.R1, rest.T1, s=10, alpha=0.4, color="#D8D5CF",
                   label=f"other sets (n={len(rest)})")
    xs = np.linspace(60, 280, 200)
    ax.plot(xs, np.polyval(lin, xs), color="#2C6E9E", lw=2,
            label="pooled linear blank line")
    ax.plot(xs, np.polyval(quad, xs), color="#B23327", lw=2,
            ls="--", label="pooled quadratic refit")
    ax.axvspan(60, r1_min, color="#F39C12", alpha=0.12)
    ax.text(62, ax.get_ylim()[0] + 40, "extrapolation zone\n(darker than every blank)",
            fontsize=9, color="#9A6206", va="bottom")
    ax.set_ylabel("T1 (counts)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    if TITLES:
        ax.set_title("The HIPS scattering correction is a regression through blanks —\n"
                 "heavily loaded filters sit beyond its support", loc="left",
                 fontsize=12)
    pm = b[(b.FilterType == "PM2.5")]
    for _, code, label in SITES:
        r1 = pm[pm.Site == code]["R1"].dropna()
        ax2.hist(r1, bins=np.arange(60, 285, 6), histtype="step", lw=1.8,
                 color=COLOR[label], density=True, label=label)
    ax2.axvline(r1_min, color="#22252A", lw=1.2, ls=":")
    ax2.set_xlabel("R1 (counts); lower = darker filter")
    ax2.set_ylabel("density")
    ax2.legend(frameon=False, fontsize=8, ncol=5, loc="upper right")
    ax2.set_xlim(60, 280)
    fig.savefig(OUT / "f3_blankline_geometry.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig_screening_cloud():
    """F4 — the 16k-variant screen: cloud, frontier, winner."""
    path = REPO / "calibration_explorer/cache/batch_results.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    df = pd.DataFrame([r for r in rows
                       if r.get("mode") == "site_heldout"
                       and (r.get("target") or "addis") == "addis"
                       and (r.get("eval_lot") or "all") == "all"
                       and r.get("heldout_R2") is not None])
    sl = df["all_deming_slope"].fillna(df["deming_slope"]).astype(float)
    ic = df["all_deming_intercept"].fillna(df["deming_intercept"]).astype(float)
    passing = df["heldout_R2"] >= 0.85
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # cap the axes at the bulk of the cloud; count what falls outside so the
    # trim is stated, never silent
    XMAX, YMAX = 3.2, 14.0
    ds, di = abs(sl - 1), abs(ic)
    shown = (ds <= XMAX) & (di <= YMAX)
    n_out = int((~shown).sum())
    m0 = (~passing) & shown
    m1 = passing & shown
    ax.scatter(ds[m0], di[m0], s=7, color="#D8D5CF",
               label=f"below the held-out floor (n={int((~passing).sum())})")
    ax.scatter(ds[m1], di[m1], s=9, color="#2C6E9E",
               alpha=0.45, label=f"held-out R² ≥ 0.85 (n={int(passing.sum())})")
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.text(0.99, 0.02, f"{n_out} extreme variants beyond the axes (all far from 1:1)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="#6E7178")
    w = df[(df.cohort == "ocec") & (df.cutoff == 450) &
           (df.spectra == "airspec") & (df.k == 9)]
    if len(w):
        ws = abs(float(w["all_deming_slope"].iloc[0]) - 1)
        wi = abs(float(w["all_deming_intercept"].iloc[0]))
        ax.scatter([ws], [wi], s=140, marker="*", color="#B23327", zorder=5,
                   label="winner: lowest-OC/EC 450 × AIRSpec, k=9")
    ax.set_xlabel("|Deming slope − 1|  (Addis, MAC 10, all pairs)")
    ax.set_ylabel("|Deming intercept| (µg/m³)")
    ax.legend(frameon=False, fontsize=9)
    if TITLES:
        ax.set_title(f"{len(df):,} scored variants; the honest (site-held-out) screen\n"
                 "selects the dense-sweep basin, confirmed out-of-country",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "f4_screening_cloud.png", dpi=200)
    plt.close(fig)


def fig_aeronet():
    """F5 — the independent column check: H by site + what predicts AAOD."""
    sc = pd.read_csv(REPO / "research/ftir_ec_phase3/output/tables/aeronet/site_comparison.csv")
    name_of = {"ETAD": "Addis", "INDH": "Delhi", "CHTS": "Beijing", "USPA": "Pasadena"}
    sc["label"] = sc["site"].map(name_of)
    sc = sc.rename(columns={"H_median_m": "H_m"}).sort_values("H_m")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                  gridspec_kw={"width_ratios": [1, 1.25]})
    ys = np.arange(len(sc))
    ax.barh(ys, sc["H_m"], height=0.5,
            color=[COLOR[v] for v in sc["label"]])
    ax.set_yticks(ys)
    ax.set_yticklabels(sc["label"])
    for y, v in zip(ys, sc["H_m"]):
        ax.text(v + 15, y, f"{v:.0f} m", va="center", fontsize=9)
    ax.set_xlabel("column AAOD₆₇₅ / surface Fabs (m)")
    if TITLES:
        ax.set_title("Addis reports more surface absorption\nthan its column supports (provisional)",
                 loc="left", fontsize=11)
    # partial-residual: 1500-1700 envelope vs AAOD, Addis, | CH + Fabs + month
    sys.path.insert(0, str(HERE))
    import phase3_common as pc
    ev = next(x for x in pc.load_addis_evaluation()
              if isinstance(x, pd.DataFrame) and "MediaId" in x.columns)
    npz = np.load(HERE.parent / "output/corrected/etad_corrected_df6.npz",
                  allow_pickle=True)
    wn = npz["wn"].astype(float)
    X = pd.DataFrame(npz["corrected"].astype(float))
    X["MediaId"] = npz["media_id"].astype(int)
    Xg = X.groupby("MediaId").mean()
    env = pd.Series(Xg.values[:, (wn >= 1500) & (wn <= 1700)].mean(axis=1),
                    index=Xg.index, name="env")
    ch = pd.Series(Xg.values[:, np.argmin(abs(wn - 2920))], index=Xg.index,
                   name="ch")
    j = (ev[["MediaId", "ExternalFilterId"]].astype({"MediaId": int})
         .merge(pd.concat([env, ch], axis=1), left_on="MediaId",
                right_index=True))
    m = pd.read_csv(HERE.parent / "output/tables/aeronet/ETAD_matched_daily.csv")
    j2 = j.merge(m, left_on="ExternalFilterId", right_on="FilterId").dropna(
        subset=["env", "Absorption_AOD[675nm]"])
    mo = pd.to_datetime(j2["SampleDate"]).dt.month
    Z = np.column_stack([np.ones(len(j2)), j2["ch"], j2["Fabs"]]
                        + [(mo == k).astype(float) for k in range(2, 13)])
    y_ = j2["Absorption_AOD[675nm]"].to_numpy(float)
    x_ = j2["env"].to_numpy(float)
    ry = y_ - Z @ np.linalg.lstsq(Z, y_, rcond=None)[0]
    rx = x_ - Z @ np.linalg.lstsq(Z, x_, rcond=None)[0]
    ax2.scatter(rx, ry, s=18, color=COLOR["Addis"], alpha=0.7)
    cf = np.polyfit(rx, ry, 1)
    xs = np.linspace(rx.min(), rx.max(), 50)
    ax2.plot(xs, np.polyval(cf, xs), color="#22252A", lw=1.6)
    r = np.corrcoef(rx, ry)[0, 1]
    ax2.set_xlabel("corrected 1500–1700 cm⁻¹ envelope (residualized)")
    ax2.set_ylabel("AAOD₆₇₅ (residualized)")
    if TITLES:
        ax2.set_title(f"…and its organic envelope predicts the column beyond\n"
                  f"loading, Fabs and season (r = {r:+.2f}, n = {len(j2)})",
                  loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "f5_aeronet_column_check.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rows = york_variants()
    fig_intercept_ladder(rows)
    fig_slope_ladder(rows)
    fig_blank_geometry()
    fig_screening_cloud()
    fig_aeronet()
    print("written:", *[p.name for p in sorted(OUT.glob("*.png"))], sep="\n  ")
