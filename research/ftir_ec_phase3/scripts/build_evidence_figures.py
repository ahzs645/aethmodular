"""Figures for the phase-3 findings that were recorded as tables or prose only.

Audit 2026-08-25: several settled results had no figure — the York cross-site
re-fit, the MAC-invariance of C, the slope trap in the five-site grid, the HIPS
instrument epochs, the AERONET true-window check, the lot blank-line swap, and
the LOCAL transfer first pass. This builds all seven from data in hand.

House style: title-free (captions carry the claim), white, flattened RGB, dpi 168.

Run:  MPLBACKEND=Agg ~/anaconda3/bin/python build_evidence_figures.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "research/ftir_ec_phase3/output/plots/evidence"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))

INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "savefig.dpi": 168})


def _finish(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- 1. York forest
def fig_york():
    """The core measurement: per-site York/EIV intercepts with z-scores."""
    rows = [("Beijing (CHTS)", 192, 0.98, +0.64, 0.15, +4.3),
            ("Bishoftu (ETBI)", 26, 0.86, -0.39, 0.45, -0.9),
            ("Addis (ETAD)", 239, 0.92, -1.38, 0.18, -7.7)]
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    for i, (site, n, slope, b, se, z) in enumerate(rows):
        sig = abs(z) >= 2
        c = ACCENT if (sig and b < 0) else (BLUE if sig else GREY)
        ax.errorbar(b, i, xerr=1.96 * se, fmt="o", color=c, ms=11, lw=3,
                    capsize=6, capthick=2.4, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.4)
        ax.text(b, i + .30, f"{b:+.2f} ± {se:.2f}", ha="center", fontsize=10.5,
                color=c, fontweight="bold")
        ax.text(2.05, i, f"n={n}   slope {slope:.2f}   z = {z:+.1f}",
                va="center", fontsize=10.5, color=INK if sig else GREY)
    ax.axvline(0, color=INK, lw=1.4, ls="--", zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlabel("York/EIV intercept  (µg m$^{-3}$, x = F$_{abs}$/10)", fontsize=11)
    ax.set_xlim(-2.4, 4.6); ax.set_ylim(-.6, len(rows) - .35)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.16)
    ax.text(.5, -.30, "per-filter weighted errors-in-variables (York 2004), empirical per-site "
            "$\\sigma$(F$_{abs}$), $\\sigma_y$ inflated until MSWD = 1\n"
            "bars are 95% CI; error inflation absorbs lack-of-fit including curvature — "
            "conservative", transform=ax.transAxes, ha="center", fontsize=9.2, color=GREY)
    _finish(fig, "f_york_cross_site.png")


# ------------------------------------------------------- 2. MAC invariance of C
def fig_mac_invariance():
    """Slopes span 3.4x; C = |b|*MAC/a barely moves."""
    setups = ["Lowest-OC/EC\n+AIRSpec", "Lowest-OC/EC\n(800)", "Ethiopia-shaped\nsmoke (300)",
              "Deployed\nSPARTAN", "Spectral\nanalogs (400)", "Biomass-smoke\n(906)"]
    a10 = [0.86, 1.59, 1.75, 1.90, 2.91, 2.65]
    C10 = [18.84, 20.25, 21.09, 21.95, 22.10, 26.08]
    C6 = [19.06, 20.34, 21.09, 21.95, 22.05, 26.08]
    x = np.arange(len(setups))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    a1.bar(x, a10, .58, color=GREY)
    a1.set_xticks(x); a1.set_xticklabels(setups, fontsize=9)
    a1.set_ylabel("Deming slope $a$ @ MAC 10", fontsize=11)
    a1.set_title("slope varies 3.4$\\times$", fontsize=11.5, color=INK, pad=9)
    for i, v in enumerate(a10):
        a1.text(i, v + .06, f"{v:.2f}", ha="center", fontsize=9.6, color=INK)
    a1.set_ylim(0, 3.45); a1.grid(axis="y", alpha=.16)
    a2.plot(x, C10, "o-", color=ACCENT, ms=9, lw=2.4, label="C @ MAC 10", zorder=3)
    a2.plot(x, C6, "o", color=ACCENT, ms=9, mfc="white", mew=2, label="C @ MAC 6", zorder=4)
    a2.axhspan(18.8, 26.1, color=ACCENT, alpha=.09, zorder=1)
    a2.axhline(21.5, color=INK, ls="--", lw=1.3, zorder=2)
    a2.text(len(setups) - .5, 21.5, " median 21.5", va="center", fontsize=10, color=INK)
    a2.set_xticks(x); a2.set_xticklabels(setups, fontsize=9)
    a2.set_ylabel("C = |b|·MAC/a   (Mm$^{-1}$)", fontsize=11)
    a2.set_title("C is invariant: 18.8 – 26.1 Mm$^{-1}$", fontsize=11.5, color=INK, pad=9)
    a2.set_ylim(0, 30); a2.legend(fontsize=9.5, frameon=False, loc="lower right")
    a2.grid(axis="y", alpha=.16)
    _finish(fig, "f_mac_invariance.png")


# ------------------------------------------------------------- 3. the slope trap
def fig_slope_trap():
    p = REPO / "calibration_explorer/cache/batch_results.jsonl"
    if not p.exists():
        print("  -- batch_results.jsonl missing, skipping slope trap"); return
    rec = []
    with p.open() as fh:
        for line in fh:
            try: d = json.loads(line)
            except Exception: continue
            if d.get("target") == "addis" and d.get("mode") == "site_heldout":
                s, b = d.get("deming_slope"), d.get("deming_intercept")
                if s is None or b is None: continue
                rec.append((s, b))
    if not rec:
        print("  -- no addis/site_heldout rows, skipping"); return
    df = pd.DataFrame(rec, columns=["slope", "intercept"])
    df["score"] = df.intercept.abs() + .5 * (df.slope - 1).abs()
    honest = df[df.slope.between(0.85, 1.18)]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.scatter(df.slope, df.intercept, s=5, c=GREY, alpha=.18, lw=0, zorder=2)
    ax.axvspan(0.85, 1.18, color=BLUE, alpha=.10, zorder=1)
    top = df.nsmallest(40, "score")
    ax.scatter(top.slope, top.intercept, s=34, c=ACCENT, lw=0, zorder=4,
               label="top 40 by naive score  |b| + 0.5|a−1|")
    if len(honest):
        th = honest.nsmallest(40, "score")
        ax.scatter(th.slope, th.intercept, s=34, c=BLUE, lw=0, zorder=5,
                   label="top 40 with slope constrained to 0.85–1.18")
    ax.axhline(0, color=INK, lw=1.2, ls="--", zorder=3)
    ax.set_xlabel("Deming slope", fontsize=11)
    ax.set_ylabel("Deming intercept  (µg m$^{-3}$)", fontsize=11)
    ax.set_xlim(0, 3.0); ax.set_ylim(-12, 4)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.grid(alpha=.15)
    ax.text(.02, .04, f"{len(df):,} Addis configurations, Option A (site-grouped)\n"
            f"naive median slope of the top 40: {top.slope.median():.2f}",
            transform=ax.transAxes, fontsize=9.6, color=GREY)
    _finish(fig, "f_slope_trap.png")


# -------------------------------------------------------- 4. instrument epochs
def fig_epochs():
    p = Path("/Users/ahmadjalil/Downloads/hips/spartan_hips_raw_all.csv")
    if not p.exists():
        print("  -- spartan_hips_raw_all.csv missing, skipping epochs"); return
    r = pd.read_csv(p, encoding="utf-8-sig")
    r.columns = [c.strip('"') for c in r.columns]
    s = r[r.ResultTypeId.eq(0)].copy()
    s["ts"] = pd.to_datetime(s.Timestamp, errors="coerce")
    s["gain"] = s.Transmittance / s.TransmittanceRaw.replace(0, np.nan)
    s = s.dropna(subset=["ts", "gain"])
    by = s.groupby(s.ts.dt.date).gain.median()
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.semilogy(pd.to_datetime(by.index), by.values, "o-", color=BLUE, ms=6, lw=1.5, zorder=3)
    for d, lab in [("2023-05-03", "collimator / optics"), ("2023-09-22", "lab move")]:
        ax.axvline(pd.Timestamp(d), color=ACCENT, lw=1.5, ls="--", zorder=2)
        ax.text(pd.Timestamp(d), 4200, f"  {d}\n  {lab}", color=ACCENT, fontsize=9.6,
                va="top", ha="left", fontweight="bold")
    for a, b, lab in [(by.index[0], "2023-05-03", "E1"), ("2023-05-03", "2023-09-22", "E2"),
                      ("2023-09-22", by.index[-1], "E3")]:
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, 25,
                lab, ha="center", fontsize=13, color=GREY, fontweight="bold")
    ax.set_ylabel("median  T1 / TransmittanceRaw   (per analysis session)", fontsize=10.5)
    ax.set_xlabel("HIPS analysis session", fontsize=11)
    ax.set_ylim(18, 6000); ax.grid(alpha=.16, which="both")
    ax.text(.5, -.26, "one point per analysis session, all SPARTAN sites pooled; "
            "T1 itself is stable (674–818) across all three epochs —\nthis marks when the "
            "instrument configuration changed, not that tau moved",
            transform=ax.transAxes, ha="center", fontsize=9.2, color=GREY)
    _finish(fig, "f_instrument_epochs.png")


# ------------------------------------------------- 5. true-window vs same-day
def fig_true_window():
    p = REPO / "research/ftir_ec_phase3/output/tables/aeronet/ETAD_true_window_match.csv"
    if not p.exists():
        print("  -- ETAD_true_window_match.csv missing, skipping"); return
    d = pd.read_csv(p)
    if not {"H", "H_naive"} <= set(d.columns):
        print("  -- unexpected columns, skipping true-window"); return
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lim = (0, max(d.H.max(), d.H_naive.max()) * 1.06)
    ax.plot(lim, lim, ls=":", color=GREY, lw=1.6, zorder=2)
    ax.scatter(d.H_naive, d.H, s=52, color=BLUE, alpha=.72, lw=0, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("H from same-UTC-day matching  (m)", fontsize=11)
    ax.set_ylabel("H from true sampling-window matching  (m)", fontsize=11)
    ax.set_aspect("equal")
    dif = (d.H - d.H_naive).abs()
    ax.text(.04, .93, f"n = {len(d)} filters\nmedian |difference| = {dif.median():.0f} m\n"
            f"identical for {100*(dif<1e-6).mean():.0f}% of filters",
            transform=ax.transAxes, fontsize=11, va="top", color=INK)
    ax.grid(alpha=.15)
    _finish(fig, "f_truewindow_vs_sameday.png")


# ------------------------------------------------------ 6. blank-line lot swap
def fig_blankline_swap():
    from pls_transfer import FTIRTransferPaths
    h = pd.read_csv(FTIRTransferPaths.defaults().spartan_hips_primary,
                    encoding="cp1252", low_memory=False).drop_duplicates("FilterId")
    h["lot"] = h.LotId.astype(str)
    L = {k: (g.Intercept.median(), g.Slope.median())
         for k, g in h.groupby("lot") if g.Intercept.notna().any()}
    if "253" not in L or "251" not in L:
        print("  -- lot lines missing, skipping blank-line swap"); return
    x = h[h.lot.eq("253") & h.R1.notna() & h.T1.notna() & (h.T1 > 0) & h.Fabs.notna()].copy()
    def tau(R1, T1, I, S):
        top = I + S * R1
        return np.log(np.where((top > 0) & (T1 > 0), top / np.maximum(T1, 1e-9), np.nan))
    I3, S3 = L["253"]; I1, S1 = L["251"]
    d = ((tau(x.R1, x.T1, I1, S1) - tau(x.R1, x.T1, I3, S3))
         * (100.0 * x.DepositArea / x.Volume))
    d = pd.Series(d).dropna()
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.hist(d, bins=44, color=BLUE, alpha=.82, edgecolor="white", linewidth=.5)
    ax.axvline(d.median(), color=ACCENT, lw=2.2, zorder=3)
    ax.text(d.median(), ax.get_ylim()[1] * .93, f"  median {d.median():+.2f} Mm$^{{-1}}$",
            color=ACCENT, fontsize=11.5, fontweight="bold", va="top")
    ax.axvline(21.5, color=INK, lw=1.8, ls="--", zorder=3)
    ax.text(21.5, ax.get_ylim()[1] * .55, " the Addis offset\n under investigation\n (21.5)",
            color=INK, fontsize=10.5, va="top")
    ax.set_xlabel("$\\Delta$F$_{abs}$ from applying lot 251's blank line to a lot-253 filter  "
                  "(Mm$^{-1}$)", fontsize=10.5)
    ax.set_ylabel("filters", fontsize=11)
    ax.set_xlim(-1, 24)
    ax.text(.5, -.30, f"n = {len(d)} lot-253 filters; the worst plausible lot-line "
            f"mis-assignment is {100*d.median()/21.5:.1f}% of the offset",
            transform=ax.transAxes, ha="center", fontsize=9.4, color=GREY)
    _finish(fig, "f_blankline_lot_swap.png")


# --------------------------------------------------------- 7. LOCAL transfer v1
def fig_local():
    rows = [("Addis", 4.53, 0.93, 0.26, "Phoenix"),
            ("Delhi", 1.37, 2.36, 0.71, "Atlanta"),
            ("Beijing", 1.39, 1.48, 0.56, "LASU2"),
            ("Pasadena", 1.04, 4.22, 0.11, "Yosemite"),
            ("Bishoftu", 0.58, 0.93, 0.27, "FRRE1")]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(rows)); w = .36
    ax.bar(x - w/2, [r[2] for r in rows], w, color=GREY, label="global winner")
    ax.bar(x + w/2, [r[1] for r in rows], w, color=BLUE, label="LOCAL v1 (per-filter kNN-PLS)")
    ax.axhline(1.0, color=INK, ls="--", lw=1.5, zorder=3)
    ax.text(len(rows) - .42, 1.02, "slope = 1", fontsize=10, color=INK, va="bottom", ha="right")
    for i, r in enumerate(rows):
        better = abs(r[1] - 1) < abs(r[2] - 1)
        ax.text(i + w/2, r[1] + .09, f"{r[1]:.2f}", ha="center", fontsize=9.8,
                color=BLUE, fontweight="bold" if better else "normal")
        ax.text(i - w/2, r[2] + .09, f"{r[2]:.2f}", ha="center", fontsize=9.8, color=GREY)
        ax.text(i, -.42, f"nbhd: {r[4]}", ha="center", fontsize=8.8, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=11.5)
    ax.set_ylabel("slope  (1.0 is the target)", fontsize=11)
    ax.set_ylim(-.6, 5.0); ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=.16)
    ax.text(.5, -.20, "k = 200 neighbours by Pearson-on-deriv2, ncomp = 8, Deming MAC-10; "
            "Delhi and Pasadena improve markedly,\nAddis degrades on a library that has no "
            "close neighbours for it", transform=ax.transAxes, ha="center",
            fontsize=9.2, color=GREY)
    _finish(fig, "f_local_transfer_v1.png")


def _run(**kw):
    """One explorer run. spectra tokens are airspec/neutral/deriv2 - anything
    else silently falls through to RAW (cost a round-trip once; see
    SPARTAN_LOT_INVENTORY_2026-08-23.md)."""
    import urllib.request
    b = dict(cohort="ocec", cutoff=800, selection_space="raw", spectra="airspec",
             mode="site_heldout", lot="all", target="addis")
    b.update(kw)
    r = urllib.request.Request("http://127.0.0.1:5058/api/run",
                               data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))


def fig_york_explainer():
    """Why the estimator matters: y-only vs errors-in-variables on one crossplot.

    Layout note: the callouts live INSIDE the axes. An earlier version put the
    intercept callout below the axis, where it collided with the x-label and the
    caption.
    """
    d = _run()
    ev = d["eval"]
    x = np.asarray(ev["ref"], float) / 10.0          # Fabs/MAC10 -> ug/m3
    y = np.asarray(ev["pred"], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    ols_s, ols_i = np.polyfit(x, y, 1)
    lam = 2.96                                        # lambda* at MAC 10
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    dem_s = ((syy - lam * sxx) + np.sqrt((syy - lam * sxx) ** 2
                                         + 4 * lam * sxy ** 2)) / (2 * sxy)
    dem_i = y.mean() - dem_s * x.mean()

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    hi = max(x.max(), y.max()) * 1.06
    ax.plot([0, hi], [0, hi], ls=":", color=GREY, lw=1.3, zorder=1)
    ax.scatter(x, y, s=26, color=BLUE, alpha=.42, edgecolor="none", zorder=2)
    xs = np.linspace(0, hi, 50)
    ax.plot(xs, ols_s * xs + ols_i, color=BLUE, lw=2.6, zorder=3,
            label=f"y-only (OLS)   {ols_s:.2f}x {ols_i:+.2f}")
    ax.plot(xs, dem_s * xs + dem_i, color=ACCENT, lw=2.6, zorder=3,
            label=f"errors-in-variables   {dem_s:.2f}x {dem_i:+.2f}")
    ax.plot([0], [dem_i], marker="D", ms=11, color=ACCENT, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)

    # callouts inside the axes, anchored well clear of the x-label
    ax.annotate("the intercept: EC predicted\nwhere the filter shows\nzero absorption",
                xy=(0, dem_i), xytext=(0.30 * hi, dem_i - 0.02 * hi),
                fontsize=10.5, color=ACCENT, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4,
                                shrinkA=0, shrinkB=6))
    ax.annotate("both axes carry error, so a y-only fit\nflattens the slope and pulls the\n"
                "intercept toward zero",
                xy=(0.72 * hi, ols_s * 0.72 * hi + ols_i),
                xytext=(0.34 * hi, 0.90 * hi),
                fontsize=10.5, color=BLUE, va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4,
                                shrinkA=0, shrinkB=6))

    ax.set_xlabel("HIPS absorption / MAC 10   ($\\mu$g m$^{-3}$ equivalent)", fontsize=11.5)
    ax.set_ylabel("FTIR-predicted EC   ($\\mu$g m$^{-3}$)", fontsize=11.5)
    ax.set_xlim(-0.45, hi); ax.set_ylim(min(dem_i * 1.5, -0.6), hi)
    ax.legend(fontsize=10.5, loc="lower right", frameon=False)
    ax.grid(alpha=.15)
    ax.text(.5, -.155, f"Addis, n = {len(x)} filters \u00b7 lowest-OC/EC-800 \u00d7 AIRSpec, "
            f"Option A (site-grouped 5-fold) \u00b7 $\\lambda$* = 2.96 at MAC 10 "
            f"\u00b7 dotted line is 1:1",
            transform=ax.transAxes, ha="center", fontsize=9.2, color=GREY)
    _finish(fig, "f_york_explainer.png")


def _deming(x, y, lam=2.96):
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    sl = ((syy - lam*sxx) + np.sqrt((syy - lam*sxx)**2 + 4*lam*sxy**2)) / (2*sxy)
    return sl, y.mean() - sl*x.mean()


def fig_deployed_shape():
    """What the network currently ships, before any of our calibration work.

    ChemSpec 'EC PM2.5' IS the deployed FTIR EC (ftir_27: circular with EC_ftir at
    r2 = 0.9997), so this is the deployed product plotted against HIPS - the
    'before' picture, not an independent check.
    """
    from data_paths import maia_data_root
    from pls_transfer import FTIRTransferPaths
    r = maia_data_root() / "Combine csv files/Chemical Speciation/PM2.5"
    h = (pd.read_csv(FTIRTransferPaths.defaults().spartan_hips_primary, encoding="cp1252",
                     usecols=["Site", "FilterId", "Fabs"], low_memory=False)
         .drop_duplicates("FilterId"))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for ax, (site, city, col) in zip(axes, [("ETAD", "Addis", ACCENT),
                                            ("INDH", "Delhi", BLUE)]):
        d = pd.read_csv(r / f"FilterBased_ChemSpecPM25_{site}.csv", skiprows=3,
                        low_memory=False)
        ec = (d[d["Parameter_Name"].eq("EC PM2.5")][["Filter_ID", "Value"]]
              .dropna().drop_duplicates("Filter_ID"))
        hs = h[h["Site"].eq(site)].copy()
        hs["base"] = hs["FilterId"].astype(str).str.extract(r"^([A-Z]{4}-\d+)")[0]
        m = hs.merge(ec, left_on="base", right_on="Filter_ID").dropna(subset=["Fabs"])
        x = (m["Fabs"] / 10.0).to_numpy(float)
        y = m["Value"].to_numpy(float)
        ds, di = _deming(x, y)
        os_, oi = np.polyfit(x, y, 1)
        hi = max(x.max(), y.max()) * 1.06
        ax.plot([0, hi], [0, hi], ls=":", color=GREY, lw=1.3)
        ax.scatter(x, y, s=30, color=col, alpha=.55, edgecolor="none")
        xs = np.linspace(0, hi, 50)
        ax.plot(xs, ds*xs + di, color=INK, lw=2.4)
        ax.plot([0], [di], marker="D", ms=10, color=INK, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.set_title(f"{city}  \u00b7  n = {len(m)}", fontsize=12.5)
        ax.set_xlabel("HIPS absorption / MAC 10   ($\\mu$g m$^{-3}$ equivalent)", fontsize=10.5)
        ax.set_ylabel("deployed FTIR EC   ($\\mu$g m$^{-3}$)" if site == "ETAD" else "",
                      fontsize=10.5)
        ax.set_xlim(0, hi); ax.set_ylim(min(di*1.4, -0.5), hi)
        ax.grid(alpha=.15)
        ax.text(.04, .95, f"Deming  {ds:.2f}x {di:+.2f}\nOLS       {os_:.2f}x {oi:+.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor=GREY, alpha=.92))
    fig.text(.5, -.02, "SPARTAN's currently deployed FTIR EC (ChemSpec 'EC PM2.5') against HIPS "
             "absorption \u2014 before any of this project's calibration work.\n"
             "Note this is the deployed product, not an independent reference: ChemSpec EC is "
             "EC_ftir rounded (ftir_27). Dotted line is 1:1; diamond marks the intercept.",
             ha="center", fontsize=9.2, color=GREY)
    _finish(fig, "f_deployed_shape.png")


def _deployed_fit(site):
    """Deming fit of SPARTAN's DEPLOYED FTIR EC against HIPS, for one site.

    Reference only: a different (ChemSpec-covered) filter set from our evaluation
    sets, so it is not a like-for-like row in the ranked comparisons.
    """
    from data_paths import maia_data_root
    from pls_transfer import FTIRTransferPaths
    r = maia_data_root() / "Combine csv files/Chemical Speciation/PM2.5"
    h = (pd.read_csv(FTIRTransferPaths.defaults().spartan_hips_primary, encoding="cp1252",
                     usecols=["Site", "FilterId", "Fabs"], low_memory=False)
         .drop_duplicates("FilterId"))
    d = pd.read_csv(r / f"FilterBased_ChemSpecPM25_{site}.csv", skiprows=3, low_memory=False)
    ec = (d[d["Parameter_Name"].eq("EC PM2.5")][["Filter_ID", "Value"]]
          .dropna().drop_duplicates("Filter_ID"))
    hs = h[h["Site"].eq(site)].copy()
    hs["base"] = hs["FilterId"].astype(str).str.extract(r"^([A-Z]{4}-\d+)")[0]
    m = hs.merge(ec, left_on="base", right_on="Filter_ID").dropna(subset=["Fabs"])
    x = (m["Fabs"] / 10.0).to_numpy(float)
    y = m["Value"].to_numpy(float)
    sl, ic = _deming(x, y)
    return sl, ic, len(m)


def _collapse_sets():
    """Top-8 configurations under two tuning strategies, joined across both sites."""
    rows = []
    with open(REPO / "calibration_explorer/cache/batch_results.jsonl") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    df = pd.DataFrame(rows)
    df = df[df["mode"].eq("site_heldout")]
    key = ["cohort", "cutoff", "selection_space", "spectra", "k"]
    a = df[df["target"].eq("addis")].set_index(key)
    d = df[df["target"].eq("indh")].set_index(key)
    both = a.join(d, rsuffix="_d", how="inner").reset_index()
    both = both.dropna(subset=["deming_slope", "deming_intercept",
                               "deming_slope_d", "deming_intercept_d"])
    sc_a = both["deming_intercept"].abs() + 0.5 * (both["deming_slope"] - 1).abs()
    sane_a = both["deming_slope"].between(0.85, 1.18)
    sane_b = sane_a & both["deming_slope_d"].between(0.85, 1.18)
    sc_j = sc_a + both["deming_intercept_d"].abs() + 0.5*(both["deming_slope_d"]-1).abs()
    tuned = both[sane_a].assign(sc=sc_a[sane_a]).nsmallest(8, "sc")
    joint = both[sane_b].assign(sc=sc_j[sane_b]).nsmallest(8, "sc")
    return tuned, joint


def _pair_panel(ax, tuned, joint, col, title, is_slope, deployed=None):
    idx = np.arange(8); w = .38
    b1 = ax.bar(idx - w/2, tuned[col].to_numpy(float), w, color=ACCENT,
                label="tuned on Addis alone")
    b2 = ax.bar(idx + w/2, joint[col].to_numpy(float), w, color=BLUE,
                label="tuned on both cities")
    if is_slope:
        ax.axhspan(.85, 1.18, color=GREY, alpha=.16, zorder=0)
        ax.axhline(1.0, color=INK, lw=1.1, ls="--", zorder=1)
    else:
        ax.axhline(0.0, color=INK, lw=1.1, ls="--", zorder=1)
    if deployed is not None:
        ax.axhline(deployed, color=AMBER, lw=2.2, ls=(0, (5, 2)), zorder=4)
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.text(-0.45, deployed + 0.022 * span,
                f"deployed SPARTAN  {deployed:+.2f}", color=AMBER, fontsize=9.8,
                va="bottom", ha="left", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none", alpha=.85))
    ax.set_title(title, fontsize=12.5)
    ax.set_xticks(idx); ax.set_xticklabels([f"#{i+1}" for i in idx], fontsize=10)
    ax.grid(axis="y", alpha=.15)
    return b1, b2


def fig_collapse_addis():
    """At Addis the Addis-tuned set genuinely wins - that is the trap."""
    tuned, joint = _collapse_sets()
    dsl, dic, dn = _deployed_fit("ETAD")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    b1, b2 = _pair_panel(axes[0], tuned, joint, "deming_slope",
                         "Deming slope at Addis", True, deployed=dsl)
    _pair_panel(axes[1], tuned, joint, "deming_intercept",
                "Deming intercept at Addis   ($\\mu$g m$^{-3}$)", False, deployed=dic)
    fig.legend([b1, b2], ["tuned on Addis alone", "tuned on both cities"],
               loc="upper center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, 1.06))
    fig.text(.5, -.14, "the 8 best configurations under each tuning strategy, ranked "
             "1\u20138 within the strategy \u00b7 Option A \u00b7 shaded band and dashed lines "
             "mark the target (slope 0.85\u20131.18, intercept 0)\n"
             "Tuning on Addis alone drives the Addis intercept almost to zero \u2014 it beats "
             "the both-cities set on Addis's own terms.\n"
             f"Amber = SPARTAN's DEPLOYED calibration on its ChemSpec-covered filters "
             f"(n = {dn}) \u2014 a reference level, not a like-for-like row.",
             ha="center", fontsize=9.4, color=GREY)
    _finish(fig, "f_collapse_1_addis.png")


def fig_collapse_delhi():
    """The same configurations at Delhi."""
    tuned, joint = _collapse_sets()
    dsl, dic, dn = _deployed_fit("INDH")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    b1, b2 = _pair_panel(axes[0], tuned, joint, "deming_slope_d",
                         "Deming slope at Delhi", True, deployed=dsl)
    _pair_panel(axes[1], tuned, joint, "deming_intercept_d",
                "Deming intercept at Delhi   ($\\mu$g m$^{-3}$)", False, deployed=dic)
    fig.legend([b1, b2], ["tuned on Addis alone", "tuned on both cities"],
               loc="upper center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, 1.06))
    fig.text(.5, -.14, "the SAME eight configurations, applied to Delhi \u00b7 slopes of 4\u20139 "
             "and intercepts of \u221215 to \u221245 $\\mu$g m$^{-3}$\n"
             "for scale, the whole Addis offset under discussion is about \u22121.4 "
             "$\\mu$g m$^{-3}$\n"
             f"Amber = SPARTAN's DEPLOYED calibration on its ChemSpec-covered filters "
             f"(n = {dn}) \u2014 a reference level, not a like-for-like row.",
             ha="center", fontsize=9.4, color=GREY)
    _finish(fig, "f_collapse_2_delhi.png")


def fig_collapse_skill():
    """And they had no held-out skill to begin with."""
    tuned, joint = _collapse_sets()
    idx = np.arange(8); w = .38
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    ax.bar(idx - w/2, tuned["heldout_R2"].to_numpy(float), w, color=ACCENT,
           label="tuned on Addis alone")
    ax.bar(idx + w/2, joint["heldout_R2"].to_numpy(float), w, color=BLUE,
           label="tuned on both cities")
    for i, (v1, v2) in enumerate(zip(tuned["heldout_R2"], joint["heldout_R2"])):
        ax.text(i - w/2, float(v1) + .02, f"{float(v1):.2f}", ha="center",
                fontsize=9, color=ACCENT)
        ax.text(i + w/2, float(v2) + .02, f"{float(v2):.2f}", ha="center",
                fontsize=9, color=BLUE)
    ax.set_ylabel("held-out TOR R$^2$", fontsize=11.5)
    ax.set_xticks(idx); ax.set_xticklabels([f"#{i+1}" for i in idx], fontsize=10)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=.15)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    fig.text(.5, -.06, "held-out TOR R$^2$ at Addis \u2014 the honest skill test, on filters the "
             "cohort selection never saw\nthe Addis-tuned set sits at 0.10\u20130.33 against "
             "0.34\u20130.80: it was overfit the whole time",
             ha="center", fontsize=9.4, color=GREY)
    _finish(fig, "f_collapse_3_skill.png")


if __name__ == "__main__":
    print(f"writing to {OUT.relative_to(REPO)}")
    for fn in (fig_york, fig_york_explainer, fig_deployed_shape,
               fig_collapse_addis, fig_collapse_delhi,
               fig_collapse_skill, fig_mac_invariance, fig_slope_trap,
               fig_epochs, fig_true_window, fig_blankline_swap, fig_local):
        try:
            fn()
        except Exception as exc:
            print(f"  !! {fn.__name__}: {type(exc).__name__}: {exc}")
