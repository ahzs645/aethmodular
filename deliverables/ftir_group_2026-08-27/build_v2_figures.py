"""Figures for the post-run-through rebuild (Ann's Aug 27 edit plan).

New/reworked: split of the blurry spectrum+histogram pair, analog-explainer
right half only, before/after baselining crossplot pair, the winner crossplot,
the filters-differ-by-site darkness figure (axes defined, no fit lines), the
cross-site spectra rework (baseline-corrected wording, CH labeled, 1700 band,
intercepts in ug/m3), and the lot-253 share rebuild with black in-plot text.

Run: MPLBACKEND=Agg EXPLORER_PORT=5058 ~/anaconda3/bin/python build_v2_figures.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "calibration_explorer"))
PORT = os.environ.get("EXPLORER_PORT", "5058")

INK, GREY, BLUE, PURPLE, ACCENT, AMBER = ("#22252A", "#8F8C84", "#2C6E9E",
                                          "#7A4FA3", "#B23327", "#C49442")
SITE = [("addis", "ETAD", "Addis", "#F39C12"), ("indh", "INDH", "Delhi", "#3498DB"),
        ("chts", "CHTS", "Beijing", "#E74C3C"), ("etbi", "ETBI", "Bishoftu", "#7A4FA3"),
        ("uspa", "USPA", "Pasadena", "#2ECC71")]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E8E6E1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white",
                     "savefig.dpi": 165})


def rgb(name):
    p = FIG / name
    im = Image.open(p)
    if im.mode != "RGB":
        im.convert("RGB").save(p)


def api(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/run",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def deming(x, y, lam=2.96):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    sxx, syy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean()
    sxy = ((x - mx) * (y - my)).mean()
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2
                                   + 4 * lam * sxy ** 2)) / (2 * sxy)
    return b, my - b * mx


def split_f14():
    """The spectrum+histogram pair rendered blurry side by side; split into
    two full-resolution figures at the whitespace gutter."""
    im = np.asarray(Image.open(FIG / "f14_teflon.png").convert("L"))
    w = im.shape[1]
    band = im[:, w // 3: 2 * w // 3].min(axis=0)
    gap = int(np.argmax(band == 255)) if (band == 255).any() else band.argmax()
    cut = w // 3 + gap
    full = Image.open(FIG / "f14_teflon.png")
    full.crop((0, 0, cut + 10, full.height)).save(FIG / "f14a_spectrum.png")
    full.crop((max(cut - 10, 0), 0, full.width, full.height)).save(
        FIG / "f14b_background_hist.png")
    rgb("f14a_spectrum.png")
    rgb("f14b_background_hist.png")
    print(f"f14 split at x={cut}")


def crop_f13():
    """Keep only the ranked histograms (right column) of the analog
    explainer; the scatter panels were diagnostic, not intuitive."""
    im = Image.open(FIG / "f13_analog_explainer.png")
    im.crop((1020, 0, im.width, im.height)).save(FIG / "f13b_analog_ranked.png")
    rgb("f13b_analog_ranked.png")
    print("f13 cropped to ranked histograms")


def fig_before_after():
    """One before/after: IMPROVE-network calibration set, raw vs
    baseline-corrected, predicting Addis. The single 'baselining matters'
    graph Ann asked for."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    for ax, (spectra, lab) in zip(axes, [("raw", "raw spectra"),
                                         ("airspec", "baseline-corrected")]):
        d = api({"cohort": "pool", "spectra": spectra, "mode": "site_heldout",
                 "target": "addis"})
        pred = np.array(d["eval"]["pred"], float)
        x = np.array(d["eval"]["ref"], float) / 10.0
        b, a = deming(x, pred)
        lim = max(x.max(), pred.max(), 1) * 1.06
        ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.2)
        ax.scatter(x, pred, s=14, color=BLUE if spectra == "airspec" else GREY,
                   alpha=0.6)
        xs = np.linspace(0, lim, 20)
        ax.plot(xs, b * xs + a, color=INK, lw=1.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(min(0, a) - 0.3, lim)
        ax.set_xlabel(f"HIPS EC-equivalent, Fabs/10 (µg/m³)")
        ax.set_ylabel("predicted FTIR EC (µg/m³)")
        ax.text(0.03, 0.97, f"{lab}\nDeming {b:.2f}x{a:+.2f}\nk = {d['k']}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(fc="white", ec="#DDDAD2", alpha=0.9))
    fig.tight_layout()
    fig.savefig(FIG / "f_before_after_baseline.png")
    plt.close(fig)
    rgb("f_before_after_baseline.png")
    print("before/after done")


def fig_winner_crossplot():
    """The crossplot of the search winner: 'the crossplot is what we look
    at' (Ann). Winner config at Addis, all pairs, Deming lambda*."""
    d = api({"cohort": "ocec", "cutoff": 450, "spectra": "airspec",
             "mode": "site_heldout", "target": "addis", "k": 9})
    pred = np.array(d["eval"]["pred"], float)
    x = np.array(d["eval"]["ref"], float) / 10.0
    m = [r for r in d["metrics"] if r["MAC"] == 10
         and r["evaluation_set"] == "all"][0]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    lim = max(x.max(), pred.max()) * 1.06
    ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.2)
    ax.scatter(x, pred, s=16, color="#F39C12", alpha=0.65)
    xs = np.linspace(0, lim, 20)
    ax.plot(xs, m["deming_slope"] * xs + m["deming_intercept"], color=INK, lw=1.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(min(0, m["deming_intercept"]) - 0.3, lim)
    ax.set_xlabel("HIPS EC-equivalent, Fabs/10 (µg/m³)")
    ax.set_ylabel("predicted FTIR EC (µg/m³)")
    ax.text(0.03, 0.97,
            (f"lowest-OC/EC 450, baseline-corrected, k=9\n"
             f"Deming {m['deming_slope']:.2f}x{m['deming_intercept']:+.2f}\n"
             f"OLS {m['ols_slope']:.2f}x{m['ols_intercept']:+.2f}\n"
             f"target R² {m['R2']:.2f} · held-out TOR R² "
             f"{d['heldout']['R2']:.2f} · n = {len(x)}"),
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(fc="white", ec="#DDDAD2", alpha=0.9))
    fig.tight_layout()
    fig.savefig(FIG / "f_winner_crossplot.png")
    plt.close(fig)
    rgb("f_winner_crossplot.png")
    print("winner crossplot done")


def fig_site_darkness():
    """The filters themselves differ by site: reflectance-counts
    distributions, axes in plain words, no fit lines, no blank cloud."""
    import hips_lab
    b = hips_lab.batch()
    pm = b[b.FilterType == "PM2.5"]
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    for _, code, label, c in SITE:
        r1 = pm[pm.Site == code]["R1"].dropna()
        ax.hist(r1, bins=np.arange(60, 285, 6), histtype="step", lw=2.0,
                color=c, density=True, label=f"{label} (n={len(r1)})")
    ax.set_xlabel("R1: reflectance at the filter (detector counts) · "
                  "lower = darker deposit")
    ax.set_ylabel("share of filters")
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "f_site_darkness.png")
    plt.close(fig)
    rgb("f_site_darkness.png")
    print("site darkness done")


def _median_corrected():
    out = {}
    npz = np.load(REPO / "research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz",
                  allow_pickle=True)
    wn = npz["wn"].astype(float)
    X = pd.DataFrame(npz["corrected"].astype(float))
    X["MediaId"] = npz["media_id"].astype(int)
    out["addis"] = (wn, X.groupby("MediaId").mean().median(axis=0).to_numpy())
    for name, _, _, _ in SITE:
        if name == "addis":
            continue
        df = pd.read_csv(REPO / f"calibration_explorer/targets/{name}/spectra_corrected.csv")
        cw = np.array([float(c) for c in df.columns[1:]])
        out[name] = (cw, np.median(df.iloc[:, 1:].to_numpy(float), axis=0))
    return out


def fig_cross_site_spectra_v2():
    """Full-slide median baseline-corrected spectra, five sites, with the
    bands the group thinks about labeled (CH ~2920, carbonyl ~1700)."""
    med = _median_corrected()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for name, code, label, c in SITE:
        wn, v = med[name]
        o = np.argsort(-wn)
        axes[0].plot(wn[o], v[o], color=c, lw=1.5, label=label)
        m = (wn >= 1350) & (wn <= 1850)
        axes[1].plot(wn[m], v[m], color=c, lw=1.7)
    for ax in axes:
        ax.invert_xaxis()
        ax.set_xlabel("wavenumber (cm⁻¹)")
    axes[0].set_ylabel("baseline-corrected absorbance (median filter)")
    axes[0].axvline(2920, color=GREY, lw=0.9, ls=":")
    axes[0].text(2920, axes[0].get_ylim()[1] * 0.95, " CH (~2920)",
                 fontsize=9, color="#6E7178")
    for ax in axes:
        ax.axvline(1700, color=GREY, lw=0.9, ls=":")
    axes[1].text(1700, axes[1].get_ylim()[1] * 0.95, " carbonyl (~1700)",
                 fontsize=9, color="#6E7178")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    axes[1].set_xlim(1850, 1350)
    fig.tight_layout()
    fig.savefig(FIG / "f_cross_site_spectra_v2.png")
    plt.close(fig)
    rgb("f_cross_site_spectra_v2.png")
    print("cross-site spectra v2 done")


def fig_band_vs_intercept():
    """Carbonyl (~1700, the band the group thinks about) per CH vs each
    site's Deming intercept in ug/m3 (consistent units with every other
    slide)."""
    med = _median_corrected()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, code, label, c in SITE:
        wn, v = med[name]
        at = lambda t: float(v[np.argmin(abs(wn - t))])  # noqa: E731
        reg = lambda lo, hi: float(np.mean(v[(wn >= lo) & (wn <= hi)]))  # noqa: E731
        co = at(1700) - (reg(1755, 1765) + reg(1645, 1655)) / 2
        ratio = co / max(at(2920), 1e-6)
        d = api({"cohort": "ocec", "cutoff": 450, "spectra": "airspec",
                 "mode": "site_heldout", "target": name, "k": 9})
        m = [r for r in d["metrics"] if r["MAC"] == 10
             and r["evaluation_set"] == "all"][0]
        ax.scatter([ratio], [m["deming_intercept"]], s=140, color=c, zorder=3)
        ax.annotate(label, (ratio, m["deming_intercept"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("carbonyl (~1700 cm⁻¹) band height per CH (~2920 cm⁻¹), "
                  "baseline-corrected median")
    ax.set_ylabel("Deming intercept (µg/m³), winner calibration")
    fig.tight_layout()
    fig.savefig(FIG / "f_band_vs_intercept_v2.png")
    plt.close(fig)
    rgb("f_band_vs_intercept_v2.png")
    print("band vs intercept done")


def fig_lot253_v2():
    """SPARTAN lot share by quarter, rebuilt: black text, inside the axes,
    nothing clipped."""
    import hips_lab
    b = pd.read_csv(hips_lab._BATCH_PATH[0] if hips_lab._BATCH_PATH
                    else hips_lab._default_batch_path(), encoding="cp1252",
                    usecols=["FilterType", "LotId", "SampleDate"])
    b["LotId"] = b["LotId"].astype(str).str.strip()
    pm = b[b.FilterType == "PM2.5"].copy()
    pm["SampleDate"] = pd.to_datetime(pm["SampleDate"], errors="coerce")
    pm = pm.dropna(subset=["SampleDate"])
    pm["q"] = pm["SampleDate"].dt.to_period("Q").dt.start_time
    lots = ["248", "250", "251", "253"]
    colors = {"248": AMBER, "250": GREY, "251": BLUE, "253": ACCENT}
    qs = sorted(pm["q"].unique())
    qs = [q for q in qs if q >= pd.Timestamp("2022-01-01")]
    shares = {lot: [] for lot in lots + ["other"]}
    for q in qs:
        g = pm[pm["q"] == q]
        n = max(len(g), 1)
        acc = 0
        for lot in lots:
            v = (g["LotId"] == lot).sum() / n * 100
            shares[lot].append(v)
            acc += v
        shares["other"].append(100 - acc)
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    ax.stackplot(qs, [shares[lot] for lot in lots] + [shares["other"]],
                 labels=[f"lot {lot}" for lot in lots] + ["other lots"],
                 colors=[colors[lot] for lot in lots] + ["#E4E1DB"],
                 edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of SPARTAN filters sampled (%)")
    ax.set_xlabel("quarter")
    ax.text(0.02, 0.06, "calibration basis: IMPROVE lots 248 + 251",
            transform=ax.transAxes, fontsize=10, color=INK,
            bbox=dict(fc="white", ec="#DDDAD2", alpha=0.9))
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "f_lot253_takeover_v2.png")
    plt.close(fig)
    rgb("f_lot253_takeover_v2.png")
    print("lot253 v2 done")


if __name__ == "__main__":
    split_f14()
    crop_f13()
    fig_site_darkness()
    fig_lot253_v2()
    fig_cross_site_spectra_v2()
    fig_before_after()
    fig_winner_crossplot()
    fig_band_vs_intercept()
    print("all v2 figures done")
