# %% [markdown]
# # ftir_34 — the offset-adjudication checks, reproduced in one place
#
# ## tl;dr
#
# (filled in by the finalize step after execution)
#
# ## Context & Methods
#
# The 2026-08-22/23 push (five-site cross-evaluation → York/EIV re-fit →
# blank-line test → band-story corrections → AERONET cross-checks) was
# adjudicated in `OFFSET_ADJUDICATION_2026-08-23.md`, but several of the
# analyses behind its numbers ran as interactive one-offs. This notebook is
# the committed reproduction: every number quoted in that doc's §1–§2b and
# §5b either comes from a committed module (`york_cross_site.py`,
# `calibration_explorer/hips_lab.py`) or is re-derived below.
#
# **Requires the calibration explorer on :5058** (predictions come from
# `/api/run` so the fits are exactly the app's; everything is file-cached, so
# re-execution is minutes, not hours).
#
# Sections:
# 1. **Five-site band tables** — including the *metric autopsy*: the
#    linear-baseline "prominence" that made Delhi look like Addis, versus the
#    interior-local-maximum test that corrected it (BAND1617_LEAD arc).
# 2. **Per-filter residual chemistry** — the 1617-band and carbonyl
#    correlations with calibration residuals at the four SPARTAN targets,
#    with the detrending that separates "tracks the slope" from "tracks the
#    scatter".
# 3. **AERONET cross-checks** — the AAE-based non-BC apportionment bound and
#    the Addis per-filter feature-attribution table (which spectral feature
#    predicts column AAOD₆₇₅ under which controls, with permutation p).
# 4. **York × blank-line variants** — the §5b table, via `hips_lab`.
#
# Winner configuration throughout: lowest-OC/EC 450 × AIRSpec, protocol A,
# k = 9, MAC 10, all pairs.

# %%
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd().resolve()
while not (REPO / "calibration_explorer").exists():
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "calibration_explorer"))
sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))
import hips_lab  # noqa: E402
import phase3_common as pc  # noqa: E402

API = "http://127.0.0.1:5058/api/run"
WINNER = {"cohort": "ocec", "cutoff": 450, "spectra": "airspec",
          "mode": "site_heldout", "k": 9}
TARGETS = REPO / "calibration_explorer/targets"
P3 = REPO / "research/ftir_ec_phase3"
rng = np.random.default_rng(20260823)


def api_run(target):
    req = urllib.request.Request(
        API, json.dumps({**WINNER, "target": target}).encode(),
        {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def corrected_target(name):
    """Corrected spectra matrix + wavenumbers for a custom target."""
    cp = pd.read_csv(TARGETS / name / "spectra_corrected.csv")
    wn = np.array([float(c) for c in cp.columns[1:]])
    return cp.iloc[:, 1:].to_numpy(float), wn


# Addis corrected spectra, per-MediaId, in the evaluation order
etad_npz = np.load(P3 / "output/corrected/etad_corrected_df6.npz",
                   allow_pickle=True)
WN = etad_npz["wn"].astype(float)
_etad = pd.DataFrame(etad_npz["corrected"].astype(float))
_etad["MediaId"] = etad_npz["media_id"].astype(int)
ETAD_BY_MEDIA = _etad.groupby("MediaId").mean()
EV = next(x for x in pc.load_addis_evaluation()
          if isinstance(x, pd.DataFrame) and "MediaId" in x.columns)

# %% [markdown]
# ## 1. Five-site band tables, and why the first read was wrong
#
# Two metrics on the same corrected median spectra:
#
# - **prominence** — A(1617) − mean(A(1700), A(1500)): the linear-baseline
#   form used in the first pass. At Delhi this reads the *flank of the
#   1700–1720 carbonyl band* as "band".
# - **interior local max** — is there a genuine local maximum in 1560–1680,
#   and where? Only a real band produces one away from the window edge.
#
# The prominence column reproduces the retracted "Delhi matches Addis
# exactly" number; the local-max column reproduces its correction.

# %%
def median_spectrum(name):
    if name == "addis":
        return ETAD_BY_MEDIA.median(axis=0).to_numpy(), WN
    M, wn = corrected_target(name)
    return np.median(M, axis=0), wn


rows = []
for name, label in [("addis", "Addis"), ("indh", "Delhi"),
                    ("chts", "Beijing"), ("etbi", "Bishoftu"),
                    ("uspa", "Pasadena")]:
    med, wn = median_spectrum(name)
    at = lambda t: float(med[np.argmin(abs(wn - t))])  # noqa: E731
    win = (wn >= 1560) & (wn <= 1680)
    iw, mw = med[win], wn[win]
    peak_i = int(np.argmax(iw))
    # "interior" = the maximum sits clearly away from either window edge
    # (>= ~4 cm-1 in); a peak on the edge is the flank of a band outside
    # the window (Delhi/Beijing: the 1700-1720 carbonyl), not a band here.
    interior = 3 <= peak_i <= win.sum() - 4
    rows.append({
        "site": label,
        "prominence_1617": round(at(1617) - (at(1700) + at(1500)) / 2, 4),
        "carbonyl_1720": round(at(1720), 4),
        "peak_at_cm1": int(round(mw[peak_i])),
        "interior_local_max": bool(interior),
    })
band_tbl = pd.DataFrame(rows).set_index("site")
print(band_tbl.to_string())
print("\nProminence says Addis == Delhi (the retracted read); the local-max "
      "column shows Delhi/Beijing/Bishoftu peak at the window edge — the "
      "carbonyl flank, not a band. Only Addis has an interior ~1620 peak on "
      "AIRSpec-corrected spectra. (Under a NEUTRAL baseline ETBI shows it "
      "too — BAND1617_LEAD correction 2 — AIRSpec's 1520–1600 anchor "
      "suppresses weak versions of the band by construction.)")

# %% [markdown]
# ## 2. Per-filter residual chemistry at the four SPARTAN targets
#
# Residual = prediction − Fabs/10 (the raw 1:1 residual), and the detrended
# residual = prediction − (each site's own Deming line). The contrast is the
# point: carbonyl correlates with the RAW residual almost perfectly at the
# slope-anomalous sites and collapses when the site's own slope is removed —
# carbonyl identifies *which sites have inflated slopes*, not which filters
# scatter.

# %%
def band_features(name):
    M, wn = corrected_target(name)
    at = lambda t: M[:, np.argmin(abs(wn - t))]          # noqa: E731
    reg = lambda lo, hi: M[:, (wn >= lo) & (wn <= hi)].mean(axis=1)  # noqa: E731
    return {"band": at(1617) - (at(1700) + at(1500)) / 2,
            "co": at(1720) - (reg(1755, 1765) + reg(1685, 1695)) / 2,
            "ch": at(2920)}


def partial_r(y, x, z):
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    return np.corrcoef(rx, ry)[0, 1]


chem = []
for name in ("etbi", "indh", "chts", "uspa"):
    d = api_run(name)
    pred = np.array(d["eval"]["pred"], float)
    x = np.array(d["eval"]["ref"], float) / 10.0
    resid = pred - x
    m10 = [m for m in d["metrics"]
           if m["MAC"] == 10 and m["evaluation_set"] == "all"][0]
    detr = pred - (m10["deming_slope"] * x + m10["deming_intercept"])
    f = band_features(name)
    ref = pd.read_csv(TARGETS / name / "reference.csv")
    vol = ref["Volume_m3"].to_numpy(float)
    band_v, co_v, ch_v = f["band"] / vol, f["co"] / vol, f["ch"] / vol
    chem.append({
        "site": name,
        "r_band_resid": round(np.corrcoef(band_v, resid)[0, 1], 2),
        "r_co_resid": round(np.corrcoef(co_v, resid)[0, 1], 2),
        "r_co_resid_partial_ch": round(partial_r(resid, co_v, ch_v), 2),
        "r_co_DETRENDED": round(np.corrcoef(co_v, detr)[0, 1], 2),
        "deming": f"{m10['deming_slope']:.2f}x{m10['deming_intercept']:+.2f}",
    })
chem_tbl = pd.DataFrame(chem).set_index("site")
print(chem_tbl.to_string())
print("\nDelhi/Pasadena: r(co, resid) ~ 0.9 raw, ~0 detrended -> carbonyl "
      "tracks the SLOPE (which sites over-read), not the scatter. The two "
      "carbonyl-rich aerosols are exactly the slope-anomalous sites.")

# %% [markdown]
# ## 3. AERONET cross-checks (Addis n = 125 matched filters)
#
# **3a — the apportionment bound.** Non-BC share of AAOD₆₇₅ under the
# textbook BC-AAE = 1 anchor extrapolated from 870 nm:
# frac = 1 − (870/675)^(1 − AAE). Blind to AAE ≈ 1–2 absorbers (char, dark
# BrC) for the same reason ftir_28 found the MA350 cannot resolve a red
# excess — do not quote it as a refutation of char.
#
# **3b — feature attribution.** Which corrected-spectrum feature predicts
# the *sun photometer's* column AAOD₆₇₅, controlling loading (CH), the
# filter's own Fabs, and month fixed-effects? The discrete 1620 peak dies
# under month control; the broad 1500–1700 envelope survives (p ≈ 0.002);
# O–H shows nothing (specificity). The photometer never touches the filter,
# so no filter artifact produces the surviving correlation.

# %%
AER = P3 / "output/tables/aeronet"
print("3a: non-BC apportionment bound at 675 nm")
for site in ("ETAD", "INDH", "CHTS", "USPA"):
    df = pd.read_csv(AER / f"{site}_matched_daily.csv").dropna(
        subset=["Absorption_AOD[675nm]",
                "Absorption_Angstrom_Exponent_440-870nm"])
    aae = df["Absorption_Angstrom_Exponent_440-870nm"]
    frac = 1 - (870 / 675) ** (1 - aae)
    surf = frac * df["Absorption_AOD[675nm]"] * 1e6 / df["H_m"]
    print(f"  {site}: n={len(df):3d}  median frac_nonBC = "
          f"{frac.median() * 100:5.1f}%  surface-equiv "
          f"{surf.median():5.2f} Mm-1")

# %%
Xg = ETAD_BY_MEDIA
at = lambda t: Xg.values[:, np.argmin(abs(WN - t))]      # noqa: E731
reg = lambda lo, hi: Xg.values[:, (WN >= lo) & (WN <= hi)].mean(axis=1)  # noqa: E731
feats = pd.DataFrame({
    "old_prominence": at(1617) - (at(1700) + at(1500)) / 2,
    "tight_1620_peak": reg(1610, 1630) - (reg(1560, 1580) + reg(1650, 1670)) / 2,
    "carbonyl": at(1720) - (reg(1755, 1765) + reg(1685, 1695)) / 2,
    "envelope_1500_1700": reg(1500, 1700),
    "oh_3100_3400": reg(3100, 3400),
    "ch": at(2920)}, index=Xg.index)
j = (EV[["MediaId", "ExternalFilterId"]].astype({"MediaId": int})
     .merge(feats, left_on="MediaId", right_index=True)
     .merge(pd.read_csv(AER / "ETAD_matched_daily.csv"),
            left_on="ExternalFilterId", right_on="FilterId")
     .dropna(subset=["Absorption_AOD[675nm]"]))
mo = pd.to_datetime(j["SampleDate"]).dt.month
month_d = [(mo == k).astype(float).to_numpy() for k in range(2, 13)]
y = j["Absorption_AOD[675nm]"].to_numpy(float)
ch_, fabs = j["ch"].to_numpy(float), j["Fabs"].to_numpy(float)


def mpartial(y_, x_, Z):
    Z = np.column_stack([np.ones(len(y_))] + Z)
    ry = y_ - Z @ np.linalg.lstsq(Z, y_, rcond=None)[0]
    rx = x_ - Z @ np.linalg.lstsq(Z, x_, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


print(f"\n3b: feature -> AAOD675 | CH, Fabs, month   (n={len(j)})")
ctrl = [ch_, fabs] + month_d
for name in ("old_prominence", "tight_1620_peak", "carbonyl",
             "envelope_1500_1700", "oh_3100_3400"):
    x = j[name].to_numpy(float)
    r = mpartial(y, x, ctrl)
    null = [mpartial(y, rng.permutation(x), ctrl) for _ in range(1000)]
    p = (np.sum(np.abs(null) >= abs(r)) + 1) / 1001
    print(f"  {name:20s} r = {r:+.3f}  perm p = {p:.3f}")

# %% [markdown]
# ## 4. York × blank-line variants (the §5b table, via `hips_lab`)
#
# The committed module recomputes each filter's Fabs from raw optics
# (R1/T1 + the lot blank line) under three blank-line forms and York-fits
# prediction vs Fabs/10 with per-filter σₓ. The delta between variants is
# independent of EC entirely; the residual −1.27 at Addis still carries the
# offset-vs-curvature degeneracy that only quartz TOR resolves.

# %%
print("blank ledger (per lot):")
for lot, L in sorted(hips_lab.blank_lines().items()):
    print(f"  lot {lot:5s} n={L['n']:3d}  rms lin/quad "
          f"{L['rms_lin']:5.1f}/{L['rms_quad']:5.1f}  "
          f"R1 {L['r1_min']:.0f}-{L['r1_max']:.0f}  "
          f"tau0 {L['tau0_mean']:+.4f}±{L['tau0_sd']:.4f}")

print(f"\n{'site':9s} {'match':>5s} {'<blank':>6s} | "
      f"{'deployed':>17s} | {'lot-linear':>17s} | {'lot-quadratic':>17s}")
for name, code in (("addis", "ETAD"), ("indh", "INDH"), ("chts", "CHTS"),
                   ("etbi", "ETBI"), ("uspa", "USPA")):
    d = api_run(name)
    if name == "addis":
        fids = EV["ExternalFilterId"].astype(str).tolist()
    else:
        fids = pd.read_csv(TARGETS / name / "reference.csv")[
            "ExternalFilterId"].astype(str).tolist()
    row = hips_lab.site_rows(d["eval"]["pred"], d["eval"]["ref"], fids, code)
    cells = []
    for kind in ("deployed", "lot_lin", "lot_quad"):
        f = row["fits"][kind]
        cells.append(f"{f['slope']:.2f}x{f['intercept']:+.2f}"
                     f"±{f['intercept_se']:.2f}" if "error" not in f
                     else f["error"])
    print(f"{name:9s} {row['n_matched']:5d} "
          f"{row['frac_below_blank_r1'] * 100:5.0f}% | "
          + " | ".join(f"{c:>17s}" for c in cells))

# %% [markdown]
# ## Takeaways
#
# (filled in by the finalize step after execution)
