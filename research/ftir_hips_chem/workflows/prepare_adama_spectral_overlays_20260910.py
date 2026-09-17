"""Ann's five-Adama-on-Addis overlays, with explicit raw-scale and season labels."""

from pathlib import Path
import hashlib
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "research/ftir_hips_chem"
sys.path.insert(0, str(ANALYSIS / "scripts"))
sys.path.insert(0, str(REPO / "research/ftir_ec_phase3/scripts"))
from config import ETHIOPIA_SEASONS, season_convention_name
from outliers import apply_exclusion_flags, get_clean_data
from plotting import PlotConfig
from plotting.utils import style_axes
from phase3_common import load_addis_evaluation, PATHS
from theory_test_suite import davis_root

PlotConfig.set(sites="all", layout="individual", font_size=12, title_size=14)
FIG = ANALYSIS / "output/plots/adama_summary_20260910"
OUT = ANALYSIS / "output/tables/adama_summary_20260910"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
adama_path = davis_root() / "DAVIS/CSU_AMOD/csu_amod_Batch_54_ShipDate_2026-05-29_spectra.csv"
adama = pd.read_csv(adama_path)
rowids = adama.iloc[:, 0].astype(str).tolist()
assert len(rowids) == 5 and len(set(rowids)) == 5
wa = adama.columns[1:].to_numpy(float)
ya = adama.iloc[:, 1:].to_numpy(float)
assert np.isfinite(ya).all()
ev, xe, we = load_addis_evaluation()
ev["date"] = pd.to_datetime(ev.SamplingStartDate)
ev = apply_exclusion_flags(ev, "Addis_Ababa")
clean = get_clean_data(ev)
keep = ev.index.isin(clean.index)
xe = xe[keep]
ev = clean.copy()
assert len(ev) == 239 and len(xe) == 239 and np.isfinite(xe).all()
summer_name = "Kiremt (Jun-Sep)"
summer_months = ETHIOPIA_SEASONS[summer_name]["months"]
summer = ev.date.dt.month.isin(summer_months).to_numpy()
assert summer.sum() > 0
# Row IDs in the spectra export are not proven FilterIds. Do not infer a date
# from sorted row order or a peak-height ranking to label a raw spectrum.
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A51A3"]
full_min = float(min(wa.min(), we.min()))
full_max = float(max(wa.max(), we.max()))
detail = (1500, 3500)
yranges = []
for lo, hi in [(full_min, full_max), detail]:
    combined = np.r_[ya[:, (wa >= lo) & (wa <= hi)].ravel(), xe[:, (we >= lo) & (we <= hi)].ravel()]
    ymin, ymax = float(combined.min()), float(combined.max())
    pad = (ymax - ymin) * 0.05
    yranges.append((ymin - pad, ymax + pad))

for subset, mask in [("all", np.ones(len(ev), dtype=bool)), ("summer", summer)]:
    fig, axs = plt.subplots(
        1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.12, 1]}, layout="constrained"
    )
    for j, (ax, (lo, hi)) in enumerate(zip(axs, [(full_min, full_max), detail])):
        for row in xe[mask]:
            ax.plot(we, row, color="#D5D5D5", alpha=0.55, lw=0.7, zorder=1)
        for i, row in enumerate(ya):
            ax.plot(wa, row, color=colors[i], lw=1.6, zorder=3, label="Adama " + rowids[i])
        ax.set(xlim=(hi, lo), ylim=yranges[j])
        style_axes(ax, "Wavenumber (cm⁻¹)", "Raw absorbance", show_legend=False)
        ax.set_title(
            "Full measured spectra" if j == 0 else "Detail: 3500–1500 cm⁻¹", loc="left", fontsize=13
        )
    handles = [Line2D([], [], color="#C7C7C7", lw=2, label=f"Addis, n={int(mask.sum())}")]
    handles += [
        Line2D([], [], color=c, lw=2, label="Adama " + sid) for c, sid in zip(colors, rowids)
    ]
    axs[0].legend(
        handles=handles,
        loc="upper left",
        fontsize=9,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )
    fig.savefig(FIG / f"spectra_{subset}.png", bbox_inches="tight")
    fig.savefig(FIG / f"spectra_{subset}.pdf", bbox_inches="tight")
    plt.close(fig)

ev[["MediaId", "ExternalFilterId", "date", "is_excluded", "exclusion_reason"]].assign(
    in_summer=summer
).to_csv(OUT / "spectral_overlay_addis_population.csv", index=False)
pd.DataFrame(
    {
        "spectral_row_id": rowids,
        "legend_label": ["Adama " + s for s in rowids],
        "filter_id_status": "Definitive crosswalk not present in the export",
    }
).to_csv(OUT / "spectral_overlay_adama_ids.csv", index=False)
manifest = dict(
    n_addis_all=len(ev),
    n_addis_summer=int(summer.sum()),
    n_adama=len(rowids),
    adama_row_ids=rowids,
    season_label=summer_name,
    months=summer_months,
    season_convention=season_convention_name(ETHIOPIA_SEASONS),
    addis_dates=[str(ev.date.min().date()), str(ev.date.max().date())],
    summer_dates=[
        str(ev.loc[summer, "date"].min().date()),
        str(ev.loc[summer, "date"].max().date()),
    ],
    scaling="Raw exported absorbance. No normalization, baseline correction, or volume scaling.",
    replicate_handling="Addis FTIR replicate scans averaged per MediaId by load_addis_evaluation.",
    x_limits=[full_max, full_min],
    y_limits=yranges,
    addis_wavenumber_range=[float(we.min()), float(we.max())],
    adama_wavenumber_range=[float(wa.min()), float(wa.max())],
    notes="Native wavenumber grids. Identical limits for all-Addis and summer-Addis slides. "
    "Selection is the existing 239-filter spectra/HIPS evaluation population, "
    "not the complete repository of every Addis scan. No new exclusions.",
    sources=[
        str(adama_path),
        str(PATHS.etad_dir / "ETAD_FTIR_spectra.csv"),
        str(PATHS.etad_dir / "ETAD_metadata.csv"),
    ],
)
manifest["source_hashes"] = {
    p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in manifest["sources"]
}
(OUT / "spectral_overlay_manifest.json").write_text(json.dumps(manifest, indent=2))
print(
    json.dumps(
        {k: v for k, v in manifest.items() if k not in ["source_hashes", "sources"]}, indent=2
    )
)
