from __future__ import annotations

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
CATCH_UP = ROOT / "research" / "catch_up"
DATA_ROOT_TEXT = (
    "/Users/ahmadjalil/Library/CloudStorage/GoogleDrive-ahzs645@gmail.com/"
    "My Drive/University/Research/Grad/UC Davis Ann/NASA MAIA/Data"
)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


COMMON = f"""
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

REPO_ROOT = Path("/Users/ahmadjalil/github/aethmodular")
FTIR_DIR = REPO_ROOT / "research" / "ftir_hips_chem"
CATCH_UP_DIR = REPO_ROOT / "research" / "catch_up"
DATA_ROOT = Path({DATA_ROOT_TEXT!r})
OUT_DIR = CATCH_UP_DIR / "output" / globals().get("NOTEBOOK_STEM", Path.cwd().name)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS_DIR = FTIR_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import SITES
from outliers import apply_exclusion_flags, apply_threshold_flags, get_clean_data
from plotting import PlotConfig, apply_default_style

apply_default_style()
PlotConfig.set(sites="all", layout="individual", show_stats=True, show_1to1=True)

SITE_CODES = {{site: cfg["code"] for site, cfg in SITES.items()}}
CODE_TO_SITE = {{v: k for k, v in SITE_CODES.items()}}
SITE_COLORS = {{site: PlotConfig.get_site_color(site) for site in SITE_CODES}}

PARAM_RENAME = {{
    "EC_ftir": "ftir_ec",
    "OC_ftir": "ftir_oc",
    "HIPS_Fabs": "hips_fabs",
    "HIPS_T1": "hips_t1",
    "HIPS_R1": "hips_r1",
    "HIPS_t": "hips_t",
    "HIPS_r": "hips_r",
    "HIPS_tau": "hips_tau",
    "ChemSpec_EC_PM2.5": "chemspec_ec",
    "ChemSpec_OC_PM2.5": "chemspec_oc",
    "ChemSpec_OM_PM2.5": "chemspec_om",
    "ChemSpec_Iron_PM2.5": "iron",
    "ChemSpec_Silicon_PM2.5": "silicon",
    "ChemSpec_Aluminum_PM2.5": "aluminum",
    "ChemSpec_Calcium_PM2.5": "calcium",
    "ChemSpec_Titanium_PM2.5": "titanium",
    "ChemSpec_Filter_PM2.5_mass": "pm25_mass",
}}

def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError("None of these paths exist:\\n" + "\\n".join(map(str, paths)))

def load_filter_long():
    path = _first_existing([
        FTIR_DIR / "Filter Data" / "unified_filter_dataset.pkl",
        DATA_ROOT / "Combine csv files" / "Filter Data" / "unified_filter_dataset.pkl",
        DATA_ROOT / "Combine csv files" / "FTIR_HIPS_Chem" / "Filter Data" / "unified_filter_dataset.pkl",
    ])
    df = pd.read_pickle(path)
    df["SampleDate"] = pd.to_datetime(df["SampleDate"])
    df["base_filter_id"] = df["FilterId"].astype(str).str.replace(r"-\\d+$", "", regex=True)
    print(f"Loaded filter data: {{path}}  rows={{len(df):,}}")
    return df

def load_filter_wide(params):
    long = load_filter_long()
    d = long[long["Parameter"].isin(params)].copy()
    meta = (
        d.sort_values(["Site", "base_filter_id", "SampleDate"])
         .groupby(["Site", "base_filter_id"], as_index=False)
         .agg(
             filter_id=("FilterId", "first"),
             date=("SampleDate", "first"),
             volume_m3=("Volume_m3", "max"),
             deposit_area_cm2=("DepositArea_cm2", "max"),
             lot_id=("LotId", "first"),
         )
    )
    conc = d.pivot_table(
        index=["Site", "base_filter_id"],
        columns="Parameter",
        values="Concentration",
        aggfunc="first",
    ).rename(columns=PARAM_RENAME)
    mass = d.pivot_table(
        index=["Site", "base_filter_id"],
        columns="Parameter",
        values="MassLoading_ug",
        aggfunc="first",
    ).rename(columns={{p: PARAM_RENAME.get(p, p) + "_mass_ug" for p in params}})
    wide = meta.merge(conc.reset_index(), on=["Site", "base_filter_id"], how="left")
    wide = wide.merge(mass.reset_index(), on=["Site", "base_filter_id"], how="left")
    wide["site"] = wide["Site"].map(CODE_TO_SITE)
    return wide

def load_aeth_site(site):
    file_map = {{
        "Beijing": "df_Beijing_9am_resampled.pkl",
        "Delhi": "df_Delhi_9am_resampled.pkl",
        "JPL": "df_JPL_9am_resampled.pkl",
        "Addis_Ababa": "df_Addis_Ababa_9am_resampled.pkl",
    }}
    repo_path = FTIR_DIR / "processed_sites" / file_map[site]
    cloud_candidates = [
        DATA_ROOT / "Aethelometry Data" / "JacrosMA350 60s Data20250804082112" / "df_Jacros_9am_resampled.pkl",
        DATA_ROOT / "Aethelometry Data" / "Kyan Data" / "Dataset" / "df_cleaned_Beijing_manual_BCc.pkl",
        DATA_ROOT / "Aethelometry Data" / "Kyan Data" / "Dataset" / "df_cleaned_Delhi_manual_BCc.pkl",
        DATA_ROOT / "Aethelometry Data" / "Kyan Data" / "Dataset" / "df_cleaned_JPL_manual_BCc.pkl",
    ]
    path = repo_path if repo_path.exists() else _first_existing([p for p in cloud_candidates if site.replace("_Ababa", "") in p.name or site == "Addis_Ababa"])
    df = pd.read_pickle(path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "day_9am" in df.columns:
        df["date"] = pd.to_datetime(df["day_9am"]).dt.normalize()
    elif "datetime_local" in df.columns:
        df["date"] = pd.to_datetime(df["datetime_local"]).dt.normalize()
    else:
        df["date"] = pd.to_datetime(df.index).normalize()
    df["site"] = site
    return df

def _to_ugm3(s):
    s = pd.to_numeric(s, errors="coerce")
    med = s.dropna().abs().median()
    return s / 1000.0 if pd.notna(med) and med > 100 else s

def aeth_metrics(site):
    df = load_aeth_site(site)
    out = df[["site", "date"]].copy()
    for wl in ["UV", "Blue", "Green", "Red", "IR"]:
        col = f"{{wl}} BCc"
        out[f"{{wl.lower()}}_bc_ugm3"] = _to_ugm3(df[col]) if col in df.columns else np.nan
    out["aeth_ir_ugm3"] = out["ir_bc_ugm3"]
    out["uv_ir_bcc_ratio"] = out["uv_bc_ugm3"] / out["ir_bc_ugm3"]
    out["green_ir_bcc_ratio"] = out["green_bc_ugm3"] / out["ir_bc_ugm3"]
    out["delta_c_ugm3"] = out["uv_bc_ugm3"] - out["ir_bc_ugm3"]
    # Raw attenuation-based apparent AAE from rolling mean dATN when available.
    uv = pd.to_numeric(df.get("delta UV ATN1 rolling mean", np.nan), errors="coerce")
    ir = pd.to_numeric(df.get("delta IR ATN1 rolling mean", np.nan), errors="coerce")
    mask = (uv > 0) & (ir > 0)
    out["aae_atn_uv_ir"] = np.where(mask, -np.log(uv / ir) / np.log(375 / 880), np.nan)
    return out.groupby(["site", "date"], as_index=False).median(numeric_only=True)

def matched_dataset(include_chem=True):
    params = ["EC_ftir", "OC_ftir", "HIPS_Fabs", "HIPS_T1", "HIPS_R1", "HIPS_t", "HIPS_r", "HIPS_tau"]
    if include_chem:
        params += [
            "ChemSpec_EC_PM2.5", "ChemSpec_OC_PM2.5", "ChemSpec_OM_PM2.5",
            "ChemSpec_Iron_PM2.5", "ChemSpec_Silicon_PM2.5", "ChemSpec_Aluminum_PM2.5",
            "ChemSpec_Calcium_PM2.5", "ChemSpec_Titanium_PM2.5", "ChemSpec_Filter_PM2.5_mass",
        ]
    filt = load_filter_wide(params)
    filt["date"] = pd.to_datetime(filt["date"]).dt.normalize()
    aeth = pd.concat([aeth_metrics(s) for s in SITE_CODES], ignore_index=True)
    m = filt.merge(aeth, on=["site", "date"], how="left")
    # Units: ChemSpec metals are usually ng/m3; convert common tracers to ug/m3 for ratios.
    for col in ["iron", "silicon", "aluminum", "calcium", "titanium"]:
        if col in m.columns:
            med = pd.to_numeric(m[col], errors="coerce").dropna().abs().median()
            if pd.notna(med) and med > 50:
                m[col + "_ugm3"] = m[col] / 1000.0
            else:
                m[col + "_ugm3"] = m[col]
    m["ec_mass_ug"] = m.get("ftir_ec_mass_ug")
    if "ftir_ec" in m.columns:
        m["ec_mass_from_volume_ug"] = m["ftir_ec"] * m["volume_m3"]
        m["ec_mass_ug"] = m["ec_mass_ug"].fillna(m["ec_mass_from_volume_ug"])
    m["ec_surface_loading_ug_cm2"] = m["ec_mass_ug"] / m["deposit_area_cm2"]
    m["hips_bc_mac10_ugm3"] = m["hips_fabs"] / 10.0
    m["hips_minus_ftir"] = m["hips_bc_mac10_ugm3"] - m["ftir_ec"]
    m["hips_to_ftir_ratio"] = m["hips_bc_mac10_ugm3"] / m["ftir_ec"]
    m["oc_ec_ratio"] = m["ftir_oc"] / m["ftir_ec"]
    return m

def add_project_exclusion_flags(df):
    # Add project exclusion/outlier flags using research/ftir_hips_chem/scripts/outliers.py.
    parts = []
    for site in SITE_CODES:
        site_df = df[df["site"] == site].copy()
        if site_df.empty:
            continue
        flag_input = site_df.copy()
        flag_input["filter_id"] = flag_input["base_filter_id"]
        flag_input["aeth_bc"] = flag_input["aeth_ir_ugm3"] * 1000.0
        flag_input["filter_ec"] = flag_input["ftir_ec"] * 1000.0
        flag_input = apply_exclusion_flags(flag_input, site)
        flag_input = apply_threshold_flags(flag_input, site)
        site_df["is_excluded"] = flag_input["is_excluded"].values
        site_df["exclusion_reason"] = flag_input.get("exclusion_reason", "").values
        site_df["is_outlier"] = flag_input["is_outlier"].values
        site_df["outlier_reason"] = flag_input.get("outlier_reason", "").values
        parts.append(site_df)
    out = pd.concat(parts, ignore_index=True) if parts else df.copy()
    out["is_clean"] = ~(out.get("is_excluded", False) | out.get("is_outlier", False))
    return out

def regression_row(df, x, y, label):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return {{"label": label, "x": x, "y": y, "n": len(d), "slope": np.nan, "intercept": np.nan, "r2": np.nan, "p": np.nan}}
    lr = stats.linregress(d[x], d[y])
    return {{"label": label, "x": x, "y": y, "n": len(d), "slope": lr.slope, "intercept": lr.intercept, "r2": lr.rvalue**2, "p": lr.pvalue}}

def save_table(df, name):
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"Wrote {{path}}")
    return path
"""


def make_nb(title: str, filename: str, cells: list):
    nb = nbf.v4.new_notebook()
    stem = Path(filename).stem
    nb["cells"] = [md(f"# {title}"), code(f'NOTEBOOK_STEM = "{stem}"')] + cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


NOTEBOOKS = {}

NOTEBOOKS["01_matched_three_method_exclusion_sensitivity.ipynb"] = make_nb(
    "Matched Three-Method and Exclusion Sensitivity",
    "01_matched_three_method_exclusion_sensitivity.ipynb",
    [
        md("""
Objective: verify that final cross-method statistics use the same matched sample set where
FTIR EC, HIPS Fabs, and aethalometer IR BC are all present, then quantify how manual and
threshold exclusions change slope/intercept/R2.
"""),
        code(COMMON),
        code("""
m = matched_dataset(include_chem=False)
required = ["ftir_ec", "hips_fabs", "aeth_ir_ugm3"]
m["has_three_methods"] = m[required].notna().all(axis=1)

coverage = (
    m.groupby("site")
     .agg(
         filters=("base_filter_id", "nunique"),
         ec_hips=("hips_fabs", lambda s: int((s.notna() & m.loc[s.index, "ftir_ec"].notna()).sum())),
         three_method=("has_three_methods", "sum"),
         aeth_missing=("aeth_ir_ugm3", lambda s: int(s.isna().sum())),
     )
     .reset_index()
)
save_table(coverage, "matched_three_method_coverage.csv")
coverage
"""),
        code("""
rows = []
comparisons = [
    ("ftir_ec", "hips_fabs", "HIPS_Fabs_vs_FTIR_EC"),
    ("ftir_ec", "aeth_ir_ugm3", "Aeth_IR_vs_FTIR_EC"),
    ("aeth_ir_ugm3", "hips_fabs", "HIPS_Fabs_vs_Aeth_IR"),
]

for site in SITE_CODES:
    site_df = m[m["site"] == site].copy()
    flag_input = site_df.copy()
    flag_input["filter_id"] = flag_input["base_filter_id"]
    flag_input["aeth_bc"] = flag_input["aeth_ir_ugm3"] * 1000.0
    flag_input["filter_ec"] = flag_input["ftir_ec"] * 1000.0
    flag_input = apply_exclusion_flags(flag_input, site)
    flag_input = apply_threshold_flags(flag_input, site)
    site_df["is_excluded"] = flag_input["is_excluded"].values
    site_df["is_outlier"] = flag_input["is_outlier"].values
    site_df["exclusion_reason"] = flag_input.get("exclusion_reason", "").values
    site_df["outlier_reason"] = flag_input.get("outlier_reason", "").values
    for x, y, label in comparisons:
        rows.append(regression_row(site_df, x, y, f"{site} | raw | {label}"))
        rows.append(regression_row(site_df[site_df["has_three_methods"]], x, y, f"{site} | matched_three_method | {label}"))
        rows.append(regression_row(site_df[~site_df["is_excluded"] & ~site_df["is_outlier"]], x, y, f"{site} | exclusions_applied | {label}"))
        rows.append(regression_row(site_df[site_df["has_three_methods"] & ~site_df["is_excluded"] & ~site_df["is_outlier"]], x, y, f"{site} | matched_three_method_exclusions | {label}"))

stats_table = pd.DataFrame(rows)
save_table(stats_table, "crossplot_stats_exclusion_sensitivity.csv")
stats_table
"""),
        code("""
excluded_rows = []
for site in SITE_CODES:
    site_df = m[m["site"] == site].copy()
    flag_input = site_df.copy()
    flag_input["filter_id"] = flag_input["base_filter_id"]
    flag_input["aeth_bc"] = flag_input["aeth_ir_ugm3"] * 1000.0
    flag_input["filter_ec"] = flag_input["ftir_ec"] * 1000.0
    flag_input = apply_threshold_flags(apply_exclusion_flags(flag_input, site), site)
    flagged = flag_input[flag_input["is_excluded"] | flag_input["is_outlier"]].copy()
    if len(flagged):
        excluded_rows.append(flagged[["site", "date", "filter_id", "ftir_ec", "hips_fabs", "aeth_ir_ugm3", "is_excluded", "exclusion_reason", "is_outlier", "outlier_reason"]])

exclusion_audit = pd.concat(excluded_rows, ignore_index=True) if excluded_rows else pd.DataFrame()
save_table(exclusion_audit, "excluded_sample_audit.csv")
exclusion_audit
"""),
    ],
)

NOTEBOOKS["02_loading_volume_operating_envelope.ipynb"] = make_nb(
    "Loading, Volume, and Operating Envelope",
    "02_loading_volume_operating_envelope.ipynb",
    [
        md("""
Objective: quantify whether Addis occupies a different filter-loading regime than the
other SPARTAN sites using volume, deposit area, EC mass, EC surface loading, Fabs, and
available HIPS R/T fields.
"""),
        code(COMMON),
        code("""
m = add_project_exclusion_flags(matched_dataset(include_chem=True))
cols = ["site", "base_filter_id", "date", "volume_m3", "deposit_area_cm2", "ftir_ec", "ec_mass_ug", "ec_surface_loading_ug_cm2", "hips_fabs", "hips_t1", "hips_r1", "hips_tau"]
analysis = m[m["is_clean"]][cols].copy()
excluded_analysis = m[~m["is_clean"]][cols + ["is_excluded", "exclusion_reason", "is_outlier", "outlier_reason"]].copy()
save_table(excluded_analysis, "excluded_rows_not_used_in_loading_envelope.csv")
summary = (
    analysis.groupby("site")
    .agg(
        n=("base_filter_id", "count"),
        volume_m3_median=("volume_m3", "median"),
        volume_m3_p05=("volume_m3", lambda s: s.quantile(0.05)),
        volume_m3_p95=("volume_m3", lambda s: s.quantile(0.95)),
        ec_mass_ug_median=("ec_mass_ug", "median"),
        ec_mass_ug_p05=("ec_mass_ug", lambda s: s.quantile(0.05)),
        ec_mass_ug_p95=("ec_mass_ug", lambda s: s.quantile(0.95)),
        ec_surface_ug_cm2_median=("ec_surface_loading_ug_cm2", "median"),
        ec_surface_ug_cm2_p05=("ec_surface_loading_ug_cm2", lambda s: s.quantile(0.05)),
        ec_surface_ug_cm2_p95=("ec_surface_loading_ug_cm2", lambda s: s.quantile(0.95)),
        fabs_median=("hips_fabs", "median"),
        fabs_p05=("hips_fabs", lambda s: s.quantile(0.05)),
        fabs_p95=("hips_fabs", lambda s: s.quantile(0.95)),
        t1_median=("hips_t1", "median"),
        r1_median=("hips_r1", "median"),
    )
    .reset_index()
)
save_table(summary, "spartan_loading_volume_envelope_summary.csv")
summary
"""),
        code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
plot_specs = [
    ("volume_m3", "Sample volume (m3)"),
    ("ec_mass_ug", "FTIR EC mass on filter (ug)"),
    ("ec_surface_loading_ug_cm2", "FTIR EC surface loading (ug/cm2)"),
    ("hips_fabs", "HIPS Fabs (1/Mm)"),
]
for ax, (col, label) in zip(axes.flat, plot_specs):
    data = [analysis.loc[analysis["site"] == s, col].dropna() for s in SITE_CODES]
    ax.boxplot(data, labels=list(SITE_CODES), showfliers=False)
    ax.set_title(label)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.25)
plt.tight_layout()
fig_path = OUT_DIR / "spartan_loading_volume_envelope_boxplots.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(fig_path)
plt.show()
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 6))
for site, color in SITE_COLORS.items():
    d = analysis[analysis["site"] == site]
    ax.scatter(d["ec_surface_loading_ug_cm2"], d["hips_fabs"], s=32, alpha=0.75, label=site, color=color)
ax.set_xlabel("FTIR EC surface loading (ug/cm2)")
ax.set_ylabel("HIPS Fabs (1/Mm)")
ax.set_title("SPARTAN HIPS response by EC surface loading")
ax.grid(True, alpha=0.25)
ax.legend()
plt.tight_layout()
fig_path = OUT_DIR / "hips_fabs_vs_ec_surface_loading_four_sites.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(fig_path)
plt.show()
"""),
    ],
)

NOTEBOOKS["03_residuals_chemistry_spectral_metrics.ipynb"] = make_nb(
    "HIPS Residuals vs Chemistry and Aethalometer Spectral Metrics",
    "03_residuals_chemistry_spectral_metrics.ipynb",
    [
        md("""
Objective: test whether HIPS residuals or HIPS/FTIR ratios track chemistry
(OC/EC, Fe, soil tracers) or aethalometer spectral metrics (UV/IR, Delta-C,
raw ATN AAE). This bridges the source/dust/organic threads to the core HIPS anomaly.
"""),
        code(COMMON),
        code("""
m = add_project_exclusion_flags(matched_dataset(include_chem=True))
m = m[m["is_clean"] & m[["ftir_ec", "hips_fabs"]].notna().all(axis=1)].copy()
m = m.replace([np.inf, -np.inf], np.nan)
predictors = [
    "ftir_oc", "oc_ec_ratio", "iron_ugm3", "silicon_ugm3", "aluminum_ugm3",
    "calcium_ugm3", "titanium_ugm3", "pm25_mass", "uv_ir_bcc_ratio",
    "green_ir_bcc_ratio", "delta_c_ugm3", "aae_atn_uv_ir", "ec_surface_loading_ug_cm2",
]
targets = ["hips_minus_ftir", "hips_to_ftir_ratio", "hips_fabs"]
rows = []
for site in list(SITE_CODES) + ["All"]:
    d = m if site == "All" else m[m["site"] == site]
    for target in targets:
        for pred in predictors:
            if pred not in d.columns:
                continue
            pair = d[[target, pred]].dropna()
            if len(pair) < 5:
                rows.append({"site": site, "target": target, "predictor": pred, "n": len(pair), "pearson_r": np.nan, "spearman_r": np.nan, "r2": np.nan, "p": np.nan})
                continue
            pear = stats.pearsonr(pair[pred], pair[target])
            spear = stats.spearmanr(pair[pred], pair[target], nan_policy="omit")
            rows.append({"site": site, "target": target, "predictor": pred, "n": len(pair), "pearson_r": pear.statistic, "spearman_r": spear.statistic, "r2": pear.statistic**2, "p": pear.pvalue})
corr = pd.DataFrame(rows).sort_values(["target", "site", "r2"], ascending=[True, True, False])
save_table(corr, "hips_residual_predictor_correlations.csv")
corr.head(30)
"""),
        code("""
focus = m[m["site"] == "Addis_Ababa"].copy()
plot_vars = ["oc_ec_ratio", "iron_ugm3", "silicon_ugm3", "uv_ir_bcc_ratio", "delta_c_ugm3", "aae_atn_uv_ir"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, pred in zip(axes.flat, plot_vars):
    d = focus[[pred, "hips_minus_ftir"]].dropna()
    ax.scatter(d[pred], d["hips_minus_ftir"], s=32, alpha=0.75, color=SITE_COLORS["Addis_Ababa"])
    if len(d) >= 3:
        lr = stats.linregress(d[pred], d["hips_minus_ftir"])
        xs = np.linspace(d[pred].min(), d[pred].max(), 100)
        ax.plot(xs, lr.intercept + lr.slope * xs, color="black", lw=1.5)
        ax.text(0.04, 0.94, f"n={len(d)}\\nR2={lr.rvalue**2:.2f}", transform=ax.transAxes, va="top")
    ax.set_title(pred)
    ax.set_ylabel("HIPS BC(MAC10) - FTIR EC")
    ax.grid(True, alpha=0.25)
plt.tight_layout()
fig_path = OUT_DIR / "addis_hips_residual_vs_chemistry_spectral_metrics.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(fig_path)
plt.show()
"""),
    ],
)

NOTEBOOKS["04_aeronet_multisite_availability_overlap.ipynb"] = make_nb(
    "AERONET Multisite Availability and Filter-Day Overlap",
    "04_aeronet_multisite_availability_overlap.ipynb",
    [
        md("""
Objective: make the missing AERONET status table for all four sites: file availability,
date range, level/header fields, daily record count, matched filter-day count, and
whether this site can support a dust/column-context analysis.
"""),
        code(COMMON),
        code("""
def read_aeronet_csv(path):
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Date(" in line and "," in line:
            header_idx = i
            break
    if header_idx is None:
        return None
    df = pd.read_csv(path, skiprows=header_idx)
    date_col = next((c for c in df.columns if c.startswith("Date(")), None)
    time_col = next((c for c in df.columns if c.startswith("Time(")), None)
    if date_col:
        if time_col:
            dt = df[date_col].astype(str) + " " + df[time_col].astype(str)
            df["datetime"] = pd.to_datetime(dt, errors="coerce", dayfirst=True)
        else:
            df["datetime"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df["date"] = df["datetime"].dt.normalize()
    df = df.replace(-999, np.nan)
    return df

def aeronet_candidates(site):
    repo = {
        "Beijing": [REPO_ROOT / "notebooks/analysis/data/aeronet_aod_beijing.csv"],
        "Delhi": [REPO_ROOT / "notebooks/analysis/data/aeronet_aod_delhi.csv"],
        "JPL": [REPO_ROOT / "notebooks/analysis/data/aeronet_aod_jpl.csv"],
        "Addis_Ababa": list((DATA_ROOT / "AERONET" / "Jacros").glob("**/*.csv")),
    }
    return [p for p in repo.get(site, []) if Path(p).exists()]

filters = load_filter_wide(["EC_ftir", "HIPS_Fabs"])
filter_days = filters.groupby("site")["date"].apply(lambda s: set(pd.to_datetime(s).dt.normalize().dropna())).to_dict()

rows = []
for site in SITE_CODES:
    paths = aeronet_candidates(site)
    if not paths:
        rows.append({"site": site, "file": None, "status": "missing_file"})
        continue
    for path in paths:
        df = read_aeronet_csv(path)
        if df is None or "date" not in df:
            rows.append({"site": site, "file": str(path), "status": "unreadable_or_no_date"})
            continue
        days = set(df["date"].dropna())
        overlap = days.intersection(filter_days.get(site, set()))
        aod_cols = [c for c in df.columns if "AOD" in c.upper()]
        fine_cols = [c for c in df.columns if "FINE" in c.upper() or "FMF" in c.upper()]
        coarse_cols = [c for c in df.columns if "COARSE" in c.upper()]
        rows.append({
            "site": site,
            "file": str(path),
            "status": "ok",
            "aeronet_rows": len(df),
            "aeronet_days": len(days),
            "aeronet_start": min(days) if days else pd.NaT,
            "aeronet_end": max(days) if days else pd.NaT,
            "filter_days": len(filter_days.get(site, set())),
            "matched_filter_days": len(overlap),
            "n_aod_columns": len(aod_cols),
            "n_fine_columns": len(fine_cols),
            "n_coarse_columns": len(coarse_cols),
            "example_aod_columns": "; ".join(aod_cols[:5]),
            "example_fine_columns": "; ".join(fine_cols[:5]),
            "example_coarse_columns": "; ".join(coarse_cols[:5]),
        })
availability = pd.DataFrame(rows)
save_table(availability, "aeronet_multisite_availability_overlap.csv")
availability
"""),
    ],
)

NOTEBOOKS["05_raw_atn_aae_green_channel_diagnostics.ipynb"] = make_nb(
    "Raw ATN AAE and Green Channel Diagnostics",
    "05_raw_atn_aae_green_channel_diagnostics.ipynb",
    [
        md("""
Objective: finish the raw attenuation/spectral diagnostic task. This checks whether
AAE computed from raw dATN behaves more plausibly than BCc-based AAE, and whether the
Addis green-channel issue is present in raw attenuation as well as processed BCc.
"""),
        code(COMMON),
        code("""
rows = []
daily = []
for site in SITE_CODES:
    df = load_aeth_site(site)
    metrics = aeth_metrics(site)
    for col in ["uv_ir_bcc_ratio", "green_ir_bcc_ratio", "delta_c_ugm3", "aae_atn_uv_ir"]:
        s = metrics[col].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"site": site, "metric": col, "n_days": len(s), "median": s.median(), "p05": s.quantile(0.05), "p95": s.quantile(0.95)})
    # Raw ATN green/IR ratios.
    for channel in ["ATN1", "ATN2"]:
        g = pd.to_numeric(df.get(f"delta Green {channel} rolling mean", np.nan), errors="coerce")
        ir = pd.to_numeric(df.get(f"delta IR {channel} rolling mean", np.nan), errors="coerce")
        ratio = (g / ir).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"site": site, "metric": f"green_ir_raw_datn_{channel}", "n_days": len(ratio), "median": ratio.median(), "p05": ratio.quantile(0.05), "p95": ratio.quantile(0.95)})
    daily.append(metrics)
summary = pd.DataFrame(rows)
save_table(summary, "raw_atn_aae_green_channel_summary.csv")
summary
"""),
        code("""
all_daily = pd.concat(daily, ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for site, color in SITE_COLORS.items():
    d = all_daily[all_daily["site"] == site]
    axes[0].hist(d["aae_atn_uv_ir"].dropna(), bins=30, alpha=0.45, label=site, color=color)
    axes[1].hist(d["green_ir_bcc_ratio"].dropna(), bins=30, alpha=0.45, label=site, color=color)
axes[0].set_title("Raw dATN AAE (UV/IR)")
axes[1].set_title("Processed BCc Green/IR ratio")
for ax in axes:
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
plt.tight_layout()
fig_path = OUT_DIR / "raw_atn_aae_and_green_ratio_distributions.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(fig_path)
plt.show()
"""),
        code("""
addis = all_daily[all_daily["site"] == "Addis_Ababa"].copy()
addis["month"] = addis["date"].dt.to_period("M").astype(str)
monthly = addis.groupby("month").agg(
    n=("date", "count"),
    green_ir_bcc_median=("green_ir_bcc_ratio", "median"),
    raw_aae_median=("aae_atn_uv_ir", "median"),
    uv_ir_bcc_median=("uv_ir_bcc_ratio", "median"),
).reset_index()
save_table(monthly, "addis_green_channel_monthly_diagnostic.csv")
monthly.tail(20)
"""),
    ],
)

NOTEBOOKS["06_delhi_seasonality_crop_burning_check.ipynb"] = make_nb(
    "Delhi Seasonality and Crop-Burning Check",
    "06_delhi_seasonality_crop_burning_check.ipynb",
    [
        md("""
Objective: make the provisional Delhi seasonality analysis explicit. This checks
whether post-monsoon/crop-residue months have different aethalometer spectral metrics
or HIPS residuals than the rest of the Delhi record.
"""),
        code(COMMON),
        code("""
def delhi_season(month):
    if month in [11, 12, 1, 2]:
        return "Winter (Nov-Feb)"
    if month in [3, 4, 5, 6]:
        return "Pre-monsoon (Mar-Jun)"
    if month in [7, 8, 9]:
        return "Monsoon (Jul-Sep)"
    if month == 10:
        return "Post-monsoon / crop-residue (Oct)"
    return "Other"

m = add_project_exclusion_flags(matched_dataset(include_chem=True))
delhi = m[m["site"] == "Delhi"].copy()
delhi["month"] = delhi["date"].dt.month
delhi["season"] = delhi["month"].map(delhi_season)
delhi["crop_residue_window"] = delhi["month"].isin([10, 11])

summary = delhi.groupby("season").agg(
    n=("base_filter_id", "count"),
    ftir_ec_median=("ftir_ec", "median"),
    hips_fabs_median=("hips_fabs", "median"),
    aeth_ir_median=("aeth_ir_ugm3", "median"),
    uv_ir_median=("uv_ir_bcc_ratio", "median"),
    delta_c_median=("delta_c_ugm3", "median"),
    raw_aae_median=("aae_atn_uv_ir", "median"),
    hips_residual_median=("hips_minus_ftir", "median"),
    ec_surface_median=("ec_surface_loading_ug_cm2", "median"),
).reset_index()
save_table(summary, "delhi_seasonality_crop_burning_summary.csv")
summary
"""),
        code("""
comparisons = []
for target in ["uv_ir_bcc_ratio", "delta_c_ugm3", "aae_atn_uv_ir", "hips_minus_ftir", "hips_to_ftir_ratio"]:
    a = delhi.loc[delhi["crop_residue_window"], target].dropna()
    b = delhi.loc[~delhi["crop_residue_window"], target].dropna()
    if len(a) >= 2 and len(b) >= 2:
        test = stats.mannwhitneyu(a, b, alternative="two-sided")
        comparisons.append({
            "metric": target,
            "crop_window_n": len(a),
            "other_n": len(b),
            "crop_window_median": a.median(),
            "other_median": b.median(),
            "mann_whitney_p": test.pvalue,
        })
crop_test = pd.DataFrame(comparisons)
save_table(crop_test, "delhi_crop_window_metric_tests.csv")
crop_test
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, col in zip(axes, ["uv_ir_bcc_ratio", "delta_c_ugm3", "hips_minus_ftir"]):
    groups = [g[col].dropna() for _, g in delhi.groupby("season", sort=False)]
    labels = [name for name, _ in delhi.groupby("season", sort=False)]
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.25)
plt.tight_layout()
fig_path = OUT_DIR / "delhi_seasonal_spectral_and_hips_residual_boxplots.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(fig_path)
plt.show()
"""),
    ],
)


for filename, nb in NOTEBOOKS.items():
    path = CATCH_UP / filename
    nbf.write(nb, path)
    print(f"Wrote {path}")
