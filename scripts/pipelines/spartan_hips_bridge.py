"""Bridge the SPARTAN public dataset to the HIPS (optical absorption)
filter-level dataset shared by Drive folder 1YVmkYP_0pzs5TQwwbcTQi7gEJ-rTl9LZ.

Inputs:
    data/drive_bridge/Spartan/SPARTAN_HIPS_Batch1-51.v2.csv
    data/drive_bridge/Spartan/SPARTAN_Site_quick_lookup.xlsx
    data/spartan/raw/FilterBased/ChemSpecPM25/*.csv   (public)

Bridging key: FilterId (e.g. "ZAJB-0041-1"). The trailing "-N" in some
public files matches the HIPS filter, others use the base "SITE-NNNN".

Outputs (under research/spartan/inventory/):
    hips_coverage_by_site.csv          rows per site, date span, MDL stats
    hips_vs_public_link.csv            FilterId-level join: HIPS Fabs vs public BC PM2.5
    figures/hips_coverage_by_site.png  per-site filter counts and date span
    figures/hips_vs_bc_scatter.png     scatter + linear fit per site (top 6)
    figures/hips_monthly_heatmap.png   site x month HIPS filter counts
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HIPS_PATH = REPO_ROOT / "data" / "drive_bridge" / "Spartan" / "SPARTAN_HIPS_Batch1-51.v2.csv"
LOOKUP_PATH = REPO_ROOT / "data" / "drive_bridge" / "Spartan" / "SPARTAN_Site_quick_lookup.xlsx"
PUBLIC_PM25 = REPO_ROOT / "data" / "spartan" / "raw" / "FilterBased" / "ChemSpecPM25"
OUT_DIR = REPO_ROOT / "research" / "spartan" / "inventory"
FIG_DIR = OUT_DIR / "figures"


def _find_header(path: Path, max_scan: int = 5) -> int:
    with open(path, "r", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                return 0
            s = line.strip()
            if not s:
                continue
            if "," in s and not s.lstrip().startswith("#"):
                if any(t in s.lower() for t in ("site_code", "year", "year_local")):
                    return i
    return 0


def _normalize_fid(s: pd.Series) -> pd.Series:
    """Some HIPS rows carry the per-replicate suffix ('-1', '-2', '-3') while
    the public files key only to the base filter ('SITE-NNNN'). Strip the
    trailing replicate so we can join both forms.
    """
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"-(\d)$", "", regex=True)
    )


def load_hips() -> pd.DataFrame:
    df = pd.read_csv(HIPS_PATH)
    df["SampleDate"] = pd.to_datetime(df["SampleDate"], errors="coerce")
    df["FilterId_base"] = _normalize_fid(df["FilterId"])
    # Replicate "-7" is the SPARTAN field blank; "*-LB*" / numeric L-codes are lab blanks.
    df["is_blank"] = (
        df["FilterId"].astype(str).str.endswith("-7")
        | df["FilterId"].astype(str).str.contains("-LB", regex=False)
        | df["SampleDate"].isna()
    )
    return df


def load_lookup() -> pd.DataFrame:
    return pd.read_excel(LOOKUP_PATH)


def load_public_bc() -> pd.DataFrame:
    """One row per (site, FilterId) carrying BC PM2.5 and Filter PM2.5 mass."""
    rows = []
    for path in sorted(PUBLIC_PM25.glob("*.csv")):
        site = path.stem.rsplit("_", 1)[-1]
        df = pd.read_csv(path, skiprows=_find_header(path), low_memory=False)
        cols = {c.lower(): c for c in df.columns}
        fid = cols.get("filter_id")
        pname = cols.get("parameter_name")
        val = cols.get("value")
        if not (fid and pname and val):
            continue
        sub = df[df[pname].isin(["BC PM2.5", "Filter PM2.5 mass"])].copy()
        sub[val] = pd.to_numeric(sub[val], errors="coerce")
        pv = sub.pivot_table(index=fid, columns=pname, values=val, aggfunc="mean")
        pv["site"] = site
        pv = pv.reset_index().rename(columns={fid: "FilterId"})
        rows.append(pv)
    out = pd.concat(rows, ignore_index=True)
    out["FilterId_base"] = _normalize_fid(out["FilterId"])
    return out


# ---------------------------------------------------------------------------
# Coverage & bridge analysis
# ---------------------------------------------------------------------------

def coverage_by_site(hips: pd.DataFrame) -> pd.DataFrame:
    hips = hips[~hips["is_blank"]]
    g = (
        hips.groupby("Site")
            .agg(
                n_filters=("FilterId", "nunique"),
                n_rows=("FilterId", "size"),
                date_min=("SampleDate", "min"),
                date_max=("SampleDate", "max"),
                fabs_median=("Fabs", "median"),
                fabs_mean=("Fabs", "mean"),
                fabs_p95=("Fabs", lambda s: s.quantile(0.95)),
                mdl_median=("MDL", "median"),
                pct_above_mdl=("Fabs", lambda s: 100 * (s > hips.loc[s.index, "MDL"]).mean()),
            )
            .reset_index()
    )
    g = g.sort_values("n_filters", ascending=False).reset_index(drop=True)
    g["date_min"] = g["date_min"].dt.strftime("%Y-%m-%d")
    g["date_max"] = g["date_max"].dt.strftime("%Y-%m-%d")
    for c in ("fabs_median", "fabs_mean", "fabs_p95", "mdl_median", "pct_above_mdl"):
        g[c] = g[c].round(2)
    return g


def bridge_hips_to_public(hips: pd.DataFrame, public: pd.DataFrame) -> pd.DataFrame:
    """One row per FilterId_base appearing in either dataset."""
    hips = hips[~hips["is_blank"]]
    h = (
        hips.groupby(["Site", "FilterId_base"])
            .agg(fabs_mean=("Fabs", "mean"),
                 fabs_n=("Fabs", "size"),
                 sample_date=("SampleDate", "min"))
            .reset_index()
            .rename(columns={"Site": "site"})
    )
    p = (
        public.groupby(["site", "FilterId_base"])
              .agg(bc_pm25=("BC PM2.5", "mean"),
                   pm25_mass=("Filter PM2.5 mass", "mean"))
              .reset_index()
    )
    merged = h.merge(p, on=["site", "FilterId_base"], how="outer", indicator=True)
    merged["sample_date"] = merged["sample_date"].dt.strftime("%Y-%m-%d")
    merged.rename(columns={"_merge": "presence"}, inplace=True)
    return merged


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_coverage(cov: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(cov))))
    d = cov.iloc[::-1]
    bars = ax.barh(d["Site"], d["n_filters"], color="teal")
    for bar, dmin, dmax, p in zip(bars, d["date_min"], d["date_max"], d["pct_above_mdl"]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())} filters · {dmin} → {dmax} · {p:.0f}% > MDL",
                va="center", fontsize=7)
    ax.set_xlabel("HIPS filters analysed")
    ax.set_title(f"SPARTAN HIPS coverage — {cov['n_filters'].sum():,} filters across {len(cov)} sites")
    ax.set_xlim(0, cov["n_filters"].max() * 1.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmap(hips: pd.DataFrame, out: Path) -> None:
    g = hips[~hips["is_blank"]].dropna(subset=["SampleDate"]).copy()
    g["month"] = g["SampleDate"].dt.to_period("M").dt.to_timestamp()
    pv = (
        g.groupby(["Site", "month"]).size()
         .unstack(fill_value=0).sort_index()
    )
    full_idx = pd.date_range(pv.columns.min(), pv.columns.max(), freq="MS")
    pv = pv.reindex(columns=full_idx, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(12, 0.18 * pv.shape[1]), max(4, 0.28 * pv.shape[0])))
    vmax = max(np.percentile(pv.values[pv.values > 0], 95), 1) if (pv.values > 0).any() else 1
    im = ax.imshow(pv.values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_yticks(range(pv.shape[0]))
    ax.set_yticklabels(pv.index, fontsize=8)
    step = max(pv.shape[1] // 14, 1)
    xt = list(range(0, pv.shape[1], step))
    ax.set_xticks(xt)
    ax.set_xticklabels([pv.columns[i].strftime("%Y-%m") for i in xt], rotation=45, ha="right", fontsize=8)
    ax.set_title("HIPS filters per site per month")
    plt.colorbar(im, ax=ax, shrink=0.7, label="filters / month")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_hips_vs_bc(bridge: pd.DataFrame, out: Path, top_n: int = 6) -> None:
    both = bridge[(bridge["presence"] == "both")
                  & bridge["fabs_mean"].notna()
                  & bridge["bc_pm25"].notna()
                  & (bridge["bc_pm25"] > 0)].copy()
    site_counts = both.groupby("site").size().sort_values(ascending=False)
    sites = site_counts.head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=False)
    for ax, site in zip(axes.flat, sites):
        sub = both[both["site"] == site]
        ax.scatter(sub["bc_pm25"], sub["fabs_mean"], s=18, alpha=0.7, color="darkblue")
        if len(sub) >= 3:
            slope, intercept = np.polyfit(sub["bc_pm25"], sub["fabs_mean"], 1)
            x = np.linspace(0, sub["bc_pm25"].max() * 1.05, 50)
            ax.plot(x, slope * x + intercept, "r-", lw=1,
                    label=f"y = {slope:.2f}x + {intercept:.2f}")
            r2 = np.corrcoef(sub["bc_pm25"], sub["fabs_mean"])[0, 1] ** 2
            ax.text(0.05, 0.95, f"n={len(sub)}, R²={r2:.2f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax.legend(loc="lower right", fontsize=8)
        ax.set_xlabel("BC PM2.5  (μg/m³, public)")
        ax.set_ylabel("HIPS Fabs (Mm⁻¹)")
        ax.set_title(site)
        ax.grid(True, alpha=0.3)
    fig.suptitle("HIPS Fabs vs. public-dataset BC PM2.5 — filter-level join (top 6 sites by n)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not HIPS_PATH.exists():
        print(f"Missing {HIPS_PATH}")
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HIPS...")
    hips = load_hips()
    print(f"  {len(hips):,} rows, {hips['FilterId'].nunique():,} unique filters,"
          f" {hips['Site'].nunique()} sites")

    print("Loading site lookup...")
    lookup = load_lookup()
    print(f"  {len(lookup)} sites including new ones: "
          f"{sorted(set(lookup['SiteCode']) - {p.stem.rsplit('_', 1)[-1] for p in PUBLIC_PM25.glob('*.csv')})}")

    print("Loading public FilterBased PM2.5 (BC + mass)...")
    public = load_public_bc()
    print(f"  {len(public):,} unique public filters carrying BC/mass")

    print("\nComputing HIPS per-site coverage...")
    cov = coverage_by_site(hips)
    cov.to_csv(OUT_DIR / "hips_coverage_by_site.csv", index=False)
    print(cov.head(15).to_string(index=False))

    print("\nBridging by FilterId (base, replicate-stripped)...")
    bridge = bridge_hips_to_public(hips, public)
    bridge.to_csv(OUT_DIR / "hips_vs_public_link.csv", index=False)
    g = bridge["presence"].value_counts()
    print(g.to_string())
    both = bridge[bridge["presence"] == "both"]
    print(f"\nFilters in BOTH HIPS and public: {len(both):,}")
    sites_both = both.groupby("site").size().sort_values(ascending=False)
    print("Top sites by overlap:")
    print(sites_both.head(10).to_string())

    # Fit summary site-by-site
    summary = []
    for site, sub in both.groupby("site"):
        m = sub.dropna(subset=["fabs_mean", "bc_pm25"])
        m = m[m["bc_pm25"] > 0]
        if len(m) < 3:
            continue
        slope, intercept = np.polyfit(m["bc_pm25"], m["fabs_mean"], 1)
        r2 = np.corrcoef(m["bc_pm25"], m["fabs_mean"])[0, 1] ** 2
        summary.append({
            "site": site, "n": len(m),
            "slope_Fabs_per_ugBC": round(float(slope), 2),
            "intercept": round(float(intercept), 2),
            "R2": round(float(r2), 3),
        })
    fit = pd.DataFrame(summary).sort_values("n", ascending=False)
    fit.to_csv(OUT_DIR / "hips_vs_bc_linear_fits.csv", index=False)
    print("\nPer-site Fabs ~ BC PM2.5 linear fits:")
    print(fit.to_string(index=False))

    print("\nPlotting...")
    plot_coverage(cov, FIG_DIR / "hips_coverage_by_site.png")
    plot_monthly_heatmap(hips, FIG_DIR / "hips_monthly_heatmap.png")
    plot_hips_vs_bc(bridge, FIG_DIR / "hips_vs_bc_scatter.png")
    print(f"  wrote figures under {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
