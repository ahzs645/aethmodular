"""Three follow-on analyses for SPARTAN:

1. Split site rankings: filter-only sites vs. nephel-equipped sites.
2. End-to-end per-parameter completeness for ETAD (Addis Ababa).
3. World map of sites colored by coverage score.

Reads cached SPARTAN files under data/spartan/raw/ and the existing
coverage.csv / top_sites.csv. Writes new artifacts into
research/spartan/inventory/ and research/spartan/inventory/figures/.
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

# scripts/ is not an installed package and the CLI runs this file by path, so
# put scripts/ on sys.path to make `common` importable. See scripts/common/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.spartan_io import (  # noqa: E402
    FIG_DIR, NEPHEL_SUBS, OUT_DIR, RAW_DIR,
    read_spartan_csv as _read, site_from_path,
)

# Full expected subproduct count across the SPARTAN product tree (the 9 keys of
# EXPECTED_STEP_H in spartan_coverage_analysis.py). The composite score divides
# a site's breadth by this fixed total so scores are comparable across groups
# and against research/spartan/inventory/top_sites.csv, which uses the same 9.
# Previously this was df["subproduct"].nunique() computed on each *subset*, so
# the nephel-equipped and filter-only tables were scored on different scales
# while the map colorbar still claimed "breadth / 9".
MAX_BREADTH = 9


# ---------------------------------------------------------------------------
# Shared loader
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# (1) split rankings
# ---------------------------------------------------------------------------

def split_rankings() -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = pd.read_csv(OUT_DIR / "coverage.csv")
    cov["completeness_pct"] = pd.to_numeric(cov["completeness_pct"], errors="coerce")
    cov["completeness_capped"] = cov["completeness_pct"].clip(upper=100)

    site_has_nephel = (
        cov[cov["subproduct"].isin(NEPHEL_SUBS)]
        .groupby("site")["n_samples"].sum().fillna(0) > 0
    )
    nephel_sites = set(site_has_nephel[site_has_nephel].index)

    def rank(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
        g = (
            df.groupby("site")
              .agg(
                  n_subproducts=("subproduct", "nunique"),
                  total_samples=("n_samples", "sum"),
                  mean_completeness=("completeness_capped", "mean"),
                  median_completeness=("completeness_capped", "median"),
                  earliest=("date_min", "min"),
                  latest=("date_max", "max"),
              )
              .reset_index()
        )
        g["mean_completeness"] = g["mean_completeness"].round(1)
        g["median_completeness"] = g["median_completeness"].round(1)
        g["score"] = (g["mean_completeness"].fillna(0) * (g["n_subproducts"] / MAX_BREADTH)).round(1)
        g["group"] = group_label
        return g.sort_values(["score", "total_samples"], ascending=False).reset_index(drop=True)

    nephel_df = rank(cov[cov["site"].isin(nephel_sites)], "nephel-equipped")
    filter_df = rank(cov[~cov["site"].isin(nephel_sites)], "filter-only")

    nephel_df.to_csv(OUT_DIR / "top_sites_nephel.csv", index=False)
    filter_df.to_csv(OUT_DIR / "top_sites_filter_only.csv", index=False)
    return nephel_df, filter_df


def plot_split_rankings(nephel: pd.DataFrame, filt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.32 * max(len(nephel), len(filt)))))
    for ax, df, title, color in [
        (axes[0], nephel, f"Nephel-equipped sites ({len(nephel)})", "steelblue"),
        (axes[1], filt,   f"Filter-only sites ({len(filt)})",       "darkorange"),
    ]:
        d = df.iloc[::-1]
        bars = ax.barh(d["site"], d["mean_completeness"], color=color)
        for bar, n, sp in zip(bars, d["total_samples"], d["n_subproducts"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{int(n):,} obs / {int(sp)}p",
                    va="center", fontsize=7)
        ax.set_xlabel("Mean completeness across present subproducts (%)")
        ax.set_title(title)
        ax.set_xlim(0, 110)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("SPARTAN sites split by capability", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "top_sites_split.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (2) ETAD parameter-level completeness
# ---------------------------------------------------------------------------

def etad_parameter_completeness() -> pd.DataFrame:
    rows: list[dict] = []
    monthly_rows: list[dict] = []
    for sub in ("ChemSpecPM10", "ChemSpecPM25", "ReconstrPM25"):
        path = RAW_DIR / "FilterBased" / sub / f"FilterBased_{sub}_ETAD.csv"
        if not path.exists():
            continue
        df = _read(path)
        cols = {c.lower(): c for c in df.columns}
        pname = cols.get("parameter_name")
        val = cols.get("value")
        fid = cols.get("filter_id")
        mdl = cols.get("mdl")
        if not (pname and val and fid):
            continue

        # filter datetime
        y, m, d = cols["start_year_local"], cols["start_month_local"], cols["start_day_local"]
        df["_dt"] = pd.to_datetime(
            dict(year=pd.to_numeric(df[y], errors="coerce"),
                 month=pd.to_numeric(df[m], errors="coerce"),
                 day=pd.to_numeric(df[d], errors="coerce")),
            errors="coerce",
        )
        n_filters = df[fid].dropna().nunique()
        span = df["_dt"].dropna()
        date_min = span.min()
        date_max = span.max()

        df[val] = pd.to_numeric(df[val], errors="coerce")
        if mdl:
            df[mdl] = pd.to_numeric(df[mdl], errors="coerce")

        for param, sub_df in df.groupby(pname):
            valid = sub_df.dropna(subset=[val])
            filters_with_param = valid[fid].dropna().nunique()
            above_mdl = int((valid[val] > 0).sum())  # zero/near-zero often used for below-MDL
            if mdl and valid[mdl].notna().any():
                above_mdl = int((valid[val] > valid[mdl]).sum())
            rows.append({
                "subproduct": sub,
                "parameter": str(param),
                "n_filters_with_value": filters_with_param,
                "n_total_filters_at_site": n_filters,
                "param_completeness_pct": round(filters_with_param / n_filters * 100, 1) if n_filters else float("nan"),
                "n_above_mdl": above_mdl,
                "frac_above_mdl_pct": round(above_mdl / filters_with_param * 100, 1) if filters_with_param else float("nan"),
                "date_min": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else "",
                "date_max": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else "",
            })
            # per-month rollup for top-row plotting
            sub_df = sub_df.dropna(subset=["_dt"])
            for period, cnt in sub_df["_dt"].dt.to_period("M").value_counts().items():
                monthly_rows.append({
                    "subproduct": sub, "parameter": str(param),
                    "month": period.to_timestamp(), "samples": int(cnt),
                })
    out = pd.DataFrame(rows).sort_values(["subproduct", "param_completeness_pct"],
                                          ascending=[True, False])
    out.to_csv(OUT_DIR / "etad_parameter_completeness.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(OUT_DIR / "etad_parameter_monthly.csv", index=False)
    return out


def plot_etad(table: pd.DataFrame) -> None:
    pm25 = table[table["subproduct"] == "ChemSpecPM25"].copy()
    if pm25.empty:
        return
    pm25 = pm25.sort_values("param_completeness_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(pm25))))
    bars = ax.barh(pm25["parameter"], pm25["param_completeness_pct"],
                   color=plt.cm.viridis(pm25["frac_above_mdl_pct"] / 100))
    for bar, n, above in zip(bars, pm25["n_filters_with_value"], pm25["frac_above_mdl_pct"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{int(n)} filters, {above:.0f}% > MDL",
                va="center", fontsize=7)
    ax.set_xlabel("% of ETAD filters with a value for this parameter")
    ax.set_title(f"ETAD (Addis Ababa) PM2.5 chemical-speciation parameter completeness — "
                 f"{int(pm25['n_total_filters_at_site'].iloc[0])} total filters")
    ax.set_xlim(0, 110)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=0, vmax=100))
    cb = plt.colorbar(sm, ax=ax, shrink=0.6)
    cb.set_label("% of valid values above MDL")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "etad_parameter_completeness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (3) World map of sites
# ---------------------------------------------------------------------------

def site_locations() -> pd.DataFrame:
    """Pull one row per site with mean lat/lon from FilterBased ChemSpecPM25."""
    rows: list[dict] = []
    for path in sorted((RAW_DIR / "FilterBased" / "ChemSpecPM25").glob("*.csv")):
        site = path.stem.rsplit("_", 1)[-1]
        df = _read(path)
        lat = pd.to_numeric(df.get("Latitude"), errors="coerce").dropna()
        lon = pd.to_numeric(df.get("Longitude"), errors="coerce").dropna()
        if lat.empty or lon.empty:
            continue
        rows.append({"site": site, "lat": float(lat.iloc[0]), "lon": float(lon.iloc[0])})
    # Sites appearing only under NephelProcd (e.g. CODC) are intentionally not
    # backfilled here: Nephel files carry no Latitude/Longitude columns, so
    # there is nothing to read. (This previously ran a full _read() over every
    # Nephel CSV and discarded the result.)
    return pd.DataFrame(rows)


def plot_world_map(locations: pd.DataFrame, ranked: pd.DataFrame) -> bool:
    """Draw the global site map. Returns False (with a note) if cartopy is absent.

    cartopy needs system GEOS/PROJ libraries, so it is an optional extra rather
    than a core dependency: `uv sync --extra geo`. Every earlier step of this
    pipeline has already written its output by the time we get here, so a
    missing cartopy must not fail the whole command.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("  SKIPPED: cartopy is not installed, so the world map was not drawn.")
        print("           Install it with:  uv sync --extra geo")
        return False

    merged = locations.merge(
        ranked[["site", "score", "mean_completeness", "n_subproducts", "total_samples"]],
        on="site", how="left",
    )
    merged["score"] = merged["score"].fillna(0)
    merged["n_subproducts"] = merged["n_subproducts"].fillna(0)

    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#e7eef5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#666666")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, color="#999999")

    sizes = 35 + merged["total_samples"].fillna(0).clip(upper=150000) / 150000 * 280
    sc = ax.scatter(
        merged["lon"], merged["lat"], c=merged["score"],
        s=sizes, cmap="viridis", vmin=0, vmax=80,
        edgecolor="black", linewidth=0.4, transform=ccrs.PlateCarree(),
        alpha=0.92, zorder=5,
    )
    for _, r in merged.iterrows():
        ax.text(r["lon"] + 2.0, r["lat"] + 0.5, r["site"],
                fontsize=7, transform=ccrs.PlateCarree(), zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7))
    cb = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Coverage score (mean completeness × breadth / 9, capped)")
    ax.set_title(f"SPARTAN sites — coverage score by site "
                 f"(marker size ∝ total observations, capped at 150k)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spartan_world_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> int:
    if not (OUT_DIR / "coverage.csv").exists():
        print("coverage.csv missing; run spartan_coverage_analysis.py first.")
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Splitting rankings: nephel-equipped vs filter-only...")
    nephel, filt = split_rankings()
    plot_split_rankings(nephel, filt)
    print(f"  nephel-equipped sites: {len(nephel)}; filter-only sites: {len(filt)}")

    print("\nNephel-equipped — top 10 by score:")
    print(nephel.head(10).to_string(index=False))
    print("\nFilter-only — top 10 by score:")
    print(filt.head(10).to_string(index=False))

    print("\nComputing ETAD parameter completeness...")
    table = etad_parameter_completeness()
    print(table[table["subproduct"] == "ChemSpecPM25"]
          [["parameter", "n_filters_with_value", "param_completeness_pct",
            "frac_above_mdl_pct"]].head(15).to_string(index=False))
    plot_etad(table)

    print("\nBuilding world map...")
    locations = site_locations()
    locations.to_csv(OUT_DIR / "site_locations.csv", index=False)
    print(f"  located {len(locations)} sites with lat/lon")
    ranked = pd.read_csv(OUT_DIR / "top_sites.csv")
    plot_world_map(locations, ranked)
    print(f"  wrote {FIG_DIR/'spartan_world_map.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
