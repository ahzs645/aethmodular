"""Visualize SPARTAN coverage and rank sites by data coverage.

Reads cached CSVs under data/spartan/raw/, computes per-(site, subproduct,
month) sample counts, and writes:

    research/spartan/inventory/figures/coverage_timeline.png
        site x month presence ribbons, one panel per subproduct
    research/spartan/inventory/figures/monthly_heatmap_filter_pm25.png
        site x month filter counts for FilterBased/ChemSpecPM25
    research/spartan/inventory/figures/monthly_heatmap_hourly_pm25_est.png
        site x month sample counts for TimeResPM25/HourlyEstPM25
    research/spartan/inventory/figures/top_sites_completeness.png
        bar chart of per-site mean completeness across all subproducts present
    research/spartan/inventory/top_sites.csv
        ranking table (mean completeness, total samples, # subproducts, span)
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
RAW_DIR = REPO_ROOT / "data" / "spartan" / "raw"
OUT_DIR = REPO_ROOT / "research" / "spartan" / "inventory"
FIG_DIR = OUT_DIR / "figures"


def _find_header_line(path: Path, max_scan: int = 5) -> int:
    with open(path, "r", errors="replace") as f:
        for i in range(max_scan):
            line = f.readline()
            if not line:
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            if "," in stripped and not stripped.lstrip().startswith("#"):
                low = stripped.lower()
                if any(tok in low for tok in ("site_code", "year", "year_local")):
                    return i
    return 0


def _load_dt(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, skiprows=_find_header_line(path), low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    y = col("year_local", "start_year_local", "year")
    m = col("month_local", "start_month_local", "month")
    d = col("day_local", "start_day_local", "day")
    h = col("hour_local", "start_hour_local", "hour")
    if not (y and m and d):
        return df, pd.Series([], dtype="datetime64[ns]")
    parts = {
        "year": pd.to_numeric(df[y], errors="coerce"),
        "month": pd.to_numeric(df[m], errors="coerce"),
        "day": pd.to_numeric(df[d], errors="coerce"),
    }
    if h:
        parts["hour"] = pd.to_numeric(df[h], errors="coerce").fillna(0).astype(int)
    frame = pd.DataFrame(parts).dropna(subset=["year", "month", "day"])
    frame = frame[(frame["year"] >= 2010) & (frame["year"] <= 2030)]
    return df, pd.to_datetime(frame, errors="coerce").dropna()


def gather_samples_per_month() -> pd.DataFrame:
    """One row per (product, subproduct, site, month) with sample count."""
    rows: list[dict] = []
    for path in sorted(RAW_DIR.glob("*/*/*.csv")):
        product = path.parents[1].name
        sub = path.parent.name
        site = path.stem.rsplit("_", 1)[-1]
        df, dt = _load_dt(path)
        if dt.empty:
            continue
        if product == "FilterBased":
            fid = next((c for c in df.columns if c.lower() == "filter_id"), None)
            if fid:
                # one observation per filter
                sub_df = df.loc[dt.index].dropna(subset=[fid]).drop_duplicates(subset=[fid])
                dt_use = dt.loc[sub_df.index]
            else:
                dt_use = dt.drop_duplicates()
        else:
            dt_use = dt.drop_duplicates()
        if dt_use.empty:
            continue
        months = dt_use.dt.to_period("M")
        counts = months.value_counts().sort_index()
        for period, n in counts.items():
            rows.append({
                "product": product, "subproduct": sub, "site": site,
                "month": period.to_timestamp(), "samples": int(n),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_timeline(monthly: pd.DataFrame, out: Path) -> None:
    """One panel per subproduct: site rows colored where data exists."""
    subs = sorted(monthly["subproduct"].unique(),
                  key=lambda s: ("Z" if "Hourly" in s else "M" if "Daily" in s else "A") + s)
    n = len(subs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1 + 0.35 * 40 + n * 0.5), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, sub in zip(axes, subs):
        m = monthly[monthly["subproduct"] == sub]
        all_sites = sorted(m["site"].unique())
        site_to_y = {s: i for i, s in enumerate(all_sites)}
        cmap = plt.get_cmap("viridis")
        max_n = m["samples"].max() if len(m) else 1
        for _, r in m.iterrows():
            ax.barh(
                site_to_y[r["site"]], 31, left=r["month"],
                height=0.85, color=cmap(min(r["samples"] / max_n, 1.0)),
                edgecolor="none",
            )
        ax.set_yticks(list(site_to_y.values()))
        ax.set_yticklabels(list(site_to_y.keys()), fontsize=7)
        ax.set_title(f"{m['product'].iloc[0]} / {sub}  ({len(all_sites)} sites)", fontsize=10, loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(pd.Timestamp("2013-01-01"), pd.Timestamp("2026-12-31"))
        ax.grid(True, axis="x", alpha=0.2)
    fig.suptitle("SPARTAN data availability — sites x months (color = sample count)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(monthly: pd.DataFrame, subproduct: str, out: Path, title: str) -> None:
    m = monthly[monthly["subproduct"] == subproduct].copy()
    if m.empty:
        return
    pv = (
        m.pivot_table(index="site", columns="month", values="samples", fill_value=0)
         .sort_index()
    )
    full_idx = pd.date_range(pv.columns.min(), pv.columns.max(), freq="MS")
    pv = pv.reindex(columns=full_idx, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(12, 0.18 * pv.shape[1]), max(4, 0.28 * pv.shape[0])))
    vmax = np.percentile(pv.values[pv.values > 0], 95) if (pv.values > 0).any() else 1
    im = ax.imshow(pv.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(vmax, 1))

    ax.set_yticks(range(pv.shape[0]))
    ax.set_yticklabels(pv.index, fontsize=8)
    step = max(pv.shape[1] // 14, 1)
    xticks = list(range(0, pv.shape[1], step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([pv.columns[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("samples / month")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def rank_sites(cov_path: Path, out: Path) -> pd.DataFrame:
    cov = pd.read_csv(cov_path)
    cov["completeness_pct"] = pd.to_numeric(cov["completeness_pct"], errors="coerce")
    # Some new FilterBased sites sample faster than the 1-in-9-days nominal,
    # which inflates raw completeness past 100%. Cap so the ranking is
    # interpretable as "fraction of nominal cadence delivered, up to 100%".
    cov["completeness_capped"] = cov["completeness_pct"].clip(upper=100)
    g = (
        cov.groupby("site")
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
    # composite score: capped completeness x breadth (max breadth = 9 subproducts)
    g["score"] = (g["mean_completeness"].fillna(0) * (g["n_subproducts"] / 9)).round(1)
    g = g.sort_values(["score", "total_samples"], ascending=False).reset_index(drop=True)
    g.to_csv(out, index=False)
    return g


def plot_top_sites(ranked: pd.DataFrame, out: Path, top: int = 20) -> None:
    g = ranked.head(top).iloc[::-1]  # flip so highest is at top of horizontal bars
    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(g))))
    bars = ax.barh(g["site"], g["mean_completeness"], color="steelblue")
    for bar, n, sp in zip(bars, g["total_samples"], g["n_subproducts"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{int(n):,} obs / {int(sp)} products",
                va="center", fontsize=8, color="black")
    ax.set_xlabel("Mean completeness across the subproducts present at the site (%)")
    ax.set_title(f"Top {top} SPARTAN sites by coverage")
    ax.set_xlim(0, max(100, g["mean_completeness"].max() * 1.15))
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not RAW_DIR.exists():
        print(f"No raw data at {RAW_DIR}")
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Aggregating monthly sample counts across every file...")
    monthly = gather_samples_per_month()
    monthly.to_csv(OUT_DIR / "monthly_samples.csv", index=False)
    print(f"  rows: {len(monthly):,}  sites: {monthly['site'].nunique()}  subproducts: {monthly['subproduct'].nunique()}")

    print("Plotting timeline...")
    plot_timeline(monthly, FIG_DIR / "coverage_timeline.png")

    print("Plotting per-subproduct heatmaps...")
    plot_heatmap(monthly, "ChemSpecPM25",
                 FIG_DIR / "monthly_heatmap_filter_pm25.png",
                 "Filter-based PM2.5 chemical speciation — filters per site per month")
    plot_heatmap(monthly, "HourlyEstPM25",
                 FIG_DIR / "monthly_heatmap_hourly_pm25_est.png",
                 "Estimated PM2.5 (hourly) — observations per site per month")
    plot_heatmap(monthly, "HourlyScaPM25",
                 FIG_DIR / "monthly_heatmap_nephel_hourly_pm25.png",
                 "Nephelometer PM2.5 (hourly) — observations per site per month")

    print("Ranking sites...")
    ranked = rank_sites(OUT_DIR / "coverage.csv", OUT_DIR / "top_sites.csv")
    plot_top_sites(ranked, FIG_DIR / "top_sites_completeness.png", top=20)

    print("\nTop 15 sites by composite coverage score:")
    print(ranked.head(15).to_string(index=False))
    print(f"\nWrote {OUT_DIR/'top_sites.csv'} and figures under {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
