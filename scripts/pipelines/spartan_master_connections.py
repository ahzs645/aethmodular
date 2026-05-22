"""Build the master cross-dataset connection matrix for every SPARTAN site
that appears in any of the sources we have:

  - Public FTP: GroupedByProduct/{FilterBased,NephelProcd,TimeResPM25}
  - HIPS Drive folder: SPARTAN_HIPS_Batch1-51.v2.csv
  - Site lookup: SPARTAN_Site_quick_lookup.xlsx

Outputs:
  research/spartan/inventory/master_connections.csv     row counts per source
  research/spartan/inventory/master_connections.md      markdown rollup
  research/spartan/inventory/figures/connection_matrix.png  visual heatmap
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "spartan" / "raw"
HIPS_PATH = REPO_ROOT / "data" / "drive_bridge" / "Spartan" / "SPARTAN_HIPS_Batch1-51.v2.csv"
LOOKUP_PATH = REPO_ROOT / "data" / "drive_bridge" / "Spartan" / "SPARTAN_Site_quick_lookup.xlsx"
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


# ---------------------------------------------------------------------------

def public_inventory() -> pd.DataFrame:
    """One row per (site, subproduct) with row count from cached public files."""
    rows = []
    for path in sorted(RAW_DIR.glob("*/*/*.csv")):
        product = path.parents[1].name
        sub = path.parent.name
        site = path.stem.rsplit("_", 1)[-1]
        df = pd.read_csv(path, skiprows=_find_header(path), low_memory=False)

        if product == "FilterBased":
            fid = next((c for c in df.columns if c.lower() == "filter_id"), None)
            count = df[fid].dropna().nunique() if fid else len(df)
            unit = "filters"
        else:
            count = len(df)
            unit = "rows"
        rows.append({"site": site, "subproduct": f"{product}/{sub}", "count": int(count), "unit": unit})
    return pd.DataFrame(rows)


def hips_inventory() -> pd.DataFrame:
    df = pd.read_csv(HIPS_PATH)
    df["is_blank"] = (
        df["FilterId"].astype(str).str.endswith("-7")
        | df["FilterId"].astype(str).str.contains("-LB", regex=False)
        | df["SampleDate"].isna()
    )
    sample = df[~df["is_blank"]].groupby("Site")["FilterId"].nunique()
    blank = df[df["is_blank"]].groupby("Site")["FilterId"].nunique()
    out = pd.DataFrame({"hips_sample_filters": sample, "hips_blank_filters": blank}).fillna(0).astype(int)
    out = out.reset_index().rename(columns={"Site": "site"})
    return out


def lookup_sites() -> pd.DataFrame:
    df = pd.read_excel(LOOKUP_PATH)
    return df.rename(columns={"SiteCode": "site",
                              "Description": "lookup_name",
                              "Latitude": "lookup_lat",
                              "Longitude": "lookup_lon"})


# ---------------------------------------------------------------------------

def build_master() -> pd.DataFrame:
    pub = public_inventory()
    pub_wide = (
        pub.pivot_table(index="site", columns="subproduct",
                        values="count", fill_value=0, aggfunc="sum")
    )
    pub_wide.columns.name = None
    hips = hips_inventory().set_index("site")
    look = lookup_sites().set_index("site")[["lookup_name", "lookup_lat", "lookup_lon"]]

    all_sites = sorted(set(pub_wide.index) | set(hips.index) | set(look.index))
    master = pd.DataFrame(index=all_sites)
    master.index.name = "site"
    master = master.join(look).join(pub_wide).join(hips)
    for c in master.columns:
        if c.startswith("FilterBased") or c.startswith("NephelProcd") or c.startswith("TimeResPM25"):
            master[c] = master[c].fillna(0).astype(int)
    for c in ("hips_sample_filters", "hips_blank_filters"):
        master[c] = master[c].fillna(0).astype(int)

    subprod_cols = [c for c in master.columns
                    if c.startswith(("FilterBased", "NephelProcd", "TimeResPM25"))]
    master["n_public_subproducts"] = (master[subprod_cols] > 0).sum(axis=1)
    master["in_public"] = master["n_public_subproducts"] > 0
    master["in_hips_data"] = master["hips_sample_filters"] > 0
    master["in_hips_blanks"] = master["hips_blank_filters"] > 0
    master["in_lookup"] = master["lookup_name"].notna()

    # priority ordering: most data first
    master["sort_key"] = (
        master["in_public"].astype(int) * 100
        + master["in_hips_data"].astype(int) * 10
        + master["n_public_subproducts"]
    )
    master = master.sort_values(["sort_key"], ascending=False).drop(columns="sort_key")
    return master


# ---------------------------------------------------------------------------

def plot_connection_matrix(master: pd.DataFrame, out: Path) -> None:
    subprod_cols = [c for c in master.columns
                    if c.startswith(("FilterBased", "NephelProcd", "TimeResPM25"))]
    extra = ["hips_sample_filters", "hips_blank_filters"]
    show_cols = subprod_cols + extra
    rename = {c: c.replace("FilterBased/", "Filt/").replace("NephelProcd/", "Neph/").replace("TimeResPM25/", "TR/") for c in show_cols}

    mat = master[show_cols].copy()
    mat_norm = mat.copy().astype(float)
    for c in mat_norm.columns:
        m = mat_norm[c].max()
        if m > 0:
            mat_norm[c] = mat_norm[c] / m
    mat_norm = mat_norm.where(mat > 0, np.nan)  # truly missing -> NaN/white

    fig, ax = plt.subplots(figsize=(11, max(8, 0.28 * len(mat))))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f1f1f1")  # color for "no data"
    im = ax.imshow(mat_norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(show_cols)))
    ax.set_xticklabels([rename[c] for c in show_cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat.iloc[i, j])
            if v > 0:
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        fontsize=6, color="white" if mat_norm.iloc[i, j] > 0.4 else "black")
    ax.set_title("SPARTAN data connections — rows/filters per site across every source", fontsize=11)
    cb = plt.colorbar(im, ax=ax, shrink=0.5)
    cb.set_label("rel. abundance within column (0-1)")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_markdown(master: pd.DataFrame, out: Path) -> None:
    subprod_cols = [c for c in master.columns
                    if c.startswith(("FilterBased", "NephelProcd", "TimeResPM25"))]
    n_total = len(master)
    pub_only = master[master["in_public"] & ~master["in_hips_data"]]
    hips_only = master[~master["in_public"] & master["in_hips_data"]]
    both = master[master["in_public"] & master["in_hips_data"]]
    blanks_only = master[master["in_hips_blanks"] & ~master["in_hips_data"]]
    lookup_only = master[master["in_lookup"] & ~master["in_public"] & ~master["in_hips_data"] & ~master["in_hips_blanks"]]
    missing_lookup = master[~master["in_lookup"]]

    lines = []
    add = lines.append
    add("# SPARTAN master connection matrix\n")
    add(f"- Total distinct sites across all sources: **{n_total}**")
    add(f"- Sites with both **public FTP data** and **HIPS data**: **{len(both)}**")
    add(f"- Sites with public data only (no HIPS): **{len(pub_only)}** "
        f"→ {', '.join(sorted(pub_only.index))}")
    add(f"- Sites with HIPS data only (not in public FTP yet): **{len(hips_only)}** "
        f"→ {', '.join(sorted(hips_only.index)) or '(none)'}")
    add(f"- Sites with HIPS blanks only (commissioning): **{len(blanks_only)}** "
        f"→ {', '.join(sorted(blanks_only.index)) or '(none)'}")
    add(f"- Sites in lookup but with no data anywhere: **{len(lookup_only)}** "
        f"→ {', '.join(sorted(lookup_only.index)) or '(none)'}")
    add(f"- Sites with data but **missing from the Drive lookup**: **{len(missing_lookup)}** "
        f"→ {', '.join(sorted(missing_lookup.index))}\n")

    # Coverage breakdown
    add("## Per-site connection grid\n")
    add("- Numbers are rows (NephelProcd / TimeResPM25) or unique filters (FilterBased / HIPS).")
    add("- 0 = source exists but site is absent there. Blank = site is missing from that source entirely.\n")
    show = master[["lookup_name"] + subprod_cols + ["hips_sample_filters", "hips_blank_filters", "n_public_subproducts"]]
    add(show.to_markdown())
    out.write_text("\n".join(lines))


# ---------------------------------------------------------------------------

def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Building master connection matrix...")
    master = build_master()
    master.to_csv(OUT_DIR / "master_connections.csv")
    print(f"  {len(master)} sites with one or more sources")
    print(master[[
        "lookup_name", "n_public_subproducts", "hips_sample_filters", "hips_blank_filters",
        "in_public", "in_hips_data", "in_lookup",
    ]].to_string())

    write_markdown(master, OUT_DIR / "master_connections.md")
    plot_connection_matrix(master, FIG_DIR / "connection_matrix.png")
    print(f"\nWrote {OUT_DIR/'master_connections.csv'}")
    print(f"Wrote {OUT_DIR/'master_connections.md'}")
    print(f"Wrote {FIG_DIR/'connection_matrix.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
