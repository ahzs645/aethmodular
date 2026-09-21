#!/usr/bin/env python3
"""Census of every figure produced by the notebooks in this repo.

Walks every .ipynb outside .venv/.git, finds the code cells that produce a
figure, records the matplotlib primitives each one uses, and maps that recipe
onto the chart taxonomy used by react-graph-gallery.com (the eight categories:
correlation, distribution, ranking, partOfWhole, evolution, map, flow).

Writes:
  chart_census.json   one record per figure cell
  chart_census.csv    the same, flat, for spreadsheets
  CENSUS.md           the human-readable rollup

Run:  python gallery/census/build_census.py
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SKIP_DIRS = {".venv", ".git", ".ipynb_checkpoints", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}

# matplotlib/seaborn primitives that actually put marks on an axes
PRIMITIVES = [
    "scatter", "plot", "bar", "barh", "hist", "hist2d", "boxplot", "violinplot",
    "hexbin", "imshow", "pcolormesh", "contourf", "fill_between", "fill_betweenx",
    "errorbar", "stackplot", "pie", "step", "stem", "heatmap", "regplot",
    "kdeplot", "quiver",
]
# decorations — they change which gallery chart is the right analogue, but
# never constitute a figure on their own
DECORATIONS = ["axhline", "axvline", "axvspan", "axhspan", "annotate", "text"]

MARK_RE = re.compile(r"\.(" + "|".join(PRIMITIVES) + r")\s*\(")
DECO_RE = re.compile(r"\.(" + "|".join(DECORATIONS) + r")\s*\(")
TITLE_RE = re.compile(r"(?:set_title|suptitle|plt\.title)\(\s*[fr]{0,2}['\"]([^'\"]{3,140})['\"]")
XLAB_RE = re.compile(r"(?:set_xlabel|plt\.xlabel)\(\s*[fr]{0,2}['\"]([^'\"]{1,80})['\"]")
YLAB_RE = re.compile(r"(?:set_ylabel|plt\.ylabel)\(\s*[fr]{0,2}['\"]([^'\"]{1,80})['\"]")
SAVE_RE = re.compile(r"savefig\(\s*[fr]{0,2}['\"]([^'\"]+\.(?:png|pdf|svg|jpg))['\"]")
MODULE_RE = re.compile(r"\b(crossplots|timeseries|distributions|comparisons|overlays)\.(\w+)\s*\(")

# signals that the x axis is time
TIME_HINT = re.compile(
    r"\b(date|time|datetime|timestamp|month|season|year|index\b|DatetimeIndex|"
    r"resample|SampleDate|dt\.)", re.I)
# signals of a fitted line / identity line sitting on top of a scatter
FIT_HINT = re.compile(r"(polyfit|linregress|\bOLS\b|deming|regress|trendline|1:1|one_to_one|np\.poly1d)", re.I)
LOG_HINT = re.compile(r"(set_[xy]scale\(\s*['\"]log|semilog|loglog)")
GRID_HINT = re.compile(r"(subplots\(\s*[2-9]|subplot2grid|GridSpec|add_subplot\(\s*[2-9])")
COLORBY_HINT = re.compile(r"\bc\s*=\s*(?!['\"])|cmap\s*=|hue\s*=")


def gallery_mapping(marks: set[str], src: str) -> tuple[str, str, list[str]]:
    """-> (gallery category, primary chart, alternative charts)

    Categories and chart names follow react-graph-gallery.com exactly.
    """
    has = marks.__contains__
    timeish = bool(TIME_HINT.search(src))

    if has("hexbin") or has("hist2d"):
        return "correlation", "2D Density", ["Hexbin", "Scatterplot", "Heatmap"]
    if has("imshow") or has("pcolormesh") or has("heatmap"):
        corr = "corr(" in src or "correlation" in src.lower()
        return "correlation", ("Correlogram" if corr else "Heatmap"), ["Heatmap", "Correlogram"]
    if has("scatter"):
        if COLORBY_HINT.search(src):
            return "correlation", "Bubble", ["Scatterplot", "2D Density", "Beeswarm"]
        if has("plot") and timeish and not FIT_HINT.search(src):
            return "correlation", "Connected Scatter", ["Scatterplot", "Line chart"]
        return "correlation", "Scatterplot", ["Bubble", "2D Density", "Correlogram"]
    if has("boxplot") or has("violinplot"):
        return "distribution", ("Violin" if has("violinplot") else "Boxplot"), ["Violin", "Boxplot", "Beeswarm", "Ridgeline"]
    if has("hist") or has("kdeplot"):
        return "distribution", ("Density" if has("kdeplot") else "Histogram"), ["Density", "Histogram", "Ridgeline", "Violin"]
    if has("stackplot"):
        return "evolution", "Stacked Area", ["Streamgraph", "Area chart"]
    if has("pie"):
        return "partOfWhole", "Donut", ["Pie Chart", "Treemap", "Circular Packing"]
    if has("fill_between") or has("fill_betweenx"):
        return "evolution", "Area chart", ["Line chart", "Timeseries", "Stacked Area", "Ridgeline"]
    if has("errorbar"):
        return "ranking", "Lollipop", ["Barplot", "Boxplot", "Beeswarm"]
    if has("bar") or has("barh"):
        return "ranking", "Barplot", ["Lollipop", "Circular Barplot", "Treemap", "Spider/Radar"]
    if has("plot") or has("step"):
        return "evolution", ("Timeseries" if timeish else "Line chart"), ["Line chart", "Timeseries", "Area chart", "Streamgraph"]
    return "correlation", "Scatterplot", []


def zone_of(rel: str) -> str:
    parts = Path(rel).parts
    if parts[0] == "research":
        return "research/" + (parts[1] if len(parts) > 1 else "")
    if parts[0] == "notebooks":
        return "notebooks/" + (parts[1] if len(parts) > 1 else "")
    if parts[0] == "deliverables":
        return "deliverables"
    return parts[0]


def is_archived(rel: str) -> bool:
    low = rel.lower()
    return "/archive/" in low or low.startswith("archive/") or "/attic/" in low or "/scratch/" in low


def walk_notebooks():
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".ipynb"):
                yield Path(dirpath) / fn


def build():
    records = []
    nb_count = 0
    for path in sorted(walk_notebooks()):
        rel = str(path.relative_to(REPO))
        try:
            nb = json.loads(path.read_text())
        except Exception:
            continue
        nb_count += 1
        cells = nb.get("cells", [])
        # carry the most recent markdown heading as section context
        heading = ""
        for idx, cell in enumerate(cells):
            if cell.get("cell_type") == "markdown":
                md = "".join(cell.get("source", []))
                hs = re.findall(r"^#{1,4}\s+(.{3,120})$", md, re.M)
                if hs:
                    heading = hs[-1].strip()
                continue
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            marks = set(MARK_RE.findall(src))
            if not marks:
                continue
            decos = sorted(set(DECO_RE.findall(src)))
            category, primary, alts = gallery_mapping(marks, src)
            titles = TITLE_RE.findall(src)
            n_png = sum(
                1 for o in (cell.get("outputs") or []) if "image/png" in (o.get("data") or {})
            )
            records.append({
                "notebook": rel,
                "zone": zone_of(rel),
                "archived": is_archived(rel),
                "cell_index": idx,
                "section": heading,
                "title": titles[0] if titles else "",
                "all_titles": titles,
                "xlabel": (XLAB_RE.findall(src) or [""])[0],
                "ylabel": (YLAB_RE.findall(src) or [""])[0],
                "recipe": "+".join(sorted(marks)),
                "marks": sorted(marks),
                "decorations": decos,
                "plot_module_calls": ["{}.{}".format(a, b) for a, b in MODULE_RE.findall(src)],
                "gallery_category": category,
                "gallery_chart": primary,
                "gallery_alternatives": alts,
                "has_fit": bool(FIT_HINT.search(src)),
                "has_log_axis": bool(LOG_HINT.search(src)),
                "is_grid": bool(GRID_HINT.search(src)),
                "color_by_third_var": bool(COLORBY_HINT.search(src)),
                "saved_to": SAVE_RE.findall(src),
                "rendered_png_outputs": n_png,
                "n_code_lines": src.count("\n") + 1,
            })

    OUT.joinpath("chart_census.json").write_text(json.dumps(records, indent=1))

    flat_cols = ["notebook", "zone", "archived", "cell_index", "section", "title",
                 "xlabel", "ylabel", "recipe", "gallery_category", "gallery_chart",
                 "has_fit", "has_log_axis", "is_grid", "color_by_third_var",
                 "rendered_png_outputs"]
    with OUT.joinpath("chart_census.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    write_markdown(records, nb_count)
    return records, nb_count


def write_markdown(records, nb_count):
    live = [r for r in records if not r["archived"]]
    by_cat = Counter(r["gallery_category"] for r in records)
    by_chart = Counter(r["gallery_chart"] for r in records)
    by_chart_live = Counter(r["gallery_chart"] for r in live)
    by_zone = Counter(r["zone"] for r in records)
    by_recipe = Counter(r["recipe"] for r in records)
    nbs = {r["notebook"] for r in records}

    chart_to_nbs = defaultdict(set)
    for r in records:
        chart_to_nbs[r["gallery_chart"]].add(r["notebook"])

    L = []
    A = L.append
    A("# Notebook chart census")
    A("")
    A("Generated by `gallery/census/build_census.py`. Every figure-producing code")
    A("cell in every notebook in this repo, mapped onto the chart taxonomy at")
    A("<https://www.react-graph-gallery.com/>.")
    A("")
    A(f"- **{nb_count}** notebooks scanned, **{len(nbs)}** of them draw something")
    A(f"- **{len(records)}** figure-producing cells (**{len(live)}** outside `archive/`, `scratch/`, `attic/`)")
    A(f"- **{sum(r['rendered_png_outputs'] for r in records)}** rendered PNG outputs still embedded in the notebooks")
    A("")
    A("## By react-graph-gallery category")
    A("")
    A("| Category | Figures | Share |")
    A("|---|---:|---:|")
    for cat, n in by_cat.most_common():
        A(f"| {cat} | {n} | {100*n/len(records):.0f} % |")
    A("")
    A("## By chart type")
    A("")
    A("| Gallery chart | Figures | Live | Notebooks | Gallery page |")
    A("|---|---:|---:|---:|---|")
    slug = {
        "Scatterplot": "scatter-plot", "Bubble": "bubble-plot", "2D Density": "2d-density-plot",
        "Connected Scatter": "connected-scatter-plot", "Heatmap": "heatmap",
        "Correlogram": "correlogram", "Boxplot": "box-plot", "Violin": "violin-plot",
        "Histogram": "histogram", "Density": "density-plot", "Ridgeline": "ridgeline-plot",
        "Beeswarm": "beeswarm", "Barplot": "barplot", "Lollipop": "lollipop",
        "Line chart": "line-chart", "Timeseries": "timeseries", "Area chart": "area-plot",
        "Stacked Area": "stacked-area-plot", "Donut": "donut", "Pie Chart": "pie-plot",
        "Treemap": "treemap",
    }
    for chart, n in by_chart.most_common():
        s = slug.get(chart, "")
        page = f"[{chart}](https://www.react-graph-gallery.com/{s})" if s else chart
        A(f"| {chart} | {n} | {by_chart_live.get(chart,0)} | {len(chart_to_nbs[chart])} | {page} |")
    A("")
    A("## By repo zone")
    A("")
    A("| Zone | Figures |")
    A("|---|---:|")
    for z, n in by_zone.most_common():
        A(f"| `{z}` | {n} |")
    A("")
    A("## Raw matplotlib recipes (top 25)")
    A("")
    A("The primitive combination each cell actually uses. This is what the mapping above is derived from.")
    A("")
    A("| Recipe | Cells |")
    A("|---|---:|")
    for rec, n in by_recipe.most_common(25):
        A(f"| `{rec}` | {n} |")
    A("")
    A("## Variant features")
    A("")
    A("| Feature | Figures |")
    A("|---|---:|")
    A(f"| fitted regression line | {sum(1 for r in records if r['has_fit'])} |")
    A(f"| colour-by-third-variable | {sum(1 for r in records if r['color_by_third_var'])} |")
    A(f"| multi-panel grid | {sum(1 for r in records if r['is_grid'])} |")
    A(f"| log axis | {sum(1 for r in records if r['has_log_axis'])} |")
    A("")
    A("## Heaviest notebooks")
    A("")
    A("| Notebook | Figures | Dominant chart |")
    A("|---|---:|---|")
    per_nb = Counter(r["notebook"] for r in records)
    for nb, n in per_nb.most_common(25):
        dom = Counter(r["gallery_chart"] for r in records if r["notebook"] == nb).most_common(1)[0][0]
        A(f"| `{nb}` | {n} | {dom} |")
    A("")
    OUT.joinpath("CENSUS.md").write_text("\n".join(L))


if __name__ == "__main__":
    recs, n = build()
    print(f"scanned {n} notebooks -> {len(recs)} figure cells")
    print(f"wrote {OUT/'chart_census.json'}")
    print(f"wrote {OUT/'chart_census.csv'}")
    print(f"wrote {OUT/'CENSUS.md'}")
