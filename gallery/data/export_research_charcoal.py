"""Export Research → Charcoal similarity: the per-filter charcoal scores of research/ftir_etad_char (char_05–char_09).

    uv run --locked --no-sync python gallery/data/export_research_charcoal.py

The char notebooks score every Addis (ETAD) filter spectrum against the charcoal archive. char_06 saves the
per-filter core (FilterId, date, season, LotId, r_char, r_raw, delta, the Gaussian-mixture component, Fabs and
EC) as output/tables/char06/per_filter_classification.csv; char_03 saves each filter's P(burned) and char_02 its
best match against the whole temperature-labelled library. The page joins these to the gallery's Addis rows in
the browser (FilterId ETAD-0017-1 → gallery id ETAD-0017), so the reader's seasons, exclusions and loading checks
apply and anything against Fabs/MAC follows the MAC. Everything else the page draws is the notebooks' summary
tables (per feedstock, per reference class, by subset), which need per-filter scores no script saves yet.

Each table is read, in order of preference, from the CSV the notebook wrote (the gitignored output/ folder,
present after a local run) or from the table the executed notebook displayed (committed with it), as
export_research_spectral.Page does. Without output/ the per-filter table is listed under `missing` and the page
shows the notebooks' tables only. Per-filter tables no script writes yet are listed with what they need.

Writes gallery/app/public/data/research/charcoal/scores.json (the Research section's data folder: private).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_research_spectral import CHAR, ROOT, Page, rel  # noqa: E402
from meeting_bundle import columns  # noqa: E402

OUT = ROOT / "gallery/app/public/data/research/charcoal"
TABLES = CHAR / "output/tables"

NB02, NB03 = "char_02_addis_vs_charcoal", "char_03_burned_vs_unburned"
NB05, NB06 = "char_05_addis_vs_unburned_feedstock", "char_06_dry_season_anomaly"
NB07, NB08 = "char_07_char05_without_dry_season", "char_08_fire_vs_furnace_char"
NB09, NB12 = "char_09_char08_dry_vs_nondry", "char_12_weakley_transferability"

# what a run of the char scripts needs that the repo does not hold
INPUTS = ("the charcoal archive (research/ftir_hips_chem/charcoal_ftir_sources/downloads/, fetched by that folder's scripts) "
          "and the AIRSpec-baselined ETAD spectra (research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz)")


class ScoresPage(Page):
    """A Page that keeps 6 decimals in the per-filter table: at the default 4, deltas the notebook had apart
    tie, and the page's Mann–Whitney p values drift from the notebook's in the third decimal."""

    def write(self) -> Path:
        path = super().write()
        bundle = json.loads(path.read_text())
        if "per_filter" in self.tables:
            bundle["tables"]["per_filter"] = columns(self.tables["per_filter"], digits=6)
        path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
        return path


def build_command(number: str) -> str:
    return f"cd {rel(CHAR)} && uv run python scripts/build_notebooks.py {number}"


def per_filter(p: Page) -> pd.DataFrame | None:
    """char_06's per-filter classification, with char_03's P(burned) and char_02's library match joined on.

    char_02 saves its matches without a FilterId, in the order of the same loader char_03 uses
    (etad_spectra.load_etad(baselined=True)); its rows are attached by position only when the two tables'
    seasons agree row for row, which is that check."""
    path06 = TABLES / "char06/per_filter_classification.csv"
    if not path06.exists():
        p.missing.append({"table": "per_filter", "what": "every Addis filter's r_char, r_raw, delta, mixture component and lot (char_06), "
                          "P(burned) (char_03) and best library match (char_02), which make the page's per-filter charts",
                          "needs": rel(path06), "run": build_command("02 03 05 06"), "after": INPUTS})
        return None
    frame = pd.read_csv(path06).rename(columns={"LotId": "lot"})
    paths = [rel(path06)]
    path03, path02 = TABLES / "char03/addis_burned_scores.csv", TABLES / "char02/addis_analogue_matches.csv"
    if path03.exists():
        burned = pd.read_csv(path03)
        if path02.exists():
            match = pd.read_csv(path02)
            if len(match) == len(burned) and (match["season_label"].to_numpy() == burned["season"].to_numpy()).all():
                burned["library_r"] = match["best_r"].to_numpy()
                burned["implied_T"] = match["median_top_temp"].to_numpy()
                paths.append(rel(path02))
            else:
                print("char_02 matches do not line up with char_03's filters: library match left out")
        frame = frame.merge(burned[[c for c in ("FilterId", "p_burned", "library_r", "implied_T") if c in burned]], on="FilterId", how="left")
        paths.append(rel(path03))
    frame = frame[[c for c in ("FilterId", "date", "season", "lot", "r_char", "r_raw", "delta", "component", "Fabs", "EC",
                               "library_r", "implied_T", "p_burned") if c in frame]]
    return p.add("per_filter", frame, notebook=f"{NB06}, {NB03}, {NB02}", read="csv", path="; ".join(paths))


def build() -> Page:
    p = ScoresPage("scores", "Charcoal similarity per filter", out=OUT, exporter=Path(__file__))
    rows = per_filter(p)

    # ------------------------------------------------------------ char_05: raw feedstock vs char
    p.shown("by_season_05", CHAR, NB05, "display(season_tab)", csv="output/tables/char05/by_season.csv")
    p.shown("feedstock_match", CHAR, NB05, "display(match_table)", csv="output/tables/char05/addis_vs_unburned_match.csv", index="reference")

    # ------------------------------------------------------------ char_06: the dry-season anomaly
    p.shown("season_medians_06", CHAR, NB06, 'display(df.groupby("season")[["r_char", "r_raw", "delta"]]', index="season")
    p.shown("lot_within_251", CHAR, NB06, "t1.to_csv", nth=1, csv="output/tables/char06/lot_control_within_lot251.csv", index="season")
    p.shown("lot_within_dry", CHAR, NB06, "t1.to_csv", nth=2, csv="output/tables/char06/lot_control_within_dry.csv", index="LotId")
    p.shown("mixture_vs_season", CHAR, NB06, "mixture_vs_season.csv", csv="output/tables/char06/mixture_vs_season.csv", index="season")
    p.shown("competing", CHAR, NB06, "res.to_csv", nth=1, csv="output/tables/char06/competing_explanations.csv", index="model")
    p.shown("quality", CHAR, NB06, "quality.to_csv", csv="output/tables/char06/spectral_quality.csv", index="group")
    # the monthly rate is drawn, not displayed: only the CSV has it (the page recomputes it from per_filter anyway)
    p.optional("monthly_06", CHAR, NB06, "output/tables/char06/monthly_anomaly_rate.csv",
               "char_06's share of filters with delta > 0.05 per calendar month")
    # within the dry season, does loading separate the two groups? (printed, Mann–Whitney)
    printed = p.stream(CHAR, NB06, "res.to_csv")
    loading = [{"measure": m, "low-similarity median": float(a), "normal median": float(b), "Mann–Whitney p": float(q)}
               for m, a, b, q in re.findall(r"(Fabs|EC)\s*: anomalous median ([\d.]+) vs normal ([\d.]+), p=([\d.]+)", printed)]
    if loading:
        p.add("loading_within_dry", pd.DataFrame(loading), notebook=NB06, read="executed notebook output (printed)",
              path=rel(CHAR / f"{NB06}.ipynb"), cell="res.to_csv")

    # ------------------------------------------------------------ char_07: char_05 without the dry season
    p.shown("headline_07", CHAR, NB07, "comparison.to_csv", csv="output/tables/char07/headline_comparison.csv")
    p.shown("rules_07", CHAR, NB07, "robust.to_csv", csv="output/tables/char07/robustness_exclusion_rules.csv")
    p.shown("feedstock_ranking", CHAR, NB07, 'feed.round(4).to_csv', csv="output/tables/char07/feedstock_ranking.csv", index="feedstock")

    # ------------------------------------------------------------ char_08 / char_09: which kind of charcoal
    p.shown("class_match", CHAR, NB08, "summary.round(4).to_csv", csv="output/tables/char08/reference_class_match.csv", index="class")
    p.shown("class_by_season", CHAR, NB08, "by_season.round(4).to_csv", csv="output/tables/char08/by_season_by_class.csv", index="class")
    p.shown("class_by_subset", CHAR, NB09, "rank.round(4).to_csv", csv="output/tables/char09/class_ranking_by_subset.csv", index="class")
    p.shown("furnace_fire", CHAR, NB09, "fvf.round(6).to_csv", csv="output/tables/char09/furnace_vs_fire_by_subset.csv")
    p.shown("relative_loss", CHAR, NB09, "loss.round(4).to_csv", csv="output/tables/char09/relative_loss_by_class.csv", index="class")
    p.shown("derivative_check", CHAR, NB12, "deriv_tbl.round(4).to_csv", csv="output/tables/char12/derivative_robustness.csv", index="measure")

    # ------------------------------------------------------------ what no script saves yet
    for table, what, script in [
        ("per_filter_feedstock", "each filter's r against each of the six raw feedstocks (char_05 [14] right, char_07 [9]), "
         "so the per-feedstock ranking would follow the reader's seasons and exclusions", "05"),
        ("per_filter_class", "each filter's best r against each of the seven reference classes (char_08 [6] furnace vs fire per filter, "
         "char_08 [16], char_09 [8][10])", "08"),
        ("per_filter_escape", "char_12's per-filter envelope-escape score (FilterId, season, lot) and the IMPROVE / ETAD envelope spectra "
         "(mean, p05, p95 per wavenumber, raw and 2nd derivative; char_12 [5][7][9][11][13])", "12"),
    ]:
        p.missing.append({"table": table, "what": what + "; the script does not save it yet: add a FilterId-keyed to_csv",
                          "needs": rel(TABLES / f"char{script}"), "run": build_command(script), "after": INPUTS})

    n = len(rows) if rows is not None else 239
    p.text["finding"] = (
        f"Each of the {n} Addis filters with a baselined spectrum is scored by its best Pearson r against McCall's charred feedstock "
        "(r_char) and its raw feedstock (r_raw); delta = r_raw − r_char. Within McCall raw biomass is the closer analogue for most filters, "
        "and the margin is largest in the dry season, where similarity to char collapses into a distinct low-similarity population "
        "(a two-component Gaussian mixture on r_char, fitted without the season) that recurs every October–February and is not "
        "the filter lot or the loading. The full charcoal library still matches Addis better than either McCall class "
        "(furnace charcoal r ≈ 0.78), so none of this says Addis is unburned biomass.")
    return p


if __name__ == "__main__":
    build().write()
