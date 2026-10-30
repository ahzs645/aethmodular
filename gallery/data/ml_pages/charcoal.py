"""Charcoal archive: unsupervised structure, analogue matching and classifiers (research/ftir_etad_char).

char_04 (silhouette by k, KMeans passes A and B, the type composition by collection, temperature and source),
char_02 (Mahalanobis distance of Addis from the charcoal cloud, the analogue matcher and its recovery check, the
baseline sensitivity), char_03 (the leave-one-feedstock-out burned/unburned classifier and its Addis scores),
char_06 (the two-component Gaussian mixture on r_char) and char_11 (the transform benchmark and what each transform
concludes for Addis).

Each table is the CSV the notebook wrote when its gitignored output/ exists, else the table the executed notebook
displayed or printed. The per-filter and per-reference tables (analogue matches, LOO scores, P(burned), the mixture
labels) only exist in output/, and are listed as missing without it. What no script saves at all (PCA scores, the
per-filter Mahalanobis distance, the Ward linkage, the AIRSpec fit traces) is listed with the run that would need
to save it. Nothing here depends on the MAC.
"""
from __future__ import annotations

import re

import pandas as pd

from ._common import CHAR, long, number, page, rel

KEY = "charcoal"

NB02, NB03, NB04 = "char_02_addis_vs_charcoal", "char_03_burned_vs_unburned", "char_04_spectral_typology"
NB06, NB11 = "char_06_dry_season_anomaly", "char_11_normalization_alternatives"
TABLES = CHAR / "output/tables"
INPUTS = ("the charcoal archive (research/ftir_hips_chem/charcoal_ftir_sources/downloads/) and the AIRSpec-baselined ETAD "
          "spectra (research/ftir_ec_phase3/output/corrected/etad_corrected_df6.npz)")


def run(number: str) -> str:
    return f"cd {rel(CHAR)} && uv run python scripts/build_notebooks.py {number}"


def printed(p, nb: str, marker: str) -> tuple[str, dict]:
    return p.stream(CHAR, nb, marker), {"notebook": nb, "read": "executed notebook output (printed)", "path": rel(CHAR / f"{nb}.ipynb"), "cell": marker}


def per_reference(p, name: str, csv: str, nb: str, what: str, number_: str, keep: list[str], rename: dict | None = None):
    """A per-filter (or per-reference-spectrum) table only a run writes: added when present, else listed as missing."""
    path = TABLES / csv
    if not path.exists():
        p.missing.append({"table": name, "what": what, "needs": rel(path), "run": run(number_), "after": INPUTS})
        return None
    frame = pd.read_csv(path).rename(columns=rename or {})
    return p.add(name, frame[[c for c in keep if c in frame]], notebook=nb, read="csv", path=rel(path))


def build():
    p = page(KEY, "Charcoal archive: structure, analogue matching and classifiers")

    # ------------------------------------------------------------ B. char_04: how many types, and what they are
    scan, src = p.read(CHAR, NB04, 'display(pd.concat([scanA.assign(pass_="A")')
    scan = scan.rename(columns={"A": "Pass A · charcoal only", "B": "Pass B · charcoal + Addis"})
    p.add("silhouette", long(scan, ["k"], "pass", "silhouette"), **src)
    s5, src5 = printed(p, NB04, 'display(pd.concat([scanA.assign(pass_="A")')
    chosen = [{"pass": "Pass A · charcoal only", "k": number(r"best k — Pass A: (\d+)", s5, int), "silhouette": number(r"Pass A: \d+ \(silhouette ([\d.]+)\)", s5)},
              {"pass": "Pass B · charcoal + Addis", "k": number(r"best k — Pass B: (\d+)", s5, int), "silhouette": number(r"Pass B: \d+ \(silhouette ([\d.]+)\)", s5)}]
    p.add("chosen_k", pd.DataFrame(chosen), **src5)

    by_coll, src = p.read(CHAR, NB04, "ct_src.to_csv", csv="output/tables/char04/passA_type_by_collection.csv", index="source")
    p.add("passA_collection", long(by_coll.rename(columns=lambda c: f"type {int(float(c))}" if re.fullmatch(r"[\d.]+", str(c)) else c), ["source"], "type", "share"), **src)
    by_t, src = p.read(CHAR, NB04, "ct_src.to_csv", nth=1, csv="output/tables/char04/passA_type_by_temperature.csv", index="temp_c")
    by_t = by_t.rename(columns=lambda c: f"type {int(float(c))}" if re.fullmatch(r"[\d.]+", str(c)) else c)
    by_t["temperature"] = by_t["temp_c"].map(lambda t: f"{int(t)} °C" if t else "0 °C (unburned)")
    p.add("passA_temperature", long(by_t.drop(columns="temp_c"), ["temperature"], "type", "share"), **src)
    s9, src9 = printed(p, NB04, "ct_src.to_csv")
    assoc = [{"against": a, "Cramér's V": number(rf"cluster with {re.escape(a)}[^:\n]*: ([\d.]+)", s9)}
             for a in ("temperature", "source collection", "species/feedstock")]
    p.add("association", pd.DataFrame(assoc), **src9)

    path = TABLES / "char04/passB_type_by_source.csv"
    if path.exists():
        b = pd.read_csv(path).rename(columns=lambda c: f"type {c}" if re.fullmatch(r"\d+", str(c)) else c)
        p.add("passB_source", long(b, ["source"], "type", "share"), notebook=NB04, read="csv", path=rel(path))
    else:
        p.missing.append({"table": "passB_source", "what": "Pass B's type composition of every source (the executed notebook shows Addis's row only)",
                          "needs": rel(path), "run": run("04"), "after": INPUTS})
    addis, src = p.read(CHAR, NB04, "comp.round(3).to_csv")
    p.add("passB_addis", addis.assign(cluster=lambda d: d["cluster"].map(lambda c: f"type {int(c)}")), **src)
    purity, src = p.read(CHAR, NB04, "comp.round(3).to_csv", nth=1)
    p.add("passB_purity", purity.assign(cluster=lambda d: d["cluster"].map(lambda c: f"type {int(c)}")), **src)

    # ------------------------------------------------------------ B. char_02 / char_03: how far outside the archive Addis sits
    s20, src20 = printed(p, NB02, "04b_mahalanobis_distance.png")
    s03, src03 = printed(p, NB03, "pca_tr = PCA(n_components=8")
    dist = [
        {"comparison": "char_02 · Addis in the charcoal PCA (6 PCs of 5,889 archive spectra)",
         "reference median": number(r"charcoal: ([\d.]+)", s20), "reference 99th pct": number(r"99th percentile \(([\d.]+)\)", s20),
         "Addis median": number(r"Addis: ([\d.]+)", s20), "Addis beyond the 99th pct (%)": number(r"99th percentile \([\d.]+\): (\d+)%", s20)},
        {"comparison": "char_03 · Addis in the McCall burned/unburned training cloud (8 PCs)",
         "reference median": number(r"McCall training spectra : median ([\d.]+)", s03), "reference 99th pct": number(r"99th pct ([\d.]+)", s03),
         "Addis median": number(r"Addis filters\s+: median ([\d.]+)", s03), "Addis beyond the 99th pct (%)": number(r"99th percentile: (\d+)%", s03)},
    ]
    p.add("mahalanobis", pd.DataFrame(dist), notebook=f"{NB02}, {NB03}", read="executed notebook output (printed)",
          path=f"{src20['path']}; {src03['path']}", cell=f"{src20['cell']}; {src03['cell']}")
    p.shown("baseline_sensitivity", CHAR, NB02, "check.to_csv", csv="output/tables/char02/baseline_sensitivity.csv", index="measure")

    # ------------------------------------------------------------ B. char_11: is there a better transform than SNV?
    p.shown("benchmark", CHAR, NB11, "bench.to_csv", csv="output/tables/char11/benchmark.csv")
    p.shown("addis_by_transform", CHAR, NB11, "addis_tab.to_csv", csv="output/tables/char11/addis_conclusion_by_transform.csv")
    p.shown("airspec_check", CHAR, NB11, "airspec_check.to_csv", csv="output/tables/char11/airspec_range_check.csv")
    p.shown("airspec_coverage", CHAR, NB11, "display(cover)")

    # ------------------------------------------------------------ C. char_02: nearest-analogue matching and its check
    s14, src14 = printed(p, NB02, "best-match correlation, Addis filters")
    if not s14:
        s14, src14 = printed(p, NB02, "addis_match = match(X_addis)")
    blocks = re.split(r"best-match correlation, held-out charcoal vs library \(known answer\):", s14)
    ref = []
    for label, text in zip(["Addis filters vs the library", "held-out charcoal vs the library (known answer)"], blocks):
        ref.append({"group": label, **{k: number(rf"{re.escape(k)}\s+([\d.]+)", text) for k in ("count", "min", "25%", "50%", "75%", "max")}})
    p.add("analogue_reference", pd.DataFrame(ref), **src14)
    p.add("recovery", pd.DataFrame([{"measure": "cross-collection temperature recovery on known charcoal, median |error| (°C)",
                                      "value": number(r"median \|error\| = (\d+)", s14, int)}]), **src14)
    per_reference(p, "analogue", "char02/addis_analogue_matches.csv", NB02,
                  "each Addis filter's best library match, implied temperature (median of the top matches) and their spread", "02",
                  ["best_r", "mean_top_r", "median_top_temp", "temp_iqr", "best_collection", "season"], {"season_label": "season"})
    p.shown("analogue_collection", CHAR, NB02, "addis_match.to_csv", nth=1)

    # ------------------------------------------------------------ C. char_03: the burned/unburned classifier
    s14c, src14c = printed(p, NB03, "cv.to_csv")
    p.add("loo_overall", pd.DataFrame([
        {"measure": "overall accuracy", "%": number(r"overall accuracy\s*: ([\d.]+)%", s14c)},
        {"measure": "unburned recall (n 24)", "%": number(r"unburned recall\s*: ([\d.]+)%", s14c)},
        {"measure": "burned recall (n 216)", "%": number(r"(?<!un)burned recall\s*: ([\d.]+)%", s14c)}]), **src14c)
    p.shown("loo_accuracy", CHAR, NB03, "cv.to_csv")
    per_reference(p, "loo", "char03/loo_feedstock_cv.csv", NB03,
                  "the leave-one-feedstock-out P(burned) of each of the 240 McCall reference spectra", "03",
                  ["feedstock", "true_burned", "p_burned", "temp_c", "correct"])
    burned = per_reference(p, "addis_burned", "char03/addis_burned_scores.csv", NB03, "each Addis filter's P(burned) from the final model", "03",
                           ["FilterId", "season", "p_burned"])
    if burned is None:
        p.shown("addis_burned_summary", CHAR, NB03, "addis_scores.to_csv", index="season")
    s15, src15 = printed(p, NB03, "final = make_model().fit(X_mc_addis, y)")
    sanity = [{"collection": m.group(1).strip(), "median P(burned)": float(m.group(2)), "% classed burned": float(m.group(3))}
              for m in re.finditer(r"^\s+(.+?)\s+median P\(burned\) = ([\d.]+)\s+→\s+(\d+)% burned", s15, re.M)]
    addis_line = re.search(r"Addis: P\(burned\) median ([\d.]+)\s+\[([\d.]+)–([\d.]+)\]\s+classified 'burned': (\d+)%", s15.replace("\n", " "))
    if addis_line:
        sanity.insert(0, {"collection": "Addis filters (unknown)", "median P(burned)": float(addis_line.group(1)), "% classed burned": float(addis_line.group(4))})
    p.add("archive_scores", pd.DataFrame(sanity), **src15)

    # ------------------------------------------------------------ C. char_06: one population or two?
    s10, src10 = printed(p, NB06, "mixture_vs_season.csv")
    p.add("mixture_fit", pd.DataFrame([
        {"model": "1 component", "BIC": number(r"BIC — 1 component: (-?[\d.]+)", s10), "means": None},
        {"model": "2 components", "BIC": number(r"2 components: (-?[\d.]+)", s10), "means": re.search(r"component means: \[([^\]]+)\]", s10).group(1).strip()},
    ]), **src10)
    p.shown("mixture_vs_season", CHAR, NB06, "mixture_vs_season.csv", csv="output/tables/char06/mixture_vs_season.csv", index="season")
    per_reference(p, "mixture", "char06/per_filter_classification.csv", NB06, "each Addis filter's r_char, season and mixture component", "06",
                  ["season", "r_char", "delta", "component"])

    # ------------------------------------------------------------ what no script saves
    for table, what, nb in [
        ("pca_scores", "PC1–PC2 scores of every archive spectrum and Addis filter, with collection, temperature and the pass A/B type "
         "(char_02 [18], char_04 [7][9][11][13] score panels)", "02 04"),
        ("mahalanobis_per_filter", "each Addis filter's Mahalanobis distance from the charcoal cloud (char_02 [20] histogram)", "02"),
        ("ward_linkage", "the Ward linkage matrix of the group means (char_04 [15] dendrogram; only the group labels are saved)", "04"),
        ("airspec_fit", "the AIRSpec baseline fitted to one charcoal and one Addis spectrum (char_11 [17])", "11"),
        ("recovery_per_spectrum", "the held-out charcoal spectra's true and recovered temperatures (char_02 [15] middle panel)", "02"),
    ]:
        p.missing.append({"table": table, "what": what + "; no script saves it yet: add a to_csv", "needs": rel(TABLES),
                          "run": run(nb), "after": INPUTS})

    p.text["finding"] = (
        "The archive's structure is charring stage, not provenance: KMeans types follow temperature (Cramér's V "
        f"{assoc[0]['Cramér' + chr(39) + 's V']:.2f}) more than the source collection ({assoc[1]['Cramér' + chr(39) + 's V']:.2f}). "
        "Add Addis and it forms a type of its own. Addis sits on the far edge of the charcoal cloud, matches it at r ≈ 0.78 "
        "where charcoal matches charcoal at 0.99, and the burned/unburned classifier's confident 'unburned' is extrapolation: "
        "most filters lie beyond its training cloud. Truncated AIRSpec on both sides gives the highest Addis–charcoal r of any "
        "transform while scoring worst on every objective check, so the asymmetric chain stays.")
    return p
