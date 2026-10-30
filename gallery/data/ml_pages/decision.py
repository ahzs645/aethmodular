"""The decision workbook: where the reference model fails inside IMPROVE, specialists, the flag on SPARTAN.

research/ftir_hips_chem/decision_workbook_20261022 §2 (out-of-site RMSE over k), §3 (failure groups [13]),
§4 (option B, specialist calibrations [17]), §6 (the flag applied to every SPARTAN filter [23]), §7 (option A,
one model over k [31]) and §8 (the decision board). §5/§6b (what the flag sees) and §7b (the local calibration)
are on the Experimental tab (export_experimental.py, export_experiments.py) and are not repeated.

The workbook writes every table to <Drive>/Aethmodular VIBES/decision_workbook_20261022/results/<run>/ (cell 39);
the newest run holding all of them is read through pls_transfer.drive_root() (AETHMODULAR_DRIVE_ROOT). Without
the Drive the page falls back to the tables the executed notebook displayed (§3 for the spline model only, §4
with its 10th–90th percentiles as printed, §6's site tables, the board) and lists the rest under `missing`.

Everything against TOR is the notebook's and does not involve the MAC. HIPS enters through Fabs/10: the
per-filter §6 table is exported as Fabs, so the page refits FTIR EC against HIPS BC at the reader's MAC
(Deming with the workbook's Addis error ratio, λ ∝ MAC²); the §3 slopes against IMPROVE HIPS and the §7 Addis
slopes are re-expressed exactly (× MAC/10, intercepts unchanged).
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from ._common import HIPS_CHEM, ROOT, page, rel

sys.path.insert(0, str(ROOT / "research/ftir_hips_chem/scripts"))
from pls_transfer import drive_root  # noqa: E402

KEY = "decision"
NB = "decision_workbook_20261022"
FOLDER = "Aethmodular VIBES/decision_workbook_20261022"
# the tables this page reads from a results run (cell 39's names)
FILES = ["section3_groups", "section4_splits", "section4_summary", "section6_sites", "section6_site_seasons",
         "section6_addis_seasons", "section6_network_filters", "section7_tradeoff", "section7_addis_by_k",
         "section7b_addis_filters", "section8_board"]
RUN_COMMAND = (f"execute {rel(HIPS_CHEM / f'{NB}.ipynb')} against the Drive bundle (cell 39 saves every table), "
               "then point AETHMODULAR_DRIVE_ROOT at the Drive")
RULES = ("same kind", "target analogs", "local")


def latest_run():
    """The newest results/<run> holding every table above, or None (no Drive, or no complete run)."""
    home = drive_root() / FOLDER / "results"
    if not home.is_dir():
        return None
    runs = sorted(p for p in home.iterdir() if p.is_dir() and all((p / f"{n}.csv").exists() for n in FILES))
    return runs[-1] if runs else None


def spec_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """§4's summary (the notebook's spec_summary) with its 10th and 90th percentiles as numbers, not a string."""
    rows = []
    for (prep, group), d in splits.groupby(["preprocessing", "group"], sort=False):
        for rule in RULES:
            col = f"{rule} rmse"
            if col not in d or d[col].isna().all():
                continue
            base = d["full rmse (local sample)"] if rule == "local" else d["full rmse"]
            ok = d[col].notna()
            change = 100 * (d.loc[ok, col] / base[ok] - 1)
            row = {"preprocessing": prep, "group": group, "rule": rule, "splits": int(ok.sum()),
                   "median n target": d.n_target.median(), "change": change.median(),
                   "change lo": change.quantile(.1), "change hi": change.quantile(.9), "wins": (change < 0).mean(),
                   "bias full": 100 * (d["full bias"] / d["mean TOR"]).median(),
                   "bias rule": 100 * (d[f"{rule} bias"] / d["mean TOR"]).median()}
            if f"{rule} rmse outside" in d:
                row["outside"] = 100 * (d[f"{rule} rmse outside"] / d["full rmse outside"] - 1).median()
            rows.append(row)
    return pd.DataFrame(rows)


def shown_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """The displayed §4 summary in the same columns (its percentiles parsed from '-3 to +21'; no rule bias shown)."""
    lo_hi = frame["RMSE change, % (10th-90th pct)"].str.extract(r"([+-]?\d+) to ([+-]?\d+)").astype(float)
    return pd.DataFrame({"preprocessing": frame.preprocessing, "group": frame.group, "rule": frame.rule,
                         "splits": frame.splits, "median n target": frame["median n target"],
                         "change": frame["RMSE change vs full, % (median)"], "change lo": lo_hi[0], "change hi": lo_hi[1],
                         "wins": frame["wins (share of splits)"], "bias full": frame["bias full, % of TOR"],
                         "outside": frame["RMSE outside its target, % change"]})


def missing(p, table: str, what: str) -> None:
    p.missing.append({"table": table, "what": what, "needs": f"<Drive>/{FOLDER}/results/<run>/{table}.csv",
                      "run": RUN_COMMAND, "after": None})


def build():
    p = page(KEY, "Decision workbook: where the reference model fails, specialists and the flag")
    nb = p.notebook(HIPS_CHEM, NB)
    nb_path = rel(HIPS_CHEM / f"{NB}.ipynb")
    run = latest_run()

    def csv(name: str) -> tuple[pd.DataFrame, dict]:
        return pd.read_csv(run / f"{name}.csv"), {"notebook": NB, "read": "csv", "path": f"Drive: {run.relative_to(drive_root()).as_posix()}/{name}.csv"}

    def shown(marker: str, nth: int = 0) -> tuple[pd.DataFrame, dict]:
        frame = nb.table(marker, nth)
        return frame.drop(columns=[c for c in ("index",) if c in frame]), {"notebook": NB, "read": "executed notebook output", "path": nb_path, "cell": marker}

    # §2: the CV choice of k per preprocessing, as printed (always in the notebook)
    cv = re.findall(r"(\w+)\s+k =\s*(\d+)\s+RMSE ([\d.]+) µg/filter\s+bias ([+-][\d.]+)\s+R² ([\d.]+)", nb.stream("OOF, KSEL, FOLD_MODELS, CURVE = {}"))
    p.add("cv_choice", pd.DataFrame([{"preprocessing": a, "k": int(k), "rmse": float(r), "bias": float(b), "r2": float(r2)} for a, k, r, b, r2 in cv]),
          notebook=NB, read="executed notebook output (printed)", path=nb_path, cell="OOF, KSEL, FOLD_MODELS, CURVE = {}")

    if run is not None:
        settings = json.loads((run / "settings.json").read_text())
        lam = float(settings["LAMBDA_ADDIS"])
        groups, src = csv("section3_groups")
        p.add("groups", groups, **src)
        splits, src = csv("section4_splits")
        p.add("specialists", spec_summary(splits), **src)
        for name, table in (("section6_sites", "sites"), ("section6_site_seasons", "site_seasons"), ("section6_addis_seasons", "addis_seasons")):
            frame, src = csv(name)
            p.add(table, frame, **src)
        net, src = csv("section6_network_filters")
        # per filter, compact: HIPS as Fabs (Mm⁻¹) so the page divides by the reader's MAC; seasons as an index into
        # `seasons` and the outside-range mark as 0/1
        names = pd.Series(net.season.fillna("").unique())
        p.add("seasons", pd.DataFrame({"season": names}), **src)
        p.add("network_filters", pd.DataFrame({
            "filter_id": net.filter_id, "code": net.code, "season": net.season.fillna("").map({v: i for i, v in names.items()}),
            "flag_spectrum": net.flag_spectrum, "flag_hips": net.flag_hips, "outside": net.outside_range.astype(int),
            "ec_model": net.ec_this_model, "ec_published": net.ec_ftir_ugm3, "fabs": net.hips_bc * 10}), **src)
        add, src = csv("section7b_addis_filters")
        p.add("addis_filters", pd.DataFrame({"filter_id": add.filter_id, "season": add.season, "ec_published": add.ec_ftir_ugm3,
                                             "fabs": add.hips_bc * 10}), **src)
        trade, src = csv("section7_tradeoff")
        p.add("tradeoff", trade, **src)
        by_k, src = csv("section7_addis_by_k")
        p.add("addis_by_k", by_k, **src)
        board, src = csv("section8_board")
        p.add("board", board, **src)
    else:
        lam = (0.531 / (2.9075 / 10.0)) ** 2   # the settings cell's LAMBDA_ADDIS at MAC 10
        groups, src = shown("FAIL = pd.DataFrame(rows)")
        p.add("groups", groups.assign(preprocessing="spline"), **src)
        summary, src = shown("SPEC = spec_summary(RUNS)")
        p.add("specialists", shown_summary(summary), **src)
        for nth, table in ((0, "sites"), (1, "addis_seasons"), (2, "site_seasons")):
            frame, src = shown("SITES = site_season_table", nth)
            p.add(table, frame, **src)
        board, src = shown("BOARD = pd.DataFrame(rows)")
        p.add("board", board, **src)
        missing(p, "section3_groups", "§3 for the raw and VIBES models (the notebook displayed spline only)")
        missing(p, "section4_splits", "§4's ten site splits (the displayed summary rounds the 10th–90th percentiles to whole per cent)")
        missing(p, "section6_network_filters", "the flag, FTIR EC and HIPS per SPARTAN filter, which the page refits at the reader's MAC "
                "(without it the site tables are the notebook's, at MAC 10, and only the top 20 site-seasons)")
        missing(p, "section7b_addis_filters", "the Addis filters' published EC and HIPS (the published line on the §7 panels)")
        missing(p, "section7_tradeoff", "§7's out-of-site RMSE on the hard groups and the rest over every k (and §2's curve)")
        missing(p, "section7_addis_by_k", "§7's Addis Deming slope and intercept against HIPS over every k")

    p.add("settings", pd.DataFrame([{"run": run.name if run is not None else None, "lambda_addis": lam,
                                     **{f"k_{r['preprocessing']}": r["k"] for r in p.tables["cv_choice"].to_dict("records")}}]),
          **({"notebook": NB, "read": "csv", "path": f"Drive: {run.relative_to(drive_root()).as_posix()}/settings.json"} if run is not None
             else {"notebook": NB, "read": "executed notebook output (printed)", "path": nb_path, "cell": "OOF, KSEL, FOLD_MODELS, CURVE = {}"}))

    g = p.tables["groups"].query("preprocessing == 'spline'").set_index("group")
    s = p.tables["specialists"]
    best = s[(s.preprocessing == "raw") & (s.rule == "local")].set_index("group")
    sites = p.tables["sites"].set_index("code")
    p.text["finding"] = (
        f"Inside IMPROVE, with TOR as the truth and whole sites held out, the reference model (spline, k = {int(p.tables['cv_choice'].set_index('preprocessing').k['spline'])}) "
        f"under-predicts the filters most like Addis: heavy loading by {-100 * g.loc['Heavy loading', 'rel_bias']:.0f} % and low OC/BC by "
        f"{-100 * g.loc['Low OC/BC', 'rel_bias']:.0f} % of mean TOR, against {100 * g.loc['None of these', 'rel_bias']:+.0f} % on the rest (§3). "
        f"A per-filter local calibration on raw spectra (each filter's 500 nearest IMPROVE spectra) cuts RMSE on every hard group "
        f"({best.change.drop('None of these').max():+.0f} to {best.change.drop('None of these').min():+.0f} %) and on ordinary filters too "
        f"({best.change['None of these']:+.0f} %), so it is a candidate for option A rather than a specialist rule (§4); its Addis readout is on "
        f"Experimental → Loading-matched analogs. Applied to every SPARTAN filter, the flag (+ HIPS) is highest at Addis "
        f"(mean {sites.loc['ETAD', 'mean flag (+ HIPS)']:.2f}) and Dhaka ({sites.loc['BDDU', 'mean flag (+ HIPS)']:.2f}); the spectrum-only flag, "
        "which does not read HIPS, points weakly the same way as the FTIR–HIPS intercept across site-seasons (§6).")
    return p
