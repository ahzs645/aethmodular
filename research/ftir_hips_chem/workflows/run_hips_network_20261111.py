"""Where HIPS under-reads across the SPARTAN network, what it depends on, and what can be done about it.

    python run_hips_network_20261111.py --summary docs/hips-network-2026-11-11.json

The 10 Nov work found that HIPS reads low as a filter darkens: on IMPROVE filters, against thermal (TOR)
EC, its optical depth follows  tau = L / (1 + L / ceiling)  with a ceiling of 2.26 [1.82, 2.70]. A filter
whose optical depth is tau therefore reports a share  tau / ceiling  less absorption than it carries. This
script applies that to every SPARTAN filter and asks:

  sites     how much each site's filters are expected to under-read
  volume    whether that comes from the air or from how much of it was drawn through the filter
  months    whether it is seasonal
  bands     which filters, wherever they are, form the affected group, and how sure the correction is
  evidence  whether each site's own HIPS and FTIR EC show the bend the IMPROVE curve predicts
  matched   Delhi filters of the same calendar months sampled at two volumes

Optical depth is  Fabs x sampled volume / (100 x 3.53 cm2), which equals the laboratory's own HIPS tau on
the 546 filters where both are on file. Inputs are committed files only: the gallery's SPARTAN table and
the 10 Nov summary (for the ceiling). No aethalometer data. FTIR EC is FTIR-derived, not thermal EC, and
HIPS is optical; no SPARTAN filter here has thermal EC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path.insert(0, str(AREA / "scripts"))
import reference_closure as rc  # noqa: E402

NETWORK = ROOT / "gallery/app/public/data/spartan_reference.json"
SITE_MAP = ROOT / "gallery/app/public/data/spartan_map.json"
CLOSURE = ROOT / "docs/addis-reference-closure-2026-11-10.json"
SUMMARY = ROOT / "docs/hips-network-2026-11-11.json"

DEPOSIT_AREA_CM2 = 3.53
MAC_VALUE = 10.0
MIN_FILTERS = 10            # a site needs this many filters with HIPS and a volume to be listed
MIN_EVIDENCE = 40           # and this many with FTIR EC and PM2.5 for its own exponent
MIN_MONTH = 4               # filters in a site's calendar month for it to be shown
VOLUMES = (7.2, 3.6, 1.8)   # the three sampled volumes SPARTAN has used (m3)
REGIMES = {"1.8 m³": (0.0, 2.6), "3.6 m³": (2.6, 5.0), "7.2 m³": (5.0, 9.0), "14 m³ or more": (9.0, 1e3)}
BANDS = (("under 5 %", 0.00, 0.05), ("5 to 10 %", 0.05, 0.10), ("10 to 20 %", 0.10, 0.20),
         ("20 to 30 %", 0.20, 0.30), ("over 30 %", 0.30, 9.99))
STAGES = ("sites", "volume", "months", "bands", "evidence", "matched")


def ceiling() -> dict[str, float]:
    """The IMPROVE ceiling and its interval, the two half-periods, and how far IMPROVE's own filters reach."""
    rows = pd.DataFrame(json.loads(CLOSURE.read_text())["improve_ceiling"]).set_index("population")
    main = rows.loc["all sites"]
    return {"central": float(main["tau_inf"]), "lo": float(main["tau_inf_lo"]), "hi": float(main["tau_inf_hi"]),
            "early": float(rows.loc["2005–2014", "tau_inf"]), "late": float(rows.loc["2015–2025", "tau_inf"]),
            "improve_reach": float(rows.loc["six urban IMPROVE sites", "max_stratum_tau"])}


def under_read(tau, top):
    """Share of the absorption on the filter that HIPS does not report: 1 - tau / L = tau / ceiling."""
    return np.clip(np.asarray(tau, dtype=float) / top, 0.0, 1.0)


def at_volume(tau, volume, new_volume, top):
    """The optical depth the same air would give at another sampled volume, through the same curve."""
    tau = np.asarray(tau, dtype=float)
    true = rc.linearise(tau, top) * new_volume / np.asarray(volume, dtype=float)
    # a filter at or past the ceiling has no finite loading under the curve; it stays at the ceiling
    return np.where(np.isfinite(true), true / (1.0 + np.where(np.isfinite(true), true, 0.0) / top), top)


def band_of(share):
    return pd.cut(share, [b[1] for b in BANDS] + [BANDS[-1][2]], labels=[b[0] for b in BANDS], right=False).astype(str)


def load(top: dict[str, float]) -> pd.DataFrame:
    """One row per counted SPARTAN filter with HIPS and a sampled volume."""
    names = {s["code"]: s.get("name") or s["code"] for s in json.loads(SITE_MAP.read_text()).get("sites", [])}
    rows = pd.DataFrame(json.loads(NETWORK.read_text())["rows"])
    rows = rows[~rows["excluded"].astype(bool)].copy()
    rows["fabs"] = rows["HIPS Fabs"].fillna(rows["BC (public HIPS)"] * MAC_VALUE)
    rows["volume"] = rows["Sampled volume"]
    rows = rows[(rows["fabs"] > 0) & (rows["volume"] > 0)].copy()
    rows["tau"] = rows["fabs"] * rows["volume"] / (100.0 * DEPOSIT_AREA_CM2)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["month"] = rows["date"].dt.month
    rows["site"] = rows["code"].map(names).fillna(rows["code"])
    rows["under_read"] = under_read(rows["tau"], top["central"])
    rows["band"] = band_of(rows["under_read"])
    rows["regime"] = pd.cut(rows["volume"], [0.0] + [hi for _, hi in REGIMES.values()], labels=list(REGIMES)).astype(str)
    rows = rows.rename(columns={"EC (FTIR)": "ftir_ec", "PM2.5 mass": "pm25", "Potassium": "potassium"})
    keep = rows.groupby("code")["tau"].transform("size") >= MIN_FILTERS
    return rows.loc[keep, ["id", "code", "site", "date", "month", "fabs", "volume", "regime", "tau", "under_read", "band",
                           "ftir_ec", "pm25", "potassium"]].reset_index(drop=True)


def stage_sites(f, top):
    out = []
    for (code, site), part in f.groupby(["code", "site"]):
        tau = part["tau"]
        out.append({
            "code": code, "site": site, "n": len(part), "first": str(part["date"].min().date()), "last": str(part["date"].max().date()),
            "median_fabs": float(part["fabs"].median()), "median_volume": float(part["volume"].median()),
            "median_tau": float(tau.median()), "p90_tau": float(tau.quantile(0.9)),
            # what the median filter's absorption would be without the bend: the air, not the filter
            "linearised_fabs": float(part["fabs"].median() / (1.0 - under_read(tau.median(), top["central"]))),
            "under_read": float(under_read(tau.median(), top["central"])),
            # the ceiling's interval, turned round: a low ceiling means a large under-read
            "under_read_lo": float(under_read(tau.median(), top["hi"])), "under_read_hi": float(under_read(tau.median(), top["lo"])),
            "share_over_10": float((part["under_read"] > 0.10).mean()), "share_over_20": float((part["under_read"] > 0.20).mean()),
            "share_over_30": float((part["under_read"] > 0.30).mean()),
            "share_beyond_improve": float((tau > top["improve_reach"]).mean()),
            "n_with_ftir_ec": int(part["ftir_ec"].notna().sum()),
        })
    sites = pd.DataFrame(out).sort_values("median_tau", ascending=False)
    sites["band"] = band_of(sites["under_read"])
    return {"sites": sites}


def stage_volume(f, top):
    """Split each site's optical depth into the air and the volume, and replay it at the other volumes."""
    t = top["central"]
    rows = []
    for (code, site), part in f.groupby(["code", "site"]):
        row = {"code": code, "site": site, "median_fabs": float(part["fabs"].median()), "median_volume": float(part["volume"].median()),
               "as_sampled": float(under_read(part["tau"].median(), t))}
        for v in VOLUMES:
            row[f"at_{v}"] = float(under_read(np.median(at_volume(part["tau"], part["volume"], v, t)), t))
        rows.append(row)
    replay = pd.DataFrame(rows).sort_values("as_sampled", ascending=False)

    # how much of the spread between sites is the air, and how much the volume
    log = np.log(replay[["median_fabs", "median_volume"]])
    total = np.var(log["median_fabs"] + log["median_volume"])
    split = pd.DataFrame([
        {"part": "absorption in the air (Fabs)", "share_of_spread": float(np.cov(log["median_fabs"], log.sum(axis=1), bias=True)[0, 1] / total)},
        {"part": "sampled volume", "share_of_spread": float(np.cov(log["median_volume"], log.sum(axis=1), bias=True)[0, 1] / total)},
    ])

    quarter = f.assign(quarter=f["date"].dt.to_period("Q").astype(str)).groupby(["code", "site", "quarter"]).agg(
        n=("tau", "size"), volume=("volume", "median"), fabs=("fabs", "median"), tau=("tau", "median")).reset_index()
    quarter = quarter[quarter["n"] >= MIN_MONTH]
    spread = quarter.groupby("code")["volume"].agg(lambda v: v.max() / v.min())
    history = quarter[quarter["code"].isin(spread[spread >= 1.5].index)].copy()
    history["under_read"] = under_read(history["tau"], t)

    regimes = f.groupby(["code", "site", "regime"]).agg(n=("tau", "size"), volume=("volume", "median"), fabs=("fabs", "median"),
                                                        tau=("tau", "median"), first=("date", "min"), last=("date", "max")).reset_index()
    regimes = regimes[regimes["n"] >= MIN_MONTH]
    regimes = regimes[regimes.groupby("code")["regime"].transform("size") > 1]
    regimes["under_read"] = under_read(regimes["tau"], t)
    regimes["first"], regimes["last"] = regimes["first"].dt.date.astype(str), regimes["last"].dt.date.astype(str)
    return {"volume_replay": replay, "volume_split": split, "volume_history": history, "volume_regimes": regimes}


def stage_months(f, top):
    g = f.groupby(["code", "site", "month"]).agg(n=("tau", "size"), median_tau=("tau", "median"),
                                                 share_over_20=("under_read", lambda s: float((s > 0.20).mean()))).reset_index()
    g = g[g["n"] >= MIN_MONTH]
    g["under_read"] = under_read(g["median_tau"], top["central"])
    g["month_name"] = pd.to_datetime(g["month"], format="%m").dt.strftime("%b")
    swing = g.groupby(["code", "site"]).agg(months=("month", "size"), lowest=("under_read", "min"), highest=("under_read", "max")).reset_index()
    swing = swing[swing["months"] >= 6]
    swing["swing"] = swing["highest"] - swing["lowest"]
    peak = g.loc[g.groupby("code")["under_read"].idxmax(), ["code", "month_name"]].rename(columns={"month_name": "worst_month"})
    best = g.loc[g.groupby("code")["under_read"].idxmin(), ["code", "month_name"]].rename(columns={"month_name": "best_month"})
    return {"months": g, "season_swing": swing.merge(peak, on="code").merge(best, on="code").sort_values("swing", ascending=False)}


def stage_bands(f, top):
    """The affected filters as one group, wherever they are, and how well the correction is known for each band."""
    rows = []
    for name, lo, hi in BANDS:
        part = f[f["band"] == name]
        tau = float(part["tau"].median()) if len(part) else float("nan")
        factor = {k: float(1.0 / (1.0 - tau / top[k])) if tau < top[k] else float("inf") for k in ("central", "lo", "hi", "early", "late")}
        counts = part["site"].value_counts()
        rows.append({
            "band": name, "tau_from": lo * top["central"], "tau_to": min(hi, 1.0) * top["central"], "filters": len(part),
            "share_of_network": len(part) / len(f), "sites": int(part["code"].nunique()),
            "mostly": ", ".join(f"{s} ({n})" for s, n in counts.head(4).items()),
            "median_tau": tau, "factor": factor["central"], "factor_lo": factor["hi"], "factor_hi": factor["lo"],
            # how far the corrected value moves across the ceiling's interval, and across IMPROVE's two half-periods
            "plus_minus": (factor["lo"] - factor["hi"]) / (2.0 * factor["central"]),
            "by_period": abs(factor["late"] - factor["early"]) / factor["central"],
            "beyond_improve": float((part["tau"] > top["improve_reach"]).mean()) if len(part) else 0.0,
        })
    by_site = f.groupby(["code", "site", "band"]).size().rename("filters").reset_index()
    by_site["share"] = by_site["filters"] / by_site.groupby("code")["filters"].transform("sum")
    return {"bands": pd.DataFrame(rows), "bands_by_site": by_site}


def stage_evidence(f, top, n_boot=500):
    """Each site's own exponent of HIPS against FTIR EC beside what the IMPROVE curve predicts for its filters."""
    rows = []
    for (code, site), part in f.groupby(["code", "site"]):
        part = part[(part[["ftir_ec", "pm25"]] > 0).all(axis=1)]
        if len(part) < MIN_EVIDENCE:
            continue
        r = rc.response_exponent(part["fabs"], part["ftir_ec"], part["pm25"], n_boot=n_boot)
        predicted = float(np.mean(1.0 - under_read(part["tau"], top["central"])))
        row = {"code": code, "site": site, "n": r["n"], "median_tau": float(part["tau"].median()), "predicted": predicted,
               "predicted_lo": float(np.mean(1.0 - under_read(part["tau"], top["lo"]))),
               "predicted_hi": float(np.mean(1.0 - under_read(part["tau"], top["hi"]))),
               "observed": r["exponent"], "lo": r["lo"], "hi": r["hi"]}
        k = part[part["potassium"] > 0]
        if len(k) >= MIN_EVIDENCE:
            p = rc.response_exponent(k["fabs"], k["ftir_ec"], k["potassium"], n_boot=n_boot)
            row |= {"observed_potassium": p["exponent"], "potassium_lo": p["lo"], "potassium_hi": p["hi"]}
        width = row["hi"] - row["lo"]
        row["reading"] = ("cannot tell" if width > 0.6 else "as predicted" if row["lo"] <= predicted <= row["hi"]
                          else "more compressed than predicted" if row["hi"] < predicted else "less compressed than predicted")
        rows.append(row)
    evidence = pd.DataFrame(rows).sort_values("median_tau", ascending=False)
    told = evidence[evidence["reading"] != "cannot tell"]
    tally = evidence["reading"].value_counts().rename_axis("reading").reset_index(name="sites")
    tally["which"] = tally["reading"].map(lambda name: ", ".join(evidence.loc[evidence["reading"] == name, "site"].str.split(",").str[0]))
    line = np.polyfit(told["predicted"], told["observed"], 1, w=1.0 / (told["hi"] - told["lo"])) if len(told) > 2 else (np.nan, np.nan)
    agreement = pd.DataFrame([{"sites_that_tell": len(told), "slope_observed_on_predicted": float(line[0]), "intercept": float(line[1]),
                               "correlation": float(np.corrcoef(told["predicted"], told["observed"])[0, 1]) if len(told) > 2 else np.nan}])
    return {"evidence": evidence, "evidence_tally": tally, "evidence_agreement": agreement}


def stage_matched(f, top, code="INDH", low="1.8 m³", high="3.6 m³", n_boot=2000, seed=20261111):
    """Same site, same calendar months, two sampled volumes: absorption per unit PM2.5 on each, beside the prediction.

    Less air through the filter means a smaller optical depth, so the curve predicts more absorption reported per
    unit of PM2.5 at the lower volume. PM2.5 is weighed, so this check uses no FTIR. The two volumes fall in
    different years, which this cannot separate.
    """
    part = f[(f["code"] == code) & f["regime"].isin([low, high]) & (f["pm25"] > 1)].copy()
    part["per_pm"] = part["fabs"] / part["pm25"]
    counts = part.groupby(["month", "regime"]).size().unstack(fill_value=0)
    months = counts[(counts.get(low, 0) >= MIN_MONTH) & (counts.get(high, 0) >= MIN_MONTH)].index
    rng = np.random.default_rng(seed)
    rows = []

    def compare(a, b, label):
        observed = a["per_pm"].median() / b["per_pm"].median()
        boot = [np.median(rng.choice(a["per_pm"].to_numpy(), len(a))) / np.median(rng.choice(b["per_pm"].to_numpy(), len(b))) for _ in range(n_boot)]
        predicted = (1.0 - under_read(a["tau"].median(), top["central"])) / (1.0 - under_read(b["tau"].median(), top["central"]))
        return {"months": label, "n_low_volume": len(a), "n_high_volume": len(b), "tau_low_volume": float(a["tau"].median()),
                "tau_high_volume": float(b["tau"].median()), "observed_ratio": float(observed), "lo": float(np.percentile(boot, 5)),
                "hi": float(np.percentile(boot, 95)), "predicted_ratio": float(predicted)}

    for month in months:
        m = part[part["month"] == month]
        rows.append(compare(m[m["regime"] == low], m[m["regime"] == high], pd.Timestamp(2000, month, 1).strftime("%B")))
    if len(months) > 1:
        m = part[part["month"].isin(months)]
        rows.append(compare(m[m["regime"] == low], m[m["regime"] == high], "these months together"))
    table = pd.DataFrame(rows)
    table.insert(0, "site", f.loc[f["code"] == code, "site"].iloc[0] if (f["code"] == code).any() else code)
    return {"volume_matched": table}


RUN = {"sites": stage_sites, "volume": stage_volume, "months": stage_months, "bands": stage_bands,
       "evidence": stage_evidence, "matched": stage_matched}


def run(stages=STAGES):
    top = ceiling()
    f = load(top)
    tables: dict[str, pd.DataFrame] = {"filters": f}
    for name in stages:
        tables |= RUN[name](f, top)
    return tables, top


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--summary", nargs="?", const=str(SUMMARY), help="write the tables (without the per-filter rows) to this JSON file")
    a = ap.parse_args(argv)
    tables, top = run(a.stages)
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.float_format", lambda v: f"{v:.3g}")
    for name, table in tables.items():
        if name != "filters":
            print(f"\n== {name}\n{table.to_string(index=False)}")
    if a.summary:
        f = tables["filters"]
        payload = {"generated_from": "research/ftir_hips_chem/workflows/run_hips_network_20261111.py", "aethalometer_used": False,
                   "inputs": {"network": str(NETWORK.relative_to(ROOT)), "ceiling": str(CLOSURE.relative_to(ROOT))},
                   "ceiling": top, "deposit_area_cm2": DEPOSIT_AREA_CM2,
                   "counts": {"filters": int(len(f)), "sites": int(f["code"].nunique()),
                              "over_10": float((f["under_read"] > 0.10).mean()), "over_20": float((f["under_read"] > 0.20).mean()),
                              "over_30": float((f["under_read"] > 0.30).mean()), "at_or_above_ceiling": int((f["tau"] >= top["central"]).sum())}}
        for name, table in tables.items():
            if name != "filters":
                payload[name] = json.loads(table.to_json(orient="records", double_precision=6))
        Path(a.summary).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
