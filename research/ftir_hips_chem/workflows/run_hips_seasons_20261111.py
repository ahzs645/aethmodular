"""Seasons and the HIPS bend: correct by site, by season or by filter, and do the season calendars hold up?

    python run_hips_seasons_20261111.py --summary docs/hips-seasons-2026-11-11.json

Two questions on the same filters as run_hips_network_20261111.py (committed inputs only, no aethalometer).

  levels     how wrong a correction is when one factor is used for the whole network, for each site, for each
             site and season, or for each site and month, against the factor of each filter; and how the spread
             of the correction divides between sites, seasons and filters
  seasons    each site and season: how dark its filters are, and whether linearising leaves HIPS proportional
             to FTIR EC within that season
  within     each site: the same check with every season's own level taken out, beside the pooled check, and
             how far Fabs per unit FTIR EC still differs between seasons after linearising
  calendars  each site's season calendar against the 1991-2020 ERA5 climate (fetch_climate_normals_era5.py):
             how much of the year's cycle in rain and temperature falls between its seasons, whether its wet,
             dry, cold and warm labels hold, the best any calendar with as many seasons could do, the scores
             of the alternative calendars on file, and how much of the monthly HIPS cycle the seasons carry

Calendars are read from gallery/app/src/siteSeasons.network.json (docs/site-seasonality-network.md); the
Addis and Delhi alternatives from research/ftir_hips_chem/scripts/config.py and src/config/multi_site_seasons.py.
FTIR EC is FTIR-derived, not thermal EC; HIPS is optical.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA = Path(__file__).resolve().parents[1]
ROOT = AREA.parents[1]
sys.path[:0] = [str(AREA / "scripts"), str(Path(__file__).resolve().parent)]
import config  # noqa: E402
import hips_loading as hl  # noqa: E402
import run_hips_network_20261111 as network  # noqa: E402

CALENDARS = ROOT / "gallery/app/src/siteSeasons.network.json"
NORMALS = ROOT / "gallery/app/public/data/weather/normals_1991_2020.json"
DAILY = ROOT / "gallery/app/public/data/weather/daily"
MULTI_SITE = ROOT / "src/config/multi_site_seasons.py"
SUMMARY = ROOT / "docs/hips-seasons-2026-11-11.json"

STAGES = ("levels", "seasons", "within", "calendars")
HEAVY = 0.10                # a site whose median filter is expected to lose more than this
MIN_SEASON = 30             # tested filters a site-season needs for its own check
MIN_SEASON_LEVEL = 15       # and for its level of Fabs per unit FTIR EC
HEADROOM = 0.05             # a calendar this far below the best possible is worth a look
WET = ("rain", "wet", "monsoon", "kiremt", "belg")
DRY = ("dry", "bega", "harmattan")
COLD = ("winter", "cold", "cool")
WARM = ("summer", "warm", "hot")
NEUTRAL = ("transition", "pre-monsoon", "post-monsoon", "inter-monsoon", "semidry", "drier", "wetter")


# --------------------------------------------------------------------------- calendars
def _months(seasons) -> dict[int, str]:
    return {month: season["name"] for season in seasons for month in season["months"]}


def load_calendars() -> dict[str, dict]:
    """Every site's calendars by key; the first key is the one the gallery uses."""
    out = {}
    for code, entry in json.loads(CALENDARS.read_text()).items():
        if not isinstance(entry, dict) or "seasons" not in entry:
            continue
        variants = entry.get("variants") or []
        calendars = {v["key"]: _months(v["seasons"]) for v in variants} if variants else {"default": _months(entry["seasons"])}
        if variants and _months(variants[0]["seasons"]) != _months(entry["seasons"]):
            raise ValueError(f"{code}: the first variant is not the calendar in use")
        out[code] = {"site": entry["site"], "confidence": entry.get("confidence", ""), "calendars": calendars}
    # the two older calendars the research code still names
    out["ETAD"]["calendars"] = {"belg_feb": out["ETAD"]["calendars"]["default"],
                                "dry_feb": {m: name for name, spec in config.resolve_seasons("dry_feb").items() for m in spec["months"]}}
    spec = importlib.util.spec_from_file_location("multi_site_seasons", MULTI_SITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out["INDH"]["calendars"] = {"imd_national": out["INDH"]["calendars"]["default"],
                                "research_2026_04": {m: name for name, s in module.CALENDARS["research_2026_04"]["Delhi"].items() for m in s["months"]}}
    out["_python"] = {"local_2026_10": {site: {m: name for name, s in seasons.items() for m in s["months"]}
                                        for site, seasons in module.CALENDARS["local_2026_10"].items()},
                      "belg_feb": {m: name for name, s in config.resolve_seasons("belg_feb").items() for m in s["months"]}}
    return out


def season_of(frame: pd.DataFrame, calendars: dict) -> pd.Series:
    """Each filter's season under its site's calendar in use."""
    first = {code: next(iter(entry["calendars"].values())) for code, entry in calendars.items() if not code.startswith("_")}
    return pd.Series([first[c][m] for c, m in zip(frame["code"], frame["month"])], index=frame.index)


def load(top):
    calendars = load_calendars()
    f = network.load(top)
    f["season"] = season_of(f, calendars)
    screened = hl.screen(f, ceiling=hl.Ceiling(top["central"], top["lo"], top["hi"], top["improve_reach"]))
    return screened, calendars


# --------------------------------------------------------------------------- levels
def stage_levels(s, top, calendars):
    """One factor for a group of filters against each filter's own: the size of the error, by how fine the group is."""
    t = top["central"]
    s = s[np.isfinite(s["factor"])].copy()
    groups = {"One factor for the network": None, "One per site": ["code"], "One per site and season": ["code", "season"],
              "One per site and month": ["code", "month"]}
    for name, keys in groups.items():
        tau = s["tau"].median() if keys is None else s.groupby(keys)["tau"].transform("median")
        s[name] = np.abs(hl.factor(tau, t) / s["factor"] - 1.0)
    s["The ceiling's own uncertainty, per filter"] = (s["factor_hi"].where(np.isfinite(s["factor_hi"])) - s["factor_lo"]) / (2.0 * s["factor"])
    names = [*groups, "The ceiling's own uncertainty, per filter"]
    heavy_codes = s.groupby("code")["under_read"].median().loc[lambda v: v > HEAVY].index
    rows = []
    for scope, part in (("Every site", s), (f"Sites that lose more than {HEAVY:.0%} on the median filter", s[s["code"].isin(heavy_codes)])):
        for name in names:
            rows.append({"scope": scope, "level": name, "filters": len(part), "median": float(part[name].median()),
                         "p90": float(part[name].quantile(0.9)), "share_over_10": float((part[name] > 0.10).mean())})
    by_site = s.groupby(["code", "site"])[names].median().reset_index()
    by_site["median_tau"] = by_site["code"].map(s.groupby("code")["tau"].median())
    by_site = by_site.sort_values("median_tau", ascending=False)

    def split(part, label):
        log = np.log(part["factor"])
        site, season = log.groupby(part["code"]).transform("mean"), log.groupby([part["code"], part["season"]]).transform("mean")
        total = ((log - log.mean()) ** 2).sum()
        return [{"scope": label, "part": "Between sites", "share": float(((site - log.mean()) ** 2).sum() / total)},
                {"scope": label, "part": "Between seasons at a site", "share": float(((season - site) ** 2).sum() / total)},
                {"scope": label, "part": "Between filters in a season", "share": float(((log - season) ** 2).sum() / total)}]

    variance = pd.DataFrame(split(s, "Every site") + split(s[s["code"].isin(heavy_codes)], f"Sites that lose more than {HEAVY:.0%}"))
    return {"levels": pd.DataFrame(rows), "levels_by_site": by_site, "variance": variance}


# --------------------------------------------------------------------------- seasons
def stage_seasons(s, top, calendars, n_boot=400):
    rows = []
    for (code, site, season), part in s.groupby(["code", "site", "season"]):
        tested = part[(part[["ftir_ec", "pm25"]] > 0).all(axis=1) & np.isfinite(part["fabs_linearised"])]
        row = {"code": code, "site": site, "season": season, "filters": len(part), "median_tau": float(part["tau"].median()),
               "under_read": float(hl.under_read(part["tau"].median(), top["central"])),
               "share_use": float((part["status"] == hl.STATUSES[0]).mean()), "share_shape_only": float((part["status"] == hl.STATUSES[2]).mean()),
               "tested_filters": len(tested)}
        if len(tested) >= MIN_SEASON_LEVEL:
            row |= {"fabs_per_ec": float((tested["fabs"] / tested["ftir_ec"]).median()),
                    "fabs_per_ec_linearised": float((tested["fabs_linearised"] / tested["ftir_ec"]).median())}
        row |= {k: v for k, v in hl.check(part, n_boot=n_boot, min_filters=MIN_SEASON).items() if k != "tested_filters"}
        rows.append(row)
    table = pd.DataFrame(rows)
    order = s.groupby("code")["tau"].median().sort_values(ascending=False)
    table["rank"] = table["code"].map({c: i for i, c in enumerate(order.index)})
    return {"site_seasons": table.sort_values(["rank", "season"]).drop(columns="rank")}


# --------------------------------------------------------------------------- within
def _within_exponent(part, column, n_boot, rng):
    """Exponent of ``column`` against FTIR EC with PM2.5 as the instrument, each season's own level taken out."""
    logs = np.log(part[[column, "ftir_ec", "pm25"]].to_numpy(dtype=float))
    codes = pd.factorize(part["season"])[0]

    def estimate(index):
        x, g = logs[index], codes[index]
        centred = x - np.vstack([x[g == k].mean(axis=0) for k in range(codes.max() + 1)])[g] if len(index) else x
        c = np.cov(centred.T)
        return c[0, 2] / c[1, 2]

    index = np.arange(len(part))
    boot = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(part), len(part))
        if len(np.unique(codes[draw])) == codes.max() + 1 and np.bincount(codes[draw]).min() > 2:
            boot.append(estimate(draw))
    return float(estimate(index)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stage_within(s, top, calendars, n_boot=400, seed=20261111):
    """Is what is left after linearising a matter within seasons (the bend) or between them (something else)?"""
    rng = np.random.default_rng(seed)
    rows = []
    for (code, site), part in s.groupby(["code", "site"]):
        tested = part[(part[["ftir_ec", "pm25"]] > 0).all(axis=1) & np.isfinite(part["fabs_linearised"])]
        counts = tested["season"].value_counts()
        tested = tested[tested["season"].isin(counts[counts >= MIN_SEASON_LEVEL].index)]
        if len(tested) < hl.MIN_TEST or tested["season"].nunique() < 2:
            continue
        pooled = hl.check(tested, n_boot=n_boot)
        if pooled["verdict"] == "cannot tell":
            continue
        within, lo, hi = _within_exponent(tested, "fabs_linearised", n_boot, rng)
        before, b_lo, b_hi = _within_exponent(tested, "fabs", n_boot, rng)
        level = lambda column: (tested[column] / tested["ftir_ec"]).groupby(tested["season"]).median()  # noqa: E731
        raw, lin = level("fabs"), level("fabs_linearised")
        spread = lambda v: float(v.max() / v.min())  # noqa: E731
        narrow = (hi - lo) <= hl.WIDE
        # each season's own verdict, where it has enough filters: a site can pass on average and fail in one season
        each = hl.check_by(tested, n_boot=n_boot, min_filters=MIN_SEASON)
        failed = each[each["verdict"].isin(["under-corrected", "over-corrected"])]
        short = lambda frame: ", ".join(f"{r.season.split(' (')[0]} {r.verdict.split('-')[0]}" for r in frame.itertuples())  # noqa: E731
        reading = ("cannot tell" if not narrow
                   else "differs by season" if len(failed) and lo <= 1 <= hi
                   else "holds within seasons and overall" if lo <= 1 <= hi and pooled["verdict"] == "corrected"
                   else "holds within seasons; the seasons sit at different levels" if lo <= 1 <= hi
                   else "still compressed within seasons" if hi < 1 else "over-corrected within seasons")
        rows.append({"code": code, "site": site, "seasons": int(tested["season"].nunique()), "tested_filters": len(tested),
                     "median_tau": float(part["tau"].median()),
                     "pooled_after": pooled["exponent_after"], "pooled_lo": pooled["after_lo"], "pooled_hi": pooled["after_hi"], "pooled_verdict": pooled["verdict"],
                     "within_before": before, "within_before_lo": b_lo, "within_before_hi": b_hi,
                     "within_after": within, "within_lo": lo, "within_hi": hi,
                     "season_spread_reported": spread(raw), "season_spread_linearised": spread(lin),
                     "highest_season": lin.idxmax(), "lowest_season": lin.idxmin(),
                     "seasons_passing": int((each["verdict"] == "corrected").sum()), "seasons_failing": short(failed), "reading": reading})
    table = pd.DataFrame(rows).sort_values("median_tau", ascending=False)
    tally = table["reading"].value_counts().rename_axis("reading").reset_index(name="sites")
    tally["which"] = tally["reading"].map(lambda name: ", ".join(table.loc[table["reading"] == name, "site"].str.split(",").str[0]))
    return {"within": table, "within_tally": tally}


# --------------------------------------------------------------------------- calendars
def _separation(values: pd.DataFrame, groups) -> dict[str, float]:
    """Share of the month-to-month spread that falls between the groups, per column and for the columns together."""
    groups = pd.Series(list(groups), index=values.index)
    out = {}
    z = (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    for name in values:
        x = values[name]
        out[name] = float(((x.groupby(groups).transform("mean") - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
    z = z.dropna(axis=1)
    out["together"] = float(((z.groupby(groups).transform("mean")) ** 2).to_numpy().sum() / (z ** 2).to_numpy().sum())
    return out


def _best_partition(values: pd.DataFrame, k: int):
    """The split of the year into k runs of consecutive months that separates the columns together best."""
    best, best_groups = -1.0, None
    for cuts in itertools.combinations(range(12), k):          # a cut after month index c
        groups = np.zeros(12, dtype=int)
        for i in range(12):
            groups[i] = sum(c < i for c in cuts) % k               # months after the last cut join the first run
        score = _separation(values, groups)["together"]
        if score > best + 1e-12:
            best, best_groups = score, groups.copy()
    return best, best_groups


def _moved(calendar: dict[int, str], best_groups) -> list[str]:
    """Months the best split puts with other months than the calendar does (after matching runs to seasons by overlap)."""
    names = pd.Series(calendar).reindex(range(1, 13))
    moved = []
    for g in np.unique(best_groups):
        months = [m for m in range(1, 13) if best_groups[m - 1] == g]
        home = names.loc[months].mode().iloc[0]
        moved += [(m, home) for m in months if names.loc[m] != home]
    return [f"{pd.Timestamp(2000, m, 1):%b} to {home.split(' (')[0]}" for m, home in sorted(moved)]


def _kind(name: str, words) -> bool:
    label = name.split(" (")[0].lower()
    return not any(w in label for w in NEUTRAL) and any(w in label for w in words)


def _labels_hold(calendar: dict[int, str], values: pd.Series, low_words, high_words):
    """True if every season named with a high word is above every season named with a low word; None if not both are named."""
    by_season = values.groupby(pd.Series(calendar).reindex(values.index)).mean()
    low = [v for name, v in by_season.items() if _kind(name, low_words)]
    high = [v for name, v in by_season.items() if _kind(name, high_words)]
    return None if not low or not high else bool(min(high) > max(low))


def stage_calendars(s, top, calendars):
    normals = json.loads(NORMALS.read_text())
    monthly = s.groupby(["code", "month"])["under_read"].agg(["median", "size"])
    rows, variants = [], []
    for code, entry in calendars.items():
        if code.startswith("_"):
            continue
        row = {"code": code, "site": entry["site"], "confidence": entry["confidence"], "calendars_on_file": len(entry["calendars"]),
               "hips_filters": int((s["code"] == code).sum())}
        climate = normals["sites"].get(code)
        daily = DAILY / f"{code}.json"
        row["gallery_weather_years"] = None
        if daily.exists():
            dates = pd.to_datetime(json.loads(daily.read_text())["date"])
            row["gallery_weather_years"] = round((dates.max() - dates.min()).days / 365.25, 1)
        if climate:
            values = pd.DataFrame({"rain": climate["precipitation"], "temperature": climate["temperature"]}, index=range(1, 13))
            for i, (key, calendar) in enumerate(entry["calendars"].items()):
                score = _separation(values, [calendar[m] for m in range(1, 13)])
                best, groups = _best_partition(values, len(set(calendar.values())))
                item = {"code": code, "site": entry["site"], "calendar": key, "in_use": i == 0, "seasons": len(set(calendar.values())),
                        "bins": "; ".join(dict.fromkeys(calendar[m] for m in range(1, 13))),
                        "rain": score["rain"], "temperature": score["temperature"], "together": score["together"], "best_possible": best,
                        "headroom": best - score["together"],
                        "rain_labels_hold": _labels_hold(calendar, values["rain"], DRY, WET),
                        "temperature_labels_hold": _labels_hold(calendar, values["temperature"], COLD, WARM),
                        "would_move": ", ".join(_moved(calendar, groups)) if best - score["together"] >= HEADROOM else ""}
                variants.append(item)
                if i == 0:
                    row |= {k: item[k] for k in ("seasons", "bins", "rain", "temperature", "together", "best_possible", "headroom",
                                                 "rain_labels_hold", "temperature_labels_hold", "would_move")}
                    row["wettest_month_mm_day"], row["driest_month_mm_day"] = float(values["rain"].max()), float(values["rain"].min())
                    row["annual_temperature_range"] = float(values["temperature"].max() - values["temperature"].min())
        if code in monthly.index.get_level_values(0):
            cycle = monthly.loc[code]
            cycle = cycle.loc[cycle["size"] >= network.MIN_MONTH, "median"]
            if len(cycle) >= 8:
                calendar = next(iter(entry["calendars"].values()))
                row["hips_cycle_between_seasons"] = _separation(cycle.to_frame("u"), [calendar[m] for m in cycle.index])["u"]
                row["hips_cycle_range"] = float(cycle.max() - cycle.min())
                if climate:
                    for name, key in (("rain", "precipitation"), ("temperature", "temperature")):
                        row[f"hips_follows_{name}"] = float(pd.Series(climate[key], index=range(1, 13)).reindex(cycle.index).corr(cycle, method="spearman"))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("code")
    variants = pd.DataFrame(variants)
    alternatives = variants[variants["code"].isin(variants.loc[~variants["in_use"], "code"])].copy()

    python = calendars["_python"]
    in_use = lambda code: next(iter(calendars[code]["calendars"].values()))  # noqa: E731
    strip = lambda cal: {m: name.split(" (")[0] for m, name in cal.items()}  # noqa: E731
    same_bins = lambda a, b: all((a[m] == a[n]) == (b[m] == b[n]) for m in range(1, 13) for n in range(1, 13))  # noqa: E731
    consistency = pd.DataFrame([
        {"pair": f"src/config/multi_site_seasons.py local_2026_10, {site}", "against": f"gallery calendar {code}",
         "same_months": same_bins(python["local_2026_10"][site], in_use(code))}
        for site, code in (("Addis_Ababa", "ETAD"), ("Beijing", "CHTS"), ("Delhi", "INDH"), ("JPL", "USPA"))
    ] + [{"pair": "research/ftir_hips_chem/scripts/config.py belg_feb", "against": "gallery calendar ETAD",
          "same_months": same_bins(python["belg_feb"], in_use("ETAD"))}])
    del strip
    structure = pd.DataFrame([{
        "sites_with_a_calendar": int(len(table)), "every_month_once": bool(all(sorted(c) == list(range(1, 13)) for e in calendars.values() if "calendars" in e
                                                                                 for c in e["calendars"].values())),
        "sites_with_normals": int(table["rain"].notna().sum()), "hips_sites_without_a_calendar": int(len(set(s["code"]) - set(table["code"]))),
        "hips_sites_with_under_a_year_of_gallery_weather": ", ".join(sorted(
            table.loc[(table["hips_filters"] > 0) & ~(table["gallery_weather_years"].fillna(0) >= 1), "code"])),
    }])
    return {"calendars": table, "calendar_alternatives": alternatives, "calendar_consistency": consistency, "calendar_structure": structure}


RUN = {"levels": stage_levels, "seasons": stage_seasons, "within": stage_within, "calendars": stage_calendars}


def run(stages=STAGES):
    top = network.ceiling()
    screened, calendars = load(top)
    tables: dict[str, pd.DataFrame] = {"filters": screened}
    for name in stages:
        tables |= RUN[name](screened, top, calendars)
    return tables, top


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--summary", nargs="?", const=str(SUMMARY), help="write the tables to this JSON file")
    a = ap.parse_args(argv)
    tables, top = run(a.stages)
    pd.set_option("display.width", 260, "display.max_rows", 300, "display.float_format", lambda v: f"{v:.3g}")
    for name, table in tables.items():
        if name != "filters":
            print(f"\n== {name}\n{table.to_string(index=False)}")
    if a.summary:
        payload = {"generated_from": "research/ftir_hips_chem/workflows/run_hips_seasons_20261111.py", "aethalometer_used": False,
                   "inputs": {"filters": str(network.NETWORK.relative_to(ROOT)), "calendars": str(CALENDARS.relative_to(ROOT)),
                              "climate": str(NORMALS.relative_to(ROOT))}, "ceiling": top}
        for name, table in tables.items():
            if name != "filters":
                payload[name] = json.loads(table.to_json(orient="records", double_precision=6))
        Path(a.summary).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
