"""Three-wavelength AERONET absorption diagnostic with quality audit.

Computes AAE(440–675) and AAE(675–870) independently.  The latter is the
red/near-IR test that a single 440–870 exponent cannot identify.  Available
exports are Level 1.5 almucantar daily products; no U27/inversion-residual
field is present, so the requested strict Level-2/U27 analysis is explicitly
reported as unavailable rather than silently approximated.  AOD440 >= 0.4 is
retained as a high-loading sensitivity subset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "output/tables/aeronet"

sys.path.insert(0, str(HERE))
from run_aeronet_spartan import addis_inversion  # noqa: E402

FILES = {
    "Addis Ababa": None,
    "Delhi (Gual Pahari)": OUT / "Gual_Pahari_inv_daily.csv",
    "Delhi (Amity Gurgaon)": OUT / "Amity_Univ_Gurgaon_inv_daily.csv",
    "Delhi (New Delhi IMD)": OUT / "New_Delhi_IMD_inv_daily.csv",
    "Beijing": OUT / "Beijing_inv_daily.csv",
    "Pasadena": OUT / "CalTech_inv_daily.csv",
}


def _find(frame: pd.DataFrame, stem: str) -> str:
    match = [column for column in frame.columns if column.lower() == stem.lower()]
    if not match:
        raise KeyError(stem)
    return match[0]


def _load(city: str, path: Path | None) -> pd.DataFrame:
    frame = addis_inversion() if path is None else pd.read_csv(path, low_memory=False)
    columns = {
        "date": _find(frame, "date"),
        "level": _find(frame, "level"),
        "aod440": _find(frame, "AOD_Coincident_Input[440nm]"),
        "aaod440": _find(frame, "Absorption_AOD[440nm]"),
        "aaod675": _find(frame, "Absorption_AOD[675nm]"),
        "aaod870": _find(frame, "Absorption_AOD[870nm]"),
    }
    selected = frame[list(columns.values())].rename(
        columns={value: key for key, value in columns.items()}
    )
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    selected["city"] = city
    for column in ("aod440", "aaod440", "aaod675", "aaod870"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    valid = selected[["aaod440", "aaod675", "aaod870"]].gt(0).all(axis=1)
    selected = selected[valid].copy()
    selected["AAE_440_675"] = np.log(selected["aaod440"] / selected["aaod675"]) / np.log(675 / 440)
    selected["AAE_675_870"] = np.log(selected["aaod675"] / selected["aaod870"]) / np.log(870 / 675)
    selected["AAE_curvature"] = selected["AAE_440_675"] - selected["AAE_675_870"]
    selected["high_AOD440"] = selected["aod440"].ge(0.4)
    selected["level2"] = selected["level"].astype(str).str.casefold().isin(
        ["lev20", "level 2.0", "2.0"]
    )
    return selected


def main() -> None:
    frames, rows = [], []
    for city, path in FILES.items():
        frame = _load(city, path)
        frames.append(frame)
        for subset, group in (
            ("all_positive_AAOD", frame),
            ("AOD440_ge_0.4", frame[frame["high_AOD440"]]),
            ("strict_level2", frame[frame["level2"]]),
        ):
            row = {
                "city": city, "subset": subset, "n": len(group),
                "source_level_values": "+".join(sorted(frame["level"].astype(str).unique())),
                "U27_available": False,
            }
            for column in ("AAE_440_675", "AAE_675_870", "AAE_curvature"):
                row[f"{column}_median"] = group[column].median()
                row[f"{column}_q25"] = group[column].quantile(0.25)
                row[f"{column}_q75"] = group[column].quantile(0.75)
            rows.append(row)
    daily = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(rows)
    tests = []
    for subset, selected in (
        ("all_positive_AAOD", daily),
        ("AOD440_ge_0.4", daily[daily["high_AOD440"]]),
    ):
        for metric in ("AAE_440_675", "AAE_675_870", "AAE_curvature"):
            groups = [group[metric].dropna().to_numpy() for _, group in selected.groupby("city")]
            groups = [group for group in groups if len(group) >= 5]
            statistic, pvalue = kruskal(*groups)
            tests.append({
                "subset": subset, "metric": metric, "n_cities": len(groups),
                "kruskal_H": statistic, "p_value": pvalue,
            })
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "three_wavelength_daily.csv", index=False)
    summary.to_csv(OUT / "three_wavelength_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "three_wavelength_site_tests.csv", index=False)
    quality = {
        "strict_level2_rows": int(daily["level2"].sum()),
        "U27_field_present": False,
        "interpretation": (
            "Exploratory Level-1.5 result only; strict Level-2/U27 confirmation "
            "cannot be run from the available exports."
        ),
    }
    (OUT / "three_wavelength_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    show = summary[summary["subset"].isin(["all_positive_AAOD", "AOD440_ge_0.4"])]
    print(show[[
        "city", "subset", "n", "AAE_440_675_median", "AAE_675_870_median",
        "AAE_curvature_median", "source_level_values",
    ]].round(3).to_string(index=False))
    print("\nStrict Level-2/U27: unavailable (0 Level-2 rows; U27 absent).")
    print(f"wrote three-wavelength tables to {OUT}")


if __name__ == "__main__":
    main()
