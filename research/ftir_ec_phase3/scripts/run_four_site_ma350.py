"""Four-site FTIR/HIPS versus independent MA350 IR-880 comparisons."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import odr, stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROCESSED = REPO / "research/ftir_hips_chem/processed_sites"
OUT = HERE.parent / "output/tables/variation_closure"

sys.path.insert(0, str(REPO / "research/ftir_hips_chem/scripts"))
from data_matching import load_filter_data  # noqa: E402

SITES = {
    "ETAD": "Addis_Ababa", "CHTS": "Beijing", "INDH": "Delhi", "USPA": "JPL",
}


def _fit(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    dof = len(x) - 2
    se_intercept = np.sqrt(
        np.sum(residual ** 2) / dof
        * (1 / len(x) + x.mean() ** 2 / np.sum((x - x.mean()) ** 2))
    )
    critical = stats.t.ppf(0.975, dof)
    model = odr.Model(lambda beta, values: beta[0] * values + beta[1])
    deming = odr.ODR(odr.RealData(x, y), model, beta0=[slope, intercept]).run()
    return {
        "n": len(x), "ols_slope": slope, "ols_intercept": intercept,
        "intercept_ci_low": intercept - critical * se_intercept,
        "intercept_ci_high": intercept + critical * se_intercept,
        "R2": np.corrcoef(x, y)[0, 1] ** 2,
        "RMSE_identity": np.sqrt(np.mean((y - x) ** 2)),
        "deming_equal_error_slope": deming.beta[0],
        "deming_equal_error_intercept": deming.beta[1],
    }


def _aeth(site_name: str) -> pd.DataFrame:
    frame = pd.read_pickle(PROCESSED / f"df_{site_name}_9am_resampled.pkl")
    output = pd.DataFrame({
        "aeth_date": pd.to_datetime(frame.loc[:, frame.columns == "day_9am"].iloc[:, 0]),
        "ma350_ir880_ugm3": pd.to_numeric(
            frame.loc[:, frame.columns == "IR BCc"].iloc[:, 0], errors="coerce"
        ) / 1000,
    }).dropna()
    return output.sort_values("aeth_date").drop_duplicates("aeth_date")


def main() -> None:
    filters = load_filter_data()
    rows, pairs = [], []
    for site, site_name in SITES.items():
        part = filters[filters["Site"].eq(site)]
        base = (part[["FilterId", "SampleDate"]].drop_duplicates("FilterId")
                .assign(filter_date=lambda frame: pd.to_datetime(frame["SampleDate"])))
        base = base.dropna(subset=["filter_date"])
        wide = (part[part["Parameter"].isin(["EC_ftir", "HIPS_Fabs"])]
                .pivot_table(index="FilterId", columns="Parameter", values="Concentration",
                             aggfunc="first").reset_index())
        base = base.merge(wide, on="FilterId", how="left", validate="one_to_one")
        aeth = _aeth(site_name)
        for match in ("exact", "nearest_1d"):
            if match == "exact":
                joined = base.merge(aeth, left_on="filter_date", right_on="aeth_date",
                                    how="inner")
            else:
                joined = pd.merge_asof(
                    base.sort_values("filter_date"), aeth, left_on="filter_date",
                    right_on="aeth_date", direction="nearest", tolerance=pd.Timedelta("1D"),
                ).dropna(subset=["ma350_ir880_ugm3"])
            joined["site"] = site
            joined["match"] = match
            joined["date_offset_days"] = (
                joined["aeth_date"] - joined["filter_date"]
            ).dt.days
            pairs.append(joined)
            for reference, column in (
                ("FTIR_EC", "EC_ftir"), ("HIPS_BC_MAC10", "HIPS_Fabs"),
            ):
                subset = joined.dropna(subset=[column, "ma350_ir880_ugm3"]).copy()
                if reference == "HIPS_BC_MAC10":
                    subset[column] = subset[column] / 10.0
                if len(subset) < 8 or subset[column].nunique() < 3:
                    continue
                rows.append({
                    "site": site, "city": site_name, "match": match,
                    "reference": reference,
                    **_fit(subset["ma350_ir880_ugm3"].to_numpy(float),
                           subset[column].to_numpy(float)),
                })
    summary = pd.DataFrame(rows)
    per_filter = pd.concat(pairs, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "four_site_ma350_summary.csv", index=False)
    per_filter.to_csv(OUT / "four_site_ma350_pairs.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(f"\nwrote four-site MA350 tables to {OUT}")


if __name__ == "__main__":
    main()
