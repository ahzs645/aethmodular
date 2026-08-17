"""monetio trial: AERONET v3 daily AOD for Addis Ababa, 2024-2025.

monetio is NOT on PyPI (404). Install from the NOAA repo:
    pip install --no-deps "git+https://github.com/noaa-oar-arl/monetio.git"
(plus cftime/netCDF4 if missing; the AERONET reader itself only needs
pandas + numpy, and dask when n_procs > 1).

Run:  python aeronet_pull.py
"""

import pandas as pd

from monetio import aeronet

# 'Addis_Ababa' is NOT a valid v3 siteid. The Addis Ababa University sites
# are 'AAU_ET' (record 2020-01-01 .. 2022-10-14, now ended) and
# 'AAU_Jackros_ET' (2022-10-15 .. present, active) — both 9.02 N, ~38.8 E,
# 2370 m, PI Araya Asfaw. Found with the bounding-box search below; a
# name-grep for 'addis'/'ethio' finds nothing.
#
# Reliability note: the AERONET web service rate-limits (~10 hits/min) and a
# pull can transiently raise "valid query but no data found" — retry after a
# pause before concluding a site is empty. monetio hits the URL twice per
# call (header sniff + read), so back-to-back pulls trip the limit quickly.
SITE = "AAU_Jackros_ET"
DATES = pd.date_range("2024-01-01", "2025-12-31", freq="D")


def list_ethiopian_sites():
    """Enumerate AERONET sites inside an Ethiopia lat/lon box."""
    sites = aeronet.get_valid_sites()  # pulls aeronet_locations_v3.txt
    lat = sites["latitude"].astype(float)
    lon = sites["longitude"].astype(float)
    return sites[lat.between(3, 15) & lon.between(33, 48)]


def main():
    print("AERONET v3 sites in the Ethiopia bounding box:")
    matches = list_ethiopian_sites()
    print(matches.to_string(index=False))

    print(f"\nPulling AOD15 daily for {SITE}, {DATES[0].date()}..{DATES[-1].date()} ...")
    df = aeronet.add_data(DATES, product="AOD15", siteid=SITE, daily=True)
    print(f"shape: {df.shape}")
    # daily=True returns 'time' as a DatetimeIndex, not a column
    print(f"time index: {df.index.min().date()} .. {df.index.max().date()}")

    aod_cols = [c for c in df.columns if c.startswith("aod_")]
    valid = {c: int(df[c].notna().sum()) for c in aod_cols}
    print(f"\nAOD wavelength columns ({len(aod_cols)}):")
    for c, n in valid.items():
        flag = "" if n else "  (all NaN at this site)"
        print(f"  {c:16s} n_valid={n}{flag}")

    extras = [c for c in df.columns if not c.startswith("aod_")]
    print(f"\nNon-AOD columns ({len(extras)}): {extras}")

    # Angstrom exponents the service computes for us
    ae_cols = [c for c in df.columns if "angstrom" in c.lower()]
    print(f"\nAngstrom-exponent columns: {ae_cols}")

    # --- inversion products (absorption AOD / SSA / AAE live here) ---------
    print("\nTrying almucantar inversion product TAB (absorption AOD), 2024 only ...")
    try:
        inv = aeronet.add_data(
            pd.date_range("2024-01-01", "2024-12-31", freq="D"),
            product="TAB", inv_type="ALM15", siteid=SITE, daily=True,
        )
        print(f"inversion shape: {inv.shape}")
        abs_cols = [c for c in inv.columns
                    if "abs" in c.lower() or "angstrom" in c.lower()]
        print(f"absorption/AAE columns: {abs_cols}")
        for c in abs_cols:
            print(f"  {c}: n_valid={int(inv[c].notna().sum())}")
    except Exception as e:
        print(f"inversion pull failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
