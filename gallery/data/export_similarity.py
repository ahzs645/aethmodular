"""Export spectral similarity between every site and site-season for the gallery.

Run with uv run --locked --no-sync python gallery/data/export_similarity.py.

Reads the frozen full-profile AIRSpec/VIBES run read-only. For every spectra
method and analog-selection mask it writes:

* the mean Pearson r between every pair of groups (a site, or one season at a
  site), computed as the mean over all filter pairs a in A, t in T with a != t;
* for every group, its 500 most similar individual filters from other sites,
  scored exactly as seasonal_analogs.mean_correlation_scores scores a library
  against a target set (mean signed Pearson r against every target filter);
* one binned, quantised trace per filter so the browser can overlay spectra.

No fitting, selection or exclusion is changed. Masks alter the similarity
channels only, exactly as in the analog selections.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AREA = ROOT / "research/ftir_hips_chem"
sys.path.insert(0, str(AREA / "scripts"))
from seasonal_analogs import centered_unit_spectra, mean_correlation_scores, spectral_region_mask  # noqa: E402
from improve_io import improve_dir, read_improve_metadata  # noqa: E402

RUN = AREA / "output/tables/vibes_colab_cloud/persistent_results/full-83dcf32e86bc0f09"
ETAD_META = AREA / "output/tables/vibes_colab_bundle/stage/data/etad_metadata.csv"
OUT = ROOT / "gallery/app/public/data/similarity"
METHODS = ("AIRSpec", "VIBES")
# Same keys and cuts as the historical mask table on the AIRSpec / VIBES tab.
MASKS = (
    ("full", "No selection mask", None, None),
    ("no_co2", "Exclude 1800–2500", 1800, None),
    ("no_co2_max3600", "Exclude 1800–2500 and >3600", 1800, 3600),
    ("no_co2_max3500", "Exclude 1800–2500 and >3500", 1800, 3500),
)
TOP_K = 500          # the analog cohorts are 500 filters
BIN = 8              # channels per displayed trace point (~10.3 cm⁻¹)
SCALE = 10000        # r stored as int16 r * SCALE
PAD = 65535          # uint16 sentinel for "no filter"

# Addis: the project's canonical convention (a), matching the Follow-up C
# figures. Every IMPROVE site: meteorological seasons. See docs/site-seasonality.md.
ADDIS_SEASONS = (
    ("Dry (Oct–Feb)", (10, 11, 12, 1, 2), "#E67E22"),
    ("Belg (Mar–May)", (3, 4, 5), "#27AE60"),
    ("Kiremt (Jun–Sep)", (6, 7, 8, 9), "#3498DB"),
)
MET_SEASONS = (
    ("Winter (Dec–Feb)", (12, 1, 2), "#6F86BA"),
    ("Spring (Mar–May)", (3, 4, 5), "#67A96A"),
    ("Summer (Jun–Aug)", (6, 7, 8), "#D99145"),
    ("Autumn (Sep–Nov)", (9, 10, 11), "#AA6B78"),
)
ADDIS = "ETAD"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def site_details(codes: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """Name and location per site code: IMPROVE's Sites sheet, Addis from its metadata."""
    sources = {}
    sheet = read_improve_metadata(improve_dir()).get("Sites")
    if sheet is None:
        raise ValueError("No IMPROVE Sites sheet found; set AETHMODULAR_IMPROVE_DIR")
    for f in sorted(set(sheet.source_file)):
        sources[Path(f).name] = sha(Path(f))
    sheet = sheet.drop_duplicates("Code").set_index("Code")
    out = {}
    for code in codes:
        if code == ADDIS:
            etad = pd.read_csv(ETAD_META)
            sources[str(ETAD_META.relative_to(ROOT))] = sha(ETAD_META)
            out[code] = dict(name="Addis Ababa", region="Ethiopia",
                             lat=round(float(etad.Latitude.iloc[0]), 4), lon=round(float(etad.Longitude.iloc[0]), 4))
            continue
        if code not in sheet.index:
            raise ValueError(f"{code}: missing from the IMPROVE Sites sheet")
        r = sheet.loc[code]
        text = lambda v: None if pd.isna(v) or str(v).strip() in ("", "Unknown") else str(v).strip()
        out[code] = dict(
            name=text(r.Site), region=text(r.State),  # County holds a FIPS code, not a name
            lat=None if pd.isna(r.Latitude) else round(float(r.Latitude), 4),
            lon=None if pd.isna(r.Longitude) else round(float(r.Longitude), 4),
            elevation=None if pd.isna(r.Elevation) else round(float(r.Elevation)),
            land_use=text(r.LandUseCode), setting=text(r.DemographicCode),
        )
    return out, sources


def calendar(site: str):
    return ADDIS_SEASONS if site == ADDIS else MET_SEASONS


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "RUN_MANIFEST.json").read_text())
    if manifest["n_failed"] or manifest["n_cases"] != 12808:
        raise ValueError("Expected the complete frozen full-profile run")
    cases = pd.read_csv(RUN / "case_audit.csv")
    if len(cases) != manifest["n_cases"] or not cases.paired_valid.all():
        raise ValueError("Case ledger does not match the run manifest")
    wn = np.load(RUN / "wn.npy")
    arrays = {m: np.load(RUN / f"corrected_{m}.npy", mmap_mode="r") for m in METHODS}
    if any(a.shape != (len(cases), len(wn)) for a in arrays.values()):
        raise ValueError("Saved case/feature alignment changed")

    # Library = every IMPROVE calibration filter plus every Addis target filter.
    # Blanks and injections are not ambient samples and are left out.
    keep = cases.kind.isin(["calibration", "target"]).to_numpy()
    rows = np.flatnonzero(keep)
    lib = cases.iloc[rows].reset_index(drop=True)
    if len(lib) >= PAD or not lib.sample_id.is_unique:
        raise ValueError("Filter table too large for uint16 indices or not unique")
    # Six Addis filters have no date in the frozen ledger. They stay in the
    # all-year site group and in no season; nothing is imputed.
    months = pd.to_datetime(lib.date, errors="raise").dt.month.fillna(0).astype(int).to_numpy()
    split = np.where(lib.kind.eq("target"), "external", lib.split).astype(str)

    # Groups: each site, then each of its seasons that has filters.
    sites = sorted(lib.Site.unique(), key=lambda s: (s != ADDIS, s))
    site_split = {s: sorted(set(split[lib.Site.eq(s)])) for s in sites}
    if any(len(v) != 1 for v in site_split.values()):
        raise ValueError("A site spans more than one split")
    groups, members = [], []
    season_of = np.full(len(lib), -1)
    for s in sites:
        at = lib.Site.eq(s).to_numpy()
        groups.append(dict(site=s, season=None, n=int(at.sum())))
        members.append(np.flatnonzero(at))
        for i, (name, ms, _) in enumerate(calendar(s)):
            idx = np.flatnonzero(at & np.isin(months, ms))
            season_of[idx] = i
            if len(idx):
                groups.append(dict(site=s, season=i, n=len(idx)))
                members.append(idx)
    if (season_of[months > 0] < 0).any():
        raise ValueError("A dated filter fell outside its site's calendar")
    undated = lib.sample_id[months == 0].tolist()
    details, detail_sources = site_details(sites)
    G = len(groups)
    member_matrix = np.zeros((G, len(lib)))
    for g, idx in enumerate(members):
        member_matrix[g, idx] = 1.0 / len(idx)
    sizes = np.array([g["n"] for g in groups], float)
    group_site = np.array([sites.index(g["site"]) for g in groups])
    filter_site = np.array([sites.index(s) for s in lib.Site])
    # Filter pairs shared by two groups (same filter on both sides, r = 1) are
    # removed so a group's score against itself or its parent site is not
    # inflated by self-pairs. Membership is 0/1, so overlap = counts product.
    binary = (member_matrix > 0).astype(float)
    overlap = binary @ binary.T

    bins = [np.arange(i, min(i + BIN, len(wn))) for i in range(0, len(wn), BIN)]
    bin_wn = np.array([wn[b].mean() for b in bins])
    mask_meta, n_pairs = [], G * (G + 1) // 2
    iu = np.triu_indices(G)

    for method in METHODS:
        X = np.asarray(arrays[method][rows], dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError(f"{method}: non-finite corrected spectra")
        # Display traces: bin means, int16 with one float32 scale per filter.
        binned = np.stack([X[:, b].mean(axis=1) for b in bins], axis=1)
        scale = np.abs(binned).max(axis=1) / 32767
        scale[scale == 0] = 1
        q = np.round(binned / scale[:, None]).astype("<i2")
        with (OUT / f"traces_{method}.bin").open("wb") as out:
            out.write(scale.astype("<f4").tobytes())
            out.write(q.tobytes())

        for key, label, co2_low, upper in MASKS:
            mask = spectral_region_mask(wn, co2_low, upper)
            U = centered_unit_spectra(X[:, mask])
            means = member_matrix @ U                     # G × channels
            total = (means @ means.T) * np.outer(sizes, sizes)
            pairs = (total - overlap) / np.maximum(np.outer(sizes, sizes) - overlap, 1)
            singleton = np.outer(sizes, sizes) - overlap == 0
            pairs[singleton] = np.nan
            if np.nanmax(np.abs(pairs)) > 1 + 1e-9:
                raise ValueError("Group mean correlation left [-1, 1]")
            tri = np.where(np.isfinite(pairs[iu]), np.round(np.clip(pairs[iu], -1, 1) * SCALE), -32768)
            (OUT / f"pairs_{method}_{key}.bin").write_bytes(tri.astype("<i2").tobytes())

            # Filter-level: identical to mean_correlation_scores with the
            # group as the target set. Spot-check parity on real groups.
            scores = np.clip(U @ means.T, -1, 1)          # filters × G
            for g in (0, 1, G // 2, G - 1):
                ref = mean_correlation_scores(X, X[members[g]], mask)
                np.testing.assert_allclose(scores[:, g], ref, atol=1e-9, rtol=0)
            top_idx = np.full((G, TOP_K), PAD, dtype="<u2")
            top_r = np.zeros((G, TOP_K), dtype="<i2")
            for g in range(G):
                cand = np.flatnonzero(filter_site != group_site[g])
                s = scores[cand, g]
                k = min(TOP_K, len(cand))
                best = np.argpartition(-s, k - 1)[:k]
                # ties broken by library order so the export is deterministic
                best = best[np.lexsort((cand[best], -s[best]))]
                top_idx[g, :k] = cand[best]
                top_r[g, :k] = np.round(s[best] * SCALE)
            with (OUT / f"top_{method}_{key}.bin").open("wb") as out:
                out.write(top_idx.tobytes())
                out.write(top_r.tobytes())
            if method == METHODS[0]:
                mask_meta.append(dict(key=key, label=label, co2_low=co2_low, upper=upper,
                                      n_channels=int(mask.sum()),
                                      bins=[bool(mask[b].all()) for b in bins]))
            print(f"{method} {key}: {mask.sum()} channels, {G} groups")

    out = dict(
        schema_version=1,
        source_signature=manifest["signature"],
        source_sha256={str((RUN / n).relative_to(ROOT)): sha(RUN / n)
                       for n in ["RUN_MANIFEST.json", "case_audit.csv", "wn.npy",
                                 "corrected_AIRSpec.npy", "corrected_VIBES.npy"]},
        exporter_sha256=sha(Path(__file__)),
        methods=list(METHODS),
        masks=mask_meta,
        default_mask="no_co2_max3500",
        scale=SCALE,
        top_k=TOP_K,
        n_pairs=n_pairs,
        bin_wn=[round(float(w), 3) for w in bin_wn],
        calendars={"ETAD": [dict(name=n, months=list(m), color=c) for n, m, c in ADDIS_SEASONS],
                   "default": [dict(name=n, months=list(m), color=c) for n, m, c in MET_SEASONS]},
        site_detail_sha256=detail_sources,
        sites=[dict(site=s, split=site_split[s][0], n=int(lib.Site.eq(s).sum()),
                    label="ETAD · Addis Ababa" if s == ADDIS else s, **details[s]) for s in sites],
        undated=undated,
        groups=[dict(site=sites.index(g["site"]), season=g["season"], n=g["n"]) for g in groups],
        filters=dict(
            sample_id=lib.sample_id.tolist(),
            site=filter_site.tolist(),
            date=lib.date.where(lib.date.notna(), None).tolist(),
            season=season_of.tolist(),
        ),
    )
    (OUT / "similarity.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(lib):,} filters, {len(sites)} sites, {G} groups -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
