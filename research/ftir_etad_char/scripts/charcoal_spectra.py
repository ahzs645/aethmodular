"""Loaders for the six published charcoal / biochar FTIR reference collections.

Source archive and provenance: ``research/ftir_hips_chem/charcoal_ftir_sources/``
(see its README.md for DOIs and licenses). Every loader returns the same
``SpectralSet`` record so the notebooks can iterate over collections without
special-casing each file's quirks.

Per-file quirks handled here so no notebook has to:

* Both Minatre CSVs carry a UTF-8 BOM (``encoding="utf-8-sig"``).
* ``Minatre_reference_spectra.csv`` rounds its wavenumber headers to integers,
  which collides 95 labels; pandas silently mangles them to ``958.1`` etc. The
  true axis is the native grid stored intact in the combustion-facility file,
  so it is borrowed from there rather than parsed from the mangled header.
* ``Maezumi_ref_data.csv`` prefixes its last column ``W_3500.32``.
* Both McCall workbooks label wavenumbers with Python ``int``, descending.
* ``McCall_barley`` has no temperature column — it is encoded in ``name``
  (bare ``BS`` = unpyrolyzed feedstock, i.e. 0 C).
* WDG is the only column-oriented file (sheet ``"FTIR data"``, with a space)
  and needs a transpose; its ``.1``-``.8`` header suffixes are the replicates.

**Normalization state** (verified empirically, not taken from the papers):
Minatre x2 and McCall x2 ship as absorbance; Maezumi and WDG ship already
**SNV-normalized per spectrum** (each spectrum mean 0, sd 1). SNV is
shape-preserving, so applying `snv()` to the absorbance sets puts all six
collections — and the Addis filter spectra — on one comparable footing. That
is the only footing on which they *are* comparable: absolute absorbance
differs by ~30x between Minatre and McCall alone.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = (
    REPO_ROOT
    / "research"
    / "ftir_hips_chem"
    / "charcoal_ftir_sources"
    / "downloads"
    / "datasets"
)

# Region where all six charcoal collections overlap.
CHARCOAL_OVERLAP = (951.0, 3500.0)

# Region where Addis filter spectra are trustworthy: below ~1425 cm-1 the
# PTFE filter dominates and the AIRSpec segmented baseline does not extend
# there (phase-3 `airspec_baseline.py`, SEG2 lower bound = 1425).
ADDIS_USABLE = (1425.0, 3500.0)

# Diagnostic bands used throughout both notebooks. Charring drives intensity
# from the aliphatic/oxygenated bands into aromatic C=C and aromatic C-H.
BANDS = {
    "OH / NH stretch": (3200, 3400),
    "aromatic CH stretch": (3040, 3070),
    "aliphatic CH stretch": (2850, 2950),
    "carbonyl C=O": (1700, 1750),
    "aromatic C=C": (1580, 1620),
    "carboxylate / C-H bend": (1400, 1450),
    "C-O / C-O-C": (1000, 1300),
}


@dataclass
class SpectralSet:
    """One published collection: spectra plus harmonized metadata.

    ``X`` is (n_spectra, n_wavenumbers) aligned to ``meta`` rows and to ``wn``,
    which is always **ascending**. ``meta`` always carries the columns
    ``sample``, ``species``, ``temp_c``, ``treatment``, ``rep``; ``temp_c`` is
    NaN for the field-burn and fossil sets, whose temperature is unknown (they
    are the application sets those papers predict onto, not training data).
    """

    key: str
    label: str
    short: str
    wn: np.ndarray
    X: np.ndarray
    meta: pd.DataFrame
    native_scale: str  # "absorbance" or "snv"
    citation: str
    notes: str = ""
    _: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def __repr__(self) -> str:
        temps = self.meta["temp_c"].dropna().unique()
        trange = f"{temps.min():.0f}-{temps.max():.0f} C" if len(temps) else "unknown T"
        return (
            f"<SpectralSet {self.key}: {self.n} spectra, "
            f"{self.wn.min():.0f}-{self.wn.max():.0f} cm-1, {trange}, {self.native_scale}>"
        )


def _ascending(wn: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(wn)
    return wn[order], X[:, order]


def _harmonize(meta: pd.DataFrame) -> pd.DataFrame:
    for col in ("sample", "species", "temp_c", "treatment", "rep"):
        if col not in meta.columns:
            meta[col] = np.nan
    return meta.reset_index(drop=True)


# --------------------------------------------------------------------------
# Minatre 2024 (Frontiers in Earth Science) — Dryad doi:10.5061/dryad.cnp5hqcbj
# --------------------------------------------------------------------------

MINATRE_SPECIES = {
    "ABCO": "Abies concolor",
    "CADE": "Calocedrus decurrens",
    "PICO": "Pinus contorta",
    "PIMO": "Pinus monticola",
    "PIPO": "Pinus ponderosa",
    "ACGL": "Acer glabrum",
    "ALRU": "Alnus rubra",
    "ARCT": "Arctostaphylos",
    "QUKE": "Quercus kelloggii",
    "QUVA": "Quercus vaccinifolia",
    "POTR": "Populus tremuloides",
}


def _minatre_grid() -> np.ndarray:
    """Native 2,646-point grid, read from the file that kept full precision."""
    header = (
        (DATA_DIR / "Minatre_combustionfacility_spectra.csv")
        .open(encoding="utf-8-sig")
        .readline()
        .strip()
        .split(",")
    )
    return np.array([float(c.strip('"')) for c in header[1:]], dtype=float)


def load_minatre_reference() -> SpectralSet:
    df = pd.read_csv(DATA_DIR / "Minatre_reference_spectra.csv", encoding="utf-8-sig")
    wn = _minatre_grid()
    X = df.iloc[:, 2:].to_numpy(float)
    assert X.shape[1] == wn.size, "reference file is not on the combustion grid"

    meta = pd.DataFrame(
        {
            "species": df["Species"].map(MINATRE_SPECIES).fillna(df["Species"]),
            "species_code": df["Species"],
            "temp_c": df["Temperature"].astype(float),
        }
    )
    meta["sample"] = meta["species_code"] + "_" + df["Temperature"].astype(str)
    meta["treatment"] = "muffle furnace"
    meta["rep"] = meta.groupby("sample").cumcount() + 1

    wn, X = _ascending(wn, X)
    return SpectralSet(
        key="minatre_ref",
        label="Minatre — muffle-furnace reference",
        short="Minatre ref",
        wn=wn,
        X=X,
        meta=_harmonize(meta),
        native_scale="absorbance",
        citation="Minatre et al. 2024, Front. Earth Sci. (doi:10.3389/feart.2024.1354080); Dryad CC0",
        notes="10 western-US tree/shrub species, 200-800 C, 30 replicates per cell.",
    )


def load_minatre_combustion() -> SpectralSet:
    df = pd.read_csv(
        DATA_DIR / "Minatre_combustionfacility_spectra.csv", encoding="utf-8-sig"
    )
    wn = df.columns[1:].astype(float).to_numpy()
    X = df.iloc[:, 1:].to_numpy(float)

    parts = df["Sample"].str.split("_", expand=True)
    meta = pd.DataFrame({"sample": df["Sample"]})
    meta["species_code"] = parts[2]
    meta["species"] = parts[2].map(MINATRE_SPECIES).fillna(parts[2])
    meta["burn_date"] = parts[0] + "_" + parts[1]
    meta["temp_c"] = np.nan  # instrumented burns; temperature is the unknown
    meta["treatment"] = "combustion facility"
    meta["rep"] = meta.groupby("sample").cumcount() + 1

    wn, X = _ascending(wn, X)
    return SpectralSet(
        key="minatre_burn",
        label="Minatre — combustion facility",
        short="Minatre burns",
        wn=wn,
        X=X,
        meta=_harmonize(meta),
        native_scale="absorbance",
        citation="Minatre et al. 2024, Front. Earth Sci.; Dryad CC0",
        notes="Instrumented open burns, 4 species. Temperature unknown by design — "
        "this is the set the reference series is used to predict onto.",
    )


# --------------------------------------------------------------------------
# Maezumi 2021 (Palaeogeography) — Zenodo 5156747
# --------------------------------------------------------------------------


def load_maezumi() -> SpectralSet:
    df = pd.read_csv(DATA_DIR / "Maezumi_ref_data.csv")
    spec_cols = df.columns[2:]
    wn = np.array([float(str(c).replace("W_", "")) for c in spec_cols])
    X = df[spec_cols].to_numpy(float)

    meta = pd.DataFrame(
        {
            "species_code": df["species"],
            "species": df["species"],
            "temp_c": df["temperature"].astype(float),
        }
    )
    meta["sample"] = meta["species_code"] + "_" + df["temperature"].astype(str)
    meta["treatment"] = "muffle furnace"
    meta["rep"] = meta.groupby("sample").cumcount() + 1

    wn, X = _ascending(wn, X)
    return SpectralSet(
        key="maezumi",
        label="Maezumi — modern analogue reference",
        short="Maezumi",
        wn=wn,
        X=X,
        meta=_harmonize(meta),
        native_scale="snv",
        citation="Maezumi et al. 2021, Palaeogeogr. (doi:10.1016/j.palaeo.2021.110580); Zenodo CC BY 4.0",
        notes="9 species, 200-700 C. Ships already SNV-normalized; species codes are "
        "case-sensitive (PC and pc are different taxa) and no legend was published.",
    )


# --------------------------------------------------------------------------
# McCall — ACS Sustainable Resource Management (two studies)
# --------------------------------------------------------------------------

MCCALL_FEEDSTOCK = {
    "BS": "barley straw",
    "CW": "chestnut wood",
    "EB": "eucalyptus bark",
    "MG": "miscanthus grass",
    "PB": "pine bark",
    "RH": "rice husk",
}


def load_mccall_multifeedstock() -> SpectralSet:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(
            DATA_DIR / "McCall_multifeedstock_FTIR_spectral_data.xlsx",
            sheet_name="FTIR_data",
        )
    meta_cols = ["Unnamed: 0", "name", "temp", "feedstock", "H.C", "O.C"]
    spec_cols = [c for c in df.columns if c not in meta_cols]
    wn = np.array(spec_cols, dtype=float)
    X = df[spec_cols].to_numpy(float)

    meta = pd.DataFrame(
        {
            "sample": df["name"],
            "species_code": df["feedstock"],
            "species": df["feedstock"].map(MCCALL_FEEDSTOCK).fillna(df["feedstock"]),
            "temp_c": df["temp"].astype(float),
            "H_C": df["H.C"],
            "O_C": df["O.C"],
        }
    )
    meta["treatment"] = np.where(meta["temp_c"] == 0, "unpyrolyzed feedstock", "pyrolysis")
    meta["rep"] = meta.groupby("sample").cumcount() + 1

    wn, X = _ascending(wn, X)
    return SpectralSet(
        key="mccall_multi",
        label="McCall — six feedstocks",
        short="McCall 6-feedstock",
        wn=wn,
        X=X,
        meta=_harmonize(meta),
        native_scale="absorbance",
        citation="McCall et al. 2025, ACS Sustain. Resour. Manage. (doi:10.1021/acssusresmgt.5c00104); CC BY",
        notes="6 feedstocks, 0 (raw) to 700 C, 3 replicates. Carries H/C and O/C "
        "atomic ratios — the only collection with elemental data attached.",
    )


def load_mccall_barley() -> SpectralSet:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(
            DATA_DIR / "McCall_barley_FTIR_spectral_data.xlsx", sheet_name="FTIR_data"
        )
    spec_cols = [c for c in df.columns if not isinstance(c, str)]
    wn = np.array(spec_cols, dtype=float)
    X = df[spec_cols].to_numpy(float)

    meta = pd.DataFrame({"sample": df["name"]})
    meta["temp_c"] = (
        df["name"].str.extract(r"BS(\d+)")[0].astype(float).fillna(0.0)
    )  # bare "BS" is the unpyrolyzed straw
    meta["species_code"] = "BS"
    meta["species"] = "barley straw"
    meta["treatment"] = np.where(meta["temp_c"] == 0, "unpyrolyzed feedstock", "pyrolysis")
    meta["rep"] = meta.groupby("sample").cumcount() + 1

    wn, X = _ascending(wn, X)
    return SpectralSet(
        key="mccall_barley",
        label="McCall — barley straw series",
        short="McCall barley",
        wn=wn,
        X=X,
        meta=_harmonize(meta),
        native_scale="absorbance",
        citation="McCall et al. 2024, ACS Sustain. Resour. Manage. (doi:10.1021/acssusresmgt.4c00148); CC BY",
        notes="Single feedstock at 50 C resolution (0, 150-700) — the finest "
        "temperature sampling in the archive.",
    )


# --------------------------------------------------------------------------
# Gosling / Cornelissen / McMichael (WDG) — Figshare 5979544
# --------------------------------------------------------------------------

WDG_SPECIES = {"AG": "Alnus glutinosa", "PC": "Panicum capillare"}
WDG_TREATMENT = {"raw": "untreated", "water": "water", "perox": "hydrogen peroxide"}


def load_wdg() -> SpectralSet:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(
            DATA_DIR / "WDG-CharcoalTemp-Data.xlsx", sheet_name="FTIR data"
        )
    wn = df["wavelength"].to_numpy(float)
    spectra = df.drop(columns=["wavelength"])
    X = spectra.to_numpy(float).T  # stored one spectrum per column

    meta = pd.DataFrame({"column": [str(c) for c in spectra.columns]})
    # pandas appends .1-.8 to the duplicated headers; those suffixes are replicates
    meta["sample"] = meta["column"].str.replace(r"\.\d+$", "", regex=True)
    meta["rep"] = meta.groupby("sample").cumcount() + 1
    meta["is_fossil"] = meta["sample"].str.startswith("Ayauchi")

    modern = meta["sample"].str.extract(r"^(AG|PC)(\d+)(raw|water|perox)$")
    meta["species_code"] = modern[0]
    meta["species"] = modern[0].map(WDG_SPECIES)
    meta["temp_c"] = pd.to_numeric(modern[1])
    meta["treatment"] = modern[2].map(WDG_TREATMENT)
    meta["depth_cm"] = pd.to_numeric(
        meta["sample"].str.extract(r"^Ayauchi_(\d+)_")[0], errors="coerce"
    )
    meta.loc[meta["is_fossil"], "species"] = "fossil charcoal (Lake Ayauchi)"
    meta.loc[meta["is_fossil"], "treatment"] = "fossil"

    wn, X = _ascending(wn, X)
    return SpectralSet(
        key="wdg",
        label="Gosling — grass/alder + Ayauchi fossil",
        short="Gosling",
        wn=wn,
        X=X,
        meta=_harmonize(meta),
        native_scale="snv",
        citation="Gosling et al. 2019, Palaeogeogr. (doi:10.1016/j.palaeo.2019.01.029); Figshare CC BY 4.0",
        notes="2 species x 6 temperatures x 3 chemical pretreatments (9 replicates each), "
        "plus 31 undated fossil fragments from Lake Ayauchi, Ecuador. Ships SNV-normalized.",
    )


LOADERS = {
    "minatre_ref": load_minatre_reference,
    "minatre_burn": load_minatre_combustion,
    "maezumi": load_maezumi,
    "mccall_multi": load_mccall_multifeedstock,
    "mccall_barley": load_mccall_barley,
    "wdg": load_wdg,
}


def load_all(keys: list[str] | None = None) -> dict[str, SpectralSet]:
    """Load every collection (or a named subset), in archive order."""
    return {k: LOADERS[k]() for k in (keys or LOADERS)}


# --------------------------------------------------------------------------
# Shared preprocessing
# --------------------------------------------------------------------------


def snv(X: np.ndarray) -> np.ndarray:
    """Standard normal variate: centre and scale each spectrum individually.

    The common footing for this project. Maezumi and WDG arrive already in this
    form, so applying it to the absorbance collections and to the Addis filter
    spectra makes all seven sources shape-comparable. It removes the additive
    baseline offset and the multiplicative path-length/loading term, which is
    exactly what separates a KBr pellet of bulk charcoal from micrograms of
    aerosol on a PTFE filter.
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def detrend(X: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """Remove a per-spectrum least-squares linear trend in wavenumber.

    Necessary whenever filter spectra are compared to bulk-material spectra. Even after
    AIRSpec baselining the Addis filter spectra retain a strong positive slope across
    1430-3500 (scattering by the deposit and the PTFE substrate), while the published
    charcoal collections carry a *negative* residual slope. Comparing the two without
    equalizing that first-order term compares baselines, not chemistry -- see
    ``ramp_score``. Apply after cropping to the comparison window and before `snv`.
    """
    A = np.vstack([wn, np.ones_like(wn)]).T
    coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)
    return X - (A @ coef).T


def ramp_score(X: np.ndarray, wn: np.ndarray) -> float:
    """Correlation of the mean spectrum with wavenumber.

    A diagnostic, not a correction. Values near +-1 mean the mean spectrum is a
    featureless slope and any "band" read off it is really a position on that slope;
    values near 0 mean the shape is carried by actual absorption features.
    """
    return float(np.corrcoef(X.mean(axis=0), wn)[0, 1])


def common_grid(step: float = 2.0, window: tuple[float, float] = CHARCOAL_OVERLAP):
    lo, hi = window
    return np.arange(lo, hi + step / 2, step)


def resample(X: np.ndarray, wn_from: np.ndarray, wn_to: np.ndarray) -> np.ndarray:
    """Interpolate onto a shared ascending grid; NaN outside the source range."""
    order = np.argsort(wn_from)
    src = wn_from[order]
    out = np.empty((X.shape[0], wn_to.size), dtype=float)
    for i in range(X.shape[0]):
        out[i] = np.interp(wn_to, src, X[i][order], left=np.nan, right=np.nan)
    return out


def prepare(X: np.ndarray, wn_from: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The canonical preprocessing chain: resample -> detrend -> SNV.

    Every cross-source comparison in char_02/03/04 goes through this, in this order.
    Cropping first keeps the excluded regions out of the detrend and SNV constants;
    detrending before SNV means the normalization is computed on band structure rather
    than on a slope. Call ``ramp_score`` afterwards to confirm it worked.
    """
    return snv(detrend(resample(X, wn_from, grid), grid))


def shape_norm(X: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """Display normalization: shift each spectrum to zero minimum, scale to unit area.

    An absorbance-like alternative to SNV for overlay *figures* only: non-negative, so
    curves read as band intensity rather than z-scores. Min-shift (rather than the
    clip-negatives convention used on AIRSpec-baselined Addis spectra alone) keeps it
    valid for the collections published already in SNV form, which are zero-mean and
    would lose half their shape to clipping. Not used for any statistic — correlations
    are affine-invariant, so the SNV numbers are unaffected by this view.
    """
    Xs = X - np.nanmin(X, axis=1, keepdims=True)
    order = np.argsort(wn)
    a = np.trapezoid(np.nan_to_num(Xs[:, order]), wn[order], axis=1)[:, None]
    a[a == 0] = np.nan
    return Xs / np.abs(a)


def prepare_shape(X: np.ndarray, wn_from: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Companion chain to ``prepare``: resample -> detrend -> shape_norm.

    Identical to the canonical chain except the final scaling is unit area instead of
    unit sd, so companion overlay figures can show the same series in absorbance-like
    units. Applied identically to both sides, like everything else in this folder.
    """
    return shape_norm(detrend(resample(X, wn_from, grid), grid), grid)


def band_area(X: np.ndarray, wn: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    """Mean value across a wavenumber window, per spectrum."""
    mask = (wn >= window[0]) & (wn <= window[1])
    if not mask.any():
        return np.full(X.shape[0], np.nan)
    return np.nanmean(X[:, mask], axis=1)


def band_table(X: np.ndarray, wn: np.ndarray, bands: dict | None = None) -> pd.DataFrame:
    bands = bands or BANDS
    return pd.DataFrame({name: band_area(X, wn, w) for name, w in bands.items()})
