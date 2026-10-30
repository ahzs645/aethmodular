"""What every ML page module shares: the research exporter's Page, its readers and folders, and the ML output folder."""
from __future__ import annotations

from export_research_spectral import CHAR, PHASE3, ROOT, Page, long, number, rel  # noqa: F401

OUT = ROOT / "gallery/app/public/data/ml"
HIPS_CHEM = ROOT / "research/ftir_hips_chem"
JUNE = ROOT / "research/ftir_ec_calibration_2026_06_25"
SPARTAN_EC = ROOT / "research/spartan_ec_2026_06_16"


def page(key: str, title: str) -> Page:
    """A page that writes to gallery/app/public/data/ml/<key>.json."""
    return Page(key, title, out=OUT, exporter=ROOT / "gallery/data/export_ml.py")
