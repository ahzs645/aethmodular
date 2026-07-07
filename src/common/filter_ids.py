"""Filter-ID normalization shared across FTIR/HIPS/ChemSpec joins.

Filter IDs come in two shapes: HIPS/FTIR analytical IDs (SITE-0000-N, with a
replicate suffix) and ChemSpec IDs (SITE-0000, no suffix). Joining across
datasets requires collapsing the replicate suffix. Consolidated from copies in
research/ftir_hips_chem.
"""

from __future__ import annotations

import re

_SUFFIX_RE = re.compile(r"^([A-Za-z]+-\d{4})-\d+$")


def base_filter_id(filter_id):
    """Strip a trailing replicate suffix: 'ETAD-0035-3' -> 'ETAD-0035'.

    IDs already in base form (no -N suffix) are returned unchanged. Non-string
    input is coerced to str; None/NaN yield None.
    """
    if filter_id is None:
        return None
    text = str(filter_id).strip()
    if not text or text.lower() == "nan":
        return None
    m = _SUFFIX_RE.match(text)
    return m.group(1) if m else text


def normalize_filter_id(filter_id):
    """Collapse any hyphen-suffixed ID to its first two dash-parts.

    Looser than base_filter_id: 'ETAD-0035-3-extra' -> 'ETAD-0035'. Use this
    when suffixes are irregular; use base_filter_id when the SITE-0000-N shape
    is guaranteed. None/NaN/empty yield None.
    """
    if filter_id is None:
        return None
    text = str(filter_id).strip()
    if not text or text.lower() == "nan":
        return None
    parts = text.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return text
