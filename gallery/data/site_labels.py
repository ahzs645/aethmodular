"""Readable labels for compact place names in the SPARTAN lookup export."""
import re

# Keep legitimate names/acronyms intact instead of splitting every capital letter.
_COMPACT_PLACES = {
    "AbuDhabi": "Abu Dhabi",
    "AddisAbaba": "Addis Ababa",
    "MexicoCity": "Mexico City",
    "PuertoRico": "Puerto Rico",
    "UnitedStates": "United States",
    "StLouis": "St Louis",
}
_COMPACT = re.compile(r"\b(" + "|".join(_COMPACT_PLACES) + r")\b")


def display_site_name(code: str, *candidates) -> str:
    """Prefer a meaningful public/map name over a missing/code-only lookup name."""
    for value in candidates:
        if value is None:
            continue
        name = str(value).strip()
        if not name or name.lower() == "nan" or name == code:
            continue
        return _COMPACT.sub(lambda match: _COMPACT_PLACES[match[0]], name)
    return code
