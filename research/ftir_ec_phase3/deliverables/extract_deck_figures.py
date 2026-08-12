"""Extract the base64-embedded figures from a standalone deck back to PNGs.

The Aug 13 deck is the only surviving copy of its 14 plots -- no builder produces
them and the upstream data sits behind a Drive mount (see README.md). Keeping the
PNGs out of git is fine precisely because this script puts them back on demand.

Figures are named from each ``<img>``'s ``alt`` text, so the output is readable
without opening the deck:

    fig09_addis_residuals_vs_score_space_extrapolation_distance_raw_vs.png

Usage
-----
    uv run python research/ftir_ec_phase3/deliverables/extract_deck_figures.py
    uv run python .../extract_deck_figures.py --deck other.html --out /tmp/figs
"""

from __future__ import annotations

import argparse
import base64
import binascii
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DECK = HERE / "satoshi_deck_2026-08-13.html"
DEFAULT_OUT = HERE / "figures"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# alt and src appear in either order depending on how the tag was written, so
# match the whole tag once and pull the attributes out separately.
IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
ALT_ATTR = re.compile(r'\balt="([^"]*)"', re.IGNORECASE)
SRC_B64 = re.compile(r'\bsrc="data:image/png;base64,([A-Za-z0-9+/=]+)"', re.IGNORECASE)


def slugify(alt: str, index: int) -> str:
    """Return ``figNN_<alt-slug>.png``, falling back when a figure has no alt."""
    slug = re.sub(r"[^a-z0-9]+", "_", alt.lower()).strip("_")[:60]
    return f"fig{index:02d}_{slug or 'untitled'}.png"


def extract(deck: Path, out_dir: Path) -> int:
    """Write every embedded PNG in ``deck`` to ``out_dir``; return the count."""
    html = deck.read_text(errors="replace")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, tag in enumerate(IMG_TAG.findall(html), start=1):
        src = SRC_B64.search(tag)
        if src is None:  # an <img> that is not an embedded PNG
            continue
        alt_match = ALT_ATTR.search(tag)
        alt = alt_match.group(1) if alt_match else ""

        try:
            raw = base64.b64decode(src.group(1), validate=True)
        except binascii.Error as exc:
            print(f"  fig{index:02d}: base64 did not decode ({exc})", file=sys.stderr)
            continue
        if not raw.startswith(PNG_MAGIC):
            print(f"  fig{index:02d}: decoded bytes are not a PNG", file=sys.stderr)
            continue

        path = out_dir / slugify(alt, index)
        path.write_bytes(raw)
        written += 1
        print(f"  {path.name}  ({len(raw) / 1024:.0f} KB)")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.deck.is_file():
        parser.error(f"deck not found: {args.deck}")

    print(f"extracting from {args.deck.name}")
    count = extract(args.deck, args.out)
    print(f"\nwrote {count} figure(s) to {args.out}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
