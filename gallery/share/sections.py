"""The gallery's flagged sections (gallery/app/src/sections/sections.json), for the publish scripts.

A section with "public": false is left out of every public build: Vite swaps its code for
src/sections/omitted.ts when VITE_PUBLIC_BUILD=1, and the publish scripts use this module to keep its
data folders (under gallery/app/public/data) out of what they publish. A section with
"public": "unlisted" is published like a public one; only the app's menu leaves its tabs out.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "app/src/sections/sections.json"
PUBLIC_BUILD_ENV = {"VITE_PUBLIC_BUILD": "1"}


def load(manifest: Path = MANIFEST) -> dict[str, dict]:
    return json.loads(manifest.read_text())["sections"]


def private_data_dirs(manifest: Path = MANIFEST) -> list[str]:
    """public/data folders that belong to sections not marked public."""
    return sorted({d.strip("/") for s in load(manifest).values() if not s.get("public") for d in s.get("data", [])})


def private_tabs(manifest: Path = MANIFEST) -> list[str]:
    return sorted({t for s in load(manifest).values() if not s.get("public") for t in s.get("tabs", [])})


def unlisted_sections(manifest: Path = MANIFEST) -> list[str]:
    """Sections published but left out of the public build's menu (reachable from a link)."""
    return sorted(k for k, s in load(manifest).items() if s.get("public") == "unlisted")


def is_private(rel: str | Path, dirs: list[str] | None = None) -> bool:
    """True when a path relative to public/data lies in a private section's folder."""
    parts = Path(rel).as_posix().split("/")
    return any(parts[: len(d.split("/"))] == d.split("/") for d in (private_data_dirs() if dirs is None else dirs))
