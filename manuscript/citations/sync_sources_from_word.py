#!/usr/bin/env python3
"""Check or synchronize Word bibliography sources and canonical CSL-JSON."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from word_sources import (
    canonical_json,
    load_csl_json,
    normalized_references,
    read_word_sources,
    update_word_sources_xml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or synchronize embedded Word sources and CSL-JSON."
    )
    parser.add_argument("docx", type=Path, help="DOCX with embedded Word sources")
    parser.add_argument(
        "--direction",
        choices=("check", "word-to-csl", "csl-to-word"),
        default="check",
        help="Default is a non-mutating drift check",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).with_name("references.csl.json"),
    )
    parser.add_argument(
        "--bib",
        type=Path,
        default=Path(__file__).with_name("references.bib"),
    )
    parser.add_argument(
        "--output-docx",
        type=Path,
        help="Required output for csl-to-word; the input is never overwritten",
    )
    parser.add_argument("--pandoc", default=shutil.which("pandoc") or "pandoc")
    return parser.parse_args()


def _copy_zip_info(item: ZipInfo) -> ZipInfo:
    copied = ZipInfo(item.filename, date_time=item.date_time)
    copied.compress_type = item.compress_type
    copied.comment = item.comment
    copied.extra = item.extra
    copied.internal_attr = item.internal_attr
    copied.external_attr = item.external_attr
    copied.create_system = item.create_system
    return copied


def write_docx_part(
    input_path: Path, output_path: Path, part_name: str, part_bytes: bytes
) -> None:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output DOCX paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".docx", delete=False
    ) as handle:
        temporary_output = Path(handle.name)
    try:
        with ZipFile(input_path) as source, ZipFile(
            temporary_output, "w", compression=ZIP_DEFLATED
        ) as target:
            for item in source.infolist():
                data = part_bytes if item.filename == part_name else source.read(item)
                target.writestr(_copy_zip_info(item), data)
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)


def assert_sources_match(word_references, csl_references) -> None:
    if normalized_references(word_references) == normalized_references(csl_references):
        return
    word_ids = {str(item.get("id")) for item in word_references}
    csl_ids = {str(item.get("id")) for item in csl_references}
    details = []
    if word_ids - csl_ids:
        details.append("Word-only IDs: " + ", ".join(sorted(word_ids - csl_ids)))
    if csl_ids - word_ids:
        details.append("CSL-only IDs: " + ", ".join(sorted(csl_ids - word_ids)))
    if not details:
        details.append("The IDs match, but source metadata differs")
    raise RuntimeError(
        "Embedded Word sources have drifted from canonical CSL-JSON. "
        + " ".join(details)
        + ". Use --direction csl-to-word to create a reconciled Word copy, "
        "or explicitly use word-to-csl if Word should replace the canonical library."
    )


def export_word_sources(
    references, json_path: Path, bib_path: Path, pandoc: str
) -> None:
    json_path = json_path.expanduser().resolve()
    bib_path = bib_path.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    bib_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aeth-source-sync-") as temp:
        temporary_json = Path(temp) / "references.csl.json"
        temporary_bib = Path(temp) / "references.bib"
        temporary_json.write_text(canonical_json(references), encoding="utf-8")
        subprocess.run(
            [
                pandoc,
                str(temporary_json),
                "--from=csljson",
                "--to=biblatex",
                "--output=%s" % temporary_bib,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        json_stage = json_path.with_suffix(json_path.suffix + ".tmp")
        bib_stage = bib_path.with_suffix(bib_path.suffix + ".tmp")
        shutil.copyfile(temporary_json, json_stage)
        shutil.copyfile(temporary_bib, bib_stage)
        os.replace(json_stage, json_path)
        os.replace(bib_stage, bib_path)


def main() -> None:
    args = parse_args()
    docx_path = args.docx.expanduser().resolve()
    json_path = args.json.expanduser().resolve()
    with ZipFile(docx_path) as archive:
        part_name, source_xml, word_references = read_word_sources(archive)

    if args.direction == "word-to-csl":
        export_word_sources(word_references, json_path, args.bib, args.pandoc)
        print(
            "Exported %d Word sources to %s and %s"
            % (len(word_references), json_path, args.bib.expanduser().resolve())
        )
        return

    csl_references = load_csl_json(json_path)
    if args.direction == "check":
        assert_sources_match(word_references, csl_references)
        print("Word and CSL source metadata match (%d sources)" % len(csl_references))
        return

    if args.output_docx is None:
        raise ValueError("--output-docx is required for --direction csl-to-word")
    updated_sources = update_word_sources_xml(source_xml, csl_references)
    write_docx_part(docx_path, args.output_docx, part_name, updated_sources)
    print(
        "Wrote %d canonical CSL sources to %s"
        % (len(csl_references), args.output_docx.expanduser().resolve())
    )


if __name__ == "__main__":
    main()
