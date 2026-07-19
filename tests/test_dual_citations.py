from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest


CITATION_DIR = Path(__file__).parents[1] / "manuscript" / "citations"
sys.path.insert(0, str(CITATION_DIR))

import build_dual_citations as build
from word_sources import B, B_NS, load_csl_json, read_word_sources, update_word_sources_xml


PANDOC = shutil.which("pandoc")
SUPERSCRIPT_STYLE = (
    Path(__file__).parent / "fixtures" / "citations" / "superscript-numeric.csl"
)


def _field_run(field_type: str) -> ET.Element:
    run = build.make_run(field_type=field_type)
    field_char = run.find(build.W + "fldChar")
    assert field_char is not None
    field_char.attrib.pop(build.W + "fldLock", None)
    field_char.attrib.pop(build.W + "dirty", None)
    return run


def _replace_paragraph_with_field(
    paragraph: ET.Element, instruction: str, result: str
) -> None:
    paragraph_properties = paragraph.find(build.W + "pPr")
    for child in list(paragraph):
        if child is not paragraph_properties:
            paragraph.remove(child)
    paragraph.extend(
        [
            _field_run("begin"),
            build.make_instruction_run(instruction),
            _field_run("separate"),
            build.make_run(result),
            _field_run("end"),
        ]
    )


def _reference() -> dict[str, object]:
    return {
        "id": "Example2024",
        "type": "article-journal",
        "title": "A connected citation test",
        "author": [{"family": "Example", "given": "Ada"}],
        "container-title": "Test Journal",
        "volume": "2",
        "issue": "1",
        "page": "1-9",
        "DOI": "10.5555/example.2024",
        "URL": "https://doi.org/10.5555/example.2024",
        "issued": {"date-parts": [[2024]]},
    }


@pytest.fixture()
def connected_docx(tmp_path: Path) -> tuple[Path, Path]:
    if PANDOC is None:
        pytest.skip("Pandoc is required for DOCX citation integration tests")
    markdown = tmp_path / "source.md"
    base_docx = tmp_path / "base.docx"
    connected = tmp_path / "connected.docx"
    bibliography = tmp_path / "references.csl.json"
    markdown.write_text(
        "Before citation CITATION_PLACEHOLDER after.\n\nBIBLIOGRAPHY_PLACEHOLDER\n",
        encoding="utf-8",
    )
    subprocess.run(
        [PANDOC, str(markdown), "--output", str(base_docx)],
        check=True,
        capture_output=True,
        text=True,
    )
    with ZipFile(base_docx) as archive:
        document_xml = archive.read("word/document.xml")
    namespaces = build.capture_namespaces(document_xml)
    root = ET.fromstring(document_xml)
    for paragraph in root.iter(build.W + "p"):
        text = build.paragraph_text(paragraph)
        if "CITATION_PLACEHOLDER" in text:
            _replace_paragraph_with_field(
                paragraph,
                r"CITATION \l 1033 Example2024",
                " (Example, 2024)",
            )
        elif "BIBLIOGRAPHY_PLACEHOLDER" in text:
            _replace_paragraph_with_field(paragraph, "BIBLIOGRAPHY", "Example (2024)")
    empty_sources = ET.tostring(
        ET.Element(B + "Sources"), encoding="UTF-8", xml_declaration=True
    )
    reference = _reference()
    source_xml = update_word_sources_xml(empty_sources, [reference])
    bibliography.write_text(json.dumps([reference], indent=2) + "\n", encoding="utf-8")
    build.write_docx(
        base_docx,
        connected,
        {
            "word/document.xml": build.serialize_xml(root, namespaces),
            "customXml/item7.xml": source_xml,
        },
    )
    return connected, bibliography


def test_citation_parser_preserves_cluster_switches() -> None:
    instruction = (
        r'CITATION \l 1033 First2020 \f "see" \p "12-14" '
        r'\s "especially Fig. 2" \m Second2021 \n \t'
    )
    markdown = build.citation_markdown(instruction, {"First2020", "Second2021"})
    assert markdown == (
        "[see @First2020, pp. 12-14 especially Fig. 2; -@Second2021]"
    )


def test_find_word_sources_without_assuming_item1(connected_docx) -> None:
    docx, _bibliography = connected_docx
    with ZipFile(docx) as archive:
        part_name, _xml, references = read_word_sources(archive)
    assert part_name == "customXml/item7.xml"
    assert references == [_reference()]


def test_csl_to_word_round_trip_preserves_current_metadata(connected_docx) -> None:
    docx, bibliography = connected_docx
    references = load_csl_json(bibliography)
    with ZipFile(docx) as archive:
        _part_name, source_xml, _word_references = read_word_sources(archive)
    updated = update_word_sources_xml(source_xml, references)
    temporary_docx = docx.with_name("round-trip.docx")
    build.write_docx(docx, temporary_docx, {"customXml/item7.xml": updated})
    with ZipFile(temporary_docx) as archive:
        _part_name, _source_xml, round_tripped = read_word_sources(archive)
    assert round_tripped == references


@pytest.mark.skipif(PANDOC is None, reason="Pandoc is required")
def test_superscript_csl_and_hyperlinks_survive_docx_build(connected_docx) -> None:
    docx, bibliography = connected_docx
    output = docx.with_name("superscript.docx")
    manifest = docx.with_name("superscript.citation-manifest.json")
    subprocess.run(
        [
            sys.executable,
            str(CITATION_DIR / "build_dual_citations.py"),
            "--mode",
            "csl",
            "--input",
            str(docx),
            "--output",
            str(output),
            "--bibliography",
            str(bibliography),
            "--csl",
            str(SUPERSCRIPT_STYLE),
            "--pandoc",
            PANDOC,
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        document_xml = archive.read("word/document.xml")
        root = ET.fromstring(document_xml)
        relationship_root = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
    assert root.find(".//" + build.W + "vertAlign") is not None
    relationship_ids = {
        node.get("Id") for node in relationship_root.findall(build.REL + "Relationship")
    }
    referenced_ids = {
        node.get(build.R + "id")
        for node in root.iter()
        if node.get(build.R + "id") is not None
    }
    assert referenced_ids <= relationship_ids
    targets = {
        node.get("Target")
        for node in relationship_root.findall(build.REL + "Relationship")
        if node.get("Type", "").endswith("/hyperlink")
    }
    assert "https://doi.org/10.5555/example.2024" in targets
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["processor"].startswith("pandoc ")
    assert manifest_data["citations"][0]["items"] == [{"id": "Example2024"}]
    assert manifest_data["output"]["sha256"] == build.sha256_file(output)


def test_serialization_declares_every_ignorable_prefix() -> None:
    xml = (
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        b'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        b'xmlns:w16="urn:test:w16" mc:Ignorable="w16"><w:body/></w:document>'
    )
    namespaces = build.capture_namespaces(xml)
    serialized = build.serialize_xml(ET.fromstring(xml), namespaces)
    start_tag = serialized.split(b"\n", 1)[1].split(b">", 1)[0].decode("utf-8")
    ignorable = re.search(r'mc:Ignorable="([^"]+)"', start_tag)
    assert ignorable is not None
    declared = set(re.findall(r"xmlns:([\w.-]+)=", start_tag))
    assert set(ignorable.group(1).split()) <= declared


def test_include_uncited_is_opt_in(tmp_path: Path) -> None:
    with_uncited = []
    without_uncited = []
    original_run = build.subprocess.run

    def fake_run(command, **kwargs):
        markdown = Path(command[1]).read_text(encoding="utf-8")
        (with_uncited if "nocite" in markdown else without_uncited).append(markdown)
        raise subprocess.CalledProcessError(1, command)

    build.subprocess.run = fake_run
    try:
        with pytest.raises(subprocess.CalledProcessError):
            build.run_citeproc(
                Path("in.docx"),
                Path("refs.json"),
                Path("style.csl"),
                ["[@Example2024]"],
                "pandoc",
                tmp_path,
            )
        with pytest.raises(subprocess.CalledProcessError):
            build.run_citeproc(
                Path("in.docx"),
                Path("refs.json"),
                Path("style.csl"),
                ["[@Example2024]"],
                "pandoc",
                tmp_path,
                include_uncited=True,
            )
    finally:
        build.subprocess.run = original_run
    assert without_uncited and with_uncited
