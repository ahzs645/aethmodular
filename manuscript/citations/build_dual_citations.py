#!/usr/bin/env python3
"""Build Word-native or CSL-styled copies of a connected-citation DOCX.

CSL-JSON is the canonical source library. Word mode exports that library into
Microsoft bibliography sources and unlocks native fields. CSL mode verifies
that Word's embedded sources have not drifted, renders the selected CSL style,
and locks the native fields beneath the rendered results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Set, Tuple
from xml.sax.saxutils import quoteattr
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from word_sources import (
    load_csl_json,
    normalized_references,
    read_word_sources,
    update_word_sources_xml,
)


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
CSL_NS = "http://purl.org/net/xbiblio/csl"
XML_NS = "http://www.w3.org/XML/1998/namespace"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
W = "{%s}" % W_NS
R = "{%s}" % R_NS
REL = "{%s}" % PKG_REL_NS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Word-native or CSL-styled citation DOCX."
    )
    parser.add_argument("--input", required=True, type=Path, help="Connected source DOCX")
    parser.add_argument("--output", required=True, type=Path, help="Output DOCX")
    parser.add_argument(
        "--mode",
        choices=("word", "csl"),
        default="csl",
        help="word unlocks native Word fields; csl renders and locks CSL text",
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=Path(__file__).with_name("references.csl.json"),
        help="Canonical CSL-JSON source library",
    )
    parser.add_argument(
        "--csl",
        type=Path,
        default=Path(__file__).with_name("styles")
        / "atmospheric-measurement-techniques.csl",
        help="CSL style; dependent styles are resolved from the same directory",
    )
    parser.add_argument(
        "--include-uncited",
        action="store_true",
        help="Include the entire source library in the bibliography",
    )
    parser.add_argument(
        "--bibliography-alignment",
        choices=("style", "left"),
        default="style",
        help="Preserve the reference-doc style or explicitly left-align bibliography entries",
    )
    parser.add_argument(
        "--pandoc",
        default=shutil.which("pandoc") or "pandoc",
        help="Pandoc executable used for citeproc rendering",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON build manifest with citation occurrences and input hashes",
    )
    return parser.parse_args()


def capture_namespaces(xml_bytes: bytes) -> Dict[str, str]:
    namespaces: Dict[str, str] = {}
    for _event, (prefix, uri) in ET.iterparse(io.BytesIO(xml_bytes), events=("start-ns",)):
        prefix = prefix or ""
        namespaces[prefix] = uri
        if prefix not in ("xml", "xmlns"):
            ET.register_namespace(prefix, uri)
    return namespaces


def serialize_xml(root: ET.Element, namespaces: Dict[str, str]) -> bytes:
    """Serialize while retaining declarations referenced only by mc:Ignorable."""

    rendered = ET.tostring(root, encoding="unicode", xml_declaration=False)
    close = rendered.find(">")
    if close < 0:
        raise ValueError("Serialized XML has no document element")
    start_tag = rendered[:close]
    declared = {
        match.group(1) or ""
        for match in re.finditer(r"\sxmlns(?::([A-Za-z_][\w.-]*))?=", start_tag)
    }
    declarations = []
    for prefix, uri in namespaces.items():
        if prefix in ("xml", "xmlns") or prefix in declared:
            continue
        name = "xmlns" if not prefix else "xmlns:%s" % prefix
        declarations.append(" %s=%s" % (name, quoteattr(uri)))
    if declarations:
        rendered = rendered[:close] + "".join(declarations) + rendered[close:]
    declaration = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    return (declaration + rendered).encode("UTF-8")


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(W + "t"))


def field_char_type(element: ET.Element) -> Optional[str]:
    field_char = element.find(".//" + W + "fldChar")
    return None if field_char is None else field_char.get(W + "fldCharType")


def field_spans(paragraph: ET.Element) -> List[Dict[str, object]]:
    children = list(paragraph)
    spans: List[Dict[str, object]] = []
    start: Optional[int] = None
    separate: Optional[int] = None
    depth = 0
    for index, child in enumerate(children):
        char_type = field_char_type(child)
        if char_type == "begin":
            if depth == 0:
                start = index
                separate = None
            depth += 1
        elif char_type == "separate" and depth == 1:
            separate = index
        elif char_type == "end" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                instruction = "".join(
                    node.text or ""
                    for child_part in children[start : index + 1]
                    for node in child_part.iter(W + "instrText")
                ).strip()
                spans.append(
                    {
                        "start": start,
                        "separate": separate,
                        "end": index,
                        "instruction": instruction,
                    }
                )
                start = None
    return spans


def citation_fields(root: ET.Element) -> List[Tuple[ET.Element, Dict[str, object]]]:
    fields: List[Tuple[ET.Element, Dict[str, object]]] = []
    for paragraph in root.iter(W + "p"):
        for span in field_spans(paragraph):
            if str(span["instruction"]).startswith("CITATION "):
                fields.append((paragraph, span))
    return fields


def _field_tokens(instruction: str) -> List[str]:
    tokens = re.findall(r'"(?:[^"]|"")*"|\\[A-Za-z*]+|[^\s]+', instruction)
    return [token[1:-1].replace('""', '"') if token.startswith('"') else token for token in tokens]


def parse_citation_items(
    instruction: str, reference_ids: Set[str]
) -> List[Dict[str, object]]:
    tokens = _field_tokens(instruction)
    if not tokens or tokens[0] != "CITATION":
        raise ValueError("Not a Word CITATION field: %s" % instruction)
    items: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in reference_ids:
            current = {"id": token}
            items.append(current)
            index += 1
            continue
        if token == "\\m":
            if index + 1 >= len(tokens) or tokens[index + 1] not in reference_ids:
                raise ValueError("Citation \\m switch has no valid source: %s" % instruction)
            current = {"id": tokens[index + 1]}
            items.append(current)
            index += 2
            continue
        if token in ("\\l", "\\v", "\\f", "\\s", "\\p"):
            if index + 1 >= len(tokens):
                raise ValueError("Citation switch %s has no argument: %s" % (token, instruction))
            value = tokens[index + 1]
            if token != "\\l":
                if current is None:
                    raise ValueError("Citation switch %s precedes a source: %s" % (token, instruction))
                names = {
                    "\\v": "volume",
                    "\\f": "prefix",
                    "\\s": "suffix",
                    "\\p": "locator",
                }
                current[names[token]] = value
            index += 2
            continue
        if token in ("\\n", "\\t"):
            if current is None:
                raise ValueError("Citation switch %s precedes a source: %s" % (token, instruction))
            if token == "\\n":
                current["suppress_author"] = True
            index += 1
            continue
        if token == "\\*":
            index += 2
            continue
        if token == "\\y":
            raise ValueError("Word's suppress-year switch is not representable in Pandoc citations")
        raise ValueError("Unsupported token %r in citation field: %s" % (token, instruction))
    if not items:
        raise ValueError("No source key found in field: %s" % instruction)
    return items


def citation_markdown(instruction: str, reference_ids: Set[str]) -> str:
    rendered_items = []
    for item in parse_citation_items(instruction, reference_ids):
        citation = "-@" if item.get("suppress_author") else "@"
        citation += str(item["id"])
        prefix = str(item.get("prefix") or "").strip()
        suffix = str(item.get("suffix") or "").strip()
        locator = str(item.get("locator") or "").strip()
        volume = str(item.get("volume") or "").strip()
        if prefix:
            citation = prefix + " " + citation
        if locator:
            label = "pp." if re.search(r"[-–,]", locator) else "p."
            citation += ", %s %s" % (label, locator)
        elif volume:
            citation += ", vol. %s" % volume
        if suffix:
            citation += " " + suffix
        rendered_items.append(citation)
    return "[" + "; ".join(rendered_items) + "]"


def resolve_independent_style(style_path: Path) -> Path:
    style_root = ET.parse(style_path).getroot()
    if style_root.find("{%s}citation" % CSL_NS) is not None:
        return style_path
    for link in style_root.findall(".//{%s}link" % CSL_NS):
        if link.get("rel") != "independent-parent":
            continue
        href = link.get("href", "")
        slug = href.rstrip("/").split("/")[-1]
        parent = style_path.with_name(slug + ".csl")
        if parent.exists():
            return parent
        raise FileNotFoundError(
            "Dependent style requires %s in %s" % (parent.name, style_path.parent)
        )
    raise ValueError("CSL style has no citation definition or independent parent: %s" % style_path)


def _inline_after_marker(paragraph: ET.Element, marker: str) -> List[ET.Element]:
    children = [
        copy.deepcopy(child) for child in list(paragraph) if child.tag != W + "pPr"
    ]
    text_nodes = [node for child in children for node in child.iter(W + "t")]
    full_text = "".join(node.text or "" for node in text_nodes)
    match = re.match(re.escape(marker) + r"\s*", full_text)
    if match is None:
        raise RuntimeError("Citeproc marker %s was not found at paragraph start" % marker)
    remaining = match.end()
    for node in text_nodes:
        value = node.text or ""
        if remaining >= len(value):
            node.text = ""
            remaining -= len(value)
        else:
            node.text = value[remaining:]
            remaining = 0
    if remaining:
        raise RuntimeError("Citeproc marker %s could not be removed" % marker)
    return children


def run_citeproc(
    input_docx: Path,
    bibliography: Path,
    style: Path,
    citation_syntax: List[str],
    pandoc: str,
    temp_dir: Path,
    include_uncited: bool = False,
) -> Tuple[Dict[str, List[ET.Element]], List[ET.Element], ET.Element, Dict[str, str]]:
    markdown_path = temp_dir / "citation-render.md"
    rendered_docx = temp_dir / "citation-render.docx"
    lines = ["---", "nocite: |", "  @*", "---", ""] if include_uncited else []
    for index, syntax in enumerate(citation_syntax):
        lines.extend(["CID%03d%s" % (index, syntax), ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    command = [
        pandoc,
        str(markdown_path),
        "--from=markdown",
        "--to=docx",
        "--citeproc",
        "--bibliography=%s" % bibliography,
        "--csl=%s" % style,
        "--reference-doc=%s" % input_docx,
        "--output=%s" % rendered_docx,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with ZipFile(rendered_docx) as archive:
        generated_xml = archive.read("word/document.xml")
        generated_relationships_xml = archive.read("word/_rels/document.xml.rels")
    generated_namespaces = capture_namespaces(generated_xml)
    generated_root = ET.fromstring(generated_xml)
    generated_relationships = ET.fromstring(generated_relationships_xml)
    rendered_citations: Dict[str, List[ET.Element]] = {}
    bibliography_paragraphs: List[ET.Element] = []
    for paragraph in generated_root.iter(W + "p"):
        text = paragraph_text(paragraph)
        marker_match = re.match(r"^(CID\d{3})", text)
        if marker_match:
            marker = marker_match.group(1)
            rendered_citations[marker] = _inline_after_marker(paragraph, marker)
        paragraph_style = paragraph.find("./" + W + "pPr/" + W + "pStyle")
        if paragraph_style is not None and paragraph_style.get(W + "val") == "Bibliography":
            bibliography_paragraphs.append(copy.deepcopy(paragraph))
    if len(rendered_citations) != len(citation_syntax):
        raise RuntimeError(
            "Citeproc returned %d of %d citations"
            % (len(rendered_citations), len(citation_syntax))
        )
    if not bibliography_paragraphs:
        raise RuntimeError("Citeproc did not return Bibliography paragraphs")
    return (
        rendered_citations,
        bibliography_paragraphs,
        generated_relationships,
        generated_namespaces,
    )


def make_run(text: Optional[str] = None, field_type: Optional[str] = None) -> ET.Element:
    run = ET.Element(W + "r")
    if field_type:
        field_char = ET.SubElement(run, W + "fldChar")
        field_char.set(W + "fldCharType", field_type)
        if field_type == "begin":
            field_char.set(W + "fldLock", "true")
            field_char.set(W + "dirty", "false")
    elif text is not None:
        text_node = ET.SubElement(run, W + "t")
        text_node.set("{%s}space" % XML_NS, "preserve")
        text_node.text = text
    return run


def make_instruction_run(instruction: str) -> ET.Element:
    run = ET.Element(W + "r")
    node = ET.SubElement(run, W + "instrText")
    node.set("{%s}space" % XML_NS, "preserve")
    node.text = " %s " % instruction
    return run


def replace_citation_results(
    root: ET.Element,
    rendered: Dict[str, List[ET.Element]],
    start_index: int = 0,
) -> int:
    fields = citation_fields(root)
    paragraph_indices: Dict[int, Tuple[ET.Element, List[int]]] = {}
    for offset, (paragraph, _span) in enumerate(fields):
        key = id(paragraph)
        if key not in paragraph_indices:
            paragraph_indices[key] = (paragraph, [])
        paragraph_indices[key][1].append(start_index + offset)
    for paragraph, indices in paragraph_indices.values():
        spans = [
            span
            for span in field_spans(paragraph)
            if str(span["instruction"]).startswith("CITATION ")
        ]
        if len(spans) != len(indices):
            raise RuntimeError("Citation-field indexing changed during replacement")
        for index, current in reversed(list(zip(indices, spans))):
            start = int(current["start"])
            separate = current["separate"]
            end = int(current["end"])
            if separate is None:
                raise ValueError("Citation field has no separator: %s" % current["instruction"])
            children = list(paragraph)
            original_result = "".join(
                node.text or ""
                for child in children[int(separate) + 1 : end]
                for node in child.iter(W + "t")
            )
            leading_space_match = re.match(r"^\s*", original_result)
            leading_space = leading_space_match.group(0) if leading_space_match else ""
            begin_char = children[start].find(".//" + W + "fldChar")
            if begin_char is not None:
                begin_char.set(W + "fldLock", "true")
                begin_char.set(W + "dirty", "false")
            for child in children[int(separate) + 1 : end]:
                paragraph.remove(child)
            insertion_index = int(separate) + 1
            if leading_space:
                paragraph.insert(insertion_index, make_run(leading_space))
                insertion_index += 1
            for element in rendered["CID%03d" % index]:
                paragraph.insert(insertion_index, copy.deepcopy(element))
                insertion_index += 1
    return start_index + len(fields)


def bibliography_span(body: ET.Element) -> Tuple[int, int, str]:
    children = list(body)
    start: Optional[int] = None
    instruction = ""
    depth = 0
    separated = False
    for index, child in enumerate(children):
        for element in child.iter():
            char_type = (
                element.get(W + "fldCharType") if element.tag == W + "fldChar" else None
            )
            if char_type == "begin":
                if depth == 0:
                    instruction = ""
                    separated = False
                depth += 1
            elif element.tag == W + "instrText" and depth:
                instruction += element.text or ""
                if "BIBLIOGRAPHY" in instruction and start is None:
                    start = index
            elif char_type == "separate" and start is not None:
                separated = True
            elif char_type == "end" and depth:
                depth -= 1
                if start is not None and separated and depth == 0:
                    return start, index, instruction.strip()
    raise ValueError("BIBLIOGRAPHY field not found")


def replace_bibliography(
    root: ET.Element,
    bibliography_paragraphs: List[ET.Element],
    alignment: str = "style",
) -> None:
    body = root.find(W + "body")
    if body is None:
        raise ValueError("DOCX document body is missing")
    start, end, instruction = bibliography_span(body)
    for child in list(body)[start : end + 1]:
        body.remove(child)
    paragraphs = [copy.deepcopy(paragraph) for paragraph in bibliography_paragraphs]
    if alignment == "left":
        for paragraph in paragraphs:
            paragraph_properties = paragraph.find("./" + W + "pPr")
            if paragraph_properties is None:
                paragraph_properties = ET.Element(W + "pPr")
                paragraph.insert(0, paragraph_properties)
            for justification in paragraph_properties.findall(W + "jc"):
                paragraph_properties.remove(justification)
            justification = ET.SubElement(paragraph_properties, W + "jc")
            justification.set(W + "val", "left")
    first = paragraphs[0]
    insertion_index = 1 if first.find("./" + W + "pPr") is not None else 0
    first.insert(insertion_index, make_run(field_type="begin"))
    first.insert(insertion_index + 1, make_instruction_run(instruction))
    first.insert(insertion_index + 2, make_run(field_type="separate"))
    paragraphs[-1].append(make_run(field_type="end"))
    for offset, paragraph in enumerate(paragraphs):
        body.insert(start + offset, paragraph)


def unlock_word_fields(root: ET.Element) -> None:
    for paragraph in root.iter(W + "p"):
        for span in field_spans(paragraph):
            instruction = str(span["instruction"])
            if not (
                instruction.startswith("CITATION ")
                or instruction.startswith("BIBLIOGRAPHY")
            ):
                continue
            begin = list(paragraph)[int(span["start"])].find(".//" + W + "fldChar")
            if begin is not None:
                begin.attrib.pop(W + "fldLock", None)
                begin.set(W + "dirty", "true")
    body = root.find(W + "body")
    if body is not None:
        try:
            start, _end, _instruction = bibliography_span(body)
            first_paragraph = list(body)[start]
            begin = next(
                node
                for node in first_paragraph.iter(W + "fldChar")
                if node.get(W + "fldCharType") == "begin"
            )
            begin.attrib.pop(W + "fldLock", None)
            begin.set(W + "dirty", "true")
        except (ValueError, StopIteration):
            pass


def relationship_part_name(story_part: str) -> str:
    path = Path(story_part)
    return str(path.parent / "_rels" / (path.name + ".rels"))


def _relationship_references(elements: Iterable[ET.Element]) -> Set[str]:
    references: Set[str] = set()
    for element in elements:
        for node in element.iter():
            for name, value in node.attrib.items():
                if name.startswith(R) and value.startswith("rId"):
                    references.add(value)
    return references


def merge_external_relationships(
    destination_xml: Optional[bytes],
    generated_root: ET.Element,
    elements: Iterable[ET.Element],
) -> Optional[bytes]:
    elements = list(elements)
    source_ids = _relationship_references(elements)
    if not source_ids:
        return destination_xml
    if destination_xml is None:
        destination_root = ET.Element(REL + "Relationships")
        destination_namespaces = {"": PKG_REL_NS}
    else:
        destination_namespaces = capture_namespaces(destination_xml)
        destination_root = ET.fromstring(destination_xml)
    generated_by_id = {
        relationship.get("Id"): relationship
        for relationship in generated_root.findall(REL + "Relationship")
    }
    existing_by_signature = {
        (
            relationship.get("Type"),
            relationship.get("Target"),
            relationship.get("TargetMode"),
        ): relationship.get("Id")
        for relationship in destination_root.findall(REL + "Relationship")
    }
    used_ids = {
        relationship.get("Id")
        for relationship in destination_root.findall(REL + "Relationship")
    }
    next_number = 1
    remapping: Dict[str, str] = {}
    for source_id in sorted(source_ids):
        relationship = generated_by_id.get(source_id)
        if relationship is None:
            raise RuntimeError("Generated relationship %s is missing" % source_id)
        if relationship.get("TargetMode") != "External":
            raise RuntimeError(
                "Cannot transplant internal generated relationship %s" % source_id
            )
        signature = (
            relationship.get("Type"),
            relationship.get("Target"),
            relationship.get("TargetMode"),
        )
        destination_id = existing_by_signature.get(signature)
        if destination_id is None:
            while "rId%d" % next_number in used_ids:
                next_number += 1
            destination_id = "rId%d" % next_number
            used_ids.add(destination_id)
            copied = copy.deepcopy(relationship)
            copied.set("Id", destination_id)
            destination_root.append(copied)
            existing_by_signature[signature] = destination_id
        remapping[source_id] = destination_id
    for element in elements:
        for node in element.iter():
            for name, value in list(node.attrib.items()):
                if name.startswith(R) and value in remapping:
                    node.set(name, remapping[value])
    return serialize_xml(destination_root, destination_namespaces)


def _copy_zip_info(item: ZipInfo) -> ZipInfo:
    copied = ZipInfo(item.filename, date_time=item.date_time)
    copied.compress_type = item.compress_type
    copied.comment = item.comment
    copied.extra = item.extra
    copied.internal_attr = item.internal_attr
    copied.external_attr = item.external_attr
    copied.create_system = item.create_system
    return copied


def write_docx(
    input_path: Path, output_path: Path, replacements: Dict[str, bytes]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output DOCX paths must differ")
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".docx", delete=False
    ) as handle:
        temporary_output = Path(handle.name)
    try:
        with ZipFile(input_path) as source, ZipFile(
            temporary_output, "w", compression=ZIP_DEFLATED
        ) as target:
            written = set()
            for item in source.infolist():
                data = replacements.get(item.filename, source.read(item))
                target.writestr(_copy_zip_info(item), data)
                written.add(item.filename)
            for name, data in replacements.items():
                if name not in written:
                    target.writestr(name, data)
        os.replace(temporary_output, output_path)
    finally:
        temporary_output.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    manifest_path: Path,
    mode: str,
    input_path: Path,
    output_path: Path,
    bibliography: Path,
    style: Path,
    include_uncited: bool,
    bibliography_alignment: str,
    pandoc: str,
    occurrences: List[Dict[str, object]],
) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    processor = None
    if mode == "csl":
        processor = subprocess.run(
            [pandoc, "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
    data = {
        "schema_version": 1,
        "citation_mode": mode,
        "include_uncited": include_uncited,
        "bibliography_alignment": bibliography_alignment,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "bibliography": {
            "path": str(bibliography),
            "sha256": sha256_file(bibliography),
        },
        "style": (
            {"path": str(style), "sha256": sha256_file(style)}
            if mode == "csl"
            else None
        ),
        "processor": processor,
        "citations": occurrences,
    }
    rendered = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        dir=manifest_path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as handle:
        handle.write(rendered)
        temporary_manifest = Path(handle.name)
    try:
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)


def story_part_names(archive: ZipFile) -> List[str]:
    names = set(archive.namelist())
    parts = ["word/document.xml"]
    patterns = (
        r"word/footnotes\.xml$",
        r"word/endnotes\.xml$",
        r"word/header\d+\.xml$",
        r"word/footer\d+\.xml$",
    )
    parts.extend(
        sorted(name for name in names if any(re.match(pattern, name) for pattern in patterns))
    )
    return [name for name in parts if name in names]


def assert_no_source_drift(word_references, csl_references) -> None:
    if normalized_references(word_references) != normalized_references(csl_references):
        raise RuntimeError(
            "Embedded Word sources differ from canonical CSL-JSON. Reconcile them with "
            "sync_sources_from_word.py before building CSL mode."
        )


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    bibliography = args.bibliography.expanduser().resolve()
    style = args.csl.expanduser().resolve()
    csl_references = load_csl_json(bibliography)
    reference_ids = {str(item["id"]) for item in csl_references}
    replacements: Dict[str, bytes] = {}

    with ZipFile(input_path) as archive:
        part_names = story_part_names(archive)
        roots = {}
        namespaces = {}
        original_xml = {}
        for name in part_names:
            original_xml[name] = archive.read(name)
            namespaces[name] = capture_namespaces(original_xml[name])
            roots[name] = ET.fromstring(original_xml[name])
        source_part, source_xml, word_references = read_word_sources(archive)
        relationship_xml = {
            name: archive.read(relationship_part_name(name))
            if relationship_part_name(name) in archive.namelist()
            else None
            for name in part_names
        }

    if args.mode == "word":
        for root in roots.values():
            unlock_word_fields(root)
        replacements[source_part] = update_word_sources_xml(source_xml, csl_references)
    else:
        assert_no_source_drift(word_references, csl_references)
        citation_syntax = [
            citation_markdown(str(span["instruction"]), reference_ids)
            for name in part_names
            for _paragraph, span in citation_fields(roots[name])
        ]
        independent_style = resolve_independent_style(style)
        with tempfile.TemporaryDirectory(prefix="aeth-citations-") as temp:
            (
                rendered,
                bibliography_paragraphs,
                generated_relationships,
                generated_namespaces,
            ) = run_citeproc(
                input_docx=input_path,
                bibliography=bibliography,
                style=independent_style,
                citation_syntax=citation_syntax,
                pandoc=args.pandoc,
                temp_dir=Path(temp),
                include_uncited=args.include_uncited,
            )
        citation_index = 0
        for name in part_names:
            field_count = len(citation_fields(roots[name]))
            local_rendered = {
                "CID%03d" % index: copy.deepcopy(rendered["CID%03d" % index])
                for index in range(citation_index, citation_index + field_count)
            }
            related_elements = [
                element
                for elements in local_rendered.values()
                for element in elements
            ]
            if name == "word/document.xml":
                related_elements.extend(bibliography_paragraphs)
            updated_relationships = merge_external_relationships(
                relationship_xml[name], generated_relationships, related_elements
            )
            if updated_relationships is not None:
                replacements[relationship_part_name(name)] = updated_relationships
            citation_index = replace_citation_results(
                roots[name], local_rendered, citation_index
            )
            namespaces[name].update(generated_namespaces)
        replace_bibliography(
            roots["word/document.xml"],
            bibliography_paragraphs,
            args.bibliography_alignment,
        )

    for name in part_names:
        replacements[name] = serialize_xml(roots[name], namespaces[name])
    write_docx(input_path, output_path, replacements)
    if args.manifest is not None:
        occurrences = []
        occurrence_index = 0
        for name in part_names:
            for _paragraph, span in citation_fields(roots[name]):
                instruction = str(span["instruction"])
                occurrences.append(
                    {
                        "index": occurrence_index,
                        "story_part": name,
                        "instruction": instruction,
                        "items": parse_citation_items(instruction, reference_ids),
                    }
                )
                occurrence_index += 1
        write_manifest(
            args.manifest,
            args.mode,
            input_path,
            output_path,
            bibliography,
            style,
            args.include_uncited,
            args.bibliography_alignment,
            args.pandoc,
            occurrences,
        )
    print("Built %s citation mode: %s" % (args.mode, output_path))


if __name__ == "__main__":
    main()
