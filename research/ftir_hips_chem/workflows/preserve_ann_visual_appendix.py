"""Retain the three original appendix chart parts and their real workbooks.

Artifact Tool authors the slide edits. Its import/export drops the embedded
workbooks, so copy the source parts to the remapped chart paths after verifying
that the appendix numeric caches are unchanged. Never invent workbook lineage.
"""

import json
import posixpath
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
CT = "{http://schemas.openxmlformats.org/package/2006/content-types}"
C = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
ET.register_namespace("", CT[1:-1])


def rels_part(part):
    return posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")


def targets(archive, owner, suffix):
    root = ET.fromstring(archive.read(rels_part(owner)))
    return [
        posixpath.normpath(node.attrib["Target"].lstrip("/") if node.attrib["Target"].startswith("/")
                          else posixpath.join(posixpath.dirname(owner), node.attrib["Target"]))
        for node in root.findall(f"{REL}Relationship")
        if node.attrib["Type"].endswith(suffix)
    ]


def numeric_values(payload):
    root = ET.fromstring(payload)
    return [
        float(pt.find(f"{C}v").text)
        for node in root.iter()
        if node.tag in (f"{C}numCache", f"{C}numLit")
        for pt in node.findall(f"{C}pt")
    ]


def preserve(source_path, candidate_path, output_path):
    if Path(output_path).resolve() in (Path(source_path).resolve(), Path(candidate_path).resolve()):
        raise ValueError("Write the repaired candidate to a separate path")
    replacements, evidence = {}, []
    with ZipFile(source_path) as source, ZipFile(candidate_path) as candidate:
        for n in [10, 11, 12]:
            owner = f"ppt/slides/slide{n}.xml"
            before, after = targets(source, owner, "/chart"), targets(candidate, owner, "/chart")
            assert len(before) == len(after) == 1
            a, b = before[0], after[0]
            assert numeric_values(source.read(a)) == numeric_values(candidate.read(b)), n
            replacements[b] = source.read(a)
            replacements[rels_part(b)] = source.read(rels_part(a))
            books = targets(source, a, "/package")
            assert len(books) == 1
            for book in books:
                assert book not in candidate.namelist(), "Unexpected workbook collision"
                replacements[book] = source.read(book)
            evidence.append(
                {"slide": n, "source_chart": a, "retained_chart": b, "workbooks": books}
            )
        content_types = ET.fromstring(candidate.read("[Content_Types].xml"))
        if not any(n.attrib.get("Extension") == "xlsx" for n in content_types):
            xlsx_type = next(
                n
                for n in ET.fromstring(source.read("[Content_Types].xml"))
                if n.attrib.get("Extension") == "xlsx"
            )
            content_types.append(xlsx_type)
        replacements["[Content_Types].xml"] = ET.tostring(
            content_types, encoding="utf-8", xml_declaration=True
        )
        with ZipFile(output_path, "w", ZIP_DEFLATED) as output:
            for entry in candidate.infolist():
                output.writestr(
                    entry, replacements.pop(entry.filename, candidate.read(entry.filename))
                )
            for name, payload in replacements.items():
                output.writestr(name, payload)
    with ZipFile(source_path) as source, ZipFile(output_path) as output:
        for item in evidence:
            assert source.read(item["source_chart"]) == output.read(item["retained_chart"])
            assert all(source.read(book) == output.read(book) for book in item["workbooks"])
    print(json.dumps(evidence))


if __name__ == "__main__":
    preserve(*sys.argv[1:])
