"""Read and write Microsoft Word bibliography source parts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile


B_NS = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"
B = "{%s}" % B_NS
ET.register_namespace("b", B_NS)

WORD_TO_CSL_TYPE = {
    "JournalArticle": "article-journal",
    "Book": "book",
    "BookSection": "chapter",
    "ConferenceProceedings": "paper-conference",
    "Report": "report",
    "InternetSite": "webpage",
    "DocumentFromInternetSite": "webpage",
    "ElectronicSource": "webpage",
}
CSL_TO_WORD_TYPE = {
    "article-journal": "JournalArticle",
    "book": "Book",
    "chapter": "BookSection",
    "paper-conference": "ConferenceProceedings",
    "report": "Report",
    "thesis": "Report",
    "webpage": "InternetSite",
    "post-weblog": "InternetSite",
    "dataset": "ElectronicSource",
    "software": "ElectronicSource",
}


def _text(source: ET.Element, name: str) -> Optional[str]:
    value = source.findtext(B + name)
    return value if value else None


def _date_parts(source: ET.Element, suffix: str = "") -> Optional[Dict[str, object]]:
    values: List[int] = []
    for name in ("Year" + suffix, "Month" + suffix, "Day" + suffix):
        value = _text(source, name)
        if not value:
            break
        try:
            values.append(int(value))
        except ValueError:
            return {"literal": value} if not values else None
    return {"date-parts": [values]} if values else None


def _names(source: ET.Element, role: str) -> List[Dict[str, str]]:
    role_node = source.find(".//%sAuthor/%s%s" % (B, B, role))
    if role_node is None:
        return []
    corporate = role_node.findtext(B + "Corporate")
    if corporate:
        return [{"literal": corporate}]
    names: List[Dict[str, str]] = []
    for person in role_node.findall(".//" + B + "Person"):
        name: Dict[str, str] = {}
        family = person.findtext(B + "Last")
        given = person.findtext(B + "First")
        suffix = person.findtext(B + "Middle")
        if family:
            name["family"] = family
        if given:
            name["given"] = given
        if suffix:
            name["suffix"] = suffix
        if name:
            names.append(name)
    return names


def source_to_csl(source: ET.Element) -> Dict[str, object]:
    """Convert one Word bibliography source to CSL-JSON without assuming a journal."""

    source_type = _text(source, "SourceType") or "Misc"
    container_title = (
        _text(source, "JournalName")
        or _text(source, "BookTitle")
        or _text(source, "ConferenceName")
        or _text(source, "PublicationTitle")
    )
    item: Dict[str, object] = {
        "id": _text(source, "Tag"),
        "type": WORD_TO_CSL_TYPE.get(source_type, "document"),
        "title": _text(source, "Title"),
        "author": _names(source, "Author"),
        "editor": _names(source, "Editor"),
        "container-title": container_title,
        "publisher": _text(source, "Publisher"),
        "publisher-place": _text(source, "City"),
        "volume": _text(source, "Volume"),
        "issue": _text(source, "Issue"),
        "page": _text(source, "Pages"),
        "edition": _text(source, "Edition"),
        "DOI": _text(source, "StandardNumber"),
        "URL": _text(source, "URL"),
        "issued": _date_parts(source),
        "accessed": _date_parts(source, "Accessed"),
    }
    return {
        key: value
        for key, value in item.items()
        if value not in (None, "", [], {})
    }


def find_bibliography_part(archive: ZipFile) -> str:
    """Find the bibliography custom XML part by its namespace and content."""

    for name in sorted(archive.namelist()):
        if not name.startswith("customXml/") or not name.endswith(".xml"):
            continue
        try:
            root = ET.fromstring(archive.read(name))
        except ET.ParseError:
            continue
        if root.tag == B + "Sources" or root.find(B + "Source") is not None:
            return name
    raise RuntimeError("No embedded Word bibliography source part found")


def read_word_sources(archive: ZipFile) -> Tuple[str, bytes, List[Dict[str, object]]]:
    part_name = find_bibliography_part(archive)
    source_xml = archive.read(part_name)
    root = ET.fromstring(source_xml)
    references = [source_to_csl(source) for source in root.findall(B + "Source")]
    if not references:
        raise RuntimeError("No embedded Word bibliography sources found")
    return part_name, source_xml, references


def _append_text(parent: ET.Element, name: str, value: object) -> None:
    if value in (None, ""):
        return
    node = ET.SubElement(parent, B + name)
    node.text = str(value)


def _append_names(parent: ET.Element, role: str, names: object) -> None:
    if not isinstance(names, list) or not names:
        return
    outer = ET.SubElement(parent, B + "Author")
    role_node = ET.SubElement(outer, B + role)
    literal_names = [name.get("literal") for name in names if isinstance(name, dict)]
    structured_names = [
        name for name in names if isinstance(name, dict) and not name.get("literal")
    ]
    if literal_names and not structured_names:
        corporate = ET.SubElement(role_node, B + "Corporate")
        corporate.text = "; ".join(str(name) for name in literal_names if name)
        return
    name_list = ET.SubElement(role_node, B + "NameList")
    for name in structured_names:
        person = ET.SubElement(name_list, B + "Person")
        _append_text(person, "Last", name.get("family"))
        _append_text(person, "First", name.get("given"))
        _append_text(person, "Middle", name.get("suffix"))


def _append_date(parent: ET.Element, value: object, suffix: str = "") -> None:
    if not isinstance(value, dict):
        return
    date_parts = value.get("date-parts")
    if not isinstance(date_parts, list) or not date_parts or not date_parts[0]:
        return
    for name, part in zip(("Year", "Month", "Day"), date_parts[0]):
        _append_text(parent, name + suffix, part)


def _guid_for(reference_id: str, existing: Optional[ET.Element]) -> str:
    if existing is not None:
        guid = existing.findtext(B + "Guid")
        if guid:
            return guid
    generated = uuid.uuid5(uuid.NAMESPACE_URL, "aethmodular:citation:" + reference_id)
    return "{%s}" % str(generated).upper()


def csl_to_source(
    reference: Dict[str, object],
    existing: Optional[ET.Element] = None,
    ref_order: Optional[int] = None,
) -> ET.Element:
    reference_id = str(reference.get("id") or "").strip()
    if not reference_id:
        raise ValueError("Every CSL reference must have a non-empty id")
    source = ET.Element(B + "Source")
    _append_text(source, "Tag", reference_id)
    word_type = CSL_TO_WORD_TYPE.get(str(reference.get("type")), "Misc")
    _append_text(source, "SourceType", word_type)
    _append_text(source, "Guid", _guid_for(reference_id, existing))
    _append_text(source, "LCID", "en-US")
    _append_names(source, "Author", reference.get("author"))
    _append_names(source, "Editor", reference.get("editor"))
    _append_text(source, "Title", reference.get("title"))
    _append_date(source, reference.get("issued"))
    container = reference.get("container-title")
    if word_type == "JournalArticle":
        _append_text(source, "JournalName", container)
    elif word_type == "BookSection":
        _append_text(source, "BookTitle", container)
    elif word_type == "ConferenceProceedings":
        _append_text(source, "ConferenceName", container)
    else:
        _append_text(source, "PublicationTitle", container)
    _append_text(source, "Publisher", reference.get("publisher"))
    _append_text(source, "City", reference.get("publisher-place"))
    _append_text(source, "Volume", reference.get("volume"))
    _append_text(source, "Issue", reference.get("issue"))
    _append_text(source, "Pages", reference.get("page"))
    _append_text(source, "Edition", reference.get("edition"))
    _append_text(source, "StandardNumber", reference.get("DOI"))
    _append_text(source, "URL", reference.get("URL"))
    _append_date(source, reference.get("accessed"), "Accessed")
    if ref_order is not None:
        _append_text(source, "RefOrder", ref_order)
    return source


def update_word_sources_xml(
    source_xml: bytes, references: Iterable[Dict[str, object]]
) -> bytes:
    """Replace the Word source list from canonical CSL-JSON references."""

    root = ET.fromstring(source_xml)
    existing_by_id = {
        source.findtext(B + "Tag"): source for source in root.findall(B + "Source")
    }
    for source in root.findall(B + "Source"):
        root.remove(source)
    for index, reference in enumerate(references, start=1):
        reference_id = str(reference.get("id") or "")
        source = csl_to_source(reference, existing_by_id.get(reference_id), index)
        root.append(source)
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def canonical_json(references: Iterable[Dict[str, object]]) -> str:
    return json.dumps(list(references), ensure_ascii=False, indent=2) + "\n"


def load_csl_json(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("CSL-JSON bibliography must be a list")
    references: List[Dict[str, object]] = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Every CSL-JSON bibliography item must be an object")
        reference_id = str(item.get("id") or "").strip()
        if not reference_id:
            raise ValueError("Every CSL-JSON bibliography item must have an id")
        if reference_id in seen:
            raise ValueError("Duplicate CSL-JSON reference id: %s" % reference_id)
        seen.add(reference_id)
        references.append(copy.deepcopy(item))
    return references


def normalized_references(references: Iterable[Dict[str, object]]) -> str:
    """Return a stable representation suitable for drift checks."""

    return json.dumps(list(references), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
