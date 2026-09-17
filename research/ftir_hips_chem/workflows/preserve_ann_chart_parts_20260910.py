"""Preserve original chart XML/workbooks after the presentation import/export.

Artifact Tool authors the slides. This narrow OOXML repair copies the source
chart parts and their existing embedded workbooks byte-for-byte, rather than
materializing new workbook snapshots from imported chart caches.
"""

import json
from pathlib import Path
import posixpath
import sys
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

REL = '{http://schemas.openxmlformats.org/package/2006/relationships}'
CT = '{http://schemas.openxmlformats.org/package/2006/content-types}'
ET.register_namespace('', CT[1:-1])


def chart_targets(archive, slide_number):
    owner = f'ppt/slides/slide{slide_number}.xml'
    rels = ET.fromstring(archive.read(f'ppt/slides/_rels/slide{slide_number}.xml.rels'))
    return [posixpath.normpath(item.attrib['Target'].lstrip('/') if item.attrib['Target'].startswith('/')
            else posixpath.join(posixpath.dirname(owner), item.attrib['Target']))
            for item in rels.findall(f'{REL}Relationship') if item.attrib['Type'].endswith('/chart')]


def preserve(source_path, candidate_path, slide_mapping=None):
    candidate_path = Path(candidate_path)
    with ZipFile(source_path) as source, ZipFile(candidate_path) as candidate:
        # Verify chart ownership still matches the inspected source after the
        # four-slide insertion. No other chart set is supported by this repair.
        for before, after in slide_mapping or [(3, 3), (10, 14), (11, 15), (12, 16)]:
            original = chart_targets(source, before)
            current = chart_targets(candidate, after)
            if len(original) != 1 or original != current:
                raise ValueError(f'Chart mapping changed: {before} -> {after}')
        parts = [name for name in source.namelist()
                 if name.startswith('ppt/slides/charts/') or name.startswith('ppt/embeddings/')]
        if len([name for name in parts if name.endswith('.xlsx')]) != 4:
            raise ValueError('Expected four original embedded chart workbooks')
        replacements = {name: source.read(name) for name in parts}
        content_types = ET.fromstring(candidate.read('[Content_Types].xml'))
        keys = {(node.tag, node.attrib.get('PartName', node.attrib.get('Extension'))) for node in content_types}
        for node in ET.fromstring(source.read('[Content_Types].xml')):
            applies = node.attrib.get('PartName', '').lstrip('/') in replacements
            applies |= node.tag == f'{CT}Default' and node.attrib.get('Extension') == 'xlsx'
            key = (node.tag, node.attrib.get('PartName', node.attrib.get('Extension')))
            if applies and key not in keys:
                content_types.append(node)
                keys.add(key)
        replacements['[Content_Types].xml'] = ET.tostring(content_types, encoding='utf-8', xml_declaration=True)
        with tempfile.NamedTemporaryFile(dir=candidate_path.parent, suffix='.pptx', delete=False) as handle:
            temp_path = Path(handle.name)
        with ZipFile(temp_path, 'w', ZIP_DEFLATED) as output:
            for entry in candidate.infolist():
                output.writestr(entry, replacements.pop(entry.filename, candidate.read(entry.filename)))
            for name, payload in replacements.items():
                output.writestr(name, payload)
    temp_path.replace(candidate_path)
    with ZipFile(source_path) as source, ZipFile(candidate_path) as candidate:
        assert all(source.read(name) == candidate.read(name) for name in parts)
    print(json.dumps({'preserved_original_chart_parts': parts}))


if __name__ == '__main__':
    preserve(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else None)
