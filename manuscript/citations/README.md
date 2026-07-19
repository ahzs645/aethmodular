# Connected citation workflow

This directory provides two explicit Microsoft Word citation outputs from one
canonical source library. It does not mix Microsoft and Zotero fields inside a
single citation.

## Source of truth

- `references.csl.json` is the canonical, versioned source library.
- `references.bib` is a derived exchange file for Zotero and other managers.
- The embedded Microsoft sources in a DOCX are a compatibility copy generated
  from CSL-JSON.
- `styles/atmospheric-measurement-techniques.csl` is the official dependent
  AMT style. `styles/copernicus-publications.csl` is its independent parent.

The build refuses CSL mode when the embedded Word metadata has drifted from
the canonical library. Check a manuscript without changing anything:

```bash
uv run python manuscript/citations/sync_sources_from_word.py \
  manuscript-word.docx --direction check
```

Write canonical CSL metadata into a new Word copy:

```bash
uv run python manuscript/citations/sync_sources_from_word.py \
  manuscript-word.docx \
  --direction csl-to-word \
  --output-docx manuscript-word-reconciled.docx
```

If Word was deliberately chosen as the new metadata authority, explicitly
replace both exchange files:

```bash
uv run python manuscript/citations/sync_sources_from_word.py \
  manuscript-word.docx --direction word-to-csl
```

That last command is intentionally explicit because it replaces the canonical
CSL-JSON library. It stages both derived files before replacing either one.

## Output modes

### Word-native compatibility copy

Word mode writes the canonical CSL library into the embedded Microsoft source
part, unlocks `CITATION` and `BIBLIOGRAPHY` fields, and marks them dirty. Open
the result in Word and update fields with `Cmd+A`, `F9` on macOS or `Ctrl+A`,
`F9` on Windows.

```bash
uv run python manuscript/citations/build_dual_citations.py \
  --mode word \
  --input manuscript-csl.docx \
  --output manuscript-word.docx
```

Word controls citation formatting in this compatibility mode. Metadata edits
made through Word are not canonical until `word-to-csl` is run deliberately.

### CSL journal-style submission copy

CSL mode uses Pandoc citeproc, preserves the generated Word run formatting and
external hyperlinks, and locks the underlying Microsoft fields so Word cannot
replace the journal formatting. Rebuild after changing citations, sources, or
the CSL style.

```bash
uv run python manuscript/citations/build_dual_citations.py \
  --mode csl \
  --input manuscript-word.docx \
  --csl manuscript/citations/styles/atmospheric-measurement-techniques.csl \
  --bibliography-alignment left \
  --output manuscript-amt.docx \
  --manifest manuscript-amt.citation-manifest.json
```

Only cited records are included in the bibliography by default. Add
`--include-uncited` when a complete reading list is intentional.
`--bibliography-alignment left` overrides a justified `Bibliography` paragraph
style in the reference DOCX; omit it when the reference style already has the
desired alignment.

The optional manifest records every citation occurrence and its ordered items,
locators, prefixes, suffixes, and suppress-author state, together with hashes
for the input, output, library, and style plus the Pandoc version.

Independent in-text CSL styles are supported, including run-level italics,
superscript, small caps, and external hyperlinks. Dependent styles work when
their independent parent is in the same directory. Note styles that must
create new Word footnotes from body citations are not yet an export target;
citations already located in Word footnotes and endnotes are processed.

The CSL output is generated and should not be used as the editable metadata
master. Coauthors may use Track Changes for prose, but citation/source changes
must be made in the selected live working copy and then rebuilt.

## Zotero status

Importing `references.bib` creates Zotero library records, but it does not
connect existing citations to Zotero. Genuine Zotero-owned fields use a
different `ADDIN ZOTERO_ITEM CSL_CITATION` protocol. If live Zotero editing is
needed, create a separate Zotero-owned working DOCX with the Zotero Word plugin;
never insert Zotero fields on top of Microsoft `CITATION` fields.

A future `zotero-live` adapter can use the same CSL-JSON IDs, but the current
implemented targets are `word` and locked `csl`.

## Validation

The regression suite covers citation switches, cited-only behavior, rich
superscript output, hyperlink relationship remapping, namespace preservation,
non-`item1.xml` Word sources, and CSL-to-Word metadata round trips:

```bash
uv run pytest tests/test_dual_citations.py
uv run ruff check manuscript/citations/*.py tests/test_dual_citations.py
```

For release builds, also open the Word-native copy, update fields, rebuild CSL
mode, and visually inspect the rendered DOCX pages.
