# CSL style provenance

These styles come from the official
[Citation Style Language style repository](https://github.com/citation-style-language/styles)
and retain their original authorship and CC BY-SA 3.0 metadata.

Verified against the repository's `master` branch on 2026-07-18:

| File | CSL style ID | SHA-256 |
|---|---|---|
| `atmospheric-measurement-techniques.csl` | `http://www.zotero.org/styles/atmospheric-measurement-techniques` | `b28f759b887b48846532c06e499b4ba4fef0f17071bca166685a383d5934946d` |
| `copernicus-publications.csl` | `http://www.zotero.org/styles/copernicus-publications` | `aa9c09672b61ab6dced0ad0455ee169bf428b09e3005670d2f514ecee5430342` |

The AMT dependent style is byte-identical to the official copy at the
verification date. The pinned Copernicus parent has the same CSL metadata and
rules as the official copy; the official repository has since applied
whitespace-only XML formatting changes, so its byte hash differs.

When updating a style, retain its complete `<info>` element, validate the XML,
run `tests/test_dual_citations.py`, rebuild the manuscript, and update this
provenance record deliberately.
