# Ann weekly meeting — 17 September 2026

**Presentation:** FTIR EC calibration: what transfers, and what does not?

**Presenter:** Ahmad Jalil. **Scope:** results through 17 September 2026. **Length:** 18 main slides plus 8 backup slides.

## Read the update in GitHub

- [Presenter notes](Presenter_Notes.md): the complete talk track, interpretation limits, discussion questions and source paths for every slide.
- [Slide text](slides.md): exact visible text extracted from the delivered PowerPoint. This is a text-only review, not a replacement for the figures.
- [Package manifest](package_manifest.json): identities, sizes and SHA-256 hashes of the approved meeting archive and primary artifacts.

The presentation begins with spectra and selected filters, then covers paired masking/cohort tests, TOR reference stability, five-site transfer, the intercept/slope screen, the separate older-product archive, remaining reference-data questions, and the next bounded Colab experiment. Main discussion ends at slide 18; the remaining eight slides support questions.

## Binary-artifact status

**The PPTX, PDF, figures and chart-data bundle are not included in this source-only commit.** The complete 4,997,659-byte archive delivered with the meeting deck is named:

`Aethmodular_Weekly_Meeting_Package_2026-09-17.zip`

No temporary ChatGPT sandbox links are presented as GitHub download links. Import that exact archive with the checked workflow below. This preserves the approved PowerPoint/PDF rather than rebuilding a different slide deck from a prose summary. The accompanying complete repository patch is an alternative way to add the same bytes.

```sh
# Run at the repository root. First validate without writing anything.
uv run python research/ftir_hips_chem/workflows/import_ann_weekly_20260917.py \
  ~/Downloads/Aethmodular_Weekly_Meeting_Package_2026-09-17.zip --check

# Import the verified package. Existing different files are never overwritten.
uv run python research/ftir_hips_chem/workflows/import_ann_weekly_20260917.py \
  ~/Downloads/Aethmodular_Weekly_Meeting_Package_2026-09-17.zip
```

The importer checks the full archive hash, primary artifact hashes, archive paths, symbolic links, expansion limits, and 26 slide / 26 notes parts. A second run skips byte-identical files. It does not fetch data, refit analyses, commit, push or modify September 10 artifacts.

After import, the presentation files are:

```text
deliverables/ann_weekly_2026-09-17/bundle/
  Aethmodular_Weekly_Meeting_2026-09-17.pptx
  Aethmodular_Weekly_Meeting_2026-09-17.pdf
  Presenter_Notes.md
  Source_Ledger.csv
  Numerical_Checks.json
  SHA256.json
  evidence/
  presentation_source/
    build_deck.js
    deck_data.json
    chart_data/
    charts/
```

Review `git status` and commit the imported bundle to publish those files. Generated PNGs in this specific approved meeting package are intentionally included; no unrelated generated images or large raw-data directories are imported.

## Rebuild a revision

After import, the portable wrapper is under the active workspace's `workflows/` directory. It invokes the original packaged PptxGenJS source. Use a Node environment where `pptxgenjs` is available, including an existing `NODE_PATH` when needed. Supply a fresh filename:

```sh
node research/ftir_hips_chem/workflows/build_ann_weekly_20260917.cjs \
  deliverables/ann_weekly_2026-09-17/ann_weekly_2026-09-17_revision.pptx
```

The wrapper refuses to overwrite an existing presentation. The original `prepare_deck_data.py` is archived provenance, not a promise of a complete raw-data rebuild from this folder alone. Scientific source packages and the large Colab data bundle remain separate.

## Interpretation boundaries

This is a retrospective research update, not acceptance of a replacement EC calibration. Keep TOR prediction separate from HIPS/MAC agreement, squared correlation separate from predictive Q², and new frozen-model applications separate from the 22-site older reported-product table. The 73 Addis reserve filters have already been evaluated. Conditional bootstrap intervals do not include complete model-selection/refitting uncertainty. The seven explicitly invalid Delhi filters, lot/period confounding, Adama crosswalk limitations, and outstanding upstream unit/QC checks remain documented in the notes.

## Importer checks

```sh
uv run python -m unittest discover -s tests -p test_import_ann_weekly_20260917.py -v
```

Eight safety tests were run for this addition, plus verification of the actual archive. No new scientific calibration was fit for repository integration. The previous meeting remains under [ann_weekly_2026-09-10](../ann_weekly_2026-09-10/).
