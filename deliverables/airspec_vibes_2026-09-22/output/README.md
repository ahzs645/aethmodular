# AIRSpec and VIBES research update

19 slides for a research-group update, approximately 15–20 minutes. Speaker notes use SAY and NOTES.

The deck draws numeric results from the completed frozen Colab run and its subsequent audit. Historical interpretations and proposed work carry explicit status labels. EC mass remains in µg/filter.

The design follows the repository ann-deck style: white background, finding-led titles, one main evidence figure per slide, and conversational notes. Quantitative charts and tables remain editable. The old explorer concentration anchors and Option A/B/B2 results describe different experiments and are not substituted for this benchmark.

Files: PowerPoint, one-page talking_points.md, full speaker_script.md, source_manifest.json.

Rebuild input data:

```sh
uv run --locked --no-sync python research/ftir_hips_chem/workflows/prepare_airspec_vibes_deck.py
```

Deck builder: ../.build/build.mjs, using the bundled Artifact Tool Node runtime.
