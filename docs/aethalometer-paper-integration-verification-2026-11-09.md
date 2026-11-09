# Paper integration: implementation verification, 9 November 2026

This receipt covers the source/coverage corrections, named Shlipak comparison,
scientific-label corrections and accepted-continuum publication gate described
in [the integration guide](aethalometer-paper-comparison.md). It is a local
implementation receipt, not evidence of deployment or a reproduction of the
paper's private instrument processing.

## Source and data scope

The branch starts at `aa0e68ad71d04fded178b14ef92e6201631d7908` in
`ahzs645/aethmodular`. Public integration used release `data-fddbc66c54ab`,
verified against its full archive SHA-256:

`fddbc66c54abf87a42dc231c4b74973ec6dcd666c674e31c1335501d512e9bf6`.

The restored continuum was the historical artifact:

`90b8e16bfcb002db7e6bc7553d190c94f462753182fcb80d5bacc1c65a2111c9`.

The shared publication gate verified the immutable review manifest, acceptance
receipt and all eight accepted scientific source files, then applied:

`f38baa0c6756fadf92583ef3b23c018b4c6f1773efd4bf52b7ce6f56db1649a2`.

The built continuum has exactly those accepted bytes. The gate preserves
35 historical top-level tables and the unaffected settings, including the
4 November target-generation date. The historical review archive remains
`data-82d701248232`; its identity is not relabelled as the newer input archive.
No accepted source predictions, manifests or scientific files were modified.

The November status bundle was regenerated through its exporter. Its loading,
carbon-share, Addis-fit, seasonal-fit, source-score, heavy-subset and historical
commit-count tables match the preceding committed bundle. Changes correct the
measurement labels, calendar qualifications and writing-status claims, and mark
the proposed writing dates as proposed.

## Checks completed

| Check | Result and scope |
|---|---|
| Focused Python regressions | 162 passed across the ten files listed below. Includes constructed physical mixtures, real firmware clipping path, UTC/DST/duplicate coverage, comparison CLI, and publication failure cases. |
| TypeScript | `npm run typecheck` passed. |
| Source Node checks | 379 passed; zero failures, skipped tests or TODOs. Chart index regenerated after the caption changes. |
| Public-data Node checks | 91 passed; zero failures, skipped tests or TODOs, using the restored public exports with the accepted overlay applied. |
| Production builds | Standard build and `VITE_PUBLIC_BUILD=1 npm run build` passed. |
| Built data | `continuum_publication.py verify-built` passed against `gallery/app/dist/data`. |
| Publication dry run | Packed 5,139 files with zero scrub substitutions into prospective snapshot `data-cfbe81e9cdca`; no upload or deployment. |
| Code and generated metadata | Scoped Ruff and `git diff --check` passed. Generated pipeline Markdown matches the registry; chart index source check passed. |
| Workflow syntax | Three YAML workflows parsed; 22 shell blocks passed `bash -n` during publication review. |
| Independent reviews | Separate reviewers checked the source/coverage arithmetic, profile facts against the supplied paper, and scientific labels against their original exporters/workflows. |

Python regression scope:

```bash
.venv/bin/pytest -q \
  tests/test_source_apportionment_model.py \
  tests/test_aethalometer_model_clipping.py \
  tests/test_outliers.py \
  tests/test_aeth_sources_export.py \
  tests/test_paper_comparison.py \
  tests/test_gallery_continuum_publication.py \
  tests/test_gallery_publication.py \
  tests/test_gallery_sections.py \
  tests/test_gallery_rebuild.py \
  tests/test_continuum_source_refresh.py
```

## Explicitly pending

**Browser acceptance is pending.** All three existing `tests/closeout.spec.ts`
journeys failed during Chromium launch, before any application assertions:
`socket() failed: Operation not permitted`. The connected browser was on a
different host, and the cloud browser could not connect to this checkout's
localhost preview (`ERR_CONNECTION_REFUSED`). No visual, mobile, shared-link or
PDF-export acceptance is claimed for this source revision. Run the existing
browser journeys in a browser-capable environment before release acceptance.

**Real instrument-data reconciliation is pending.** The coverage exporter was
tested with synthetic inputs, including an export/readback, but the private
MAIA hourly files and daily pickles were not available for a full rebuild.
Older hourly exports remain readable; their biomass fraction is withheld until
independent paired-hour coverage exists. Signed firmware source components
remain preserved; this batch does not invent additional physical-range QC.

**Exact paper reproduction is pending.** The named command has been exercised
end to end with synthetic data, not the paper's Central/Jacros observations.
Supporting-information settings and deployment identities must be established
before a real comparison can claim matching processing. Optical fractions,
AAE class frequencies, chemical PMF and thermal EC remain distinct quantities.

This batch is prepared for review. Merge, a new data release, installation of
the canonical deployment workflow, deployment, and live receipt/byte readback
are separate actions. Local checks conserve GitHub Actions usage; commit skip
markers do not replace the pending browser/release checks.
