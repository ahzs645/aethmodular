"""Create and execute the notebook that supplies the visual audit slide section."""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / 'research/ftir_hips_chem'
NAME = 'matched_sample_audit_figures.ipynb'

# Keep the repository's documented setup as the first code cell.
guide = (ROOT / 'AGENTS.md').read_text()
setup = guide.split('## Standard notebook setup cell', 1)[1].split('```python\n', 1)[1].split('```', 1)[0]
setup = setup.replace(
    "PlotConfig.set(sites='all', layout='individual', show_stats=True, show_1to1=True)",
    "PlotConfig.set(sites='all', layout='individual', show_stats=False, show_1to1=False,\n"
    "               figsize=(14.8, 6.2), font_size=20, title_size=23)",
)
setup += '\nfrom IPython.display import display\nfrom plotting.matched_sample_audit import MatchedSampleAuditFigures\n'

pages = [
    ('01_pair_availability', 'pair_availability', '546 same-filter pairs, 347 IR date candidates',
     'All finite HIPS/FTIR pairs appear here, including the registered Delhi exclusion. '
     'The darker bars are a subset with finite raw IR within the legacy ±1-calendar-day search. '
     'These 347 candidates do not establish a shared sampling interval or observed coverage. '
     'This catalog is separate from the 239-filter seasonal calibration cohort in the weekly presentation.'),
    ('02_metadata_availability', 'metadata_availability', '317 HIPS filters have recovered collection windows',
     'Each complete bar contains the HIPS/FTIR pairs from the previous figure. Colored portions have '
     'positive-duration collection bounds recovered from portal records. Grey portions have no usable '
     'window in the inspected inputs. The current source directory supplied Addis and JPL exports only. '
     'Zeros for Beijing/Delhi describe that source coverage, not whether such metadata exist elsewhere. '
     'Recovered bounds still require active-period verification for intermittent collection.'),
    ('03_timestamp_hours', 'timestamp_hours', 'Saved timestamps cluster at 15:00',
     'Labels give the complete saved row counts at each local hour. Marker area increases with count '
     'but every nonempty hour remains visible. The old builder shifted by 15 hours despite its 9 AM label. '
     'The corrected code now defines local [09:00, next 09:00) intervals and accounts for elapsed DST minutes. '
     'Existing saved files have not been rebuilt. Relabeling their timestamps would not repair their averages.'),
    ('04_completeness', 'completeness', 'Saved completeness needs an observation audit',
     'Empirical cumulative distributions include all finite saved completeness values, including zeros '
     'and the value above 100%. Denominators are all saved rows for each represented site. Addis has no '
     'such field and does not receive an invented zero. The original field counted first-channel non-null '
     'records relative to 1,440 minutes. It does not verify observed IR minutes, duplicate handling, '
     'DST duration, or interpolation history. The corrected pipeline requires explicit observation '
     'provenance to distinguish observed minutes from mere input availability.'),
    ('05_timing_status', 'timing_status', '23 timing issues in 453 portal records',
     'Hours agree means a positive-duration window and reported sampled hours agreeing within 0.05 hours. '
     'It is a metadata consistency check, not proof of continuous operation. All 453 portal filter records '
     'remain in the count: Addis 181 consistent, 6 differing, 1 invalid; JPL 249 consistent, 16 differing. '
     'The right panel enlarges the issue counts using the same denominators. The separate one-day '
     'SampleDate disagreement for ETAD-0243 is not counted as an additional record.'),
    ('06_collection_durations', 'collection_durations', '20 filters span several days of collection',
     'Every portal filter contributes to a duration group. Coincident points are grouped by site, elapsed '
     'hours and sampled hours, with marker area increasing with group count. The dashed segment denotes '
     'equal hours; no regression or validation statistic is fitted. The horizontal axis has a wider range '
     'than the vertical axis. Five Addis filters report 24 sampled hours across 192–216 elapsed hours; '
     '15 JPL filters report 48 sampled hours across 237–238 elapsed hours. The other three timing issues '
     'are ETAD-0163 (24 vs 22.8 h), ETAD-0178 (0 vs 24 h), and USPA-0243 (23 vs 24 h). '
     'UTC differences account for the local JPL daylight-saving change.'),
    ('07_duration_examples', 'duration_examples', 'Sampled hours cover only part of the collection window',
     'Examples are selected explicitly to show the two intermittent patterns and an invalid record, '
     'not to estimate their prevalence. ETAD-0241 spans 7–15 August 2024 locally: 192 elapsed hours, '
     '24 reported sampled hours. USPA-0329 spans 6–16 March 2024 locally: 237 elapsed hours, 48 sampled '
     'hours. ETAD-0178 reports identical local start/end on 24 February 2024 despite 24 sampled hours. '
     'The bars compare totals only. They do not place sampled hours at the beginning of the window or '
     'identify any active interval. Active on/off records are needed before computing a matched aethalometer mean.'),
    ('08_ec_conflicts', 'ec_conflicts', '500 filters contain conflicting ChemSpec EC values',
     'The left chart counts physical-filter groups whose ChemSpec_EC_PM2.5 concentrations conflict in the '
     'unified table. This is a count of inconsistent groups, not a count of independent EC references '
     'or chemically invalid filters. CHTS-0658 illustrates the issue: source rows 5859 and 5860 carry '
     '0.93 and 0.06 µg/m³ under the same parameter name and calibration label. Source-row positions are '
     'zero-based iloc offsets in the hashed input. The audit retains both and does not average or choose '
     'the first value. Existing repository provenance work identifies the concentration product as FTIR EC '
     'and ChemSpec BC as HIPS/10, subject to rounding. Independent EC validation requires a separate '
     'reference. The next comparison needs resolved parameter definitions, active sampling intervals, '
     'observed coverage and optical conversions before evaluating adjustments in withheld calendar periods.'),
]


def main():
    nb = nbf.v4.new_notebook(metadata={
        'kernelspec': {'display_name':'Python 3 (aethmodular)', 'language':'python', 'name':'python3'},
        'language_info': {'name':'python', 'version':'3.13'},
    })
    nb.cells = [nbf.v4.new_code_cell(setup), nbf.v4.new_markdown_cell(
        '# Matched-sample audit figures\n\n'
        'Eight reproducible figures for the audit section of the weekly presentation. '
        'Run this active notebook from `research/ftir_hips_chem/` in the repository uv environment. '
        'The executed companion is saved under `notebooks/archive/executed/`.\n\n'
        'This is an inventory and metadata audit, so all flags remain visible. The source audit already '
        'applies the exclusion registry non-destructively. We do not call `get_clean_data` to discard '
        'flagged records, do not re-match observations, and do not fit or validate a calibration. '
        'Each input hash must match the audit manifest before plotting.\n\n'
        'Source report: `docs/matched-sample-audit-2026-09-10.md`. '
        'The figures use the canonical site colors and plotting defaults. Reusable figure code lives '
        'in `scripts/plotting/matched_sample_audit.py`. PNG and SVG exports go under '
        '`output/plots/matched_sample_audit/`, with plotted data and provenance under '
        '`output/tables/matched_sample_audit_figures/`.'),
        nbf.v4.new_code_cell('audit = MatchedSampleAuditFigures()\ndisplay(audit.counts)')]
    for key, method, title, explanation in pages:
        nb.cells.extend([nbf.v4.new_markdown_cell(f'## {title}\n\n{explanation}'),
                        nbf.v4.new_code_cell(f'fig = audit.{method}()\n'
                                             f'png = audit.save(fig, "{key}")\n'
                                             'display(fig)\nplt.close(fig)\nprint(png)')])
    nb.cells.append(nbf.v4.new_code_cell('manifest_path = audit.write_manifest()\nprint(manifest_path)'))
    nb.cells.append(nbf.v4.new_markdown_cell(
        '## Source definitions\n\n'
        'All measurements, row links, recovered collection bounds and registered flags come from '
        '`output/tables/matched_sample_audit/`. The coverage figure reads the four saved pickles '
        'identified by `config.SITES` and checks them against the same manifest. '
        'ChemSpec provenance is documented in `docs/open-items.md`. '
        'No on/off schedule or independent EC reference is inferred from the displayed counts.'))
    nbf.write(nb, ACTIVE / NAME)
    NotebookClient(nb, timeout=180, kernel_name='python3',
                   resources={'metadata':{'path':str(ACTIVE)}}).execute()
    destination = ACTIVE / 'notebooks/archive/executed' / NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, destination)
    figures = sum('image/png' in out.get('data', {}) for cell in nb.cells for out in cell.get('outputs', []))
    assert figures == len(pages), (figures, len(pages))
    print(f'Executed {figures} figures: {destination}')


if __name__ == '__main__':
    main()
