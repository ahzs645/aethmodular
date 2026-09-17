"""Figures for the matched-sample audit, reproduced by its active notebook.

These are inventory and metadata diagnostics. All registry/timing flags remain
visible. No measurement regression, calibration, or threshold selection occurs.
The categorical audit plots extend the standard plotting package; the duration
scatter uses its shared axes-level crossplot with fitted statistics disabled.
"""

from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_ROOT, PROCESSED_SITES_DIR, SITES
from . import PlotConfig
from .overlays import crossplot_on_axes
from .utils import style_axes


INK = '#17384A'
MUTED = '#536574'
PALE = '#DCE5EB'
ISSUE = '#B54435'
ORDER = ['Beijing', 'Delhi', 'JPL', 'Addis_Ababa']


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


class MatchedSampleAuditFigures:
    """Load source-backed audit records and export figure data with provenance."""

    def __init__(self, audit_dir=None, output_dir=None, table_dir=None):
        self.audit = Path(audit_dir or DATA_ROOT / 'output/tables/matched_sample_audit')
        self.output = Path(output_dir or DATA_ROOT / 'output/plots/matched_sample_audit')
        self.tables = Path(table_dir or DATA_ROOT / 'output/tables/matched_sample_audit_figures')
        self.output.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((self.audit / 'manifest.json').read_text())
        self.used_sources = []
        self.exports = []
        outputs = {Path(x['path']).name: x for x in self.manifest['outputs']}
        for name in ['matched_sample_candidates.parquet', 'filter_measurements.parquet',
                     'portal_filter_metadata.parquet', 'aeth_inventory.json']:
            source = self.audit / name
            self._verify(source, outputs[name]['sha256'])
        self.catalog = pd.read_parquet(self.audit / 'matched_sample_candidates.parquet')
        self.measurements = pd.read_parquet(self.audit / 'filter_measurements.parquet')
        self.portal = pd.read_parquet(self.audit / 'portal_filter_metadata.parquet')
        self.inventory = json.loads((self.audit / 'aeth_inventory.json').read_text())
        assert not self.catalog.duplicated(['site', 'base_filter_id']).any()
        assert not self.portal.duplicated(['site_code', 'base_filter_id']).any()

        rows = []
        for site in ORDER:
            c = self.catalog.loc[self.catalog.site.eq(site)]
            rows.append(dict(
                site=site, physical_filters=len(c),
                hips_ftir_pairs=int(c.has_hips_ftir_pair.sum()),
                hips_ir_candidates=int((c.has_hips_ftir_pair & np.isfinite(c.candidate_ir_ebc_ugm3)).sum()),
                hips_window=int((c.has_hips_ftir_pair & c.portal_window_available).sum()),
                ec_conflicts=int(c.chemspec_ec_ugm3_conflict.sum()),
                excluded_pairs=int((c.has_hips_ftir_pair & c.is_excluded).sum()),
            ))
        self.counts = pd.DataFrame(rows).set_index('site')
        inputs = {Path(x['path']).name: x for x in self.manifest['inputs']}
        coverage = []
        for site in ORDER:
            path = PROCESSED_SITES_DIR / SITES[site]['file']
            self._verify(path, inputs[path.name]['sha256'])
            frame = pd.read_pickle(path)
            values = (pd.to_numeric(frame['data_completeness_pct'], errors='coerce')
                      if 'data_completeness_pct' in frame else pd.Series(np.nan, index=frame.index))
            coverage.append(pd.DataFrame({'site': site, 'source_row': np.arange(len(frame)),
                                          'saved_completeness_pct': values.to_numpy(),
                                          'coverage_field_present': 'data_completeness_pct' in frame,
                                          'source_file': str(path)}))
        self.coverage_rows = pd.concat(coverage, ignore_index=True)
        self.save_table(self.counts.reset_index(), 'site_counts')
        self.save_table(self.coverage_rows, 'saved_completeness_rows')

    def _verify(self, path, expected):
        observed = digest(path)
        if observed != expected:
            raise ValueError(f'{path} changed since the audit. Rerun the audit first.')
        self.used_sources.append({'path': str(path), 'sha256': observed})

    def save_table(self, frame, name):
        frame.to_parquet(self.tables / f'{name}.parquet', index=False)
        return frame

    def _figure(self, panels=1, widths=None):
        fig, axes = plt.subplots(1, panels, figsize=PlotConfig.get('figsize'),
                                 gridspec_kw={'width_ratios': widths} if widths else None)
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.19, top=0.84, wspace=0.50)
        return fig, np.atleast_1d(axes)

    @staticmethod
    def _style(ax, xlabel='', ylabel=''):
        style_axes(ax, xlabel, ylabel, show_legend=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=PlotConfig.get('font_size'))
        ax.grid(False)
        ax.set_axisbelow(True)

    @staticmethod
    def label(site):
        return site.replace('_', ' ')

    def pair_availability(self):
        fig, (ax,) = self._figure()
        y = np.arange(len(ORDER))
        for i, site in enumerate(ORDER):
            row = self.counts.loc[site]
            color = SITES[site]['color']
            ax.barh(i - 0.18, row.hips_ftir_pairs, height=0.30, color=color, alpha=0.28)
            ax.barh(i + 0.18, row.hips_ir_candidates, height=0.30, color=color)
            for dy, value in [(-0.18, row.hips_ftir_pairs), (0.18, row.hips_ir_candidates)]:
                ax.text(value + 2, i + dy, f'{value:,}', va='center', fontsize=20, color=INK)
        # Neutral swatches encode subset; hue consistently identifies the site.
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=MUTED, alpha=0.28, label='HIPS + FTIR EC'),
                           Patch(facecolor=MUTED, label='Also has an IR date candidate')],
                   loc='upper center', bbox_to_anchor=(0.57, 0.99), ncol=2, frameon=False, fontsize=18)
        ax.set_yticks(y, [self.label(s) for s in ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0, 218)
        self._style(ax, 'Physical filters')
        ax.grid(axis='x', alpha=0.17)
        return fig

    def metadata_availability(self):
        fig, (ax,) = self._figure()
        for i, site in enumerate(ORDER):
            row = self.counts.loc[site]
            available = int(row.hips_window)
            unavailable = int(row.hips_ftir_pairs - available)
            ax.barh(i, available, color=SITES[site]['color'], height=0.54)
            ax.barh(i, unavailable, left=available, color=PALE, height=0.54)
            if available:
                ax.text(available / 2, i, str(available), ha='center', va='center', fontsize=21, color=INK)
            if unavailable >= 10:
                ax.text(available + unavailable / 2, i, str(unavailable), ha='center', va='center', fontsize=21, color=INK)
            ax.text(row.hips_ftir_pairs + 3, i, f'{available}/{row.hips_ftir_pairs}', va='center', fontsize=21, color=INK)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=MUTED, label='Collection window recovered'),
                            Patch(facecolor=PALE, label='Window unavailable in inspected inputs')],
                   loc='upper center', bbox_to_anchor=(0.57, 1.0), ncol=1, frameon=False, fontsize=18)
        fig.subplots_adjust(top=0.80)
        ax.set_yticks(range(4), [self.label(s) for s in ORDER])
        ax.invert_yaxis()
        ax.set_xlim(0, 230)
        self._style(ax, 'Filters with HIPS and FTIR EC')
        ax.grid(axis='x', alpha=0.17)
        return fig

    def timestamp_hours(self):
        fig, (ax,) = self._figure()
        hour_rows = []
        for i, site in enumerate(ORDER):
            item = next(x for x in self.inventory if x['site'] == site)
            for hour, n in item['local_timestamp_hours'].items():
                hour, n = int(hour), int(n)
                hour_rows.append(dict(site=site, local_hour=hour, rows=n))
                ax.scatter(hour, i, s=120 + 1.3*n, color=SITES[site]['color'],
                           edgecolors=INK, linewidths=0.6, zorder=3)
                ax.annotate(f'{n:,}', (hour, i), xytext=(0, 24 if n > 2 else -27),
                            textcoords='offset points', ha='center', va='center', fontsize=20,
                            bbox={'facecolor':'white', 'edgecolor':'none', 'pad':0.5})
        self.save_table(pd.DataFrame(hour_rows), 'local_timestamp_hours')
        ax.axvline(9, color=MUTED, linestyle='--', linewidth=1.5, zorder=1)
        ax.text(9, -0.70, '09:00 target', ha='center', va='center', fontsize=19, color=MUTED,
                bbox={'facecolor':'white', 'edgecolor':'none', 'pad':1})
        ax.set_yticks(range(4), [self.label(s) for s in ORDER])
        ax.set_ylim(3.65, -0.90)
        ax.set_xlim(7.6, 17.1)
        ax.set_xticks([8, 9, 10, 12, 14, 15, 16], ['08:00','09:00','10:00','12:00','14:00','15:00','16:00'])
        self._style(ax, 'Saved local timestamp (labels show row counts)')
        ax.grid(axis='x', alpha=0.13)
        return fig

    def completeness(self):
        fig, (ax,) = self._figure()
        summaries = []
        for site in ORDER:
            group = self.coverage_rows.loc[self.coverage_rows.site.eq(site)]
            values = group.saved_completeness_pct.dropna().sort_values().to_numpy()
            summaries.append(dict(site=site, rows=len(group), finite_values=len(values),
                                  zero_values=int((values == 0).sum()),
                                  above_100=int((values > 100).sum()),
                                  maximum=float(values.max()) if len(values) else None))
            if not len(values):
                continue
            percent = 100*np.arange(1, len(values)+1)/len(values)
            ax.step(values, percent, where='post', color=SITES[site]['color'], linewidth=3,
                    label=f'{self.label(site)} (n={len(values):,})')
        ax.axvline(100, color=MUTED, linestyle='--', linewidth=1.4)
        maximum = self.coverage_rows.saved_completeness_pct.max()
        ax.annotate(f'JPL maximum: {maximum:.1f}%', (maximum, 100), xytext=(65, 74),
                    arrowprops={'arrowstyle':'-', 'color':MUTED}, fontsize=19, color=INK)
        ax.text(0.04, 0.27, 'Addis Ababa: no saved coverage field', transform=ax.transAxes,
                fontsize=19, color=MUTED)
        ax.set_xlim(-1.5, 108)
        ax.set_ylim(0, 105)
        self._style(ax, 'Saved completeness (%)', 'Cumulative share of saved rows (%)')
        ax.grid(alpha=0.16)
        ax.legend(loc='lower center', bbox_to_anchor=(0.68, 0.01), frameon=False, fontsize=18)
        self.save_table(pd.DataFrame(summaries), 'saved_completeness_summary')
        return fig

    def timing_status(self):
        fig, axes = self._figure(2, [1.5, 1])
        names = ['Hours agree', 'Hours differ', 'Invalid window']
        columns = ['consistent', 'different', 'invalid']
        records = []
        for site in ['Addis_Ababa','JPL']:
            p = self.portal.loc[self.portal.site_code.eq(SITES[site]['code'])]
            invalid = p.portal_timing_status.str.contains('invalid_or_nonpositive')
            consistent = p.portal_continuous_schedule_consistent
            records.append(dict(site=site, consistent=int(consistent.sum()), invalid=int(invalid.sum()),
                                different=int((~consistent & ~invalid).sum()), records=len(p)))
        statuses = pd.DataFrame(records).set_index('site')
        for j, site in enumerate(statuses.index):
            values = statuses.loc[site, columns].to_numpy(dtype=int)
            bars = axes[0].bar(np.arange(3) + (j-0.5)*0.32, values, width=0.32,
                              color=SITES[site]['color'], label=self.label(site))
            axes[0].bar_label(bars, padding=4, fontsize=19)
        axes[0].set_xticks(range(3), names)
        axes[0].set_ylim(0, 280)
        self._style(axes[0], ylabel='Portal filter records')
        axes[0].legend(frameon=False, fontsize=18)
        axes[0].grid(axis='y', alpha=0.17)
        issues = statuses[['different','invalid']].sum(axis=1)
        bars = axes[1].bar(range(2), issues, width=0.52,
                           color=[SITES[s]['color'] for s in issues.index])
        axes[1].bar_label(bars, labels=[f'{n}/{statuses.loc[s,"records"]}' for s,n in issues.items()],
                          padding=5, fontsize=22)
        axes[1].set_xticks(range(2), ['Addis Ababa','JPL'])
        axes[1].set_ylim(0, 20)
        axes[1].set_yticks([0,5,10,15,20])
        self._style(axes[1], ylabel='Records with a timing issue')
        axes[1].grid(axis='y', alpha=0.17)
        self.save_table(statuses.reset_index(), 'timing_status')
        return fig

    def collection_durations(self):
        fig, axes = self._figure(2)
        groups = self.portal.groupby(['site_code','portal_elapsed_hours','portal_hours_sampled'],
                                     dropna=False).size().rename('filters').reset_index()
        self.save_table(groups, 'collection_duration_groups')
        for ax, site in zip(axes, ['Addis_Ababa','JPL']):
            p = groups.loc[groups.site_code.eq(SITES[site]['code'])]
            # Full grouped inventory, including the zero-length window. Fit-free
            # scatter is a metadata comparison, not a measurement regression.
            crossplot_on_axes(ax, p.portal_elapsed_hours, p.portal_hours_sampled,
                              'Elapsed collection time (h)', 'Reported sampled time (h)',
                              color=SITES[site]['color'], one_to_one=False, equal_axes=False,
                              fit_line=False, stats_box=False, stats={})
            points = ax.collections[0]
            points.set_sizes(80 + 4*p.filters.to_numpy())
            points.set_alpha(0.70)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.plot([0,60],[0,60], '--', color=MUTED, linewidth=1.3)
            ax.set_xlim(-12, 260)
            ax.set_ylim(0, 64)
            ax.set_xticks([0,24,100,200,240])
            ax.set_yticks([0,24,48,60])
            ax.set_title(self.label(site), fontsize=22, color=INK, pad=16)
            self._style(ax, 'Elapsed collection time (h)', 'Reported sampled time (h)')
            ax.grid(alpha=0.14)
            if site == 'Addis_Ababa':
                ax.annotate('181 filters\n24 h / 24 h', (24,24), xytext=(65,39), fontsize=18,
                            arrowprops={'arrowstyle':'-', 'color':MUTED})
                ax.annotate('5 filters\n192–216 h / 24 h', (200,24), xytext=(130,6), fontsize=18,
                            arrowprops={'arrowstyle':'-', 'color':MUTED})
                ax.annotate('2 other timing issues', (8,23.5), xytext=(38,54), fontsize=17,
                            arrowprops={'arrowstyle':'-', 'color':MUTED})
            else:
                ax.annotate('249 filters\n24 h / 24 h', (24,24), xytext=(67,7), fontsize=18,
                            arrowprops={'arrowstyle':'-', 'color':MUTED})
                ax.annotate('15 filters\n237–238 h / 48 h', (237.7,48), xytext=(96,54), fontsize=18,
                            arrowprops={'arrowstyle':'-', 'color':MUTED})
                ax.annotate('1 filter: 23 h / 24 h', (23,24), xytext=(57,35), fontsize=17,
                            arrowprops={'arrowstyle':'-', 'color':MUTED})
        return fig

    def duration_examples(self):
        fig, (ax,) = self._figure()
        ids = ['ETAD-0241','USPA-0329','ETAD-0178']
        examples = self.portal.set_index('base_filter_id').loc[ids].reset_index()
        for i, row in examples.iterrows():
            site = next(s for s in ORDER if SITES[s]['code']==row.site_code)
            for dy, column, alpha in [(-0.18,'portal_elapsed_hours',0.30),(0.18,'portal_hours_sampled',1)]:
                value = row[column]
                ax.barh(i+dy, value, height=0.29, color=SITES[site]['color'], alpha=alpha)
                ax.text(value+3, i+dy, f'{value:g} h', va='center', fontsize=22, color=INK)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=MUTED, alpha=0.3,label='Elapsed collection time'),
                           Patch(facecolor=MUTED,label='Reported sampled time')],
                  loc='lower right', frameon=False, fontsize=19)
        ax.set_yticks(range(3), ids)
        ax.invert_yaxis()
        ax.set_xlim(0,280)
        self._style(ax, 'Hours')
        ax.grid(axis='x',alpha=0.17)
        self.save_table(examples, 'duration_examples')
        return fig

    def ec_conflicts(self):
        fig, axes = self._figure(2, [1.2,1])
        values = self.counts.ec_conflicts
        bars=axes[0].barh(range(4),values,height=0.55,color=[SITES[s]['color'] for s in ORDER])
        axes[0].bar_label(bars,padding=6,fontsize=21)
        axes[0].set_yticks(range(4),[self.label(s) for s in ORDER])
        axes[0].invert_yaxis()
        axes[0].set_xlim(0,205)
        self._style(axes[0],'Filters with conflicting EC values')
        axes[0].xaxis.label.set_fontsize(18)
        axes[0].grid(axis='x',alpha=0.17)
        e=self.measurements.loc[self.measurements.base_filter_id.eq('CHTS-0658') &
                                self.measurements.Parameter.eq('ChemSpec_EC_PM2.5')].sort_values('Concentration')
        assert len(e)==2 and e.source_row.nunique()==2
        for i,row in enumerate(e.itertuples()):
            axes[1].barh(i,row.Concentration,height=0.43,color=SITES['Beijing']['color'])
            axes[1].text(row.Concentration+0.025,i,f'{row.Concentration:.2f}',va='center',fontsize=22)
        axes[1].set_yticks(range(2),['Source row\n'+str(n) for n in e.source_row])
        axes[1].set_ylim(1.7,-0.7)
        axes[1].set_xlim(0,1.13)
        axes[1].set_xticks([0,0.5,1.0])
        axes[1].set_title('Example: CHTS-0658',fontsize=22,pad=18,color=INK)
        self._style(axes[1],'ChemSpec EC (µg/m³)')
        axes[1].grid(axis='x',alpha=0.17)
        self.save_table(e,'ec_conflict_example')
        self.save_table(self.catalog.loc[self.catalog.chemspec_ec_ugm3_conflict], 'ec_conflicting_filters')
        return fig

    def save(self, fig, key):
        png = self.output / f'{key}.png'
        svg = self.output / f'{key}.svg'
        fig.savefig(png, facecolor='white')
        fig.savefig(svg, facecolor='white', metadata={'Date':None})
        self.exports.append({'key':key,'png':str(png),'png_sha256':digest(png),
                             'svg':str(svg),'svg_sha256':digest(svg)})
        return png

    def write_manifest(self):
        document={'sources':self.used_sources,'figures':self.exports,
                  'code_sha256':digest(__file__),
                  'selection':'All audit records retained; grouping is descriptive, not exclusion.',
                  'saved_coverage':'Unverified input availability, not observed interval coverage.'}
        path=self.tables/'figure_manifest.json'
        path.write_text(json.dumps(document,indent=2)+'\n')
        return path
