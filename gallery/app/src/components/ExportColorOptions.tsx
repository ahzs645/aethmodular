import { Segmented } from '@/components/ChartFrame'
import { seasonColorRows, seasonExportColor } from '@/lib/exportSeasons'
import { useEffect, useState, type ReactNode } from 'react'
import { DEFAULT_EXPORT_COLORS, effectiveExportColors, hexColor, recommendExportColors, siteExportColor, type ExportColors, type ExportPalette } from '@/lib/exportColors'
import { FIT_STYLE } from '@/lib/regressionMethods'

const STANDARD_SWATCHES = [
  ['Black', '#000000'], ['Gray', '#777777'], ['White', '#ffffff'], ['Red', '#d62728'],
  ['Orange', '#ff7f0e'], ['Gold', '#d4a017'], ['Green', '#2ca02c'], ['Teal', '#008080'],
  ['Blue', '#1f77b4'], ['Navy', '#17365d'], ['Purple', '#9467bd'], ['Pink', '#e377c2'],
] as const

function ColorChoice({ label, value, standard, onChange, onStandard, icon, note }: { label: string; value: string; standard: string; onChange: (color: string) => void; onStandard?: () => void; icon?: ReactNode; note?: string }) {
  const [draft, setDraft] = useState(value)
  const [swatches, setSwatches] = useState(false)
  useEffect(() => setDraft(value), [value])
  return <div>
    <div className="export-color-choice">
      <span className="export-color-name" title={label}>{icon}{label}{note && <small>{note}</small>}</span>
      <button type="button" className="btn quiet export-swatch-toggle" onClick={() => setSwatches(!swatches)} aria-expanded={swatches}
        aria-label={`Standard colors for ${label}`} title="Choose a standard color">▦</button>
      <input type="color" value={value} onChange={e => onChange(e.target.value)} aria-label={`${label} color`} title="Choose a custom color" />
      <input type="text" className="select export-color-hex" value={draft} aria-label={`${label} hex color`} spellCheck={false}
        aria-invalid={!hexColor(draft)} onChange={e => { setDraft(e.target.value); const hex = hexColor(e.target.value); if (hex && /^#[\da-f]{6}$/i.test(e.target.value.trim())) onChange(hex) }}
        onBlur={() => { const hex = hexColor(draft); if (hex && hex !== value) onChange(hex); setDraft(hex ?? value) }} />
    </div>
    {swatches && <div className="export-color-swatches" role="group" aria-label={`${label} standard colors`}>
      <button type="button" className="btn quiet" onClick={() => onStandard ? onStandard() : onChange(standard)} title={`Original chart color: ${standard}`}>Chart standard</button>
      {STANDARD_SWATCHES.map(([name, color]) => <button type="button" key={name} className="export-color-swatch" style={{ background: color }}
        aria-label={`${label}: ${name} (${color})`} title={`${name} · ${color}`} aria-pressed={hexColor(value) === color} onClick={() => onChange(color)} />)}
    </div>}
  </div>
}

export function ExportColorOptions({ palette, value, onChange }: { palette: ExportPalette; value: ExportColors; onChange: (value: ExportColors) => void }) {
  if (!palette.sites.length && !palette.seasons.length && !palette.fits.length) return null
  const shown = effectiveExportColors(value)
  const customize = (patch: Partial<ExportColors>) => onChange({ ...value, ...patch, mode: 'custom' })
  const recommend = () => {
    const next = recommendExportColors(palette, shown.sameSites, value.consolidateSeasons)
    onChange({ ...next, seasons: { ...value.seasons, ...next.seasons }, sites: { ...value.sites, ...next.sites }, fits: { ...value.fits, ...next.fits } })
  }
  return <fieldset className="export-colors">
    <legend>Export colors</legend>
    <Segmented label="Color scheme" value={value.mode} options={['standard', 'custom'] as const} optionLabel={mode => mode === 'standard' ? 'Standard' : 'Custom'}
      onChange={mode => onChange({ ...value, mode })} title="Standard uses the chart's original colors. Custom restores your saved choices." />
    <p className="export-color-note">{value.mode === 'standard' ? 'Using the chart’s standard colors. Pick a swatch or enter a color to customize.' : 'Using your custom colors. Switching to Standard keeps these choices for later.'}</p>
    <div className="export-color-actions">
      <button type="button" className="btn" onClick={recommend} title="Recommend contrasting site, season and fit colors for a light background">Recommend colors</button>
      <button type="button" className="btn quiet" onClick={() => onChange({ ...DEFAULT_EXPORT_COLORS })}>Reset colors</button>
    </div>
    {palette.sites.length > 0 && <>
      <label className="export-same-color"><input type="checkbox" checked={shown.sameSites} onChange={e => customize({ sameSites: e.target.checked })} /> Same color for all sites</label>
      <div className="export-site-colors" aria-label="Site colors">
        {shown.sameSites
          ? <ColorChoice label={palette.sites.length === 1 ? palette.sites[0].label : 'All sites'} value={shown.shared} standard={DEFAULT_EXPORT_COLORS.shared} onChange={shared => customize({ shared })} />
          : palette.sites.map(site => <ColorChoice key={site.key} label={site.label} value={siteExportColor(site, shown)} standard={site.color}
            onChange={color => customize({ sites: { ...value.sites, [site.key]: color } })} />)}
      </div>
    </>}
    {palette.seasons.length > 0 && <>
      <label className="export-same-color"><input type="checkbox" checked={value.consolidateSeasons} onChange={e => onChange({ ...value, consolidateSeasons: e.target.checked })} /> Consolidate matching seasons across sites</label>
      <div className="export-site-colors" aria-label="Season colors">
        {seasonColorRows(palette.seasons, value.consolidateSeasons).map(season => <ColorChoice key={season.key} label={season.label}
          value={seasonExportColor(season, shown, palette.seasons)} standard={seasonExportColor(season, { ...DEFAULT_EXPORT_COLORS, consolidateSeasons: value.consolidateSeasons }, palette.seasons)} onChange={color => customize({ seasons: { ...value.seasons, [season.key]: color } })} />)}
      </div>
      <p className="export-color-note">Matching names and months share a color and one legend entry with combined counts. The consolidated legend is reflowed below the chart when “As on screen” is selected. Fits keep their original populations.</p>
    </>}
    {palette.fits.length > 0 && <div className="export-fit-colors" aria-label="Fit line colors">
      <strong>Fit lines</strong>
      {palette.fits.map(fit => {
        const color = shown.fits[fit.key] ?? fit.color
        return <ColorChoice key={fit.key} label={fit.label} value={color} standard={fit.color} onStandard={() => { const fits = { ...value.fits }; delete fits[fit.key]; customize({ fits }) }} note={!shown.fits[fit.key] && fit.mixed ? 'per group' : undefined}
          icon={<svg width="24" height="12" viewBox="0 0 24 12" aria-hidden="true"><path d="M0 6H24" fill="none" stroke={color} strokeWidth="2" strokeDasharray={FIT_STYLE[fit.key].dash} /></svg>}
          onChange={color => customize({ fits: { ...value.fits, [fit.key]: color } })} />
      })}
    </div>}
    <p className="export-color-note">Recommendations spread category hues apart and use darker fit colors for a light background. Fit dash patterns stay visible.</p>
  </fieldset>
}
