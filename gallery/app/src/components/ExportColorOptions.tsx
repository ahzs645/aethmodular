import { useEffect, useState, type ReactNode } from 'react'
import { DEFAULT_EXPORT_COLORS, hexColor, recommendExportColors, siteExportColor, type ExportColors, type ExportPalette } from '@/lib/exportColors'
import { FIT_STYLE } from '@/lib/regressionMethods'

function ColorChoice({ label, value, onChange, icon, note }: { label: string; value: string; onChange: (color: string) => void; icon?: ReactNode; note?: string }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  return <div className="export-color-choice">
    <span className="export-color-name" title={label}>{icon}{label}{note && <small>{note}</small>}</span>
    <input type="color" value={value} onChange={e => onChange(e.target.value)} aria-label={`${label} color`} />
    <input type="text" className="select export-color-hex" value={draft} aria-label={`${label} hex color`} spellCheck={false}
      aria-invalid={!hexColor(draft)} onChange={e => { setDraft(e.target.value); const hex = hexColor(e.target.value); if (hex && /^#[\da-f]{6}$/i.test(e.target.value.trim())) onChange(hex) }}
      onBlur={() => { const hex = hexColor(draft); if (hex && hex !== value) onChange(hex); setDraft(hex ?? value) }} />
  </div>
}

export function ExportColorOptions({ palette, value, onChange }: { palette: ExportPalette; value: ExportColors; onChange: (value: ExportColors) => void }) {
  if (!palette.sites.length && !palette.fits.length) return null
  const recommend = () => {
    const next = recommendExportColors(palette, value.sameSites)
    onChange({ ...next, sites: { ...value.sites, ...next.sites }, fits: { ...value.fits, ...next.fits } })
  }
  return <fieldset className="export-colors">
    <legend>Export colors</legend>
    <div className="export-color-actions">
      <button type="button" className="btn" onClick={recommend} title="Recommend contrasting site and fit colors for a light background">Recommend colors</button>
      <button type="button" className="btn quiet" onClick={() => onChange({ ...DEFAULT_EXPORT_COLORS })}>Reset colors</button>
    </div>
    {palette.sites.length > 0 && <>
      <label className="export-same-color"><input type="checkbox" checked={value.sameSites} onChange={e => onChange({ ...value, sameSites: e.target.checked })} /> Same color for all sites</label>
      <div className="export-site-colors" aria-label="Site colors">
        {value.sameSites
          ? <ColorChoice label={palette.sites.length === 1 ? palette.sites[0].label : 'All sites'} value={value.shared} onChange={shared => onChange({ ...value, shared })} />
          : palette.sites.map(site => <ColorChoice key={site.key} label={site.label} value={siteExportColor(site, value)}
            onChange={color => onChange({ ...value, sites: { ...value.sites, [site.key]: color } })} />)}
      </div>
    </>}
    {palette.fits.length > 0 && <div className="export-fit-colors" aria-label="Fit line colors">
      <strong>Fit lines</strong>
      {palette.fits.map(fit => {
        const color = value.fits[fit.key] ?? fit.color
        return <ColorChoice key={fit.key} label={fit.label} value={color} note={!value.fits[fit.key] && fit.mixed ? 'per group' : undefined}
          icon={<svg width="24" height="12" viewBox="0 0 24 12" aria-hidden="true"><path d="M0 6H24" fill="none" stroke={color} strokeWidth="2" strokeDasharray={FIT_STYLE[fit.key].dash} /></svg>}
          onChange={color => onChange({ ...value, fits: { ...value.fits, [fit.key]: color } })} />
      })}
    </div>}
    <p className="export-color-note">Recommendations spread site hues apart and use darker fit colors for a light background. Fit dash patterns stay visible.</p>
  </fieldset>
}
