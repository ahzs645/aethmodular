import { Combobox } from '@/components/Combobox'
import { DYN_COLORS, DYN_VARS, presetsFor, type DynCond, type DynRule, type DynSeason } from '@/lib/dynamicSeasons'
import { XIcon } from '@/components/Icons'

const CUSTOM = 'custom'

/**
 * A site's dynamic seasons (lib/dynamicSeasons): start from a preset, then edit. Rule seasons are
 * checked top to bottom, each day taking the first whose conditions all hold; the last takes the
 * rest. Any edit turns the preset into "Custom". Everything travels in the link (dyn=).
 */
export function DynamicSeasonEditor({ code, rule, onChange }: { code: string; rule: DynRule; onChange: (r: DynRule) => void }) {
  const presets = presetsFor(code)
  const edit = (r: DynRule) => onChange({ ...r, preset: undefined })
  const pick = (key: string) => {
    const p = presets.find((x) => x.key === key)
    if (p) onChange(structuredClone({ ...p.rule, preset: p.key }))
  }
  const num = (value: number, set: (v: number) => void, opts: { step?: number; min?: number; width?: number; label: string }) => (
    <input type="number" className="select dyn-num" style={{ width: opts.width ?? 58 }} step={opts.step ?? 1} min={opts.min} value={value} aria-label={opts.label}
      onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) set(v) }} />
  )
  return (
    <div className="dyn-editor">
      <div className="dyn-preset">
        <span className="dyn-label">Start from</span>
        <Combobox label="" title="Preset rule for this site" value={rule.preset ?? CUSTOM} minWidth={170}
          options={[...presets.map((p) => ({ key: p.key, label: p.label })), ...(rule.preset ? [] : [{ key: CUSTOM, label: 'Custom (edited)' }])]}
          onChange={pick} />
      </div>
      {rule.kind === 'onset' ? (
        <div className="dyn-onset">
          <label>{num(rule.window, (v) => v >= 3 && edit({ ...rule, window: Math.round(v) }), { min: 3, label: 'Window (days)' })} day mean</label>
          <label>rains begin above {num(rule.onset, (v) => v >= 0 && v < rule.heavy && edit({ ...rule, onset: v }), { step: 0.5, min: 0, label: 'Rains begin (mm/day)' })} mm/day</label>
          <label>heavy above {num(rule.heavy, (v) => v > rule.onset && edit({ ...rule, heavy: v }), { step: 0.5, min: 0, label: 'Heavy rains (mm/day)' })} mm/day</label>
        </div>
      ) : (
        <RuleSeasons rule={rule} onChange={edit} num={num} />
      )}
    </div>
  )
}

type Num = (value: number, set: (v: number) => void, opts: { step?: number; min?: number; width?: number; label: string }) => React.ReactNode

function RuleSeasons({ rule, onChange, num }: { rule: Extract<DynRule, { kind: 'rules' }>; onChange: (r: DynRule) => void; num: Num }) {
  const last = rule.seasons.length - 1
  const setSeasons = (seasons: DynSeason[]) => onChange({ ...rule, seasons })
  const setSeason = (i: number, s: DynSeason) => setSeasons(rule.seasons.map((x, j) => (j === i ? s : x)))
  const setCond = (i: number, k: number, c: DynCond) => setSeason(i, { ...rule.seasons[i], when: rule.seasons[i].when.map((x, j) => (j === k ? c : x)) })
  const addCond = (i: number) => setSeason(i, { ...rule.seasons[i], when: [...rule.seasons[i].when, { v: 'p', w: 30, op: '>', x: 50, pct: true }] })
  const addSeason = () => setSeasons([...rule.seasons.slice(0, last), { name: `Season ${rule.seasons.length + 1}`, when: [{ v: 'p', w: 30, op: '>', x: 50, pct: true }] }, rule.seasons[last]])
  return (
    <div className="dyn-seasons">
      {rule.seasons.map((s, i) => (
        <div key={i} className="dyn-season">
          <div className="dyn-season-head">
            <span className="season-swatch" style={{ background: DYN_COLORS[i % DYN_COLORS.length] }} />
            <input className="select" value={s.name} maxLength={40} aria-label={`Season ${i + 1} name`} onChange={(e) => setSeason(i, { ...s, name: e.target.value })} />
            {rule.seasons.length > 2 && <button type="button" className="btn quiet x-btn" aria-label={`Remove ${s.name}`} title="Remove this season"
              onClick={() => setSeasons(rule.seasons.filter((_, j) => j !== i))}><XIcon /></button>}
          </div>
          {i === last ? <div className="dyn-cond dyn-otherwise">every other day</div> : <>
            {s.when.map((c, k) => {
              const unit = DYN_VARS.find((d) => d.key === c.v)?.unit ?? ''
              return <div key={k} className="dyn-cond">
                <span className="dyn-label">{k === 0 ? 'when' : 'and'}</span>
                {num(c.w, (v) => v >= 1 && setCond(i, k, { ...c, w: Math.round(v) }), { min: 1, width: 48, label: 'Window (days)' })}
                <span className="dyn-label">day</span>
                <select className="select" value={c.v} aria-label="Variable" onChange={(e) => setCond(i, k, { ...c, v: e.target.value as DynCond['v'] })}>
                  {DYN_VARS.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
                </select>
                <select className="select" value={c.op} aria-label="Above or below" onChange={(e) => setCond(i, k, { ...c, op: e.target.value as DynCond['op'] })}>
                  <option value=">">above</option><option value="<">below</option>
                </select>
                {num(c.x, (v) => setCond(i, k, { ...c, x: c.pct ? Math.min(100, Math.max(0, v)) : v }), { step: c.pct ? 5 : 0.5, width: 62, label: 'Threshold' })}
                <select className="select" value={c.pct ? 'pct' : 'abs'} aria-label="Threshold unit"
                  title="A value in the variable's own unit, or a percentile of this site's own rolling means"
                  onChange={(e) => setCond(i, k, { ...c, pct: e.target.value === 'pct' })}>
                  <option value="abs">{unit}</option><option value="pct">th pct (site)</option>
                </select>
                <button type="button" className="btn quiet x-btn" aria-label="Remove this condition" onClick={() => setSeason(i, { ...s, when: s.when.filter((_, j) => j !== k) })}><XIcon /></button>
              </div>
            })}
            <button type="button" className="btn quiet dyn-add" onClick={() => addCond(i)}>+ condition</button>
          </>}
        </div>
      ))}
      <div className="site-picker-bar">
        <button type="button" className="btn quiet" onClick={addSeason} disabled={rule.seasons.length >= 8}>+ Add season</button>
        <label className="dyn-minrun" title="Spells shorter than this join the season before them, so a few wet days don't make a season">
          shortest spell {num(rule.minRun, (v) => v >= 1 && onChange({ ...rule, minRun: Math.round(v) }), { min: 1, width: 48, label: 'Shortest spell (days)' })} days
        </label>
      </div>
    </div>
  )
}
