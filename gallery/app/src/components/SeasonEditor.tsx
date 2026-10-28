import { useState } from 'react'
import type { SeasonMeta } from '@/lib/types'
import { XIcon } from '@/components/Icons'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
/** colours for custom seasons, in order of creation */
export const CUSTOM_SEASON_COLORS = ['#6F86BA', '#67A96A', '#D99145', '#AA6B78', '#428EAF', '#B8860B', '#8A6FB0', '#5A9E9A']

/** "Dry (Nov–Mar)": the month range is added to the name, so labels say what the bin is. */
export function customSeasonLabel(name: string, months: number[]) {
  if (!months.length) return name
  // describe the run in calendar order, starting after a gap so Nov–Mar reads as one range
  const set = new Set(months)
  const start = MONTHS.findIndex((_, i) => set.has(i + 1) && !set.has(((i + 11) % 12) + 1))
  const ordered = start < 0 ? [...months].sort((a, b) => a - b) : Array.from({ length: 12 }, (_, k) => ((start + k) % 12) + 1).filter((m) => set.has(m))
  const contiguous = ordered.every((m, i) => i === 0 || m === (ordered[i - 1] % 12) + 1)
  const range = contiguous ? (ordered.length === 1 ? MONTHS[ordered[0] - 1] : `${MONTHS[ordered[0] - 1]}–${MONTHS[ordered[ordered.length - 1] - 1]}`) : ordered.map((m) => MONTHS[m - 1]).join(', ')
  return `${name} (${range})`
}

/** Custom seasons from the URL: [[name, [months]], …]; null when absent or malformed. */
export function parseCustomSeasons(v: string | null): { name: string; months: number[] }[] | null {
  if (!v) return null
  try {
    const raw = JSON.parse(v) as [string, number[]][]
    const out = raw.map(([name, months]) => ({ name: String(name).slice(0, 40), months: months.filter((m) => m >= 1 && m <= 12) })).filter((s) => s.name && s.months.length)
    return out.length ? out : null
  } catch { return null }
}
export const customSeasonsParam = (s: { name: string; months: number[] }[] | null) => (s ? JSON.stringify(s.map((x) => [x.name, x.months])) : null)

/** SeasonMeta for the charts: labelled with their month range, coloured in order. */
export function customSeasonMeta(s: { name: string; months: number[] }[]): SeasonMeta[] {
  return s.map((x, i) => ({ name: customSeasonLabel(x.name, x.months), months: x.months, color: CUSTOM_SEASON_COLORS[i % CUSTOM_SEASON_COLORS.length] }))
}

/**
 * Edit custom month bins: pick a season, then click months to paint them into it. Every
 * month always belongs to exactly one season, so the bins can never overlap or leave a gap.
 */
export function SeasonEditor({ seasons, onChange }: { seasons: { name: string; months: number[] }[]; onChange: (s: { name: string; months: number[] }[]) => void }) {
  const [active, setActive] = useState(0)
  const owner = (m: number) => seasons.findIndex((s) => s.months.includes(m))
  const paint = (m: number) => {
    if (active >= seasons.length) return
    // moving the last month out of a season would empty it; keep at least one month each
    const from = owner(m)
    if (from === active) return
    if (from >= 0 && seasons[from].months.length === 1) return
    onChange(seasons.map((s, i) => ({ ...s, months: i === active ? [...s.months, m].sort((a, b) => a - b) : s.months.filter((x) => x !== m) })))
  }
  const rename = (i: number, name: string) => onChange(seasons.map((s, j) => (j === i ? { ...s, name } : s)))
  const add = () => {
    // a new season takes one month from the largest season, so all twelve stay covered
    const big = seasons.reduce((b, s, i) => (s.months.length > seasons[b].months.length ? i : b), 0)
    if (seasons[big].months.length < 2) return
    const m = seasons[big].months[seasons[big].months.length - 1]
    onChange([...seasons.map((s, i) => (i === big ? { ...s, months: s.months.filter((x) => x !== m) } : s)), { name: `Season ${seasons.length + 1}`, months: [m] }])
    setActive(seasons.length)
  }
  const remove = (i: number) => {
    if (seasons.length < 2) return
    // its months go to the season before it (or after, for the first)
    const heir = i === 0 ? 1 : i - 1
    onChange(seasons.flatMap((s, j) => (j === i ? [] : j === heir ? [{ ...s, months: [...s.months, ...seasons[i].months].sort((a, b) => a - b) }] : [s])))
    setActive(Math.max(0, heir > i ? heir - 1 : heir))
  }
  return (
    <div className="season-editor">
      <div className="season-months" role="group" aria-label="Months; click to move a month into the selected season">
        {MONTHS.map((label, k) => {
          const i = owner(k + 1)
          return <button key={label} type="button" className="season-month" title={`${label} → ${seasons[active]?.name ?? ''}`}
            style={{ ['--chip' as string]: CUSTOM_SEASON_COLORS[i % CUSTOM_SEASON_COLORS.length] }} onClick={() => paint(k + 1)}>{label[0]}<span>{label.slice(1)}</span></button>
        })}
      </div>
      {seasons.map((s, i) => (
        <div key={i} className={`season-edit-row ${i === active ? 'on' : ''}`}>
          <button type="button" className="season-swatch" aria-label={`Paint months into ${s.name}`} aria-pressed={i === active}
            style={{ background: CUSTOM_SEASON_COLORS[i % CUSTOM_SEASON_COLORS.length] }} onClick={() => setActive(i)} />
          <input className="select" value={s.name} maxLength={40} aria-label={`Season ${i + 1} name`} onFocus={() => setActive(i)} onChange={(e) => rename(i, e.target.value)} />
          <span className="season-edit-range">{customSeasonLabel('', s.months).replace(/^ \(|\)$/g, '')}</span>
          {seasons.length > 1 && <button type="button" className="btn quiet x-btn" aria-label={`Remove ${s.name}`} onClick={() => remove(i)}><XIcon /></button>}
        </div>
      ))}
      <div className="site-picker-bar">
        <button type="button" className="btn quiet" onClick={add} disabled={seasons.length >= 8}>+ Add season</button>
        <span className="site-picker-count">pick a season, then click months</span>
      </div>
    </div>
  )
}
