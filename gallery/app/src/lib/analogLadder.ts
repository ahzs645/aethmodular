/**
 * One seasonal calibration out of the analog ladder (gallery/data/export_analog_ladder.py): the meeting's
 * calibration refitted at every cohort size from 20 to 1000, under each selection rule. A different count is a
 * different PLS fit, so these are that fit's own predictions, not the saved 500's filtered down.
 * Pure, so it is node-tested against the saved calibrations (analogLadder.test.mjs).
 */

type Cols = Record<string, (number | string | null)[]>

export interface LadderRule { key: string; label: string; note: string; per_group: boolean }
export interface LadderManifest {
  schema_version: number
  n_min: number
  n_max: number
  default_rule: string
  default_n: number
  rules: LadderRule[]
  method_slug: Record<string, string>
  group_slug: Record<string, string>
  /** the Addis filters in the order of the binary's rows */
  addis_media_id: number[]
  /** a rule, baseline and group that pick the same analogs as another, and so share its file */
  alias: Record<string, string>
}
export interface LadderHead {
  rule: string; method: string; group: string
  n_min: number; n_max: number; n_addis: number
  /** a size the protocol could not fit (its top filters span fewer than five sites) has k and the rest null */
  fits: Record<'train_n' | 'test_n' | 'test_sites', number[]>
    & Record<'k' | 'TOR_test_R2' | 'TOR_test_RMSE_ug' | 'lambda_tor_fabs' | 'lambda_ftir_tor' | 'lambda_ftir_fabs', (number | null)[]>
  /** the ranked analogs' references, rank 1 first */
  analogs: Cols
}
/** a calibration at one size, in the page's own shapes: rows for its held-out analogs and the Addis test set, its fit */
export interface LadderView {
  n: number
  analogs: Cols
  addis: Cols
  fit: { cohort_n: number; train_n: number; test_n: number; k: number | null; TOR_test_R2: number | null; TOR_test_RMSE_ug: number | null }
  lambda: { tor_vs_fabs: number | null; ftir_vs_tor: number | null; ftir_vs_fabs: number | null }
}

/** the header and predictions of a rule, baseline and group, relative to the export's folder */
export function ladderPath(m: LadderManifest, ruleKey: string, method: string, group: string) {
  const rule = m.rules.find((r) => r.key === ruleKey)
  const g = rule && !rule.per_group ? 'All Addis' : group
  const slug = m.method_slug[method], groupSlug = m.group_slug[g]
  if (!rule || !slug || !groupSlug) return null
  const stem = `${ruleKey}.${slug}.${groupSlug}`
  return m.alias?.[stem] ?? stem
}

/**
 * The calibration of size `n`. `addis` is the page's Addis table and `fitId` a saved fit of it: its rows give each
 * filter's date, season and HIPS; the binary replaces their predictions. `model`, `method` and `group` label the rows,
 * so the page's point builders find them as they find a saved calibration's.
 */
export function ladderView(head: LadderHead, bin: ArrayBuffer, manifest: LadderManifest, n: number,
  { addis, fitId, model, method, group }: { addis: Cols; fitId: string; model: string; method: string; group: string }): LadderView {
  const at = n - head.n_min, sizes = head.n_max - head.n_min + 1, maskBytes = Math.ceil(head.n_max / 8)
  if (at < 0 || at >= sizes) throw new RangeError(`no calibration of size ${n}`)
  const view = new DataView(bin)
  const addisStart = at * head.n_addis * 4
  const maskBase = sizes * head.n_addis * 4, heldBase = maskBase + sizes * maskBytes
  const before = head.fits.test_n.slice(0, at).reduce((a, b) => a + b, 0)

  // held-out analogs, rank order
  const ranks: number[] = []
  for (let r = 0; r < n; r++) {
    if (view.getUint8(maskBase + at * maskBytes + (r >> 3)) & (0x80 >> (r & 7))) ranks.push(r)
  }
  const ftir = ranks.map((_, i) => view.getFloat32(heldBase + (before + i) * 4, true))
  const analogs: Cols = Object.fromEntries(Object.entries(head.analogs).map(([c, v]) => [c, ranks.map((r) => v[r])]))
  const fill = (v: string) => ranks.map(() => v)
  Object.assign(analogs, { family: fill('seasonal'), model: fill(model), method: fill(method), group: fill(group), role: fill('test'), ftir_ec_ugm3: ftir })

  // the Addis table of one saved fit, with this calibration's predictions
  const pos = new Map(manifest.addis_media_id.map((id, i) => [String(id), i]))
  const rows = addis.fit_id.flatMap((f, i) => f === fitId ? [i] : [])
  const out: Cols = Object.fromEntries(Object.keys(addis).map((c) => [c, rows.map((i) => addis[c][i])]))
  out.ftir_ec_ugm3 = rows.map((i) => {
    const p = pos.get(String(addis.MediaId[i]))
    const y = p == null ? NaN : view.getFloat32(addisStart + p * 4, true)
    return Number.isFinite(y) ? y : null
  })

  const f = head.fits
  return {
    n, analogs, addis: out,
    fit: { cohort_n: n, train_n: f.train_n[at], test_n: f.test_n[at], k: f.k[at], TOR_test_R2: f.TOR_test_R2[at], TOR_test_RMSE_ug: f.TOR_test_RMSE_ug[at] },
    lambda: { tor_vs_fabs: f.lambda_tor_fabs[at], ftir_vs_tor: f.lambda_ftir_tor[at], ftir_vs_fabs: f.lambda_ftir_fabs[at] },
  }
}
