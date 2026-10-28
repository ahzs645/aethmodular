/**
 * The undo / redo stacks behind ⌘Z (lib/undo): every gallery setting lives in the URL (lib/url), so one
 * step is one URL state, query and hash together. Pure, so the stepping rules are tested without a page.
 *
 *   begin(before)  a change is starting from `before` (the URL as it was settled): the first call after a
 *                  settle remembers it; later calls before the next settle join the same step, so a drag,
 *                  a typed value or a burst of writes from one click is one undo
 *   settle(now)    the burst is over: a step from the remembered URL to `now` (none if they are equal)
 *   undo(now) / redo(now)   the URL to go to, or null at the end of the stack
 */
export interface UndoStep { from: string; to: string }

export class UrlHistory {
  private past: UndoStep[] = []
  private future: UndoStep[] = []
  private start: string | null = null
  private readonly limit: number

  constructor(limit = 100) { this.limit = limit }

  begin(before: string) {
    if (this.start === null) this.start = before
  }

  /** true when a burst is open (begin without a settle) */
  get pending() { return this.start !== null }

  settle(now: string): UndoStep | null {
    const from = this.start
    this.start = null
    if (from === null || from === now) return null
    const step = { from, to: now }
    this.past.push(step)
    if (this.past.length > this.limit) this.past.shift()
    this.future = []
    return step
  }

  undo(now: string): UndoStep | null {
    this.settle(now)
    const step = this.past.pop()
    if (!step) return null
    this.future.push(step)
    return step
  }

  redo(now: string): UndoStep | null {
    this.settle(now)
    const step = this.future.pop()
    if (!step) return null
    this.past.push(step)
    return step
  }

  get canUndo() { return this.past.length > 0 }
  get canRedo() { return this.future.length > 0 }
}

/**
 * The settings a step changed, for the "Undid …" note: query and hash keys whose values differ between two
 * URLs (search + hash). Dashboard tile prefixes (t3.) are folded into "dashboard tile".
 */
export function changedKeys(a: string, b: string): string[] {
  const parse = (u: string) => {
    const hashAt = u.indexOf('#')
    const search = hashAt < 0 ? u : u.slice(0, hashAt)
    const hash = hashAt < 0 ? '' : u.slice(hashAt + 1)
    const out = new Map<string, string>()
    for (const [k, v] of new URLSearchParams(search.replace(/^\?/, ''))) out.set(k, v)
    for (const [k, v] of new URLSearchParams(hash)) out.set(k, v)
    return out
  }
  const pa = parse(a), pb = parse(b)
  const keys = new Set([...pa.keys(), ...pb.keys()])
  const names = [...keys].filter((k) => pa.get(k) !== pb.get(k)).map((k) => (/^t\d+\./.test(k) ? 'dashboard tile' : k))
  return [...new Set(names)]
}

/** reader-facing names for the URL keys App writes (App.tsx writeParams); others show as their last segment */
const KEY_NAMES: Record<string, string> = {
  sites: 'sites', ssites: 'sites', isites: 'sites', net: 'network', dscope: 'dashboard scope',
  drop: 'exclusions', incl: 'exclusions', keep: 'exclusions', dropwhy: 'exclusions', excl: 'exclusions',
  from: 'dates', to: 'dates', seasons: 'seasons', seas: 'seasons', cs: 'seasons', cal: 'seasons', sse: 'seasons', dyn: 'seasons', rr: 'seasons',
  x: 'axis fields', y: 'axis fields', f: 'measurement', mac: 'MAC', basis: 'units', fb: 'FTIR baseline',
  ax: 'axes', eq: 'axes', log: 'axes', pax: 'panel axes', fits: 'fits', ci: 'fits', tab: 'tab', dash: 'dashboard',
  pin: 'pinned filter', grp: 'groupings', ratios: 'ratios', pmf: 'PMF', pmfon: 'PMF', pmfonly: 'PMF', pmfg: 'PMF',
}

/** "sites and dates", for the note after an undo */
export function describeChange(a: string, b: string): string {
  const names = [...new Set(changedKeys(a, b).map((k) => KEY_NAMES[k] ?? k.split('.').pop()!))]
  if (!names.length) return 'the last change'
  if (names.length > 3) return `${names.slice(0, 3).join(', ')} and more`
  return names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
}
