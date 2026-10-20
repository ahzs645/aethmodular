/**
 * Calculated fields for rule editors (components/GuardrailEditor): a small, safe arithmetic
 * language over a chart's numeric fields, parsed to JsonLogic so the same engine that runs the
 * rules (lib/guardrails) evaluates it. Nothing is passed to eval or Function: the parser only
 * builds JsonLogic nodes, and only the names it is given resolve.
 *
 *   abs(intercept)              slope * 10 / 6              max(drySlope, belgSlope) - min(drySlope, belgSlope)
 *
 * Grammar: + - * / ^, unary minus, parentheses, numbers, field names, and the functions below.
 * A missing value anywhere makes the result missing (null), so a rule on it fails rather than
 * treating the gap as zero.
 */

export const FORMULA_FUNCTIONS: Record<string, { arity: [number, number]; help: string }> = {
  abs: { arity: [1, 1], help: 'absolute value' },
  sqrt: { arity: [1, 1], help: 'square root' },
  log10: { arity: [1, 1], help: 'base-10 logarithm' },
  ln: { arity: [1, 1], help: 'natural logarithm' },
  exp: { arity: [1, 1], help: 'e to the power' },
  min: { arity: [2, 8], help: 'smallest of the arguments' },
  max: { arity: [2, 8], help: 'largest of the arguments' },
}

export type Logic = number | { var: string } | { [op: string]: Logic[] }
export class FormulaError extends Error {
  at: number
  constructor(message: string, at: number) { super(message); this.at = at }
}

type Tok = { kind: 'num' | 'name' | 'op'; text: string; at: number }

function tokenize(src: string): Tok[] {
  const out: Tok[] = []
  let i = 0
  while (i < src.length) {
    const c = src[i]
    if (/\s/.test(c)) { i++; continue }
    const num = /^(\d+\.?\d*|\.\d+)(e[+-]?\d+)?/i.exec(src.slice(i))
    if (num) { out.push({ kind: 'num', text: num[0], at: i }); i += num[0].length; continue }
    const name = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i))
    if (name) { out.push({ kind: 'name', text: name[0], at: i }); i += name[0].length; continue }
    if ('+-*/^(),'.includes(c)) { out.push({ kind: 'op', text: c, at: i }); i++; continue }
    throw new FormulaError(`unexpected '${c}'`, i)
  }
  return out
}

/** Parse a formula over `names` (the fields it may use) to JsonLogic; throws FormulaError. */
export function parseFormula(src: string, names: readonly string[]): Logic {
  const toks = tokenize(src)
  let p = 0
  const peek = () => toks[p]
  const end = () => new FormulaError('the formula ends too soon', src.length)
  const expect = (text: string) => {
    const t = toks[p]
    if (!t) throw end()
    if (t.text !== text) throw new FormulaError(`expected '${text}'`, t.at)
    p++
  }
  const expr = (): Logic => {
    let left = term()
    while (peek()?.kind === 'op' && (peek().text === '+' || peek().text === '-')) { const op = toks[p++].text; left = { [op]: [left, term()] } }
    return left
  }
  const term = (): Logic => {
    let left = unary()
    while (peek()?.kind === 'op' && (peek().text === '*' || peek().text === '/')) { const op = toks[p++].text; left = { [op]: [left, unary()] } }
    return left
  }
  // as in mathematics, -x^2 is -(x^2): the sign applies after the power
  const unary = (): Logic => {
    if (peek()?.text === '-') { p++; return { '*': [-1, unary()] } }
    if (peek()?.text === '+') { p++; return unary() }
    return power()
  }
  const power = (): Logic => {
    const base = primary()
    if (peek()?.text === '^') { p++; return { pow: [base, unary()] } }   // right-associative: 2^3^2 = 2^9
    return base
  }
  const primary = (): Logic => {
    const t = toks[p++]
    if (!t) throw end()
    if (t.kind === 'num') return Number(t.text)
    if (t.text === '(') { const e = expr(); expect(')'); return e }
    if (t.kind === 'name') {
      if (peek()?.text === '(') {
        const fn = FORMULA_FUNCTIONS[t.text.toLowerCase()]
        if (!fn) throw new FormulaError(`unknown function '${t.text}'`, t.at)
        p++
        const args: Logic[] = []
        if (peek()?.text !== ')') { args.push(expr()); while (peek()?.text === ',') { p++; args.push(expr()) } }
        expect(')')
        if (args.length < fn.arity[0] || args.length > fn.arity[1]) throw new FormulaError(`${t.text} takes ${fn.arity[0] === fn.arity[1] ? fn.arity[0] : `${fn.arity[0]}–${fn.arity[1]}`} argument${fn.arity[1] > 1 ? 's' : ''}`, t.at)
        return { [`f_${t.text.toLowerCase()}`]: args }
      }
      if (!names.includes(t.text)) throw new FormulaError(`unknown field '${t.text}'`, t.at)
      return { var: t.text }
    }
    throw new FormulaError(`unexpected '${t.text}'`, t.at)
  }
  if (!toks.length) throw new FormulaError('the formula is empty', 0)
  const out = expr()
  if (p < toks.length) throw new FormulaError(`unexpected '${toks[p].text}'`, toks[p].at)
  return out
}

/** The field names a parsed formula reads. */
export function formulaFields(logic: Logic): string[] {
  if (typeof logic === 'number') return []
  if ('var' in logic && typeof logic.var === 'string') return [logic.var]
  return [...new Set(Object.values(logic as Record<string, Logic[]>).flat().flatMap(formulaFields))]
}

/** Evaluate parsed JsonLogic arithmetic on one row, null when any input is missing or the result is not finite. */
export function evalFormula(logic: Logic, row: Record<string, unknown>): number | null {
  const ev = (n: Logic): number => {
    if (typeof n === 'number') return n
    if ('var' in n && typeof n.var === 'string') { const v = row[n.var]; return typeof v === 'number' ? v : NaN }
    const [op, args] = Object.entries(n as Record<string, Logic[]>)[0]
    const a = args.map(ev)
    switch (op) {
      case '+': return a[0] + a[1]
      case '-': return a[0] - a[1]
      case '*': return a[0] * a[1]
      case '/': return a[0] / a[1]
      case 'pow': return a[0] ** a[1]
      case 'f_abs': return Math.abs(a[0])
      case 'f_sqrt': return Math.sqrt(a[0])
      case 'f_log10': return Math.log10(a[0])
      case 'f_ln': return Math.log(a[0])
      case 'f_exp': return Math.exp(a[0])
      case 'f_min': return Math.min(...a)
      case 'f_max': return Math.max(...a)
      default: return NaN
    }
  }
  const v = ev(logic)
  return Number.isFinite(v) ? v : null
}
