import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { QueryBuilder, type Field, type OptionGroup, type RuleGroupType } from 'react-querybuilder'
import 'react-querybuilder/dist/query-builder-layout.css'
import { FORMULA_FUNCTIONS } from '@/lib/formula'
import {
  describeGuardrail, evaluateGuardrail, formulaName, guardFields, guardRows, templatesFor,
  type Formula, type Guardrail, type GuardField, type GuardrailSource,
} from '@/lib/guardrails'

/**
 * The guardrail editor: a popup where a reader writes the rule that picks a specification
 * curve's red dots, instead of choosing from fixed presets. The chart's curve sits at the top
 * of the popup (`preview`, kept in view while the options scroll) and every edit is sent back
 * (`onPreview`), so the red dots move while the rule is written; Apply keeps it, Cancel puts
 * back what was there. react-querybuilder draws the rule
 * (AND / OR groups, NOT, a comparison per rule, against a value or another field); calculated
 * fields add formulas over the numeric fields (lib/formula); lib/guardrails runs it. The count
 * of passing configurations updates as the rule changes, before anything is applied.
 *
 * Named rule sets are kept in this browser (localStorage); the applied rule travels in the URL.
 */
const STORE = 'gallery.guardrails.v1'
interface SavedSet { name: string; rule: Guardrail }
const readSaved = (): SavedSet[] => { try { const v = JSON.parse(localStorage.getItem(STORE) ?? '[]'); return Array.isArray(v) ? v : [] } catch { return [] } }
const writeSaved = (sets: SavedSet[]) => { try { localStorage.setItem(STORE, JSON.stringify(sets)) } catch { /* private window: sets last this visit only */ } }

const NUMBER_OPS = ['<=', '<', '>=', '>', 'between', 'notBetween', '=', '!=', 'null', 'notNull']
const CATEGORY_OPS = ['in', 'notIn', '=', '!=']
const OP_LABELS: Record<string, string> = {
  '<=': '≤', '<': '<', '>=': '≥', '>': '>', between: 'between', notBetween: 'not between', '=': '=', '!=': '≠',
  null: 'is missing', notNull: 'is present', in: 'is one of', notIn: 'is not one of',
}

/** the editor's field list: grouped, with each kind's operators and value editor */
function builderFields(fields: GuardField[]): OptionGroup<Field>[] {
  const groups: OptionGroup<Field>[] = []
  for (const f of fields) {
    const field: Field = {
      name: f.name, value: f.name, label: f.unit ? `${f.label} (${f.unit})` : f.label, kind: f.kind, title: f.help,
      ...(f.kind === 'number' ? {
        inputType: 'number', defaultOperator: '<=', operators: NUMBER_OPS.map((name) => ({ name, value: name, label: OP_LABELS[name] })),
        valueSources: ['value', 'field'], comparator: 'kind',
      } : f.kind === 'category' ? {
        defaultOperator: 'in', operators: CATEGORY_OPS.map((name) => ({ name, value: name, label: OP_LABELS[name] })),
        valueEditorType: (op: string) => (op === 'in' || op === 'notIn' ? 'multiselect' : 'select'),
        values: f.values?.map((v) => ({ name: v.name, value: v.name, label: v.label })),
      } : {
        defaultOperator: '=', defaultValue: true, operators: [{ name: '=', value: '=', label: 'is' }],
        valueEditorType: 'checkbox',
      }),
    }
    const g = groups.find((x) => x.label === f.group)
    if (g) g.options.push(field)
    else groups.push({ label: f.group, options: [field] })
  }
  return groups
}

const CLASSES = {
  combinators: 'select', fields: 'select', operators: 'select', value: 'select', valueSource: 'select',
  addRule: 'btn quiet', addGroup: 'btn quiet', removeRule: 'btn quiet', removeGroup: 'btn quiet', notToggle: 'toggle',
}

export function GuardrailEditor({ title, src, initial, onApply, onPreview, preview, onClose }: {
  title: string
  src: GuardrailSource
  initial: Guardrail
  onApply: (rule: Guardrail) => void
  /** each edit, as it is made (null while a calculated field does not parse) */
  onPreview?: (rule: Guardrail | null) => void
  /** the live chart drawn above the options */
  preview?: ReactNode
  onClose: () => void
}) {
  const [query, setQuery] = useState<RuleGroupType>(initial.query)
  const [formulas, setFormulas] = useState<Formula[]>(initial.formulas)
  const [saved, setSaved] = useState<SavedSet[]>(readSaved)
  const [saveName, setSaveName] = useState('')
  const dialog = useRef<HTMLDivElement>(null)
  // focus the popup once when it opens; Esc cancels (via a ref, so typing never re-runs the focus)
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    dialog.current?.focus()
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') close.current() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [])

  const rows = useMemo(() => guardRows(src), [src])
  const baseFields = useMemo(() => guardFields(src), [src])
  const fields = useMemo(() => guardFields(src, formulas.filter((f) => f.formula.trim())), [src, formulas])
  const numericNames = baseFields.filter((f) => f.kind === 'number').map((f) => f.name)
  const rule: Guardrail = { query, formulas: formulas.filter((f) => f.formula.trim()) }
  const result = useMemo(() => evaluateGuardrail(rule, rows, numericNames), [query, formulas, rows]) // eslint-disable-line react-hooks/exhaustive-deps
  const passing = result.pass.filter(Boolean).length
  // the chart previews the rule being written
  const sendPreview = useRef(onPreview)
  sendPreview.current = onPreview
  useEffect(() => { sendPreview.current?.(result.error ? null : rule) }, [result]) // eslint-disable-line react-hooks/exhaustive-deps
  const groups = useMemo(() => builderFields(fields), [fields])
  const templates = templatesFor(src)

  const load = (r: Guardrail) => { setQuery(structuredClone(r.query)); setFormulas(structuredClone(r.formulas)) }
  const addFormula = () => {
    const label = `Calculated ${formulas.length + 1}`
    setFormulas((fs) => [...fs, { name: formulaName(label, [...baseFields.map((f) => f.name), ...fs.map((f) => f.name)]), label, formula: '' }])
  }
  const editFormula = (i: number, patch: Partial<Formula>) => setFormulas((fs) => fs.map((f, j) => {
    if (j !== i) return f
    const next = { ...f, ...patch }
    // the name follows the label (rules refer to the name); renaming a field in use leaves its rules empty
    if (patch.label !== undefined) next.name = formulaName(patch.label || 'calc', [...baseFields.map((x) => x.name), ...fs.filter((_, k) => k !== i).map((x) => x.name)])
    return next
  }))
  const saveSet = () => {
    const name = saveName.trim()
    if (!name) return
    const next = [...saved.filter((s) => s.name !== name), { name, rule }]
    setSaved(next); writeSaved(next); setSaveName('')
  }
  const removeSet = (name: string) => { const next = saved.filter((s) => s.name !== name); setSaved(next); writeSaved(next) }

  return createPortal(
    <div className="export-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div ref={dialog} tabIndex={-1} className="export-dialog guard-dialog" role="dialog" aria-modal="true" aria-label={`Guardrails for ${title}`}>
        <div className="guard-top">
          <header className="guard-head">
            <div><strong>Guardrails</strong><span className="export-title">{title}</span></div>
            <button type="button" className="btn quiet" onClick={onClose} aria-label="Cancel" title="Cancel (Esc): put the red dots back">✕</button>
          </header>
          {preview && <div className="guard-preview">{preview}</div>}
        </div>

        <div className="guard-start">
          <span className="control">Start from</span>
          {templates.map((t) => <button key={t.key} type="button" className="btn quiet" onClick={() => load(t.rule)}>{t.label}</button>)}
          <button type="button" className="btn quiet" onClick={() => load({ query: { combinator: 'and', rules: [] }, formulas: [] })}>Empty</button>
          {saved.length > 0 && <span className="guard-saved">
            <span className="control">Saved</span>
            {saved.map((s) => <span key={s.name} className="guard-chip">
              <button type="button" className="btn quiet" onClick={() => load(s.rule)} title="Load this rule set">{s.name}</button>
              <button type="button" className="btn quiet" onClick={() => removeSet(s.name)} aria-label={`Delete ${s.name}`} title="Delete">×</button>
            </span>)}
          </span>}
        </div>

        <section className="guard-builder">
          <QueryBuilder fields={groups} query={query} onQueryChange={setQuery} controlClassnames={CLASSES}
            showNotToggle resetOnFieldChange addRuleToNewGroups
            translations={{ addRule: { label: '+ Condition', title: 'Add a condition' }, addGroup: { label: '+ Group', title: 'Add a group of conditions' },
              removeRule: { label: '×', title: 'Remove this condition' }, removeGroup: { label: '×', title: 'Remove this group' }, notToggle: { label: 'Not', title: 'Pass when this group does NOT hold' } }}
            combinators={[{ name: 'and', value: 'and', label: 'All of' }, { name: 'or', value: 'or', label: 'Any of' }]} />
        </section>

        <section className="guard-formulas">
          <div className="guard-formulas-head">
            <strong>Calculated fields</strong>
            <span className="control">a formula over the fields below, usable in any condition (Calculated group)</span>
            <button type="button" className="btn quiet" onClick={addFormula}>+ Calculated field</button>
          </div>
          {formulas.map((f, i) => <div key={i} className="guard-formula">
            <input className="select" aria-label="Calculated field name" value={f.label} onChange={(e) => editFormula(i, { label: e.target.value })} style={{ width: 180 }} />
            <span>=</span>
            <input className="select guard-formula-text" aria-label={`Formula for ${f.label}`} value={f.formula} placeholder="e.g. abs(slope - 1) * 10"
              onChange={(e) => editFormula(i, { formula: e.target.value })} spellCheck={false} />
            <code className="guard-name" title="Its name in other formulas">{f.name}</code>
            <button type="button" className="btn quiet" onClick={() => setFormulas((fs) => fs.filter((_, j) => j !== i))} aria-label={`Remove ${f.label}`}>×</button>
          </div>)}
          {(formulas.length > 0) && <p className="guard-help">
            Fields: {baseFields.filter((f) => f.kind === 'number').map((f) => <code key={f.name} title={f.label}>{f.name}</code>)}
            {' '}· Functions: {Object.entries(FORMULA_FUNCTIONS).map(([k, v]) => <code key={k} title={v.help}>{k}()</code>)} · Operators: + − * / ^ ( )
          </p>}
        </section>

        <footer className="guard-foot">
          <div className="guard-result" aria-live="polite">
            {result.error ? <span className="guard-error">{result.error}</span>
              : <><strong>{passing.toLocaleString()}</strong> of {rows.length.toLocaleString()} configurations pass
                <span className="guard-words">{describeGuardrail(rule, fields)}</span></>}
          </div>
          <div className="guard-actions">
            <input className="select" placeholder="Name this rule set" value={saveName} onChange={(e) => setSaveName(e.target.value)} aria-label="Rule set name"
              onKeyDown={(e) => { if (e.key === 'Enter') saveSet() }} style={{ width: 170 }} />
            <button type="button" className="btn quiet" onClick={saveSet} disabled={!saveName.trim()}>Save</button>
            <span style={{ flex: 1 }} />
            <button type="button" className="btn quiet" onClick={onClose}>Cancel</button>
            <button type="button" className="btn" onClick={() => { onApply(rule); onClose() }} disabled={!!result.error}>Apply</button>
          </div>
        </footer>
      </div>
    </div>,
    document.body,
  )
}
