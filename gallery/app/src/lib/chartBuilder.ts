import type { IChannelScales, IChart, IMutField, IRow } from '@kanaries/graphic-walker'
import type { FilterRow, MetaFile } from './types.ts'
import { HIDDEN_DUPLICATES } from '../fieldLineage.ts'
import { withUnit } from './theme.ts'
import walkerContract from './graphicWalkerContract.json' with { type: 'json' }

export const WALKER_VERSION = walkerContract.testedVersion
const DIMENSIONS: IMutField[] = [
  { fid: 'id', name: 'Filter ID', analyticType: 'dimension', semanticType: 'nominal' },
  { fid: 'site', name: 'Site', analyticType: 'dimension', semanticType: 'nominal' },
  { fid: 'code', name: 'Site code', analyticType: 'dimension', semanticType: 'nominal' },
  { fid: 'date', name: 'Sampling date', analyticType: 'dimension', semanticType: 'temporal' },
  { fid: 'season', name: 'Season (selected calendar)', analyticType: 'dimension', semanticType: 'nominal' },
  { fid: 'year', name: 'Year', analyticType: 'dimension', semanticType: 'ordinal' },
  { fid: 'month', name: 'Month', analyticType: 'dimension', semanticType: 'ordinal' },
  { fid: 'excluded', name: 'Flagged exclusion', analyticType: 'dimension', semanticType: 'nominal' },
  { fid: 'exclusion_reason', name: 'Exclusion reason', analyticType: 'dimension', semanticType: 'nominal' },
]

// Vega treats dots/brackets as paths, and the query engine treats names as SQL identifiers.
// Encode the original key and unit without collisions, independent of field ordering or subset.
export const builderFieldId = (field: string, unit = '') => 'm_' + Array.from(`${field}\0${unit}`).map(c => c.codePointAt(0)!.toString(16)).join('_')

export function builderFields(meta: MetaFile): IMutField[] {
  return [
    ...DIMENSIONS.map(f => ({ ...f })),
    ...[...new Set(meta.fields)].filter(f => !HIDDEN_DUPLICATES.includes(f)).map(field => ({
      fid: builderFieldId(field, meta.field_units[field]), name: withUnit(field, meta.field_units),
      semanticType: 'quantitative' as const, analyticType: 'measure' as const,
    })),
  ]
}

/** Accept only the gallery's declared measurements; never infer numeric values or expose other row properties. */
export function builderRows(rows: FilterRow[], meta: MetaFile): IRow[] {
  const measures = [...new Set(meta.fields)].filter(f => !HIDDEN_DUPLICATES.includes(f))
    .map(field => [field, builderFieldId(field, meta.field_units[field])] as const)
  return rows.map(row => Object.fromEntries([
    ...DIMENSIONS.map(({ fid }) => {
      const value = row[fid]
      return [fid, fid === 'date' ? (typeof value === 'string' && value && Number.isFinite(Date.parse(value)) ? value : null)
        : typeof value === 'number' && !Number.isFinite(value) ? null : value ?? null]
    }),
    ...measures.map(([field, fid]) => [fid, typeof row[field] === 'number' && Number.isFinite(row[field]) ? row[field] : null]),
  ]))
}

export const builderSchema = (fields: IMutField[]) => JSON.stringify(fields.map(f => [f.fid, f.name, f.analyticType, f.semanticType]).sort((a, b) => String(a[0]).localeCompare(String(b[0]))))

/** Keep the gallery's site/calendar colours when those categories are placed on Color. */
export function builderScales(meta: MetaFile): IChannelScales {
  const colours = new Map([
    ...meta.sites.flatMap(site => [[site.name, site.color], [site.code, site.color]] as [string, string][]),
    ...meta.seasons.map(season => [season.name, season.color] as [string, string]),
  ])
  return { color: ({ semanticType, values }) => {
    if (semanticType === 'quantitative') return {}
    const domain = [...new Set(values.filter(value => value != null))]
    return domain.length && domain.every(value => typeof value === 'string' && colours.has(value))
      ? { domain, range: domain.map(value => colours.get(value)!) } : {}
  } }
}

export interface BuilderDraft {
  format: 'aeth-gallery-chart-builder'
  version: 1
  graphicWalkerVersion: typeof WALKER_VERSION
  fields: IMutField[]
  charts: IChart[]
  context: { galleryHash: string; exportedAt: string; dataGenerated: string; filterIds: string[]; basis: string; mac: number; seasonConvention: string }
}

export function builderDraft(charts: IChart[], fields: IMutField[], rows: FilterRow[], meta: MetaFile, galleryHash: string): BuilderDraft {
  return {
    format: 'aeth-gallery-chart-builder', version: 1, graphicWalkerVersion: WALKER_VERSION, fields, charts,
    context: { galleryHash, exportedAt: new Date().toISOString(), dataGenerated: meta.generated, filterIds: rows.map(r => r.id),
      basis: meta.basis ?? 'conc', mac: meta.mac_value, seasonConvention: meta.season_convention },
  }
}

/** This format carries chart definitions and selection provenance, never a substitute data source. */
export function readBuilderDraft(value: unknown, fields: IMutField[]): BuilderDraft {
  if (!value || typeof value !== 'object') throw new Error('Choose an Air Gallery chart-builder draft.')
  const draft = value as BuilderDraft
  if (draft.format !== 'aeth-gallery-chart-builder' || draft.version !== 1 || !walkerContract.readableDraftVersions.includes(draft.graphicWalkerVersion))
    throw new Error('This draft uses an unsupported chart-builder format or version.')
  if (!Array.isArray(draft.fields) || draft.fields.some(f => !f || typeof f.fid !== 'string') || builderSchema(draft.fields) !== builderSchema(fields))
    throw new Error('The draft’s fields or units differ from this view. Reopen its original gallery selection and try again.')
  if (!Array.isArray(draft.charts) || !draft.charts.length || draft.charts.some(chart =>
    !chart || typeof chart.visId !== 'string' || typeof chart.name !== 'string' || !chart.config || !Array.isArray(chart.config.geoms)
    || !chart.layout || !chart.encodings || ['dimensions', 'measures', 'rows', 'columns', 'filters'].some(key => !Array.isArray(chart.encodings[key as keyof typeof chart.encodings]))))
    throw new Error('The draft does not contain valid chart definitions.')
  if (!draft.context || !Array.isArray(draft.context.filterIds) || draft.context.filterIds.some(id => typeof id !== 'string'))
    throw new Error('The draft is missing its original filter selection.')
  return draft
}
