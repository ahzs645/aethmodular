import { useCallback, useMemo, useRef, type ReactNode } from 'react'
import * as d3 from 'd3'
import { useChartOptions } from '@/lib/chartOptions'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { wrapCategory } from '@/components/CategoryAxis'
import { Legend, useLegend } from '@/components/Legend'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { fmt } from '@/lib/stats'
import { INK, FONT, inkOnFill } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'
import { categoricalScale, hasNamedColours } from '@/lib/palettes'
import type { FilterRow, MetaFile, SeasonMeta } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { useChartLabel, useFitHeight } from '@/lib/chartFit'
import { fitInCircle, fitInRect } from '@/lib/fitLabel'
import { useHighlightActions } from '@/lib/highlight'
import { PanelAxesControl, usePanelAxes } from '@/lib/panelAxes'
import { collapsePanelAxes } from '@/lib/axes'
import { calendarShortName, seasonOfDate, seasonOnly } from '@/lib/seasons'
import { CompositionTimeline, stripSeasons, TIME_AXES, TIMELINE_H, TIME_SHAPES, TIMELINE_W, TimelineSizeControl, timelineAutoHeight, type TimeAxis, type TimePoint, type TimeShape, type TimelineSize } from '@/charts/partOfWhole/CompositionTimeline'
import { useSingleView } from '@/lib/singleView'
import { ComponentSetBar, ComponentSetPicker, insertInOrder, useComponentSets, useSetEditor, type BasisInfo, type Underlying } from '@/components/ComponentSetPicker'
import { describeField, fieldOrigin } from '@/lib/glossary'
import { useSearchParam } from '@/lib/url'
import { BUILT_IN_SETS, resolveSet, type Basis, type ComponentSetDef } from '@/lib/componentSets'
import { RECON_FIELDS, RECON_SOURCE, reconMeta, useReconstructedRows } from '@/lib/spartanReconstruction'

// Stacked bars are the recommended view (PI decision, 23 Sep 2026); the
// hierarchy encodings stay selectable for the within-site reading.
const MODES = ['Stacked bars', 'Treemap', 'Circle pack', 'Donut'] as const
const STATS = ['Mean', 'Median'] as const
const GROUPINGS = ['per site', 'by season', 'over time'] as const
const VIEW_LABEL: Record<string, string> = { 'over time': 'over time (each filter)' }
// sparse species (nitrite, bromide) leave few complete filters, so every filter with any is drawn by default
const MISSING = ['Leave missing out', 'Complete filters only'] as const
const SIDEBAR = 'Season calendar'
const UNIT_MODES = ['concentration (µg/m³)', 'composition (%)'] as const
/** the carbon at the foot of each bar, OC under EC */
const CARBON = ['OC (FTIR)', 'EC (FTIR)']
const RECON_FAMILY = 'SPARTAN reconstruction'
const RECON_SUBTITLE = "SPARTAN's own reconstruction of PM2.5 mass, the components of its network composition figure, filter by filter: with all twelve shown they sum to each filter's PM2.5 (35% RH), so a bar is the mean or median measured mass. Only filters with SPARTAN's mass balance are used. Dust is SPARTAN's fine soil (Al, Si, Ca, Fe and Ti with its mineral factors); TEO the trace element oxides; Sodium and Chloride split SPARTAN's sea salt (not the sodium ion); Water and Residual are by difference (Residual is mostly the non-carbon mass of the organics). Filters without FTIR OC or BC keep that mass in the residual, as in SPARTAN's balance. Components picks a ready-made set (composition (%) then divides by the shown total), the measured species (which do not close the mass), or your own: + makes one, with its components, stacking order and colours."

interface Kid { name: string; family: string; value: number; median: number; n: number }
/** one stacked bar: a site, or a site × season */
interface Bar { key: string; label: string; site: string; siteColor: string; barColor: string; pm25_mean: number | null; nRows: number; rows: FilterRow[]; children: Kid[] }
const DOMINANCE = ['20', '30', '40', '50', '60'] as const
interface Node { name: string; family?: string; children?: Node[]; value?: number; median?: number; n?: number }

/**
 * Treemap / circle pack / donut / stacked bars — PM2.5 composition per site, per season, or over time.
 *
 * One chart for every stacked composition in the gallery: the chemistry here, the FTIR functional
 * groups (FunctionalGroups) and the PMF sources (charts/pmf/SourceStack) all draw through it, so
 * they share their options. View: per site and by season stack the mean or median of each species;
 * over time stacks every filter on a date axis (CompositionTimeline: stacked area, daily bars or
 * streamgraph, with a season strip). A caller can give its own share (`share`, the PMF paper's
 * normalised fractions), extra season strips (`calendars`) and the nouns its readouts use.
 *
 * The estate has almost nothing in this category (1 stackplot, 2 pie calls),
 * because matplotlib makes it awkward. It is the natural reading of the
 * ChemSpec columns, so it goes in, with react-graph-gallery's two-level
 * hierarchy: measurement family (carbon / ions / crustal elements / trace metals) → species,
 * so "how much of Addis is crustal" is a border, not a sum in your head.
 * Stacked bars are the one encoding that compares *across* sites at a
 * glance; the others compare within one.
 */
export function CompositionTreemap({ rows: allRows, meta, initialGrouping = 'per site', id = 'composition',
  title = 'Stacked bars — what the PM2.5 is made of, per site and season', species: speciesOverride, notes, stackedOnly = false,
  palette = 'SPARTAN composition', subtitle, provenance, legendTopDown = false, views = GROUPINGS,
  initialUnits = 'concentration (µg/m³)', initialShape = 'Stacked area', share, massField = 'PM2.5 mass', calendars = [], initialCalendar = SIDEBAR,
  itemNoun = 'species', totalNoun = 'speciated', recordSource = 'the composition chart', extraTip, extraFields, dominantOf,
}: { rows: FilterRow[]; meta: MetaFile; initialGrouping?: (typeof GROUPINGS)[number]; id?: string;
  /** the views offered (the meeting page keeps its per-site and per-season bars) */
  views?: readonly (typeof GROUPINGS)[number][]
  initialUnits?: (typeof UNIT_MODES)[number]; initialShape?: TimeShape
  /** a filter's share of a species, in place of its µg/m³ over the shown total (PMF: the normalised fractions) */
  share?: (row: FilterRow, field: string) => number | null
  /** the measured mass the bars are read against; null for none (PMF sources) */
  massField?: string | null
  /** season strips for the time view besides the sidebar's calendar */
  calendars?: { value: string; seasons: SeasonMeta[]; title?: string }[]
  initialCalendar?: string
  /** what the stacked things are called ("species", "sources") and what their total is ("speciated", "apportioned") */
  itemNoun?: string; totalNoun?: string
  /** the filter record's source line, and a caller's own tooltip lines and record fields for a filter */
  recordSource?: string
  extraTip?: (row: FilterRow) => string[]
  extraFields?: (row: FilterRow) => [string, string | null][]
  /** the one thing that makes up most of a filter (PMF: its dominant source and share): the bars then count the filters
   *  one source dominates above a chosen share */
  dominantOf?: (row: FilterRow) => { label: string; share: number } | null
  title?: string; species?: string[]; notes?: ReactNode; stackedOnly?: boolean
  /** the categorical palette the chart opens on (lib/palettes); the reader can switch it */
  palette?: string
  subtitle?: string; provenance?: string
  /** the legend lists the species top of the stack first, as the bars read (a published figure's key) */
  legendTopDown?: boolean }) {
  const chartLabel = useChartLabel()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  // the composition chart itself (no caller's species) opens on SPARTAN's own mass reconstruction, the components of
  // its network figure. Its Components picker (components/ComponentSetPicker) offers ready-made sets on that basis or
  // on the measured species, and the reader's own (lib/componentSets); a set's basis decides which rows are drawn.
  // Where no filter has SPARTAN's balance (IMPROVE) only the measured species are offered
  const offersRecon = !speciesOverride
  const sets = useComponentSets()
  const [setId, setSetId] = useSearchParam<string>(`${id}.parts`, BUILT_IN_SETS[0].id)
  // over time, the reader's panel size: `<width>x<height>` in px, either 0 for Auto, '' for both Auto
  const [sizeParam, setSizeParam] = useSearchParam<string>(`${id}.size`, '')
  const timeSize = useMemo((): TimelineSize => {
    const [w, h] = sizeParam.split('x').map((v) => Math.round(Number(v)) || 0)
    const clamp = (v: number | undefined, [lo, hi]: [number, number]) => (v ? Math.min(hi, Math.max(lo, v)) : 0)
    return { w: clamp(w, TIMELINE_W), h: clamp(h, TIMELINE_H) }
  }, [sizeParam])
  const single = useSingleView().on
  const recon = useReconstructedRows(allRows, offersRecon)
  const reconAvailable = offersRecon && (recon === null || recon.rows.length > 0)
  const measuredDefault = BUILT_IN_SETS.find((x) => x.basis === 'measured')!
  const pickedSet = BUILT_IN_SETS.find((x) => x.id === setId) ?? sets.list.find((x) => x.id === setId) ?? BUILT_IN_SETS[0]
  const chosenSet = pickedSet.basis === 'recon' && !reconAvailable ? measuredDefault : pickedSet
  const reconOn = reconAvailable && chosenSet.basis === 'recon'
  const ownSet = 'colors' in chosenSet || Array.isArray(chosenSet.fields) ? (chosenSet as ComponentSetDef) : null
  // Species in the exporter's family order. PM2.5 mass is a separate
  // comparison measurement; the visible species total is the percentage denominator.
  const measuredFields = useMemo(() => {
    // a caller's own list may name fields the meta does not (the PMF sources): kept where a row has them
    if (speciesOverride) return speciesOverride.filter((f) => meta.fields.includes(f) || allRows.some((r) => typeof r[f] === 'number'))
    const wanted = new Set(['Ions', 'Crustal elements', 'Trace metals'])
    const fromGroups = meta.field_groups.filter((g) => wanted.has(g.label)).flatMap((g) => g.fields)
    // the FTIR carbon, not its ChemSpec restatement (fieldLineage RESTATES): the same product, and it covers as many
    // filters or more (every network row carries both; the focus sites' own FTIR has 597 against 543)
    return [...CARBON, ...fromGroups].filter((f) => meta.fields.includes(f))
  }, [meta, speciesOverride, allRows]) // eslint-disable-line react-hooks/exhaustive-deps
  // the set's components that the view has, in the set's order (a ready-made reconstruction set: the figure's legend
  // order, black carbon at the foot), under the reader's names for them: the rows carry each renamed component under
  // its new name too, so the stack, legend, labels, tooltips and filter record all read the reader's name
  const basisFields = reconOn ? RECON_FIELDS : measuredFields
  const setFields = useMemo(() => (offersRecon ? resolveSet(chosenSet, basisFields) : measuredFields), [offersRecon, chosenSet, basisFields, measuredFields])
  const names = ownSet?.labels
  const nameOf = useCallback((f: string) => names?.[f]?.trim() || f, [names])
  const fieldOf = useMemo(() => new Map(basisFields.map((f) => [nameOf(f), f])), [basisFields, nameOf])
  const origOf = (label: string) => fieldOf.get(label) ?? label
  const speciesFields = useMemo(() => setFields.map(nameOf), [setFields, nameOf])
  const baseRows = reconOn ? recon?.rows ?? [] : allRows
  const rows = useMemo(() => {
    const renamed = setFields.filter((f) => nameOf(f) !== f)
    if (!renamed.length) return baseRows
    return baseRows.map((r) => {
      const c: FilterRow = { ...r }
      for (const f of renamed) c[nameOf(f)] = r[f]
      return c
    })
  }, [baseRows, setFields, nameOf])
  const familyOf = (label: string) => {
    const f = origOf(label)
    return reconOn ? RECON_FAMILY : f.startsWith('OC') || f.startsWith('EC') ? 'Carbon' : meta.field_groups.find((g) => g.fields.includes(f))?.label ?? 'Other'
  }
  const families = useMemo(() => [...new Set(speciesFields.map(familyOf))], [speciesFields, reconOn, fieldOf]) // eslint-disable-line react-hooks/exhaustive-deps
  // the legend keys only species with a value in view: a site that never measured FTIR carbon (Buenos Aires,
  // before SPARTAN's stretched-Teflon filters) does not list it. Colours still follow speciesFields, so they stay put
  const shownSpecies = useMemo(() => speciesFields.filter((f) => rows.some((r) => typeof r[f] === 'number' && Number.isFinite(r[f] as number))), [rows, speciesFields])
  const shownFamilies = useMemo(() => families.filter((fm) => shownSpecies.some((f) => familyOf(f) === fm)), [families, shownSpecies]) // eslint-disable-line react-hooks/exhaustive-deps

  // filters with some but not all of the species: the time view either leaves them out or stacks what they have
  const incomplete = useMemo(() => rows.filter((r) => {
    const have = speciesFields.filter((f) => typeof r[f] === 'number' && Number.isFinite(r[f] as number)).length
    return have > 0 && have < speciesFields.length
  }).length, [rows, speciesFields])
  const [panelAxes] = usePanelAxes()

  // the options and when each applies (lib/chartOptions): the encodings, statistic and family level belong to the
  // summary views, shape and season strip to the time view; units to both (stacked bars or over time). Bars by
  // season need more than one season in view; a streamgraph of shares would only wobble a flat band
  const o = useChartOptions(id, [
    { key: 'parts', node: () => <ComponentSetPicker value={chosenSet.id} onChange={useSet} sets={sets} bases={bases} editor={editor} showing={showing} hiddenNote={hiddenNote} />, when: () => offersRecon },
    { key: 'bars', label: 'View', choices: views.map((v) => ({ value: v, label: VIEW_LABEL[v] ?? v, when: v === 'by season' ? (c) => (c.nSeasons as number) > 1 : undefined })),
      default: initialGrouping, title: 'One bar (or panel) per site, one per season within each site, or every filter on a date axis' },
    { key: 'mode', label: 'Encoding', choices: MODES, default: 'Stacked bars', hidden: 'Stacked bars', when: (c) => !stackedOnly && c.bars !== 'over time' },
    { key: 'units', label: 'Units', choices: UNIT_MODES, default: initialUnits, hidden: 'concentration (µg/m³)', when: (c) => c.mode === 'Stacked bars' },
    { key: 'shape', label: 'Shape', choices: TIME_SHAPES.map((v) => ({ value: v, when: v === 'Streamgraph' ? (c) => c.units !== 'composition (%)' : undefined })), default: initialShape,
      when: (c) => c.bars === 'over time', title: "Daily bars: one bar per sampling day on the date axis (the PMF paper's Figure 7). Streamgraph (concentration only): wiggle offset, inside-out order, no baseline, better shape" },
    { key: 'axis', label: 'Time axis', choices: TIME_AXES, default: 'Calendar dates', hidden: 'Calendar dates', when: (c) => c.bars === 'over time',
      title: 'Calendar dates: one continuous date axis. Seasons side by side: one block per season and year (Belg 2023, Belg 2024 …), grouped by season, the months between left out; filter one season in the sidebar to compare it year over year' },
    { key: 'stat', label: 'Statistic', choices: STATS, default: 'Mean', when: (c) => c.bars !== 'over time' },
    { key: 'twoLevel', label: 'Group by family', toggle: true, default: true, hidden: true, when: (c) => c.mode !== 'Stacked bars' && c.bars !== 'over time' && (c.nFamilies as number) > 1, title: 'Two levels of hierarchy: family → species' },
    { key: 'missing', label: `Missing ${itemNoun}`, choices: MISSING, default: 'Leave missing out', when: (c) => c.bars === 'over time' && (c.incomplete as number) > 0,
      title: `Leave missing out: every filter with any ${itemNoun === 'species' ? 'species' : itemNoun.replace(/s$/, '')} shown is drawn, its missing ones adding nothing, so its stack dips (the tooltip names them). Complete filters only: a filter is drawn when it has every one shown (hide sparse ones in the legend to admit more)` },
    { key: 'strip', label: 'Season strip', toggle: true, default: false, hidden: false, when: (c) => c.bars === 'over time',
      title: 'A strip of season bands along the top of each panel over time (off: the dates alone)' },
    { key: 'calendar', label: 'Seasons', choices: [{ value: SIDEBAR, label: `Sidebar: ${calendarShortName(meta.season_convention)}` }, ...calendars.map((c) => ({ value: c.value }))], default: initialCalendar, style: 'select',
      when: (c) => (c.bars === 'over time' && !!c.strip) || (c.bars === 'by season' && calendars.length > 0),
      title: "The seasons of the bars by season and of the strip over time: the Filters sidebar's season calendar (it follows any change there)" + calendars.map((c) => `, or ${c.title ?? c.value}`).join('') },
    { key: 'above', label: 'Count filters dominated above', choices: DOMINANCE.map((v) => ({ value: v, label: `${v} %` })), default: '30', style: 'select',
      when: (c) => !!dominantOf && c.bars !== 'over time' && c.mode === 'Stacked bars',
      title: `The share one ${itemNoun.replace(/s$/, '')} needs before a filter counts as dominated by it: counted under each bar and in the note below` },
    { key: 'pax', node: () => <PanelAxesControl dims={['site']} />, when: (c) => c.bars === 'over time' && (c.nSites as number) > 1 },
    { key: 'size', node: () => <TimelineSizeControl size={timeSize} auto={{ w: width, h: timelineAutoHeight(fit, composition.sites.length, single) }}
      onChange={(s) => setSizeParam(s.w || s.h ? `${s.w}x${s.h}` : '')} />, when: (c) => c.bars === 'over time' },
  ], { reconAvailable, nSeasons: new Set(rows.map((r) => r.season)).size, nFamilies: families.length, incomplete, nSites: new Set(rows.map((r) => r.site)).size })
  const mode = o.values.mode as (typeof MODES)[number]
  const stat = o.values.stat as (typeof STATS)[number]
  const grouping = o.values.bars as (typeof GROUPINGS)[number]
  const unitMode = o.values.units as (typeof UNIT_MODES)[number]
  const twoLevel = (o.values.twoLevel as boolean) && families.length > 1
  const overTime = grouping === 'over time'
  const shape = o.values.shape as TimeShape
  const hl = useHighlightActions()
  // a caller's own calendar (the PMF paper's seasons) groups the bars by season and draws the time strip
  const strip = calendars.find((c) => c.value === o.values.calendar)
  const colours = useColourScheme(id, 'categorical', palette)
  const lg = useLegend()
  const fam = useLegend()

  const fit = useFitHeight()
  const siteCount = new Set(rows.map(r => r.site)).size
  const cols = Math.max(1, Math.min(siteCount, width >= 600 ? 2 : 1))
  const panelW = Math.max(200, Math.floor((width - 24 * (cols - 1)) / cols))
  const panelH = fit === null ? 260 : Math.max(100, fit / Math.max(1, Math.ceil(siteCount / cols)))
  const UNITS = 'µg/m³'

  // Built from the live subset so the season/site filters above apply here too.
  const pct = unitMode === 'composition (%)'
  // a caller's share (PMF fractions), as a percentage, stands in for µg/m³ when the units are a share
  const ownShare = pct && !!share
  const UNITS_SHOWN = ownShare ? '%' : UNITS
  const composition = useMemo(() => {
    const pick = (r: FilterRow, f: string) => {
      if (!ownShare) return r[f]
      const v = share!(r, f)
      return v === null ? null : v * 100
    }
    const kidsOf = (sub: FilterRow[]) =>
      speciesFields
        .map((f): Kid | null => {
          const vals = sub.map((r) => pick(r, f)).filter((v): v is number => typeof v === 'number' && Number.isFinite(v)).sort(d3.ascending)
          if (!vals.length) return null
          return { name: f, family: familyOf(f), value: d3.mean(vals) ?? 0, median: d3.quantile(vals, 0.5) ?? 0, n: vals.length }
        })
        .filter((c): c is Kid => !!c)
    const pmOf = (sub: FilterRow[]) => {
      if (!massField) return null
      const pm = sub.map((r) => r[massField]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      return pm.length ? (d3.mean(pm) ?? null) : null
    }
    const sites = meta.sites
      .filter((s) => rows.some((r) => r.site === s.name))
      .map((s) => {
        const sub = rows.filter((r) => r.site === s.name)
        return { code: s.code, name: s.name, color: s.color, pm25_mean: pmOf(sub), children: kidsOf(sub) }
      })
    // One bar per season per site, from the season each row already carries
    // (App relabels it under the calendar picked in the subset bar), ordered as
    // that site's calendar lists them; under the per-site calendar the names are
    // site-qualified ("Addis Ababa · Dry …"), so the prefix is dropped for the label.
    const seasonBars: Bar[] = meta.sites
      .filter((s) => rows.some((r) => r.site === s.name))
      .flatMap((s) => {
        const sub = rows.filter((r) => r.site === s.name)
        const cal = strip?.seasons ?? seasonsForSite(meta, s.name)
        const seasonOf = (r: FilterRow) => (strip ? seasonOfDate(r.date, r.month as number | null, strip.seasons)?.name ?? '' : r.season)
        const order = [...cal.map((c) => c.name), ...new Set(sub.map(seasonOf).filter((n) => !cal.some((c) => c.name === n)))]
        return order
          .map((name): Bar | null => {
            const ss = sub.filter((r) => seasonOf(r) === name)
            const kids = kidsOf(ss)
            // a season with filters but no speciated species (e.g. Delhi winter) has no bar to stack
            if (!ss.length || !kids.length) return null
            const label = name.startsWith(`${s.name} · `) ? name.slice(s.name.length + 3) : name
            return { key: `${s.code}|${name}`, label, site: s.name, siteColor: s.color, barColor: cal.find((c) => c.name === name)?.color ?? INK.muted, pm25_mean: pmOf(ss), nRows: ss.length, rows: ss, children: kids }
          })
          .filter((b): b is Bar => !!b)
      })
    return { units: UNITS, sites, seasonBars }
  }, [rows, meta, speciesFields, ownShare, share, massField, strip]) // eslint-disable-line react-hooks/exhaustive-deps

  // colours keyed on every component the basis offers, so a component keeps its colour when others are left out of
  // the set (and its palette colour when renamed); the reader's own colours (a component set's) over the palette's
  const ownColors = ownSet?.colors
  const paletteFor = useCallback((b: Basis) => categoricalScale(colours.name, b === 'recon' ? RECON_FIELDS : measuredFields, colours.reverse), [colours.name, colours.reverse, measuredFields])
  const color = useMemo(() => {
    const base = paletteFor(reconOn ? 'recon' : 'measured')
    const domain = base.domain()
    return d3.scaleOrdinal<string, string>().domain(domain.map(nameOf)).range(domain.map((f) => ownColors?.[f] || base(f)))
  }, [paletteFor, reconOn, ownColors, nameOf])
  // the measured species under a reconstructed component, from each filter's own row (the reconstruction's rows carry
  // its sodium, nitrate … under the ions' names)
  const origById = useMemo(() => new Map(allRows.map((r) => [r.id, r])), [allRows])
  const underOf = useCallback((f: string, rs: FilterRow[]): Underlying[] => (RECON_SOURCE[f]?.measured ?? []).filter((m) => meta.fields.includes(m)).map((m) => {
    const v = rs.map((r) => origById.get(r.id)?.[m]).filter((x): x is number => typeof x === 'number' && Number.isFinite(x))
    return { field: m, mean: v.length ? d3.mean(v)! : NaN, n: v.length }
  }), [meta.fields, origById])
  const sourceOf = (f: string, b: Basis) => (b === 'recon' ? RECON_SOURCE[f]?.from ?? '' : fieldOrigin(f) || describeField(f))
  const bases: Partial<Record<Basis, BasisInfo>> = {
    ...(reconAvailable ? { recon: { available: RECON_FIELDS, rows: recon?.rows ?? [], meta: reconMeta(meta), source: (f: string) => sourceOf(f, 'recon'), under: underOf } } : {}),
    measured: { available: measuredFields, rows: allRows, meta, source: (f: string) => sourceOf(f, 'measured') },
  }
  // what is showing: the set less anything hidden in the legend, the editor's starting point; clicking a set clears
  // the legend's hidden list, since it names the old set's components
  const hiddenFields = setFields.filter((f) => !lg.show(nameOf(f)))
  const showing: ComponentSetDef = { id: chosenSet.id, name: chosenSet.name, basis: chosenSet.basis, fields: setFields.filter((f) => !hiddenFields.includes(f)),
    ...(ownSet?.colors ? { colors: ownSet.colors } : {}), ...(ownSet?.labels ? { labels: ownSet.labels } : {}), ...(ownSet?.legend ? { legend: ownSet.legend } : {}) }
  const hiddenNote = hiddenFields.length ? `${hiddenFields.map(nameOf).join(', ')} ${hiddenFields.length === 1 ? 'is' : 'are'} hidden in the legend, so ${hiddenFields.length === 1 ? 'it starts' : 'they start'} unticked here: tick to keep.` : undefined
  const useSet = (setKey: string) => { setSetId(setKey); lg.setHidden(new Set()) }
  const editor = useSetEditor({ sets, bases, standardColor: (f, b) => paletteFor(b)(f), onUse: useSet })
  const isOwnSet = sets.list.some((x) => x.id === chosenSet.id)
  const editShowing = (fields = showing.fields) => (isOwnSet
    ? editor.open({ ...(chosenSet as ComponentSetDef), fields }, 'own', hiddenNote)
    : editor.open({ ...showing, fields, id: `c${Date.now().toString(36)}`, name: `${chosenSet.name} (edited)` }, 'copy', hiddenNote))
  const notShown = basisFields.filter((f) => !setFields.includes(f) && baseRows.some((r) => typeof r[f] === 'number'))
  const legendTop = offersRecon ? ownSet?.legend === 'top' : legendTopDown
  const familyColor = useMemo(() => categoricalScale(colours.name, families, colours.reverse, 'family'), [families, colours.name, colours.reverse])
  const val = (c: { value: number; median: number }) => (stat === 'Mean' ? c.value : c.median)
  // the family legend is live wherever families are drawn; hiding a family or a species drops it from every total
  // one family (the functional groups) has nothing to group: no family marks or legend bands
  const famOn = (twoLevel || mode === 'Stacked bars') && families.length > 1
  const vis = (k: Kid) => lg.show(k.name) && (!famOn || fam.show(k.family))
  const dimK = (k: { name: string; family?: string }) => lg.dim(k.name) * (famOn ? fam.dim(k.family ?? '') : 1)
  const famDimOf = (f: string) => (famOn ? fam.dim(f, 0.3) : 1)
  const shownSites = composition.sites.map((s) => ({ ...s, children: s.children.filter(vis) }))
  const bySeason = grouping === 'by season'
  const bars: Bar[] = bySeason
    ? composition.seasonBars.map((b) => ({ ...b, children: b.children.filter(vis) }))
    : shownSites.map((s) => ({ key: s.code, label: chartLabel(s.name), site: s.name, siteColor: s.color, barColor: s.color, pm25_mean: s.pm25_mean, nRows: rows.filter((r) => r.site === s.name).length, rows: rows.filter((r) => r.site === s.name), children: s.children }))

  /** family → species hierarchy (or flat), summed on the chosen statistic */
  const rootFor = (siteName: string, kids: Kid[]) => {
    const data: Node = twoLevel
      ? { name: siteName, children: families.map((fam) => ({ name: fam, children: kids.filter((k) => k.family === fam) })).filter((f) => f.children!.length) }
      : { name: siteName, children: kids }
    return d3.hierarchy<Node>(data).sum((d) => (d.children ? 0 : val(d as Kid))).sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  }
  const leafTip = (leaf: d3.HierarchyNode<Node>, total: number) => [
    leaf.data.name,
    `${leaf.data.family ?? ''}`,
    `${fmt(leaf.value, 3)} ${UNITS}`,
    `${(((leaf.value ?? 0) / total) * 100).toFixed(1)} % of ${totalNoun} mass`,
    `n = ${leaf.data.n}`,
  ]

  // the time view: the shown species bottom first, each site's season strip, the filter record on a click
  const timeSpecies = speciesFields.filter((f) => lg.show(f) && (!famOn || fam.show(familyOf(f))))
  const seasonsFor = useCallback((site: string): readonly SeasonMeta[] => strip?.seasons ?? seasonsForSite(meta, site), [strip, meta])
  const showStrip = overTime && (o.values.strip as boolean)
  const stripKey = showStrip ? stripSeasons(composition.sites.map((s) => seasonsFor(s.name))) : []
  const recordOf = (p: TimePoint, seasonName: string) => hl.openRecord({
    id: p.row.id, source: recordSource, site: p.row.site, date: p.row.date,
    fields: [
      ['season', seasonName],
      ...timeSpecies.map((f): [string, string] => {
        const c = typeof p.row[f] === 'number' ? (p.row[f] as number) : null
        const sh = share?.(p.row, f)
        return [f, c === null && sh == null ? '—' : [sh != null ? `${(sh * 100).toFixed(1)} %` : null, c !== null ? `${fmt(c, 2)} ${UNITS}` : null].filter(Boolean).join(' · ')]
      }),
      ...(extraFields?.(p.row) ?? []),
      // the measured species under the reconstructed components shown, from the filter's own row
      ...(reconOn ? [...new Set(timeSpecies.flatMap((l) => RECON_SOURCE[origOf(l)]?.measured ?? []))].filter((m) => meta.fields.includes(m)).map((m): [string, string] => {
        const v = origById.get(p.row.id)?.[m]
        return [`${m} (measured)`, typeof v === 'number' ? `${fmt(v, 3)} ${UNITS}` : '—']
      }) : []),
    ],
  })
  // the slice tooltips: what a component is made from, and the measured species under it in that bar's filters
  const kidSource = offersRecon ? (b: Bar, label: string) => {
    const f = origOf(label)
    const src = sourceOf(f, reconOn ? 'recon' : 'measured')
    const under = reconOn ? underOf(f, b.rows).filter((u) => u.n > 0) : []
    return [
      ...(label !== f ? [`(${f})`] : []),
      ...(src ? [src.length > 140 ? `${src.slice(0, 137)}…` : src] : []),
      ...(under.length ? [`measured here: ${under.map((u) => `${u.field} ${fmt(u.mean, 2)}`).join(' · ')} µg/m³`] : []),
    ]
  } : undefined

  // the dominance counts: under each bar, in each slice's tooltip, and in the note below the chart
  const thr = Number(o.values.above) / 100
  const showDominance = !!dominantOf && !overTime && mode === 'Stacked bars'
  const dominated = (rs: FilterRow[]) => rs.filter((r) => (dominantOf?.(r)?.share ?? 0) >= thr).length
  const dominance = showDominance ? {
    footer: (b: Bar) => `${dominated(b.rows)} > ${o.values.above}%`,
    kidTip: (b: Bar, name: string) => [
      `dominant on ${b.rows.filter((r) => dominantOf!(r)?.label === name).length} of ${b.nRows} filters`,
      ...(share ? [`above ${o.values.above} % on ${b.rows.filter((r) => (share(r, name) ?? 0) >= thr).length} of ${b.nRows}`] : []),
    ],
  } : undefined

  return (
    <ChartFrame
      id={id}
      title={title}
      subtitle={subtitle ?? (reconOn ? RECON_SUBTITLE : "Available species grouped by measurement family (carbon, ions, crustal elements, trace metals). Crustal elements and trace metals are element masses, without SPARTAN's oxide factors. Concentration stacks the selected mean or median in µg/m³. Composition (%) divides each positive, visible species statistic by the sum of those statistics; it is not a percentage of measured PM2.5 or an average of individual-filter percentages. Coverage can differ by species. Missing values are omitted, never filled with zero. Negative or zero aggregate species values have no positive slice. Hiding a species or family changes the percentage denominator. PM2.5 mass is a separate measurement: the difference is not a measured residual or full mass closure.")}
      provenance={provenance ?? (reconOn ? "SPARTAN public FilterBased ReconstrPM25 and ChemSpecPM25 (potassium), joined by filter ID (spartan_composition.json, gallery/data/export_spartan_composition.py) · SPARTAN's October 2026 network composition figure" : "1 stackplot + 2 pie calls in the estate · react-graph-gallery.com/barplot (stacked) · /treemap · /circular-packing · /donut")}
      controls={<>{o.controls()}{colours.picker}</>}
    >
      <div ref={wrapRef} className="chart-wrap" data-export-fixed-colours={[...color.range(), ...familyColor.range()].join(' ')} style={{ display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        {reconOn && recon === null ? <Empty>Loading SPARTAN's mass reconstruction…</Empty>
          : composition.sites.length === 0 ? <Empty>{reconOn ? "No filter in this subset has SPARTAN's mass balance: choose Measured species." : 'No sites in this subset.'}</Empty>
          : speciesFields.length === 0 && <Empty>No components chosen: pick some under Components.</Empty>}
        {overTime ? (
          <div style={{ width: '100%' }}>
            <CompositionTimeline rows={rows} sites={composition.sites} species={timeSpecies} color={color} dim={(f) => dimK({ name: f, family: familyOf(f) })}
              pct={pct} share={share} shape={shape} timeAxis={o.values.axis as TimeAxis} completeOnly={o.values.missing === 'Complete filters only'} sharedAxes={collapsePanelAxes(panelAxes, 'site') === 'Shared'}
              seasonsFor={seasonsFor} strip={showStrip} size={timeSize} tip={tip} units={UNITS} itemNoun={itemNoun} onRecord={recordOf} extraTip={extraTip} />
          </div>
        ) : mode === 'Stacked bars' ? (
          <div style={{ width: '100%' }}>
            <StackedBars bars={bars} grouped={bySeason} pct={pct} families={families} famMarks={famOn} solid={hasNamedColours(colours.name) || !!ownColors} val={val} dim={dimK} famDim={fam.dim} color={color} familyColor={familyColor} width={width} tip={tip} units={UNITS_SHOWN} statistic={stat.toLowerCase()} itemNoun={itemNoun} totalNoun={totalNoun} mass={!!massField} dominance={dominance} kidSource={kidSource} />
          </div>
        ) : (
          (bySeason ? composition.seasonBars.map(b => ({ code: b.key, name: `${b.site} · ${b.label}`, color: b.siteColor, pm25_mean: b.pm25_mean, children: b.children.filter(vis) })) : shownSites).map((site) => {
            const kids = site.children.filter((c) => val(c) > 0)
            const total = d3.sum(kids, val)
            const root = rootFor(site.name, kids)
            return (
              <div key={site.code} style={{ width: panelW }}>
                <p className="facet-title">
                  {chartLabel(site.name)}
                  <span style={{ fontWeight: 400, color: 'var(--ink-muted)', fontSize: 11.5 }}>
                    {massField && `PM2.5 ${fmt(site.pm25_mean, 1)} µg/m³ · `}{totalNoun} {fmt(total, 1)}
                  </span>
                </p>
                <svg width={panelW} height={panelH} className="animated">
                  {kids.length > 0 && mode === 'Treemap' && (() => {
                    d3.treemap<Node>().size([panelW - 1, panelH - 1]).paddingInner(2).paddingOuter(twoLevel ? 3 : 0).paddingTop(twoLevel ? 16 : 0).round(true)(root)
                    return (
                      <>
                        {twoLevel && root.children?.map((fam: any) => (
                          <g key={fam.data.name} pointerEvents="none">
                            <rect x={fam.x0} y={fam.y0} width={Math.max(0, fam.x1 - fam.x0)} height={Math.max(0, fam.y1 - fam.y0)} fill={familyColor(fam.data.name)} fillOpacity={0.1 * famDimOf(fam.data.name)} stroke={familyColor(fam.data.name)} strokeOpacity={0.5 * famDimOf(fam.data.name)} rx={3} />
                            {fam.x1 - fam.x0 > 60 && <text x={fam.x0 + 5} y={fam.y0 + 12} fontSize={10} fontWeight={600} fill={familyColor(fam.data.name)} fontFamily={FONT.family}>{fam.data.name} · {(((fam.value ?? 0) / total) * 100).toFixed(0)}%</text>}
                          </g>
                        ))}
                        {root.leaves().map((leaf: any, i) => {
                          const w = leaf.x1 - leaf.x0
                          const h = leaf.y1 - leaf.y0
                          if (w <= 0 || h <= 0) return null
                          const pct = ((leaf.value ?? 0) / total) * 100
                          // the name shrinks, then wraps, to stay inside its tile; the share line
                          // goes under it when there is room, otherwise the name takes the tile alone
                          const PCT_H = 12
                          const roomW = w - 10, roomH = h - 8
                          const withName = roomW > 14 ? fitInRect(leaf.data.name, roomW, roomH - PCT_H) : null
                          const label = withName ?? (roomW > 14 ? fitInRect(leaf.data.name, roomW, roomH) : null)
                          const withPct = !!withName
                          return (
                            <g key={i} opacity={dimK(leaf.data)} onMouseEnter={(e) => tip.show(e, leafTip(leaf, total))} onMouseLeave={tip.hide}>
                              <rect x={leaf.x0} y={leaf.y0} width={w} height={h} fill={color(leaf.data.name)} fillOpacity={0.85} stroke={INK.panel} rx={2} />
                              {label && (
                                <text x={leaf.x0 + 5} y={leaf.y0 + 4 + label.size} fontSize={label.size} fill={inkOnFill(color(leaf.data.name))} fontFamily={FONT.family} pointerEvents="none">
                                  {label.lines.map((line, j) => <tspan key={j} x={leaf.x0 + 5} dy={j ? label.lineHeight : 0}>{line}</tspan>)}
                                </text>
                              )}
                              {label && withPct && (
                                <text x={leaf.x0 + 5} y={leaf.y0 + 4 + label.size + label.lines.length * label.lineHeight + 1} fontSize={Math.min(9.5, label.size)} fill={inkOnFill(color(leaf.data.name))} fillOpacity={0.85} fontFamily={FONT.mono} pointerEvents="none">{pct.toFixed(1)}%</text>
                              )}
                            </g>
                          )
                        })}
                      </>
                    )
                  })()}
                  {kids.length > 0 && mode === 'Circle pack' && (() => {
                    d3.pack<Node>().size([panelW - 2, panelH - 2]).padding(twoLevel ? 4 : 2)(root)
                    return (
                      <g transform="translate(1,1)">
                        {twoLevel && root.children?.map((fam: any) => (
                          <g key={fam.data.name} pointerEvents="none">
                            <circle cx={fam.x} cy={fam.y} r={fam.r} fill={familyColor(fam.data.name)} fillOpacity={0.08 * famDimOf(fam.data.name)} stroke={familyColor(fam.data.name)} strokeOpacity={0.5 * famDimOf(fam.data.name)} />
                            {fam.r > 28 && <text x={fam.x} y={fam.y - fam.r + 11} textAnchor="middle" fontSize={9.5} fontWeight={600} fill={familyColor(fam.data.name)} fontFamily={FONT.family}>{fam.data.name}</text>}
                          </g>
                        ))}
                        {root.leaves().map((leaf: any, i) => {
                          // shrink, then wrap, so the name stays inside its circle
                          const label = leaf.r > 12 ? fitInCircle(leaf.data.name, leaf.r) : null
                          return (
                            <g key={i} opacity={dimK(leaf.data)} onMouseEnter={(e) => tip.show(e, leafTip(leaf, total))} onMouseLeave={tip.hide}>
                              <circle cx={leaf.x} cy={leaf.y} r={leaf.r} fill={color(leaf.data.name)} fillOpacity={0.85} stroke={INK.panel} />
                              {label && (
                                <text x={leaf.x} y={leaf.y - ((label.lines.length - 1) * label.lineHeight) / 2} textAnchor="middle" fontSize={label.size} fill={inkOnFill(color(leaf.data.name))} fontFamily={FONT.family} pointerEvents="none">
                                  {label.lines.map((line, j) => <tspan key={j} x={leaf.x} dy={j ? label.lineHeight : '0.35em'}>{line}</tspan>)}
                                </text>
                              )}
                            </g>
                          )
                        })}
                      </g>
                    )
                  })()}
                  {mode === 'Donut' && (() => {
                    const r = Math.min(panelW, panelH) / 2 - 8
                    const arcs = d3.pie<Kid>().value(val).sort(null)(kids)
                    const arc = d3.arc<d3.PieArcDatum<Kid>>().innerRadius(r * 0.55).outerRadius(r)
                    // family ring: a thin outer band per family, the two-level reading of a donut
                    const famArcs = d3.pie<{ name: string; v: number }>().value((d) => d.v).sort(null)(families.map((f) => ({ name: f, v: d3.sum(kids.filter((k) => k.family === f), val) })))
                    const famArc = d3.arc<d3.PieArcDatum<{ name: string; v: number }>>().innerRadius(r + 2).outerRadius(r + 7)
                    return (
                      <g transform={`translate(${panelW / 2},${panelH / 2})`}>
                        {arcs.map((a, i) => (
                          <path
                            key={i} d={arc(a) ?? ''} fill={color(a.data.name)} fillOpacity={0.88 * dimK(a.data)} stroke={INK.panel} strokeWidth={1}
                            onMouseEnter={(e) => tip.show(e, [a.data.name, a.data.family, `${fmt(val(a.data), 3)} ${UNITS}`, `${((val(a.data) / total) * 100).toFixed(1)} %`, `n = ${a.data.n}`])}
                            onMouseLeave={tip.hide}
                          />
                        ))}
                        {twoLevel && famArcs.map((a, i) => (
                          <path key={`f${i}`} d={famArc(a) ?? ''} fill={familyColor(a.data.name)} fillOpacity={0.7 * famDimOf(a.data.name)} onMouseEnter={(e) => tip.show(e, [a.data.name, `${fmt(a.data.v, 2)} ${UNITS}`, `${((a.data.v / total) * 100).toFixed(1)} % of ${totalNoun} mass`])} onMouseLeave={tip.hide} />
                        ))}
                        <text textAnchor="middle" dy="0.35em" fontSize={12} fill={INK.muted} fontFamily={FONT.mono}>{fmt(total, 1)}</text>
                      </g>
                    )
                  })()}
                </svg>
              </div>
            )
          })
        )}
        {tip.node}
      </div>
      <Legend items={[
        ...(famOn ? shownFamilies.map(f => ({ label: f, color: familyColor(f), shape: 'band' as const })) : []),
        ...(legendTop ? shownSpecies.slice().reverse() : shownSpecies).map(f => ({ label: f, parent: famOn ? familyOf(f) : undefined, color: color(f), shape: 'square' as const })),
      ]} hidden={new Set([...fam.hidden, ...lg.hidden])} onToggle={label => families.includes(label) ? fam.toggle(label) : lg.toggle(label)} onHover={label => { fam.setHover(label && families.includes(label) ? label : null); lg.setHover(label && !families.includes(label) ? label : null) }} highlighted={fam.hover ?? lg.hover} />
      {offersRecon && <ComponentSetBar notShown={notShown} hidden={hiddenFields} nameOf={nameOf} colorOf={(f) => ownColors?.[f] || paletteFor(reconOn ? 'recon' : 'measured')(f)} builtIn={!isOwnSet}
        onAdd={(f) => (isOwnSet ? sets.save({ ...(chosenSet as ComponentSetDef), fields: insertInOrder(setFields, f, basisFields) }) : editShowing(insertInOrder(showing.fields, f, basisFields)))}
        onDrop={() => { if (isOwnSet) { sets.save({ ...(chosenSet as ComponentSetDef), fields: showing.fields }); lg.setHidden(new Set()) } else editShowing() }}
        onEdit={() => editShowing()} />}
      {editor.node}
      {reconOn && recon && recon.missing > 0 && rows.length > 0 && <p className="chart-note">{recon.missing.toLocaleString()} of {allRows.length.toLocaleString()} filters in view have no SPARTAN mass balance (no reconstruction published for them) and are left out, so the PM2.5 read against a stack is over the same filters.</p>}
      {showDominance && <p className="chart-note">{dominated(rows)} of {rows.length} filters have one {itemNoun.replace(/s$/, '')} above {o.values.above} % of the filter's {totalNoun} total (the count under each bar is that bar's).</p>}
      {overTime && stripKey.length > 0 && <div className="legend">{stripKey.map((s) => <span key={s.name} className="legend-item"><span className="swatch square" style={{ background: s.color }} />{seasonOnly(s.name)}</span>)}
        <span className="legend-note">top strip: {strip ? strip.title ?? strip.value : meta.season_convention === 'rain_rule' ? "the rain rule, from this year's rain" : 'the selected season calendar'}</span></div>}
      {notes}
    </ChartFrame>
  )
}

/**
 * Stacked bars, species in family order with family boundaries marked. `pct`
 * normalises each bar to its speciated total (composition %); otherwise the
 * stack is the mean species concentration in µg/m³ on one shared axis.
 * `grouped` lays season bars out in per-site clusters with the site underneath.
 */
function StackedBars({ bars, grouped, pct, families, famMarks, solid, val, dim, famDim, color, familyColor, width, tip, units, statistic, itemNoun, totalNoun, mass, dominance, kidSource }: {
  bars: Bar[]
  grouped: boolean
  pct: boolean
  families: string[]
  /** the family brackets beside each bar (off with one family) */
  famMarks: boolean
  /** a palette of fixed, published colours (lib/palettes `named`): drawn opaque, so the bars show those colours exactly */
  solid: boolean
  val: (c: Kid) => number
  dim: (k: Kid) => number
  famDim: (family: string) => number
  color: d3.ScaleOrdinal<string, string>
  familyColor: d3.ScaleOrdinal<string, string>
  width: number
  tip: ReturnType<typeof useTooltip>
  units: string
  statistic: string
  itemNoun: string
  totalNoun: string
  /** a measured mass to read each bar against (PM2.5) */
  mass: boolean
  /** the filters one thing dominates: a count under each bar, and lines in each slice's tooltip */
  dominance?: { footer: (b: Bar) => string; kidTip: (b: Bar, name: string) => string[] }
  /** a slice's tooltip lines on what it is made from (the component set's measurement, the species under it) */
  kidSource?: (b: Bar, name: string) => string[]
}) {
  const chartLabel = useChartLabel()
  const margin = { top: 16, right: 20, bottom: grouped ? 100 : 64, left: 60 }
  const plotWidth = Math.max(width, (bars.length + new Set(bars.map(b => b.site)).size - 1) * 112 + margin.left + margin.right)
  const innerW = plotWidth - margin.left - margin.right
  const fit = useFitHeight()
  const innerH = fit === null ? 320 : Math.max(100, fit - margin.top - margin.bottom)
  // a spacer slot between site clusters so the seasons of one site read as a group
  const slots: string[] = []
  const siteOrder = [...new Set(bars.map((b) => b.site))]
  siteOrder.forEach((site, i) => {
    if (grouped && i > 0) slots.push(`__gap${i}`)
    for (const b of bars) if (b.site === site) slots.push(b.key)
  })
  const x = d3.scaleBand<string>().domain(slots).range([0, innerW]).padding(grouped ? 0.22 : 0.3)
  const bw = x.bandwidth()
  const gapW = x.step() - bw
  const kidsOf = (b: Bar) => families.flatMap((f) => b.children.filter((k) => k.family === f && val(k) > 0))
  const maxTotal = d3.max(bars, (b) => d3.sum(kidsOf(b), val)) ?? 1
  const y = d3.scaleLinear().domain([0, pct ? 1 : maxTotal || 1]).range([innerH, 0]).nice()
  const yLabel = pct ? `share of visible ${itemNoun} total (%)` : `${statistic} concentration (${units})`
  const famText = (v: number, total: number) => (pct ? `${Math.round((v / total) * 100)}%` : fmt(v, 1))
  return (
    <div className="chart-scroll" tabIndex={0} role="region" aria-label="Stacked bars, scroll horizontally for all seasons"><svg width={plotWidth} height={innerH + margin.top + margin.bottom} className="animated">
      <g transform={`translate(${margin.left},${margin.top})`}>
        {y.ticks(5).map((t) => (
          <g key={t} transform={`translate(0,${y(t)})`}>
            <line x2={innerW} stroke={INK.grid} />
            <text x={-8} dy="0.32em" textAnchor="end" fontSize={10.5} fill={INK.muted} fontFamily={FONT.family}>{pct ? `${Math.round(t * 100)}%` : fmt(t, 1)}</text>
          </g>
        ))}
        <text transform={`translate(${-margin.left + 13},${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill={INK.axis} fontFamily={FONT.family}>{yLabel}</text>
        {bars.map((bar) => {
          const kids = kidsOf(bar)
          const total = d3.sum(kids, val)
          const norm = pct ? total || 1 : 1
          const where = grouped ? `${bar.site} · ${bar.label}` : bar.site
          let acc = 0
          const bx = x(bar.key) ?? 0
          const famBounds: { name: string; v: number; y0: number; y1: number }[] = []
          let famAcc = 0
          for (const f of families) {
            const v = d3.sum(kids.filter((k) => k.family === f), val)
            if (v > 0) famBounds.push({ name: f, v, y0: famAcc / norm, y1: (famAcc + v) / norm })
            famAcc += v
          }
          // "Belg (Feb-May, short rains)" -> "Belg" over "Feb-May"; the full name is in the <title>
          const paren = bar.label.indexOf(' (')
          const seasonName = paren > 0 ? bar.label.slice(0, paren) : bar.label
          const seasonMonths = paren > 0 ? bar.label.slice(paren + 2).replace(/\)$/, '').split(',')[0] : ''
          return (
            <g key={bar.key}>
              {kids.map((k) => {
                const y0 = acc / norm
                acc += val(k)
                const y1 = acc / norm
                const h = y(y0) - y(y1)
                const share = total ? (val(k) / total) * 100 : 0
                return (
                  <g key={k.name} opacity={dim(k)} onMouseEnter={(e) => tip.show(e, [k.name, `${where} · ${k.family}`, `${fmt(val(k), 3)} ${units}`, `${share.toFixed(1)} % of the ${totalNoun} ${fmt(total, 1)} ${units}`, `${mass ? `PM2.5 ${fmt(bar.pm25_mean, 1)} µg/m³ · ` : ''}n = ${k.n} of ${bar.nRows} filters`, ...(dominance?.kidTip(bar, k.name) ?? []), ...(kidSource?.(bar, k.name) ?? [])])} onMouseLeave={tip.hide}>
                    <rect x={bx} y={y(y1)} width={bw} height={Math.max(0, h)} fill={color(k.name)} fillOpacity={solid ? 1 : 0.88} stroke={INK.panel} strokeWidth={0.6} />
                    {h > 13 && bw > 70 && <text x={bx + bw / 2} y={y(y1) + h / 2} dy="0.35em" textAnchor="middle" fontSize={9.5} fill={inkOnFill(color(k.name))} fontFamily={FONT.family} pointerEvents="none">{k.name} {pct ? `${share.toFixed(0)}%` : fmt(val(k), 1)}</text>}
                  </g>
                )
              })}
              {famMarks && famBounds.map((f) => (
                <g key={f.name} pointerEvents="none">
                  <rect x={bx + bw + 3} y={y(f.y1)} width={Math.min(5, gapW / 3)} height={Math.max(0, y(f.y0) - y(f.y1))} fill={familyColor(f.name)} fillOpacity={0.8 * famDim(f.name)} rx={1} />
                  {y(f.y0) - y(f.y1) > 12 && gapW > 26 && <text x={bx + bw + 10} y={(y(f.y0) + y(f.y1)) / 2} dy="0.35em" fontSize={9} fill={familyColor(f.name)} fontFamily={FONT.family}>{famText(f.v, total)}</text>}
                </g>
              ))}
              {grouped ? (
                <>
                  <rect x={bx} y={innerH + 3} width={bw} height={3} fill={bar.barColor} rx={1}><title>{bar.label}</title></rect>
                  <text x={bx + bw / 2} y={innerH + 17} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={INK.text} fontFamily={FONT.family}>{wrapCategory(seasonName, x.step() - 10).map((line, i) => <tspan x={bx + bw / 2} dy={i ? 13 : 0} key={i}>{line}</tspan>)}</text>
                  {seasonMonths && <text x={bx + bw / 2} y={innerH + 47} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.family}>{seasonMonths}</text>}
                  <text x={bx + bw / 2} y={innerH + 60} textAnchor="middle" fontSize={9} fill={INK.muted} fontFamily={FONT.mono}>{pct ? `n=${bar.nRows}` : fmt(total, 1)}{dominance ? ` · ${dominance.footer(bar)}` : ''}</text>
                </>
              ) : (
                <>
                  <text x={bx + bw / 2} y={innerH + 16} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={bar.siteColor} fontFamily={FONT.family}>{bar.label}</text>
                  {/* two short lines, so neighbouring bars' readouts never run into each other; on a narrow
                      bar the unit and the 'of PM2.5' words go (about 6 px a character at 10 px mono) */}
                  {(() => {
                    const room = bw + 10, wide = `${fmt(total, 1)} ${units}`.length * 6 <= room
                    return <>
                      <text x={bx + bw / 2} y={innerH + 30} textAnchor="middle" fontSize={10} fill={INK.muted} fontFamily={FONT.mono}>{wide ? `${fmt(total, 1)} ${units}` : fmt(total, 1)}</text>
                      {mass && <text x={bx + bw / 2} y={innerH + 42} textAnchor="middle" fontSize={9.5} fill={INK.muted} fontFamily={FONT.mono}>
                        <title>{`${fmt(total, 1)} of PM2.5 ${fmt(bar.pm25_mean, 1)} ${units}`}</title>{`of PM₂.₅ ${fmt(bar.pm25_mean, 1)}`.length * 5.8 <= room ? `of PM₂.₅ ${fmt(bar.pm25_mean, 1)}` : `/ ${fmt(bar.pm25_mean, 1)}`}</text>}
                      {!mass && dominance && <text x={bx + bw / 2} y={innerH + 42} textAnchor="middle" fontSize={9.5} fill={INK.muted} fontFamily={FONT.mono}>{dominance.footer(bar)}</text>}
                    </>
                  })()}
                </>
              )}
            </g>
          )
        })}
        {grouped && siteOrder.map((site) => {
          const mine = bars.filter((b) => b.site === site)
          const x0 = x(mine[0].key) ?? 0
          const x1 = (x(mine[mine.length - 1].key) ?? 0) + bw
          return (
            <g key={site} pointerEvents="none">
              <line x1={x0} x2={x1} y1={innerH + 71} y2={innerH + 71} stroke={mine[0].siteColor} strokeWidth={1.5} />
              <text x={(x0 + x1) / 2} y={innerH + 88} textAnchor="middle" fontSize={11.5} fontWeight={600} fill={mine[0].siteColor} fontFamily={FONT.family}>{chartLabel(site)}</text>
            </g>
          )
        })}
      </g>
    </svg></div>
  )
}
