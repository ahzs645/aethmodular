import { useMemo } from 'react'
import { HeatMatrix, type HeatMatrixProps, type MatrixItem } from '@/components/HeatMatrix'
import { monthColumns } from '@/lib/heatMatrix'

export interface SiteMonthMatrixProps extends Omit<HeatMatrixProps, 'columns' | 'cell'> {
  /** 'YYYY-MM' (consecutive, lib/notebookAvailability calendarMonths) or calendar months '01'…'12' / 1…12 */
  months: readonly (string | number)[]
  /** the cell of row i in a month (the month key as given, as a string) */
  cell: (i: number, month: string, j: number) => ReturnType<HeatMatrixProps['cell']>
  /**
   * years: only the year at each January (and the first month); months: the year over the month
   * numbers where they fit; none: no column labels
   */
  monthLabels?: 'years' | 'months' | 'none'
}

/**
 * Sites (or instruments, or lanes) by month: the coverage and value grids of the Availability,
 * Evolution, AERONET and aethalometer pages, defined once on components/HeatMatrix. Year-months run
 * as consecutive columns grouped by year (a label at each year's first month, a rule at each
 * January); calendar months are labelled Jan…Dec (J…D when narrow). Everything else, from cell
 * colours and tooltips to tile fitting and the ramp, is HeatMatrix's.
 */
export function SiteMonthMatrix({ months, cell, monthLabels, ...rest }: SiteMonthMatrixProps) {
  const cols = useMemo(() => monthColumns(months), [months])
  const yearMonths = cols.some((c) => c.group)
  const labels = monthLabels ?? (yearMonths ? 'years' : 'months')
  const columns = useMemo<MatrixItem[]>(() => cols.map((c) => ({
    key: c.key, title: c.title, group: labels === 'none' ? undefined : c.group,
    label: labels === 'months' ? c.label : yearMonths || labels === 'none' ? '' : c.label, short: c.short,
  })), [cols, labels, yearMonths])
  const keys = useMemo(() => cols.map((c) => c.key), [cols])
  const byMonth = useMemo(() => (i: number, j: number) => cell(i, keys[j], j), [cell, keys])
  return <HeatMatrix colLabels={labels === 'none' || (yearMonths && labels === 'years') ? 'none' : 'fit'} colLabelSize={10.5}
    colGroupRules={yearMonths} cellWidth={yearMonths ? [3, 42] : [18, 56]} {...rest} columns={columns} cell={byMonth} />
}
