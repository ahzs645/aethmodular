import { timeFormat } from 'd3'

/** Time-series axes keep short spans legible; long histories retain month/year labels. */
export function timeTickFormat([start, end]: [number, number]): (value: unknown) => string {
  const span = Math.abs(end - start), day = 86400000
  const pattern = span <= day ? '%H:%M' : span <= 14 * day ? '%d %b %H:%M' : span <= 120 * day ? '%d %b' : '%b %Y'
  const format = timeFormat(pattern)
  return (value) => format(value instanceof Date ? value : new Date(Number(value)))
}
