# Network rank notebook views and controls

Network rank (`#netrank`) shows every eligible site once. The comparison
background can be IMPROVE, SPARTAN or both; IMPROVE is grey and other SPARTAN
sites are blue by default. Active sites from Samples always remain visible
as stars in their Samples colours, including when the comparison background
is the other network. A star replaces its site's background dot. Clicking
any site opens its network, rank, filter count, median and interquartile
range in the right sidebar; the searchable Site details picker provides the
same action by keyboard. On phones it uses the gallery's standard bottom
sheet.

## Notebook views

The source and Markdown cells of 232 repository notebooks were searched for
site rankings involving IMPROVE or SPARTAN. Three contain this rank-plot
family. Spectral-analog rankings and lists of individual matched filters are
different analyses and are not site-median rank plots.

| Notebook | Section or figures | Choices in the original plots |
|---|---|---|
| [Follow up mass and ratio plots](../research/spartan/follow_up_mass_and_ratio_plots.ipynb) | §4–5, figs 02–03; addendum A1/A3, figs 08–09 | fAbs/EC, OC/EC, OC/fAbs, iron/fAbs, iron/EC and iron/PM2.5. IMPROVE median and IQR in grey; SPARTAN inserted into the IMPROVE ranking. Four focus sites for the carbon views, all matched HIPS sites for the iron views. At least 50 IMPROVE or 10 SPARTAN filters; logarithmic axes for the iron panels. Iron/PM2.5 shown as percent. |
| [Site ratios in mass space](../research/spartan/site_ratios_mass_space.ipynb) | §2, figure R1 | OC mass/EC mass, effective MAC from optical depth and EC mass, and OC mass/optical depth. Three panels with the core four sites placed in the IMPROVE ranking; a mass-versus-concentration equivalence check follows. |
| [Priority figure export](../research/ftir_ec_phase3/ftir_51_priority_figure_export.ipynb) | §11, figures 11a/11b | Corrected OC/EC and OC/fAbs views using FTIR carbon for SPARTAN and TOR carbon for IMPROVE. Per-filter ratios, then site medians and IQR. OC, EC and fAbs must all be positive; at least 50 IMPROVE or 10 SPARTAN filters. |

The older follow-up and mass-space notebooks call public ChemSpec carbon
“thermal” and sometimes draw it alongside FTIR. Those are duplicate reports
of the same FTIR product, not independent measurements. The corrected
priority-figure notebook supersedes those labels and duplicated series.
Network rank retains one SPARTAN series per site.

## Gallery controls

| Control | Behaviour |
|---|---|
| Comparison networks | IMPROVE + SPARTAN, SPARTAN only, or IMPROVE only. This selects the background; active Samples sites stay visible in all modes. A single background network ranks against itself, with other active sites placed into that ordering. |
| Y axis title | Click the quantity label to open the same searchable measurement and ratio picker as the scatter plots. Site ratios contains the six notebook ratios; Define a ratio… builds a custom one. The separate Notebook view dropdown is removed. Existing notebook-view links still reopen their chosen quantity. |
| Ranking | Rank both networks together, or reproduce the notebook's SPARTAN placements in the IMPROVE ordering. Available when both networks are shown. |
| SPARTAN sites | All sites by default, or sites and filters in the current gallery selection. |
| SPARTAN background colours | One network colour, or the gallery's individual site colours where available. Active stars always retain their Samples colours. |
| Filters | OC, EC and fAbs all positive, or every filter with the requested measurement. Ratios require both parts to be positive on the same filter. |
| Minimum counts | Separate IMPROVE and SPARTAN background thresholds, defaulting to 50 and 10. Active stars need at least one valid filter under the population rule; sites without a valid value are named below the plot. |
| Scale | Linear or logarithmic. A zero or negative lower quartile triggers a labelled linear fallback. |
| Site details | Search for a site or click its mark to open its statistics in the right sidebar. |

IMPROVE uses the fixed per-site quantile export across all dates and seasons.
SPARTAN's All sites mode uses the full bundled network across all dates and
seasons, following the gallery's exclusion rules. Sites / filters in view
uses the current source and selection. Active stars always use the current
Samples filters, dates, seasons, source and exclusions. Their colour is the
site's current Samples colour. The background's exported medians and ranks
stay fixed; an active star is placed among the comparison sites other than
itself, and replaces its background dot. An active site never adds a site to
the reference population. Legend hiding changes background visibility and
the vertical range; active stars remain visible and the comparison
population stays intact. Network controls recompute ranks. Ratios are computed per physical filter before
site summarisation, never by dividing two site medians. Iron/PM2.5 is a
fraction in the gallery; multiply by 100 to compare with the notebook's
percent axis.

Active-site labels are drawn in a final foreground layer above all site marks,
with a panel-coloured halo for legibility in light and dark themes. That layer
does not intercept clicks on the marks below it.

## Units and interpretation

SPARTAN carbon is FTIR-derived; IMPROVE carbon is thermal-optical (TOR).
Network placements compare those products and cannot establish SPARTAN EC
accuracy. HIPS absorption is an optical measurement, and HIPS BC uses the
reader's MAC.

The mass-space notebook converts fAbs to filter optical depth using volume
and deposit area. The gallery's mass view converts concentration fields to
mass per filter and leaves fAbs in Mm⁻¹, so fAbs/EC in that view is not the
notebook's volume-free effective MAC. Network rank blocks comparisons with
the IMPROVE concentration quantiles when concentration powers do not cancel.
Ratios such as OC/EC and iron/EC can be compared in either basis when the
same filter volume is available for both parts. Optical-depth rank presets
are not provided without the corresponding volume and deposit-area
transformation.

Implementation: [NetworkRankPlacement](../gallery/app/src/charts/ranking/NetworkRankPlacement.tsx)
and [referenceRank](../gallery/app/src/lib/referenceRank.ts).
