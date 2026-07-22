# Charcoal and biochar FTIR source archive

Downloaded 2026-07-22. The large files live under `downloads/`, which is
already excluded by the repository's top-level `downloads/` ignore rule.

## Machine-readable spectra

| Local file | Coverage | Shape | Source and license |
|---|---|---:|---|
| `downloads/datasets/Minatre_reference_spectra.csv` | Ten tree and shrub species heated at 200-800 C in a muffle furnace | 2,100 spectra x 2,648 columns | [Dryad dataset](https://doi.org/10.5061/dryad.cnp5hqcbj), CC0 1.0 |
| `downloads/datasets/Minatre_combustionfacility_spectra.csv` | Four vegetation species burned under instrumented combustion-facility conditions | 1,934 spectra x 2,647 columns | [Dryad dataset](https://doi.org/10.5061/dryad.cnp5hqcbj), CC0 1.0 |
| `downloads/datasets/McCall_multifeedstock_FTIR_spectral_data.xlsx` | **Unburned (0 C) and burned (200-700 C)** barley straw, chestnut wood, eucalyptus bark, miscanthus grass, pine bark, and rice husk | 162 spectra x 1,615 columns | [Biochar Stability Revealed by FTIR and Machine Learning](https://doi.org/10.1021/acssusresmgt.5c00104), [Europe PMC](https://europepmc.org/article/PMC/12105012), CC BY |
| `downloads/datasets/McCall_barley_FTIR_spectral_data.xlsx` | **Unburned barley straw and char made at 150-700 C** | 78 spectra x 1,792 columns | [Predicting Stability of Barley Straw-Derived Biochars Using FTIR Spectroscopy](https://doi.org/10.1021/acssusresmgt.4c00148), [Europe PMC](https://europepmc.org/article/PMC/11449111), CC BY |
| `downloads/datasets/Maezumi_ref_data.csv` | Nine plant species heated at 200-700 C; companion analogue-matching R code is also included | 1,260 spectra x 1,325 columns | [Zenodo record 5156747](https://doi.org/10.5281/zenodo.5156747), CC BY 4.0 |
| `downloads/datasets/WDG-CharcoalTemp-Data.xlsx` | Grass and alder charcoal at 200-700 C, including untreated, water-treated, and peroxide-treated material; includes ancient charcoal | 1,478 wavenumber rows x 355 spectrum columns | [Figshare dataset](https://doi.org/10.6084/m9.figshare.5979544.v1), CC BY 4.0 |

The five collections contain about 5,889 individual spectra in total. The
Minatre, Maezumi, and McCall files store one spectrum per row, whereas the WDG
workbook stores spectra in columns. `Minatre_Dryad_README.md` is the original
Dryad data dictionary.

## Figure-digitized estimates — keep separate from measured spectra

The following two curves were reconstructed by tracing a published raster
figure. They are **not instrument-exported spectra** and must not be pooled with
the downloaded numerical datasets as equivalent measurements.

| Derived spectrum | Provenance flag | Appropriate use | Local file |
|---|---|---|---|
| Apple-wood charcoal, 600 C, KBr pellet | `FIGURE_DIGITIZED_ESTIMATE` | Qualitative spectral shape and approximate peak locations only | `../output/tables/charcoal_ftir/figure8_kbr_600C_digitized.csv` |
| Walnut-shell charcoal, 600 C, KBr pellet | `FIGURE_DIGITIZED_ESTIMATE` | Qualitative spectral shape and approximate peak locations only | `../output/tables/charcoal_ftir/figure8_kbr_600C_digitized.csv` |

Every row in that CSV carries `provenance_flag`, `data_origin`,
`is_instrument_export`, and `recommended_use` fields. Its transmittance,
absorbance, and uncertainty columns are also explicitly labeled as estimated.
The associated plot is titled `DIGITIZED ESTIMATE` and the tracing overlay is
retained for visual auditing. All sources in the numerical-data tables below
are deposited or author-supplied files unless their individual check result
says otherwise.

## Additional machine-readable sources downloaded and audited

The second source sweep added **209 spectral series**. Some series are
replicates, averages, or normalized versions rather than independent samples.
The file-level audit is in
`downloads/metadata/new_sources_spectral_audit.csv` and
`downloads/metadata/new_sources_spectral_audit.json`.

| Local directory | Material and coverage | Spectral series | Check result | Source |
|---|---|---:|---|---|
| `downloads/datasets/PLOS_MCPA_biochars/` | Raw and 350, 500, and 800 C poultry-manure, rice-hull, and **wood-pellet chars** | 12 | Excellent: complete diamond-ATR absorbance spectra, 399.9-3999.9 cm-1, 14,935 points each | [PLOS ONE article and data](https://doi.org/10.1371/journal.pone.0291398), CC0 |
| `downloads/datasets/Mendeley_Shangchen_black_carbon/` | Shangchen sediment black carbon plus experimentally burnt vegetation at 250, 300, 400, 500, and 600 C | 142 | Excellent numerical coverage: 36 sediment spectra and 106 combustion-series columns, 399.2-4001.6 cm-1. The vegetation workbook mixes replicates, averages, and normalized columns and does not include a species-code key. | [Mendeley Data](https://doi.org/10.17632/22f4cx9666.1), CC BY 4.0 |
| `downloads/datasets/Mendeley_oil_palm_frond/` | Two oil-palm-frond biochars at 500 C | 2 | Good: native PerkinElmer ASCII, MIRacle ATR percent transmittance, 650-4000 cm-1, 3,351 points | [Mendeley Data](https://doi.org/10.17632/ktgcfwf9bx.1), CC BY 4.0 |
| `downloads/datasets/Mendeley_HTP_biochar/` | Birch, Miscanthus, and straw chars at 550 C, untreated versus hydrothermal-pretreatment/steam-explosion counterparts; duplicate runs | 12 | **Recoverable but not analysis-ready:** 7,469 intensity values per file, but wavenumbers were exported with only three significant digits, producing 6,568 duplicate wavenumbers. Paper method: diamond ATR absorbance with multiplicative-scatter correction. | [Mendeley Data](https://doi.org/10.17632/ypbr4mb24w.3), CC BY 4.0 |
| `downloads/datasets/Mendeley_rice_straw_biochar/` | Pristine, KOH-modified, and Pb-loaded rice-straw biochar | 3 | Good: JASCO transmittance, 399.2-4000.6 cm-1, 3,736 points | [Mendeley Data](https://doi.org/10.17632/zcshgj7kyy.3), CC BY 4.0 |
| `downloads/datasets/Mendeley_jamun_seed_biochar/` | Seven numbered Jamun-seed spectra, apparently raw plus chars spanning 250-750 C | 7 | Numerically good: 400-4000 cm-1, 1,801 points. **Metadata limitation:** the workbook does not explicitly map Sample 1-7 to temperature. | [Mendeley Data](https://doi.org/10.17632/pphv3ygkfk.1), CC BY 4.0 |
| `downloads/datasets/Mendeley_sewage_sludge_char/` | Sewage-sludge feed, pyrolysis char, and bio-oil | 3 | Good: percent transmittance, 450-4000 cm-1, 3,551 points | [Mendeley Data](https://doi.org/10.17632/r4tb4nbsbc.1), CC BY 4.0 |
| `downloads/datasets/Mendeley_sorghum_stalk_biochar/` | Unmodified, FeCl3-modified, and cyanide-loaded sorghum-stalk biochar | 3 | Good: percent transmittance, 400-4000.6 cm-1, 1,800 points | [Mendeley Data](https://doi.org/10.17632/3vm6gwbhz6.1), CC BY 4.0 |
| `downloads/datasets/Zenodo_olive_pomace_hydrochar/` | Raw olive pomace and hydrochars at 180, 215, and 250 C with multiple residence times and replicates | 22 readable DPT spectra plus native OPUS files | Good: 399.2-3996.3 cm-1, 1,866 points. This is hydrothermal carbonization, not conventional pyrolysis charcoal. | [Zenodo record](https://doi.org/10.5281/zenodo.11353931), CC BY 4.0 |
| `downloads/datasets/Mendeley_activated_biochar_DRIFT/` | Activated biochar and Pd/Al-loaded variants | 3 | Specialized only: pyridine-DRIFT surface-acidity window, 1101-1801 cm-1, 364 points; large arbitrary vertical offsets make direct comparison with bulk FTIR inappropriate. | [Mendeley Data](https://doi.org/10.17632/v38nbb9rtp.3), CC BY 4.0 |

Each Mendeley directory includes `mendeley_manifest.json` with the repository
file list, original SHA-256 values, download URLs, and locally verified hashes.
All 31 selected Mendeley files matched their declared checksums; the Zenodo
FTIR ZIP also matched its published MD5 checksum.

## Papers and supplementary material

| Local file or directory | Contents | Source |
|---|---|---|
| `downloads/papers/Minatre_2024_Frontiers_FTIR_charcoal.pdf` | Controlled muffle-furnace and combustion-facility charcoal, 200-800 C | [Frontiers paper](https://doi.org/10.3389/feart.2024.1354080) |
| `downloads/papers/Maezumi_2021_Palaeogeography_FTIR_charcoal.pdf` | Modern analogue matching across plant species and temperature | [Paper](https://doi.org/10.1016/j.palaeo.2021.110580) |
| `downloads/supplements/Maezumi_2021_supplement.docx` | Maezumi supplementary methods/tables | [UvA repository](https://pure.uva.nl/en/publications/a-modern-analogue-matching-approach-to-characterize-fire-temperatu) |
| `downloads/papers/Hu_2022_four_waste_biomass_biochars.pdf` | Biochars made from four waste biomasses | [BioResources PDF](https://bioresources.cnr.ncsu.edu/wp-content/uploads/2022/10/BioRes_17_4_6464_Hu_LYLL_Prep_Charact_Biochars_Four_Waste_Biomass_20066.pdf) |
| `downloads/papers/Agricultural_byproducts_FTIR_charcoal.pdf` | Biochars from agricultural by-products | [Publisher PDF](https://www.ndpublisher.in/admin/issues/IJAEBv13n4f.pdf) |
| `downloads/papers/Tea_apple_wheat_walnut_biochars_FTIR.pdf` | Effect of pyrolysis temperature and feedstock (tea waste, apple wood, wheat straw, walnut shell) | [Journal PDF](https://jast.modares.ac.ir/article_16428_50959eee56aa5f4695971b375b69354f.pdf) |
| `downloads/papers/new_sources/Activated_carbon_KBr_pellet_method.pdf` | Activated-carbon KBr-pellet method: 0.40-0.50 mg carbon with 300 mg KBr; 4000-400 cm-1; plotted spectra only | [Chemistry Journal of Moldova](https://doi.org/10.19261/cjm.2015.10(1).16) |
| `downloads/papers/new_sources/Jackfruit_biochar_KBr_thesis.pdf` | Jackfruit peel/seed biochar mixed with KBr and pressed into discs; raw, before-adsorption, and after-adsorption curves are figure-only | [UPSI repository record](https://ir.upsi.edu.my/detailsg.php?det=8570) |
| `downloads/papers/new_sources/HTP_biochar_steel_FTIR.pdf` | Methods and interpretation for the downloaded 550 C birch/Miscanthus/straw ATR spectra | [Swansea repository PDF](https://cronfa.swan.ac.uk/Record/cronfa70134) |
| `downloads/papers/new_sources/Jamun_seed_biochar_FTIR.pdf` | Companion characterization paper for Jamun-seed biochar; contains plotted temperature comparisons | [NM-AIST repository PDF](https://dspace.nm-aist.ac.tz/items/44a2f00e-8473-4dbe-9335-4bbfd9f0a8c3) |
| `downloads/supplements/PMC11449111/` | Barley study workbook, supplementary PDF, and article figures | [Europe PMC supplementary bundle](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11449111/supplementaryFiles) |
| `downloads/supplements/PMC12105012/` | Six-feedstock workbook, supplementary PDF, and article figures | [Europe PMC supplementary bundle](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12105012/supplementaryFiles) |
| `downloads/supplements/PMC9424292/` | Oak/pine charcoal carbonisation-temperature supplement and figures | [Article](https://doi.org/10.1038/s41598-022-17836-2) |
| `downloads/supplements/PMC6562222/` | Fresh and aged hardwood-biochar FTIR figure assets | [Article](https://doi.org/10.1016/j.dib.2019.104073) |

The original Europe PMC ZIP bundles are retained beside their extracted
directories. Repository metadata snapshots are under `downloads/metadata/`.

## Additional sources without downloadable raw spectra or full text

| Source | What it contains | Status |
|---|---|---|
| [Gosling et al. 2019](https://doi.org/10.1016/j.palaeo.2019.01.029) | Paper associated with the downloaded WDG workbook | Publisher full text was not openly downloadable; the complete Figshare spectral workbook is present. |
| [Keiluweit et al. 2010](https://doi.org/10.1021/es9031419) | Temperature-dependent molecular structure of plant-biomass chars; FTIR figures in supporting information | ACS returned HTTP 403 for the supporting PDF. |
| [Labbé et al. 2006](https://doi.org/10.1021/jf053062n) | FTIR study of wood charcoal | Publisher access only. |
| [Cohen-Ofri et al. 2006](https://doi.org/10.1016/j.jas.2005.08.008) | FTIR/Raman characterization of archaeological charcoal | Publisher access only. |
| [Guo and Bustin 1998](https://doi.org/10.1016/S0166-5162(98)00019-6) | FTIR and reflectance study of wood char from 100-700 C | Publisher access only. |
| [Janu et al. 2021](https://doi.org/10.1016/j.crcon.2021.01.003) | Biochar structure across pyrolysis conditions | Publisher access only. |
| [Sajdak et al. 2018](https://doi.org/10.1016/j.biombioe.2018.02.019) | FTIR classification of biomass and charcoal | Publisher access only. |
| [Rice- and soybean-straw KBr-pellet paper](https://doi.org/10.9734/ijecc/2024/v14i104478) | Figure-only KBr-pellet spectra | The PDF is openly viewable in the browser, but the publisher returned HTTP 403 to automated download. |
| [Young-durian-fruit KBr-pellet biochar](https://doi.org/10.1039/D5RA05229G) | Figure-only KBr-pellet spectrum | RSC and its mirror returned HTTP 403/404 to automated download; full text remains openly viewable online. |

## Integrity check

From this directory, verify the primary files with:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```
