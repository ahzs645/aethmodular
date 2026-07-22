# Charcoal and biochar FTIR source archive

Downloaded 2026-07-22. The large files live under `downloads/`, which is
already excluded by the repository's top-level `downloads/` ignore rule.

## Machine-readable spectra

| Local file | Coverage | Shape | Source and license |
|---|---|---:|---|
| `downloads/datasets/McCall_multifeedstock_FTIR_spectral_data.xlsx` | **Unburned (0 C) and burned (200-700 C)** barley straw, chestnut wood, eucalyptus bark, miscanthus grass, pine bark, and rice husk | 162 spectra x 1,615 columns | [Biochar Stability Revealed by FTIR and Machine Learning](https://doi.org/10.1021/acssusresmgt.5c00104), [Europe PMC](https://europepmc.org/article/PMC/12105012), CC BY |
| `downloads/datasets/McCall_barley_FTIR_spectral_data.xlsx` | **Unburned barley straw and char made at 150-700 C** | 78 spectra x 1,792 columns | [Predicting Stability of Barley Straw-Derived Biochars Using FTIR Spectroscopy](https://doi.org/10.1021/acssusresmgt.4c00148), [Europe PMC](https://europepmc.org/article/PMC/11449111), CC BY |
| `downloads/datasets/Maezumi_ref_data.csv` | Nine plant species heated at 200-700 C; companion analogue-matching R code is also included | 1,260 spectra x 1,325 columns | [Zenodo record 5156747](https://doi.org/10.5281/zenodo.5156747), CC BY 4.0 |
| `downloads/datasets/WDG-CharcoalTemp-Data.xlsx` | Grass and alder charcoal at 200-700 C, including untreated, water-treated, and peroxide-treated material; includes ancient charcoal | 1,478 wavenumber rows x 355 spectrum columns | [Figshare dataset](https://doi.org/10.6084/m9.figshare.5979544.v1), CC BY 4.0 |

The four collections contain about 1,855 individual spectra in total. Their
orientations differ: the Maezumi and McCall files store one spectrum per row,
whereas the WDG workbook stores spectra in columns.

## Papers and supplementary material

| Local file or directory | Contents | Source |
|---|---|---|
| `downloads/papers/Minatre_2024_Frontiers_FTIR_charcoal.pdf` | Controlled muffle-furnace and combustion-facility charcoal, 200-800 C | [Frontiers paper](https://doi.org/10.3389/feart.2024.1354080) |
| `downloads/papers/Maezumi_2021_Palaeogeography_FTIR_charcoal.pdf` | Modern analogue matching across plant species and temperature | [Paper](https://doi.org/10.1016/j.palaeo.2021.110580) |
| `downloads/supplements/Maezumi_2021_supplement.docx` | Maezumi supplementary methods/tables | [UvA repository](https://pure.uva.nl/en/publications/a-modern-analogue-matching-approach-to-characterize-fire-temperatu) |
| `downloads/papers/Hu_2022_four_waste_biomass_biochars.pdf` | Biochars made from four waste biomasses | [BioResources PDF](https://bioresources.cnr.ncsu.edu/wp-content/uploads/2022/10/BioRes_17_4_6464_Hu_LYLL_Prep_Charact_Biochars_Four_Waste_Biomass_20066.pdf) |
| `downloads/papers/Agricultural_byproducts_FTIR_charcoal.pdf` | Biochars from agricultural by-products | [Publisher PDF](https://www.ndpublisher.in/admin/issues/IJAEBv13n4f.pdf) |
| `downloads/papers/Tea_apple_wheat_walnut_biochars_FTIR.pdf` | Effect of pyrolysis temperature and feedstock (tea waste, apple wood, wheat straw, walnut shell) | [Journal PDF](https://jast.modares.ac.ir/article_16428_50959eee56aa5f4695971b375b69354f.pdf) |
| `downloads/supplements/PMC11449111/` | Barley study workbook, supplementary PDF, and article figures | [Europe PMC supplementary bundle](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11449111/supplementaryFiles) |
| `downloads/supplements/PMC12105012/` | Six-feedstock workbook, supplementary PDF, and article figures | [Europe PMC supplementary bundle](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12105012/supplementaryFiles) |
| `downloads/supplements/PMC9424292/` | Oak/pine charcoal carbonisation-temperature supplement and figures | [Article](https://doi.org/10.1038/s41598-022-17836-2) |
| `downloads/supplements/PMC6562222/` | Fresh and aged hardwood-biochar FTIR figure assets | [Article](https://doi.org/10.1016/j.dib.2019.104073) |

The original Europe PMC ZIP bundles are retained beside their extracted
directories. Repository metadata snapshots are under `downloads/metadata/`.

## Located but not downloaded automatically

| Source | What it contains | Status |
|---|---|---|
| [Dryad dataset 10.5061/dryad.cnp5hqcbj](https://doi.org/10.5061/dryad.cnp5hqcbj) | `Minatre_reference_spectra.csv` (56.7 MB), `Minatre_combustionfacility_spectra.csv` (54.6 MB), and README; 200-800 C reference and real-combustion spectra | Dryad's public page is readable, but its file stream presented an automated-client verification gate and its API download route required a bearer token. Metadata and the complete file inventory were saved locally; no gate was bypassed. |
| [Gosling et al. 2019](https://doi.org/10.1016/j.palaeo.2019.01.029) | Paper associated with the downloaded WDG workbook | Publisher full text was not openly downloadable; the complete Figshare spectral workbook is present. |
| [Keiluweit et al. 2010](https://doi.org/10.1021/es9031419) | Temperature-dependent molecular structure of plant-biomass chars; FTIR figures in supporting information | ACS returned HTTP 403 for the supporting PDF. |
| [Labbé et al. 2006](https://doi.org/10.1021/jf053062n) | FTIR study of wood charcoal | Publisher access only. |
| [Cohen-Ofri et al. 2006](https://doi.org/10.1016/j.jas.2005.08.008) | FTIR/Raman characterization of archaeological charcoal | Publisher access only. |
| [Guo and Bustin 1998](https://doi.org/10.1016/S0166-5162(98)00019-6) | FTIR and reflectance study of wood char from 100-700 C | Publisher access only. |
| [Janu et al. 2021](https://doi.org/10.1016/j.crcon.2021.01.003) | Biochar structure across pyrolysis conditions | Publisher access only. |
| [Sajdak et al. 2018](https://doi.org/10.1016/j.biombioe.2018.02.019) | FTIR classification of biomass and charcoal | Publisher access only. |

## Integrity check

From this directory, verify the primary files with:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

