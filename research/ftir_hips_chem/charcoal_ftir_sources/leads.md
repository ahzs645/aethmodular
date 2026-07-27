# Additional Charcoal / Biochar FTIR Leads

This file tracks relevant leads that are not yet clean automated pulls in
`sources.json`, either because they are figure/supplement driven, require manual
inspection, or are reference-library comparisons rather than charcoal datasets.

## KBr / Transmission Method Leads

| Priority | Lead | Why it matters | Status |
|---|---|---|---|
| High | Almond shell-derived biochar for Pb adsorption, MDPI Molecules 2025, DOI `10.3390/molecules30204121` | Explicitly labels FTIR spectra as KBr-pellet spectra and states that representative instrumental parameters and replicate raw data are in supplementary material. | Manual follow-up; inspect supplement for raw spectra before adding as `direct`. |
| Medium | Ryan et al. high-intensity fire events / south-eastern Australia | KBr pressed-disc FTIR on many sediment/charcoal subsamples with Nicolet 6700 and OMNIC. | Figure/supplement availability unclear. |
| Medium | Dias Junior, "Infrared spectroscopy analysis on charcoal" | KBr pellets, 4000-400 cm^-1, 4 cm^-1 resolution, 128 scans. | Likely figure-based or author-request. |
| Medium | Coconut shell charcoal activation paper | KBr pellet activated-carbon/charcoal method and peak comparison. | Likely figure-based. |
| Medium | Soot deposits FTIR/Raman paper | Traditional KBr pellet method, dried samples plus KBr, 2 cm^-1, 100 scans. | Useful soot/carbon-black comparison if raw spectra can be located. |

## Reference-Library Comparison Leads

| Lead | Use |
|---|---|
| OpenSpecy | Pipeline/library matching support for `.asp`, `.csv`, `.jdx`, `.spc`, `.spa`, `.0`, and `.zip`; useful for importing mixed spectral formats from this folder. |
| IRUG | Conservation reference spectra, including carbon black and bone black entries. |
| University of Tartu ATR-FTIR pigment database | Bone black, ivory black, Mars black, and related pigment references. |
| SpectraBase / Wiley carbon black | Commercial/reference comparison; not an open bulk pull. |

## Method Warning

Keep ATR and transmission/KBr spectra separated during analysis. Carbon black
and highly carbonized materials can produce distorted bands in diamond ATR; Ge
ATR and transmission/KBr methods may not be directly comparable without a
method-specific correction/normalization step.

## White Charcoal / Binchotan Leads

### Automated Pulls

| Source id | What we pulled | Why it matters |
|---|---|---|
| `pereira_ufscar_binchotan_thesis` | Thesis PDF, extracted text, and license RDF from UFSCar. | Strong binchotan-specific material characterization: Japan/Myanmar binchotan vs calcined petroleum coke, including proximate analysis, FTIR, SEM-EDS, TGA, density, HHV, resistivity, and XRD. |
| `kwon_woodj_white_charcoal_quality` | Article PDF and three downloadable table files. | White-charcoal quality data: proximate analysis, density, pH, EMC, calorific value, hardness, and refinement degree. |

### Manual / Literature Leads

| Priority | Lead | Download status | Notes |
|---|---|---|---|
| High | Pereira et al. 2024, ACS Sustainable Chemistry & Engineering, DOI `10.1021/acssuschemeng.4c03756` | Article PDF/HTML, no raw SI found. | Compact article version of the UFSCar thesis binchotan characterization. |
| High | Pijarn et al. 2021, Journal of Materials Research and Technology, DOI `10.1016/j.jmrt.2021.04.082` | Open PDF via NSTDA repository. | White-charcoal characterization across bamboo, miscellaneous wood, coconut shell, and coconut spathe; tables include proximate/ultimate analysis, EDS/ED-XRF, XRD, TGA, BET, COD/pH treatment. |
| Medium-high | Chia et al. 2014, Journal of Analytical and Applied Pyrolysis, DOI `10.1016/j.jaap.2014.06.009` | Article PDF via MPG.PuRe; no raw SI found. | Classic white-charcoal microstructure paper; useful for fixed carbon, XPS/NMR/SEM/TEM/XRD/BET-style characterization. |
| Medium | Yu et al. 2020, Aerosol and Air Quality Research | Open PDF tables. | Binchotan combustion/emissions context with proximate analysis and emission factors. |
| Medium-low | Syarif et al. 2022, AIMS Energy, DOI `10.3934/energy.2022016` | HTML/PDF/figures, supplement table empty. | Binchotan-derived carbon dispersion for fuel-cell gas diffusion layers; FTIR/XRD figures but not raw spectra. |
| Medium-low | Syarif et al. 2020, IOP, DOI `10.1088/1757-899X/796/1/012057` | PDF only. | Binchotan carbon nanodots; FTIR/XRD/SEM embedded as figures. |
