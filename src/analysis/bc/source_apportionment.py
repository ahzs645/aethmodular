"""Instrument-aware source-apportionment summaries for aethalometer data.

The no-argument analyzer still accepts the existing BC column families and
returns its established summary keys. Calculations now use explicit wavelength
pairs and the nonlinear two-component optical model. This is conditional
optical source estimation, not independent chemical source identification.
"""

from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.base import BaseAnalyzer
from .source_model import (
    MA350_ATTENUATION_CROSS_SECTIONS,
    MA350_WAVELENGTHS_NM,
    two_source_absorption,
)


# BC1..BC7 belong to the AE33 family, not the MA350's named wavelength / spot
# columns. No AE33 cross-section is guessed from MA350 values.
AE33_WAVELENGTHS_NM = {
    'BC1': 370, 'BC2': 470, 'BC3': 520, 'BC4': 590,
    'BC5': 660, 'BC6': 880, 'BC7': 950,
}


class SourceApportionmentAnalyzer(BaseAnalyzer):
    """Summarize a declared optical two-source model without silently dropping rows.

    Existing ``SourceApportionmentAnalyzer()`` and ``analyze(data)`` calls remain
    valid. Named MA350 columns identify that family; BC1..BC7 identify the AE33
    family. Numeric BC_470 aliases and mixed families require ``instrument``.
    The default pair is Blue/IR (470/880); UV/IR is used only when Blue is absent.
    Other pairs must be selected by their exact column names.

    ``signal_basis='bcc'`` reconstructs an attenuation proxy using documented
    MA350 firmware cross-sections or explicitly supplied per-column constants.
    Its interpretation assumes a common scattering factor across the pair.
    AE33 BC concentrations always require caller-supplied cross-sections.
    For already corrected optical inputs use ``'absorption'``; for known
    attenuation inputs use ``'attenuation'`` and acknowledge the same assumption.
    """

    def __init__(
        self,
        *,
        instrument: Optional[str] = None,
        wavelength_pair: Optional[Tuple[str, str]] = None,
        signal_basis: str = "bcc",
        cross_sections: Optional[Mapping[str, float]] = None,
        alpha_fossil: float = 1.0,
        alpha_biomass: float = 2.0,
        clip: bool = True,
    ):
        try:
            super().__init__("SourceApportionmentAnalyzer")
        except TypeError:
            super().__init__()
        if instrument is not None and instrument not in ("MA350", "AE33"):
            raise ValueError("instrument must be MA350 or AE33")
        if signal_basis not in ("bcc", "absorption", "attenuation"):
            raise ValueError("signal_basis must be bcc, absorption or attenuation")
        self.instrument = instrument
        self.wavelength_pair = wavelength_pair
        self.signal_basis = signal_basis
        self.cross_sections = dict(cross_sections) if cross_sections is not None else None
        self.aae_fossil = alpha_fossil
        self.aae_biomass = alpha_biomass
        self.clip = clip
        self.bc_columns = [
            *(f"{name} BCc" for name in MA350_WAVELENGTHS_NM),
            *AE33_WAVELENGTHS_NM,
            *(f"BC_{nm}" for nm in sorted(set(MA350_WAVELENGTHS_NM.values())
                                         | set(AE33_WAVELENGTHS_NM.values()))),
        ]

    def _instrument_for(self, data):
        if self.instrument is not None:
            return self.instrument
        named_ma350 = any(f"{c} BCc" in data for c in MA350_WAVELENGTHS_NM)
        named_ae33 = any(c in data for c in AE33_WAVELENGTHS_NM)
        if named_ma350 != named_ae33:
            return "MA350" if named_ma350 else "AE33"
        raise ValueError("Declare instrument explicitly for mixed or numeric-only channel names")

    def _columns(self, data, instrument):
        named = ({f"{c} BCc": nm for c, nm in MA350_WAVELENGTHS_NM.items()}
                 if instrument == "MA350" else dict(AE33_WAVELENGTHS_NM))
        aliases = {f"BC_{nm}": nm for nm in named.values()}
        return {c: nm for c, nm in sorted({**named, **aliases}.items(), key=lambda x: (x[1], x[0]))
                if c in data}

    def _pair(self, columns, instrument):
        if self.wavelength_pair is not None:
            if len(self.wavelength_pair) != 2:
                raise ValueError("wavelength_pair must contain exactly two column names")
            pair = tuple(self.wavelength_pair)
            if any(c not in columns for c in pair):
                raise ValueError("Requested wavelength_pair is missing or belongs to another instrument")
            if columns[pair[0]] >= columns[pair[1]]:
                raise ValueError("wavelength_pair must be ordered short then long")
            return pair, "explicit column pair"
        uv_nm = 375 if instrument == "MA350" else 370
        for short_nm in (470, uv_nm):
            matches = [[c for c, nm in columns.items() if nm == target] for target in (short_nm, 880)]
            if all(matches):
                if any(len(m) != 1 for m in matches):
                    raise ValueError("Ambiguous wavelength aliases; select wavelength_pair explicitly")
                selection = "Blue/IR default" if short_nm == 470 else "UV/IR fallback (Blue absent)"
                return (matches[0][0], matches[1][0]), selection
        raise ValueError("No default Blue/IR or UV/IR pair; select wavelength_pair explicitly")

    def _cross_sections(self, pair, columns, instrument):
        if self.signal_basis != "bcc":
            return {c: 1.0 for c in pair}
        if self.cross_sections is not None:
            missing = [c for c in pair if c not in self.cross_sections]
            if missing:
                raise ValueError(f"Missing declared cross-sections for: {missing}")
            values = {c: float(self.cross_sections[c]) for c in pair}
        elif instrument == "MA350":
            by_nm = {MA350_WAVELENGTHS_NM[c]: v for c, v in MA350_ATTENUATION_CROSS_SECTIONS.items()}
            values = {c: by_nm[columns[c]] for c in pair}
        else:
            raise ValueError("AE33 BC inputs require explicitly supplied cross_sections; MA350 constants do not apply")
        if not all(np.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("Cross-sections must be finite positive values")
        return values

    def _model(self, data, pair, columns, instrument):
        if any(list(data.columns).count(c) != 1 for c in pair):
            raise ValueError("Selected input columns must have unique names")
        sigma = self._cross_sections(pair, columns, instrument)
        short, long = (pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=float) for c in pair)
        with np.errstate(over="ignore", invalid="ignore"):
            short_optical, long_optical = short * sigma[pair[0]], long * sigma[pair[1]]
        model = two_source_absorption(
            short_optical, long_optical,
            wavelength_short_nm=columns[pair[0]], wavelength_long_nm=columns[pair[1]],
            alpha_fossil=self.aae_fossil, alpha_biomass=self.aae_biomass, clip=self.clip,
        )
        return model, long, sigma

    @staticmethod
    def _stats(values, *, extrema=False, total=False):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        result = {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "median": float(np.median(finite)) if len(finite) else None,
        }
        if extrema:
            result.update(min=float(finite.min()) if len(finite) else None,
                          max=float(finite.max()) if len(finite) else None)
        if total:
            result["total"] = float(finite.sum()) if len(finite) else None
        return result

    def _contributions(self, model, reference):
        with np.errstate(invalid="ignore"):
            bb = reference * model.biomass_fraction
            ff = reference * model.fossil_fraction
        total = float(reference[model.valid].sum())
        result = {
            "biomass_fraction": self._stats(model.biomass_fraction),
            "fossil_fraction": self._stats(model.fossil_fraction),
            "biomass_fraction_unbounded": self._stats(model.biomass_fraction_unbounded, extrema=True),
            "valid_measurements": int(model.valid.sum()),
            "reference_weighted_biomass_fraction": float(bb[model.valid].sum() / total) if total > 0 else None,
        }
        # Preserve legacy BC keys for BC inputs: optical amplitudes cannot
        # silently acquire a BC mass label.
        key = "bc" if self.signal_basis == "bcc" else "reference_optical"
        result[f"biomass_{key}"] = self._stats(bb, total=True)
        result[f"fossil_{key}"] = self._stats(ff, total=True)
        result["totals_note"] = "Sums of retained input values, not time-integrated mass or exposure"
        return result

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return established summaries plus signal eligibility and model provenance."""
        results = {"total_measurements": len(data)}
        try:
            instrument = self._instrument_for(data)
            columns = self._columns(data, instrument)
            results["available_columns"] = list(columns)
            pair, selection = self._pair(columns, instrument)
            model, reference, sigma = self._model(data, pair, columns, instrument)
            invalid = model.invalid_reason[~model.valid]
            reasons, counts = np.unique(invalid, return_counts=True)
            results.update(
                wavelength_pair={"bc1": pair[0], "bc2": pair[1]},
                aae_statistics=self._stats(model.aae, extrema=True),
                source_contributions=self._contributions(model, reference),
                signal_validity={
                    "total": len(data), "valid": int(model.valid.sum()),
                    "invalid": int((~model.valid).sum()),
                    "invalid_reasons": {str(r): int(n) for r, n in zip(reasons, counts)},
                    "clipped_low": int(model.clipped_low.sum()), "clipped_high": int(model.clipped_high.sum()),
                    "note": "Numerical signal eligibility only; not instrument QA/QC or validated source identity",
                },
                source_model={
                    "model": "two_component_power_law_absorption",
                    "instrument": instrument, "instrument_selection": "declared" if self.instrument else "column-name family",
                    "pair_selection": selection, "wavelengths_nm": [columns[c] for c in pair],
                    "input_basis": self.signal_basis,
                    "calculation_basis": "absorption" if self.signal_basis == "absorption" else "attenuation_proxy",
                    "cross_sections": sigma if self.signal_basis == "bcc" else None,
                    "cross_section_provenance": ("caller supplied" if self.cross_sections is not None else
                                                  "MA350 firmware attenuation constants") if self.signal_basis == "bcc" else None,
                    "scattering_assumption": None if self.signal_basis == "absorption" else "A common wavelength-independent scattering correction cancels in the ratio",
                    "alpha_fossil": float(self.aae_fossil), "alpha_biomass": float(self.aae_biomass),
                    "clip": self.clip, "fraction_reference": "long-wavelength optical absorption under the stated signal assumptions",
                    "bc_partition_assumption": "A common reference-wavelength mass absorption cross-section for both components" if self.signal_basis == "bcc" else None,
                    "component_labels": "Model end members; not unique chemical source identities",
                    "uncertainty": "Not estimated; end-member and measurement uncertainties require a separate sensitivity analysis",
                },
            )
            if len(columns) >= 3:
                results["multi_wavelength"] = self._multi_wavelength_analysis(data, list(columns))
            results["summary"] = self._generate_summary(results)
        except (TypeError, ValueError, KeyError) as exc:
            results["error"] = f"Analysis unavailable: {exc}"
        return results

    def _calculate_aae(self, data, bc1_col, bc2_col):
        instrument = self._instrument_for(data)
        columns = self._columns(data, instrument)
        pair = tuple(sorted((bc1_col, bc2_col), key=columns.__getitem__))
        model, _, _ = self._model(data, pair, columns, instrument)
        return pd.Series(model.aae, index=data.index)

    def _estimate_source_contributions(self, data, bc1_col, bc2_col, aae=None):
        # Retain ``aae`` in the private compatibility signature, but always use
        # the declared optical signals in the physical mixing equation.
        instrument = self._instrument_for(data)
        columns = self._columns(data, instrument)
        pair = tuple(sorted((bc1_col, bc2_col), key=columns.__getitem__))
        model, reference, _ = self._model(data, pair, columns, instrument)
        return self._contributions(model, reference)

    def _multi_wavelength_analysis(self, data, bc_cols):
        instrument = self._instrument_for(data)
        columns = self._columns(data, instrument)
        pairs, skipped = [], []
        for short, long in combinations(bc_cols, 2):
            if columns[short] == columns[long]:
                continue
            try:
                values = self._calculate_aae(data, short, long)
                stats = self._stats(values)
                pairs.append({"wavelength_pair": f"{short}-{long}", "aae_mean": stats["mean"],
                              "aae_std": stats["std"], "valid_measurements": int(np.isfinite(values).sum())})
            except (ValueError, KeyError) as exc:
                skipped.append({"wavelength_pair": f"{short}-{long}", "reason": str(exc)})
        means = [self._stats(pd.to_numeric(data[c], errors="coerce"))["mean"] for c in bc_cols]
        reference = means[-1]
        return {
            "aae_all_pairs": pairs, "unavailable_pairs": skipped,
            "spectral_profile": {
                "columns": bc_cols, "mean_concentrations": means,
                "concentration_ratios": [v / reference if v is not None and reference is not None and reference > 0 else None for v in means],
                "basis": self.signal_basis, "reference_column": bc_cols[-1],
            },
        }

    def _generate_summary(self, results):
        if "error" in results:
            return results["error"]
        contributions = results["source_contributions"]
        if not contributions["valid_measurements"]:
            return "No eligible paired optical signals; source fractions are unavailable"
        bb = contributions["biomass_fraction"]["mean"]
        ff = contributions["fossil_fraction"]["mean"]
        return (f"Model fractions, mean over eligible observations: {bb:.1%} biomass-like, "
                f"{ff:.1%} fossil-like; optical AAE {results['aae_statistics']['mean']:.2f}. "
                "Conditional on the declared end members and optical signal assumptions.")
