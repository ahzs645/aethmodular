"""Digitize the 600 C apple-wood and walnut-shell FTIR traces in Figure 8.

This is a raster-to-data estimate, not a replacement for the instrument files.
It extracts the embedded Figure 8 panels from the source PDF, follows the red
600 C curve with a color-aware continuity constraint, calibrates the printed
axes, and exports values on a 10 cm-1 grid.
"""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PDF = (
    REPO_ROOT
    / "research/ftir_hips_chem/charcoal_ftir_sources/downloads/papers"
    / "Tea_apple_wheat_walnut_biochars_FTIR.pdf"
)
TEMP_DIR = REPO_ROOT / "tmp/pdfs/charcoal_figure8"
TABLE_DIR = REPO_ROOT / "research/ftir_hips_chem/output/tables/charcoal_ftir"
PLOT_DIR = REPO_ROOT / "research/ftir_hips_chem/output/plots/charcoal_ftir"
CSV_PATH = TABLE_DIR / "figure8_kbr_600C_digitized.csv"
SPECTRA_PLOT_PATH = PLOT_DIR / "figure8_kbr_600C_estimated_spectra.png"
OVERLAY_PATH = PLOT_DIR / "figure8_kbr_600C_digitization_overlay.png"


@dataclass(frozen=True)
class PanelConfig:
    image_number: int
    feedstock: str
    panel: str
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int
    anchor_x: int
    anchor_y: int
    y_slope: float
    y_intercept: float
    smoothness: float


PANELS = (
    PanelConfig(
        image_number=5,
        feedstock="apple wood",
        panel="c",
        x_left=33,
        x_right=598,
        y_top=15,
        y_bottom=300,
        anchor_x=270,
        anchor_y=119,
        y_slope=-0.0470387356,
        y_intercept=103.800857,
        smoothness=1.5,
    ),
    PanelConfig(
        image_number=6,
        feedstock="walnut shell",
        panel="d",
        x_left=35,
        x_right=602,
        y_top=15,
        y_bottom=300,
        anchor_x=300,
        anchor_y=20,
        y_slope=-0.0544657032,
        y_intercept=100.880506,
        smoothness=2.0,
    ),
)


def extract_embedded_images() -> None:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    pdfimages = shutil.which("pdfimages")
    if pdfimages is None:
        raise RuntimeError("pdfimages was not found on PATH")
    prefix = TEMP_DIR / "extracted"
    subprocess.run(
        [
            pdfimages,
            "-f",
            "10",
            "-l",
            "10",
            "-j",
            str(SOURCE_PDF),
            str(prefix),
        ],
        check=True,
    )


def red_emission(rgb: np.ndarray) -> np.ndarray:
    """Score red curve pixels while suppressing gray annotations and white."""
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    score = np.maximum(0.0, red - (green + blue) / 2.0)
    score += 0.5 * np.maximum(0.0, red - blue)
    score[(255.0 - rgb.mean(axis=2)) < 5.0] = 0.0
    return score


def directional_trace(
    emission: np.ndarray,
    columns: list[int],
    start_y: int,
    smoothness: float,
    jump: int = 8,
) -> list[int]:
    n_y = emission.shape[0]
    previous = np.full(n_y, -1e9, dtype=float)
    previous[start_y] = 0.0
    backpointers: list[np.ndarray] = []

    for column in columns:
        current = np.full(n_y, -1e9, dtype=float)
        back = np.zeros(n_y, dtype=np.int16)
        for y_value in range(n_y):
            lower = max(0, y_value - jump)
            upper = min(n_y, y_value + jump + 1)
            candidate_y = np.arange(lower, upper)
            candidates = previous[lower:upper] - smoothness * np.abs(
                candidate_y - y_value
            )
            best = int(np.argmax(candidates))
            current[y_value] = candidates[best] + emission[y_value, column]
            back[y_value] = lower + best
        previous = current
        backpointers.append(back)

    endpoint = int(np.argmax(previous))
    result: list[int] = []
    for back in reversed(backpointers):
        result.append(endpoint)
        endpoint = int(back[endpoint])
    return list(reversed(result))


def trace_panel(config: PanelConfig) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    image_path = TEMP_DIR / f"extracted-{config.image_number:03d}.jpg"
    source = Image.open(image_path).convert("RGB")
    pixels = np.asarray(source, dtype=float)
    rgb = pixels[
        config.y_top : config.y_bottom + 1,
        config.x_left : config.x_right + 1,
    ]
    emission = red_emission(rgb)

    anchor_column = config.anchor_x - config.x_left
    anchor_y = config.anchor_y - config.y_top
    n_columns = emission.shape[1]
    path = np.zeros(n_columns, dtype=int)
    path[anchor_column] = anchor_y

    forward_columns = list(range(anchor_column + 1, n_columns))
    forward = directional_trace(
        emission, forward_columns, anchor_y, config.smoothness
    )
    if forward:
        path[anchor_column + 1 :] = forward

    backward_columns = list(range(anchor_column - 1, -1, -1))
    backward = directional_trace(
        emission, backward_columns, anchor_y, config.smoothness
    )
    for column, y_value in zip(backward_columns, backward):
        path[column] = y_value

    absolute_y = path + config.y_top
    local_score = np.empty(n_columns, dtype=float)
    for column, y_value in enumerate(path):
        lower = max(0, y_value - 2)
        upper = min(emission.shape[0], y_value + 3)
        local_score[column] = float(emission[lower:upper, column].max())
    return source, absolute_y, local_score


def interpolate_spectrum(
    config: PanelConfig,
    pixel_y: np.ndarray,
    color_score: np.ndarray,
) -> list[dict[str, object]]:
    wavenumbers = np.arange(4000, 399, -10, dtype=float)
    x_pixels = config.x_left + (4000.0 - wavenumbers) * (
        config.x_right - config.x_left
    ) / 3600.0
    source_x = np.arange(config.x_left, config.x_right + 1, dtype=float)
    y_values = np.interp(x_pixels, source_x, pixel_y)
    scores = np.interp(x_pixels, source_x, color_score)
    transmittance = config.y_slope * y_values + config.y_intercept
    absorbance = -np.log10(np.clip(transmittance, 1e-6, None) / 100.0)

    rows: list[dict[str, object]] = []
    for wn, pct_t, absorb, score in zip(
        wavenumbers, transmittance, absorbance, scores
    ):
        if score >= 30:
            quality, uncertainty = "high", 0.20
        elif score >= 12:
            quality, uncertainty = "medium", 0.40
        else:
            quality, uncertainty = "low", 0.80
        rows.append(
            {
                "provenance_flag": "FIGURE_DIGITIZED_ESTIMATE",
                "data_origin": "raster_trace_from_published_figure",
                "is_instrument_export": False,
                "recommended_use": "qualitative_shape_and_peak_location_only",
                "feedstock": config.feedstock,
                "pyrolysis_temperature_C": 600,
                "kbr_sample_mg": 1,
                "kbr_mg": 300,
                "wavenumber_cm-1": int(wn),
                "transmittance_pct_estimated": round(float(pct_t), 4),
                "absorbance_estimated": round(float(absorb), 6),
                "uncertainty_pctT_estimated": uncertainty,
                "digitization_quality": quality,
                "wavenumber_uncertainty_cm-1": 7,
                "source_figure": f"Figure 8{config.panel}",
                "source_pdf_page": 10,
            }
        )
    return rows


def save_csv(rows: list[dict[str, object]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_overlay(
    traced: list[tuple[PanelConfig, Image.Image, np.ndarray]],
) -> None:
    panels: list[tuple[PanelConfig, Image.Image]] = []
    for config, source, pixel_y in traced:
        overlay = source.copy()
        draw = ImageDraw.Draw(overlay)
        draw.line(
            [
                (x_value, int(y_value))
                for x_value, y_value in zip(
                    range(config.x_left, config.x_right + 1), pixel_y
                )
            ],
            fill=(0, 255, 255),
            width=2,
        )
        panels.append((config, overlay))

    label_height = 24
    panel_height = max(panel.height for _, panel in panels)
    canvas = Image.new(
        "RGB",
        (sum(panel.width for _, panel in panels), panel_height + label_height),
        "white",
    )
    canvas_draw = ImageDraw.Draw(canvas)
    offset = 0
    for config, panel in panels:
        canvas.paste(panel, (offset, 0))
        canvas_draw.text(
            (offset + 10, panel_height + 6),
            f"Cyan overlay: recovered 600 C {config.feedstock} path",
            fill=(0, 0, 0),
            font=ImageFont.load_default(),
        )
        offset += panel.width
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OVERLAY_PATH)


def save_spectra_plot(rows: list[dict[str, object]]) -> None:
    plt.style.use("default")
    fig, axis = plt.subplots(figsize=(10, 5.8), dpi=180)
    colors = {"apple wood": "#B33A3A", "walnut shell": "#4C6FA8"}
    for feedstock in colors:
        subset = [row for row in rows if row["feedstock"] == feedstock]
        axis.plot(
            [row["wavenumber_cm-1"] for row in subset],
            [row["transmittance_pct_estimated"] for row in subset],
            label=f"{feedstock.title()} charcoal, 600 C",
            color=colors[feedstock],
            linewidth=1.8,
        )
    axis.set_xlim(4000, 400)
    axis.set_xlabel("Wavenumber (cm$^{-1}$)")
    axis.set_ylabel("Estimated transmittance (%)")
    axis.set_title("DIGITIZED ESTIMATE — figure-derived KBr-pellet FTIR spectra")
    axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False)
    axis.text(
        0.01,
        0.02,
        "FIGURE-DIGITIZED ESTIMATE from Reyhanitabar et al. (2020), Figure 8; "
        "not instrument-exported data",
        transform=axis.transAxes,
        fontsize=8,
        color="#8B0000",
        weight="bold",
    )
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SPECTRA_PLOT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    extract_embedded_images()
    all_rows: list[dict[str, object]] = []
    traced: list[tuple[PanelConfig, Image.Image, np.ndarray]] = []
    for config in PANELS:
        source, pixel_y, color_score = trace_panel(config)
        all_rows.extend(interpolate_spectrum(config, pixel_y, color_score))
        traced.append((config, source, pixel_y))
    save_csv(all_rows)
    save_overlay(traced)
    save_spectra_plot(all_rows)
    print(CSV_PATH)
    print(SPECTRA_PLOT_PATH)
    print(OVERLAY_PATH)


if __name__ == "__main__":
    main()
