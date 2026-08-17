"""PyMieScatt trial: MAC of BC at 633 nm (the HIPS He-Ne wavelength).

Grounds the project's HIPS MAC 6-vs-10 m^2/g fork with Mie theory:
- monodisperse MAC(D) for two literature BC refractive indices
- lognormal-ensemble MAC over GMD 100-250 nm, GSD 1.5-1.8
- coated-sphere (core-shell) absorption enhancement E_abs

Compatibility note: PyMieScatt 1.8.1.1 imports `scipy.integrate.trapz`,
removed in SciPy >= 1.14 — shim it BEFORE importing PyMieScatt.

Run:  python mac_sweep.py   (writes mac_vs_diameter.png next to itself)
"""

from pathlib import Path

import numpy as np
import scipy.integrate

if not hasattr(scipy.integrate, "trapz"):  # SciPy >= 1.14 shim
    scipy.integrate.trapz = np.trapezoid

import matplotlib.pyplot as plt
import PyMieScatt as ps

HERE = Path(__file__).resolve().parent

WAVELENGTH_NM = 633.0            # HIPS He-Ne laser
DENSITY_G_CM3 = 1.8              # BC material density
RI_CASES = {                     # label -> complex refractive index
    "m=1.95+0.79j (Bond & Bergstrom)": 1.95 + 0.79j,
    "m=1.85+0.71j": 1.85 + 0.71j,
}
GMD_NM = np.arange(100, 251, 25)         # number geometric mean diameter
GSD = [1.5, 1.65, 1.8]                   # geometric standard deviation


def mac_monodisperse(m, d_nm):
    """MAC (m^2/g) of a bare BC sphere: 3 Qabs / (2 rho D)."""
    qabs = np.array([ps.MieQ(m, WAVELENGTH_NM, d)[2] for d in np.atleast_1d(d_nm)])
    d_m = np.atleast_1d(d_nm) * 1e-9
    rho_g_m3 = DENSITY_G_CM3 * 1e6
    return 3.0 * qabs / (2.0 * rho_g_m3 * d_m)


def mac_lognormal(m, gmd_nm, gsd, n_bins=400, d_lo=10.0, d_hi=5000.0):
    """Ensemble MAC (m^2/g) for a lognormal number size distribution.

    Integrates Qabs * (pi/4) D^2 and (pi/6) rho D^3 over the same
    discretised distribution, so the ratio is exactly Babs / mass.
    """
    d = np.logspace(np.log10(d_lo), np.log10(d_hi), n_bins)          # nm
    lnd = np.log(d)
    pdf = np.exp(-((lnd - np.log(gmd_nm)) ** 2) / (2 * np.log(gsd) ** 2))
    qabs = np.array([ps.MieQ(m, WAVELENGTH_NM, di)[2] for di in d])
    d_m = d * 1e-9
    cabs = qabs * (np.pi / 4) * d_m**2                               # m^2/particle
    mass = (DENSITY_G_CM3 * 1e6) * (np.pi / 6) * d_m**3              # g/particle
    babs = np.trapezoid(cabs * pdf, lnd)
    mtot = np.trapezoid(mass * pdf, lnd)
    return babs / mtot


def coated_enhancement(m_core, core_d_nm, shell_ratio, m_shell=1.55 + 0j):
    """E_abs = Cabs(coated) / Cabs(bare), MAC referenced to the BC core mass."""
    d_shell = core_d_nm * shell_ratio
    qabs_coated = ps.MieQCoreShell(
        m_core, m_shell, WAVELENGTH_NM, core_d_nm, d_shell
    )[2]
    cabs_coated = qabs_coated * (np.pi / 4) * (d_shell * 1e-9) ** 2
    qabs_bare = ps.MieQ(m_core, WAVELENGTH_NM, core_d_nm)[2]
    cabs_bare = qabs_bare * (np.pi / 4) * (core_d_nm * 1e-9) ** 2
    return cabs_coated / cabs_bare


def main():
    # ---- table: lognormal-ensemble MAC ------------------------------------
    print(f"Lognormal-ensemble MAC of bare BC spheres at {WAVELENGTH_NM:.0f} nm, "
          f"rho = {DENSITY_G_CM3} g/cm^3  (m^2/g)\n")
    header = "| m | GSD | " + " | ".join(f"GMD {g} nm" for g in GMD_NM) + " |"
    print(header)
    print("|" + "---|" * (len(GMD_NM) + 2))
    results = {}
    for label, m in RI_CASES.items():
        for gsd in GSD:
            row = [mac_lognormal(m, g, gsd) for g in GMD_NM]
            results[(label, gsd)] = row
            cells = " | ".join(f"{v:.2f}" for v in row)
            print(f"| {label.split()[0]} | {gsd} | {cells} |")
    all_vals = np.array(list(results.values()))
    print(f"\nSweep range: {all_vals.min():.2f} - {all_vals.max():.2f} m^2/g")

    # ---- monodisperse curve + peak ----------------------------------------
    d_grid = np.logspace(np.log10(20), np.log10(1000), 200)
    mono = {label: mac_monodisperse(m, d_grid) for label, m in RI_CASES.items()}
    for label, curve in mono.items():
        i = int(np.argmax(curve))
        print(f"Monodisperse peak, {label}: MAC = {curve[i]:.2f} m^2/g "
              f"at D = {d_grid[i]:.0f} nm")

    # ---- coated-sphere enhancement ----------------------------------------
    print("\nCore-shell absorption enhancement E_abs "
          "(non-absorbing shell m = 1.55+0j):")
    print("| m_core | core D (nm) | Dshell/Dcore 1.2 | 1.5 | 1.8 | 2.0 |")
    print("|---|---|---|---|---|---|")
    for label, m in RI_CASES.items():
        for core in (150, 200):
            e = [coated_enhancement(m, core, r) for r in (1.2, 1.5, 1.8, 2.0)]
            print(f"| {label.split()[0]} | {core} | "
                  + " | ".join(f"{v:.2f}" for v in e) + " |")

    # ---- figure -----------------------------------------------------------
    colors = {list(RI_CASES)[0]: "#2a78d6", list(RI_CASES)[1]: "#eb6834"}
    styles = {1.5: "-", 1.65: "--", 1.8: ":"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.axhspan(6, 10, color="#e1e0d9", alpha=0.45, zorder=0)
        ax.axhline(6, color="#898781", lw=1, ls="-")
        ax.axhline(10, color="#898781", lw=1, ls="-")
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_ylabel("MAC at 633 nm (m$^2$/g)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for label, curve in mono.items():
        ax1.plot(d_grid, curve, color=colors[label], lw=2, label=label)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sphere diameter (nm)")
    ax1.set_title("Monodisperse bare BC sphere", fontsize=11)
    ax1.legend(fontsize=8, frameon=False)
    ax1.annotate("MAC 6-10 window", xy=(22, 8), fontsize=8, color="#52514e")

    for (label, gsd), row in results.items():
        ax2.plot(GMD_NM, row, color=colors[label], ls=styles[gsd], lw=2,
                 marker="o", ms=4,
                 label=f"{label.split()[0]}, GSD {gsd}")
    ax2.set_xlabel("Geometric mean diameter (nm)")
    ax2.set_title("Lognormal ensemble (bare spheres)", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=7, frameon=False, ncol=2)

    fig.suptitle("Mie MAC of BC at 633 nm (HIPS wavelength), "
                 r"$\rho$ = 1.8 g cm$^{-3}$", fontsize=12)
    fig.tight_layout()
    out = HERE / "mac_vs_diameter.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
