import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def capability_analysis(data, USL, LSL, target=None):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    mean = float(data.mean())
    std = float(data.std(ddof=1))
    if target is None:
        target = (USL + LSL) / 2

    cp = (USL - LSL) / (6 * std) if std > 0 else np.nan
    cpu = (USL - mean) / (3 * std) if std > 0 else np.nan
    cpl = (mean - LSL) / (3 * std) if std > 0 else np.nan
    cpk = min(cpu, cpl) if std > 0 else np.nan
    pp = cp
    ppk = cpk
    tau = np.sqrt(std ** 2 + (mean - target) ** 2)
    cpm = (USL - LSL) / (6 * tau) if tau > 0 else np.nan
    sigma_level = 3 * cpk + 1.5 if not np.isnan(cpk) else np.nan

    z_u = (USL - mean) / std if std > 0 else np.nan
    z_l = (mean - LSL) / std if std > 0 else np.nan
    ppm_above = (1 - stats.norm.cdf(z_u)) * 1e6
    ppm_below = stats.norm.cdf(-z_l) * 1e6
    dpmo = ppm_above + ppm_below

    sample = data[:500] if len(data) > 500 else data
    try:
        _, p_norm = stats.shapiro(sample)
    except Exception:
        p_norm = float("nan")

    return {
        "Cp": cp, "Cpk": cpk, "Pp": pp, "Ppk": ppk, "Cpm": cpm,
        "sigma_level": sigma_level, "dpmo": dpmo,
        "normality_p": p_norm, "mean": mean, "std": std,
        "USL": USL, "LSL": LSL, "target": target,
    }


def capability_histogram(data, USL, LSL, target=None):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if target is None:
        target = (USL + LSL) / 2
    idx = capability_analysis(data, USL, LSL, target)
    mean, std = idx["mean"], idx["std"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=30, density=True, color="steelblue",
            edgecolor="white", alpha=0.7)
    lo = min(data.min(), LSL) - 0.5 * std
    hi = max(data.max(), USL) + 0.5 * std
    xs = np.linspace(lo, hi, 300)
    ax.plot(xs, stats.norm.pdf(xs, mean, std), color="darkorange",
            linewidth=2, label="Normal fit")
    ax.axvline(USL, color="red", linestyle="--", linewidth=1.5, label=f"USL={USL}")
    ax.axvline(LSL, color="red", linestyle="--", linewidth=1.5, label=f"LSL={LSL}")
    ax.axvline(target, color="green", linestyle="--", linewidth=1.5, label=f"Target={target}")
    ax.axvline(mean, color="black", linestyle="-", linewidth=1.5, label=f"Mean={mean:.2f}")

    text = (
        f"Cp   = {idx['Cp']:.3f}\n"
        f"Cpk  = {idx['Cpk']:.3f}\n"
        f"Pp   = {idx['Pp']:.3f}\n"
        f"Ppk  = {idx['Ppk']:.3f}\n"
        f"Cpm  = {idx['Cpm']:.3f}\n"
        f"Sigma= {idx['sigma_level']:.2f}\n"
        f"DPMO = {idx['dpmo']:.0f}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.set_title("Process Capability Histogram")
    ax.set_xlabel("Measurement (y)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
