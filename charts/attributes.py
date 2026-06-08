import numpy as np
import matplotlib.pyplot as plt


def _attr_plot(ax, values, center, ucl, lcl, title, ylabel):
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values))
    ucl_arr = np.asarray(ucl, dtype=float) if np.ndim(ucl) else np.full_like(values, ucl, dtype=float)
    lcl_arr = np.asarray(lcl, dtype=float) if np.ndim(lcl) else np.full_like(values, lcl, dtype=float)
    ax.plot(x, values, color="steelblue", linewidth=1.2, marker="o", markersize=3)
    ooc = (values > ucl_arr) | (values < lcl_arr)
    if np.any(ooc):
        ax.plot(x[ooc], values[ooc], "o", color="red", markersize=7, label="OOC")
    ax.axhline(center, color="green", linestyle="--", label="CL")
    if np.ndim(ucl):
        ax.plot(x, ucl_arr, color="red", linestyle="--", label="UCL")
        ax.plot(x, lcl_arr, color="red", linestyle="--", label="LCL")
    else:
        ax.axhline(ucl, color="red", linestyle="--", label="UCL")
        ax.axhline(lcl, color="red", linestyle="--", label="LCL")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def p_chart(defects, sample_sizes):
    defects = np.asarray(defects, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    p = defects / n
    p_bar = defects.sum() / n.sum()
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    ucl = p_bar + 3 * sigma
    lcl = np.maximum(0.0, p_bar - 3 * sigma)
    fig, ax = plt.subplots(figsize=(10, 5))
    _attr_plot(ax, p, p_bar, ucl, lcl, "p Chart - Proportion Defective", "Proportion")
    fig.tight_layout()
    return fig, {"center": float(p_bar)}


def np_chart(defects, sample_size):
    defects = np.asarray(defects, dtype=float)
    n = float(sample_size)
    p_bar = defects.mean() / n
    np_bar = n * p_bar
    sigma = np.sqrt(n * p_bar * (1 - p_bar))
    ucl = np_bar + 3 * sigma
    lcl = max(0.0, np_bar - 3 * sigma)
    fig, ax = plt.subplots(figsize=(10, 5))
    _attr_plot(ax, defects, np_bar, ucl, lcl, "np Chart - Number Defective", "Defective count")
    fig.tight_layout()
    return fig, {"center": float(np_bar)}


def c_chart(counts):
    counts = np.asarray(counts, dtype=float)
    c_bar = counts.mean()
    sigma = np.sqrt(c_bar)
    ucl = c_bar + 3 * sigma
    lcl = max(0.0, c_bar - 3 * sigma)
    fig, ax = plt.subplots(figsize=(10, 5))
    _attr_plot(ax, counts, c_bar, ucl, lcl, "c Chart - Count of Defects", "Defect count")
    fig.tight_layout()
    return fig, {"center": float(c_bar)}


def u_chart(counts, sample_sizes):
    counts = np.asarray(counts, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    u = counts / n
    u_bar = counts.sum() / n.sum()
    sigma = np.sqrt(u_bar / n)
    ucl = u_bar + 3 * sigma
    lcl = np.maximum(0.0, u_bar - 3 * sigma)
    fig, ax = plt.subplots(figsize=(10, 5))
    _attr_plot(ax, u, u_bar, ucl, lcl, "u Chart - Defects per Unit", "Defects/unit")
    fig.tight_layout()
    return fig, {"center": float(u_bar)}
