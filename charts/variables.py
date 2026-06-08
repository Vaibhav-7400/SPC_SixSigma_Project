import numpy as np
import matplotlib.pyplot as plt


D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704,
      8: 2.847, 9: 2.970, 10: 3.078}
D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076,
      8: 0.136, 9: 0.184, 10: 0.223}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924,
      8: 1.864, 9: 1.816, 10: 1.777}
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419,
      8: 0.373, 9: 0.337, 10: 0.308}
A3 = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182,
      8: 1.099, 9: 1.032, 10: 0.975}
B3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.030, 7: 0.118,
      8: 0.185, 9: 0.239, 10: 0.284}
B4 = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882,
      8: 1.815, 9: 1.761, 10: 1.716}
C4 = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594,
      8: 0.9650, 9: 0.9693, 10: 0.9727}


def _plot_panel(ax, data, center, ucl, lcl, title, ylabel):
    data = np.asarray(data, dtype=float)
    x = np.arange(len(data))
    ax.plot(x, data, color="steelblue", linewidth=1.2, marker="o", markersize=3)
    ooc = (data > ucl) | (data < lcl)
    if np.any(ooc):
        ax.plot(x[ooc], data[ooc], "o", color="red", markersize=7, label="OOC")
    ax.axhline(center, color="green", linestyle="--", linewidth=1.2, label="CL")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=1.2, label="UCL")
    ax.axhline(lcl, color="red", linestyle="--", linewidth=1.2, label="LCL")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def imr_chart(data):
    data = np.asarray(data, dtype=float)
    mr = np.abs(np.diff(data))
    mr_bar = float(np.mean(mr))
    x_bar = float(np.mean(data))
    sigma = mr_bar / D2[2]
    ucl_i = x_bar + 3 * sigma
    lcl_i = x_bar - 3 * sigma
    ucl_mr = D4[2] * mr_bar
    lcl_mr = D3[2] * mr_bar

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    _plot_panel(axes[0], data, x_bar, ucl_i, lcl_i,
                "Individuals (I) Chart - y (test bench time, s)", "y")
    mr_with_pad = np.concatenate(([np.nan], mr))
    _plot_panel(axes[1], np.nan_to_num(mr_with_pad, nan=mr_bar),
                mr_bar, ucl_mr, lcl_mr,
                "Moving Range (MR) Chart", "MR")
    fig.tight_layout()
    return fig, {"mean": x_bar, "sigma": sigma, "ucl": ucl_i, "lcl": lcl_i}


def _subgroup(data, k):
    data = np.asarray(data, dtype=float)
    m = (len(data) // k) * k
    return data[:m].reshape(-1, k)


def xbar_r_chart(data, subgroup_size=5):
    k = subgroup_size
    groups = _subgroup(data, k)
    xbar = groups.mean(axis=1)
    r = groups.max(axis=1) - groups.min(axis=1)
    xbarbar = float(xbar.mean())
    rbar = float(r.mean())
    ucl_x = xbarbar + A2[k] * rbar
    lcl_x = xbarbar - A2[k] * rbar
    ucl_r = D4[k] * rbar
    lcl_r = D3[k] * rbar

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    _plot_panel(axes[0], xbar, xbarbar, ucl_x, lcl_x,
                f"Xbar Chart (n={k})", "Subgroup mean")
    _plot_panel(axes[1], r, rbar, ucl_r, lcl_r, "R Chart", "Subgroup range")
    fig.tight_layout()
    return fig, {"mean": xbarbar, "sigma": rbar / D2[k],
                 "ucl": ucl_x, "lcl": lcl_x}


def xbar_s_chart(data, subgroup_size=5):
    k = subgroup_size
    groups = _subgroup(data, k)
    xbar = groups.mean(axis=1)
    s = groups.std(axis=1, ddof=1)
    xbarbar = float(xbar.mean())
    sbar = float(s.mean())
    ucl_x = xbarbar + A3[k] * sbar
    lcl_x = xbarbar - A3[k] * sbar
    ucl_s = B4[k] * sbar
    lcl_s = B3[k] * sbar

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    _plot_panel(axes[0], xbar, xbarbar, ucl_x, lcl_x,
                f"Xbar Chart (n={k})", "Subgroup mean")
    _plot_panel(axes[1], s, sbar, ucl_s, lcl_s, "S Chart", "Subgroup std dev")
    fig.tight_layout()
    return fig, {"mean": xbarbar, "sigma": sbar / C4[k],
                 "ucl": ucl_x, "lcl": lcl_x}


def ewma_chart(data, lambda_=0.2):
    data = np.asarray(data, dtype=float)
    mu = float(data.mean())
    sigma = float(data.std(ddof=1))
    z = np.zeros_like(data)
    z[0] = mu
    for i in range(1, len(data)):
        z[i] = lambda_ * data[i] + (1 - lambda_) * z[i - 1]
    idx = np.arange(1, len(data) + 1)
    factor = (lambda_ / (2 - lambda_)) * (1 - (1 - lambda_) ** (2 * idx))
    ucl = mu + 3 * sigma * np.sqrt(factor)
    lcl = mu - 3 * sigma * np.sqrt(factor)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(data))
    ax.plot(x, z, color="steelblue", linewidth=1.2, marker="o", markersize=3, label="EWMA")
    ax.axhline(mu, color="green", linestyle="--", label="CL")
    ax.plot(x, ucl, color="red", linestyle="--", label="UCL")
    ax.plot(x, lcl, color="red", linestyle="--", label="LCL")
    ooc = (z > ucl) | (z < lcl)
    if np.any(ooc):
        ax.plot(x[ooc], z[ooc], "o", color="red", markersize=7)
    ax.set_title(f"EWMA Chart (lambda={lambda_})")
    ax.set_ylabel("EWMA")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, {"mean": mu, "sigma": sigma}


def cusum_chart(data, target=None, k=0.5, h=5):
    data = np.asarray(data, dtype=float)
    if target is None:
        target = float(data.mean())
    sigma = float(data.std(ddof=1))
    K = k * sigma
    H = h * sigma
    sh = np.zeros_like(data)
    sl = np.zeros_like(data)
    for i in range(len(data)):
        prev_h = sh[i - 1] if i > 0 else 0.0
        prev_l = sl[i - 1] if i > 0 else 0.0
        sh[i] = max(0.0, prev_h + (data[i] - target) - K)
        sl[i] = min(0.0, prev_l + (data[i] - target) + K)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(data))
    ax.plot(x, sh, color="steelblue", label="C+ (upper)")
    ax.plot(x, sl, color="darkorange", label="C- (lower)")
    ax.axhline(H, color="red", linestyle="--", label=f"H=+{H:.2f}")
    ax.axhline(-H, color="red", linestyle="--", label=f"H=-{H:.2f}")
    ax.axhline(0, color="green", linestyle="--")
    if np.any(sh > H):
        ax.plot(x[sh > H], sh[sh > H], "o", color="red", markersize=7)
    if np.any(sl < -H):
        ax.plot(x[sl < -H], sl[sl < -H], "o", color="red", markersize=7)
    ax.set_title(f"Tabular CUSUM (k={k}, h={h}, target={target:.2f})")
    ax.set_ylabel("Cumulative deviation")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, {"mean": target, "sigma": sigma, "H": H, "K": K}
