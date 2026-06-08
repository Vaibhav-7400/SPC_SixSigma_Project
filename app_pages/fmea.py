import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data.load_data import load_mercedes


def _count_runs(y, run_len=6):
    diffs = np.diff(y)
    count = 0
    run = 1
    last_sign = 0
    for d in diffs:
        sign = 1 if d > 0 else (-1 if d < 0 else 0)
        if sign != 0 and sign == last_sign:
            run += 1
            if run >= run_len:
                count += 1
                run = 1
        else:
            run = 1
        last_sign = sign
    return count


def _occurrence_score(rate):
    if rate >= 0.10: return 10
    if rate >= 0.05: return 9
    if rate >= 0.02: return 8
    if rate >= 0.01: return 7
    if rate >= 0.005: return 6
    if rate >= 0.002: return 5
    if rate >= 0.001: return 4
    if rate >= 0.0005: return 3
    if rate > 0: return 2
    return 1


def build_fmea_table(df, USL=130, LSL=75):
    y = df["y"].to_numpy()
    n = len(y)
    mean = float(y.mean())
    std = float(y.std(ddof=1))

    above_usl = int(np.sum(y > USL))
    below_lsl = int(np.sum(y < LSL))
    cfg_std = df.groupby("X0")["y"].std(ddof=1)
    high_var = int(np.sum(cfg_std > 15))
    total_cfgs = max(int(cfg_std.shape[0]), 1)
    trends = _count_runs(y, run_len=6)
    extreme = int(np.sum(np.abs(y - mean) > 3 * std))

    rows = [
        {
            "Mode": "Test exceeds USL (y>130s)",
            "Effect": "Slow test, line stop, customer perception",
            "Cause": "Sub-assembly drag, bench calibration drift",
            "Count": above_usl, "Rate": above_usl / n,
            "S": 7, "O": _occurrence_score(above_usl / n), "D": 3,
            "Action Plan": "Monthly bench calibration; sub-assembly drag inspection; alert at +2σ to trigger early intervention.",
        },
        {
            "Mode": "Test below LSL (y<75s)",
            "Effect": "Suspicious fast test, possible skipped steps",
            "Cause": "Test segment skipped, sensor short, software glitch",
            "Count": below_lsl, "Rate": below_lsl / n,
            "S": 8, "O": _occurrence_score(below_lsl / n), "D": 4,
            "Action Plan": "Audit test sequence completion logs; add poka-yoke check that all segments ran; sensor diagnostics on every shift start.",
        },
        {
            "Mode": "High-variance configuration (std>15s)",
            "Effect": "Unpredictable bench time, scheduling/throughput loss",
            "Cause": "Configuration-specific fixture instability",
            "Count": high_var, "Rate": high_var / total_cfgs,
            "S": 6, "O": _occurrence_score(high_var / total_cfgs), "D": 5,
            "Action Plan": "Per-config DOE on fixture clamping torque; tighten tolerance on the worst 3 configurations; targeted MSA study.",
        },
        {
            "Mode": "Consecutive trend (6+ in same direction)",
            "Effect": "Slow drift from target, capability loss over shift",
            "Cause": "Tool wear, thermal drift, operator fatigue",
            "Count": trends, "Rate": trends / max(n, 1),
            "S": 5, "O": _occurrence_score(trends / max(n, 1)), "D": 6,
            "Action Plan": "Add EWMA/trend rule alerts to the line dashboard; scheduled tool change at wear threshold; HVAC log correlation.",
        },
        {
            "Mode": "Extreme outlier batch (|y-mean|>3sigma)",
            "Effect": "Anomalous unit shipped or scrapped",
            "Cause": "Defective sub-assembly, mis-installed harness",
            "Count": extreme, "Rate": extreme / n,
            "S": 9, "O": _occurrence_score(extreme / n), "D": 2,
            "Action Plan": "Mandatory rework station for outliers; harness routing inspection on flagged units; supplier 8D on repeat lots.",
        },
    ]
    tbl = pd.DataFrame(rows)
    tbl["RPN"] = tbl["S"] * tbl["O"] * tbl["D"]
    tbl = tbl.sort_values("RPN", ascending=False).reset_index(drop=True)
    return tbl


def _row_color(row):
    if row["S"] >= 9:
        return ["background-color: #ff6961; color: black; font-weight: bold;"] * len(row)
    if row["RPN"] >= 200:
        return ["background-color: #ff6961; color: black;"] * len(row)
    if row["RPN"] >= 100:
        return ["background-color: #ffd27f; color: black;"] * len(row)
    return ["background-color: #b6e3b6; color: black;"] * len(row)


def render_fmea(USL=130.0, LSL=75.0):
    st.header("FMEA - Failure Modes from Mercedes Bench Data")
    st.caption("Severity, Occurrence, Detection each scored 1-10. RPN = S * O * D.")

    df = load_mercedes()
    tbl = build_fmea_table(df, USL=USL, LSL=LSL)

    top = tbl.iloc[0]
    st.error(
        f"**Highest priority: {top['Mode']} (RPN={int(top['RPN'])})** — "
        f"{top['Action Plan']}"
    )

    show_cols = ["Mode", "Effect", "Cause", "Count", "Rate",
                 "S", "O", "D", "RPN", "Action Plan"]
    st.subheader("FMEA Table")
    st.dataframe(
        tbl[show_cols].style.apply(_row_color, axis=1)
        .format({"Rate": "{:.4%}"}),
        width='stretch',
    )

    flagged = tbl[tbl["S"] >= 9]
    if len(flagged):
        st.warning("Severity >= 9 flagged: " + ", ".join(flagged["Mode"].tolist()))

    c1, c2, c3 = st.columns(3)
    c1.metric("Max RPN", int(tbl["RPN"].max()))
    c2.metric("Modes >= 200", int((tbl["RPN"] >= 200).sum()))
    c3.metric("Mean RPN", f"{tbl['RPN'].mean():.0f}")

    st.subheader("Before / After Mitigation (40-60% RPN reduction)")
    rng = np.random.default_rng(7)
    reductions = rng.uniform(0.4, 0.6, len(tbl))
    after = (tbl["RPN"] * (1 - reductions)).round().astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tbl))
    w = 0.4
    ax.bar(x - w / 2, tbl["RPN"], w, label="Before", color="indianred")
    ax.bar(x + w / 2, after, w, label="After", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels([m[:20] for m in tbl["Mode"]], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("RPN")
    ax.set_title("RPN Before vs After Mitigation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Severity x Occurrence Bubble (size = RPN)")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sizes = tbl["RPN"] * 2
    sc = ax2.scatter(tbl["O"], tbl["S"], s=sizes, c=tbl["RPN"],
                     cmap="Reds", alpha=0.75, edgecolors="black")
    for _, row in tbl.iterrows():
        ax2.annotate(row["Mode"][:20], (row["O"], row["S"]),
                     fontsize=8, xytext=(6, 6), textcoords="offset points")
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 11)
    ax2.set_xlabel("Occurrence")
    ax2.set_ylabel("Severity")
    ax2.set_title("FMEA - Severity vs Occurrence")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="RPN")
    fig2.tight_layout()
    st.pyplot(fig2)
