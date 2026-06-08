import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from data.load_data import load_mercedes, get_ooc_by_config


FISHBONE = {
    "Man": "Operator handling at test bench; shift-to-shift variation in fixture loading.",
    "Machine": "Test bench calibration drift; sensor wear; clamping fixture wear.",
    "Method": "Test sequence definition; pass/fail thresholds; warm-up procedure.",
    "Material": "Sub-assembly tolerance stack-up; supplier lot variation in subsystems.",
    "Measurement": "Timer resolution; bench-to-bench bias; data-logging latency.",
    "Mother Nature": "Bay temperature/humidity affecting electronic test response times.",
}


def render_pareto(USL=130.0, LSL=75.0):
    st.header("Pareto Analysis & Root Cause (Mercedes Test Bench)")
    st.caption("Group test failures by car configuration (X0) to find the vital few.")

    df = load_mercedes()
    grouped = get_ooc_by_config(df, USL=USL, LSL=LSL)
    total_ooc = int(grouped["ooc_count"].sum())
    top_config = grouped.iloc[0]["X0"] if len(grouped) else "n/a"
    top3_share = (grouped["ooc_count"].head(3).sum() / total_ooc * 100) if total_ooc else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total OOC parts", f"{total_ooc:,}")
    c2.metric("Top offending config", f"X0={top_config}")
    c3.metric("Top-3 share of OOC", f"{top3_share:.1f}%")

    plot_df = grouped[grouped["ooc_count"] > 0].copy()
    if len(plot_df) == 0:
        st.info("No OOC parts found with current spec limits.")
        return

    plot_df["cum_pct"] = plot_df["ooc_count"].cumsum() / plot_df["ooc_count"].sum() * 100
    plot_df = plot_df.head(20)

    fig, ax1 = plt.subplots(figsize=(10, max(4, 0.4 * len(plot_df))))
    y = np.arange(len(plot_df))[::-1]
    ax1.barh(y, plot_df["ooc_count"], color="steelblue", edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(plot_df["X0"])
    ax1.set_xlabel("OOC count")
    ax1.set_title("Pareto: OOC parts by car configuration (X0)")
    ax1.grid(True, axis="x", alpha=0.3)

    ax2 = ax1.twiny()
    ax2.plot(plot_df["cum_pct"], y, color="darkorange", marker="o", linewidth=1.5,
             label="Cumulative %")
    ax2.axvline(80, color="red", linestyle="--", linewidth=1.0, label="80% line")
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Cumulative %")
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    top_share = grouped.iloc[0]["ooc_count"] / total_ooc * 100 if total_ooc else 0.0
    top_rate = grouped.iloc[0]["ooc_rate"] * 100
    st.success(
        f"**Recommended Action:** Focus DMAIC project on configuration "
        f"**X0={top_config}** — fixing this alone could reduce OOC by "
        f"**{top_share:.1f}%** of all failures "
        f"(its in-config OOC rate is {top_rate:.1f}% vs the facility-wide baseline)."
    )

    st.subheader("6M Fishbone (Test Bench Context)")
    cols = st.columns(3)
    for i, (cat, desc) in enumerate(FISHBONE.items()):
        with cols[i % 3]:
            st.markdown(f"**{cat}**")
            st.write(desc)

    st.subheader("Detail Table (sorted by OOC count)")
    st.dataframe(grouped, width='stretch')
