import numpy as np
import streamlit as st

from data.load_data import load_mercedes, get_measurement, get_ooc_by_config
from charts.variables import (
    imr_chart, xbar_r_chart, xbar_s_chart, ewma_chart, cusum_chart,
)
from charts.capability import capability_analysis, capability_histogram
from rules.weco import weco_rules
from rules.nelson import nelson_rules
from app_pages.pareto import render_pareto
from app_pages.fmea import render_fmea, build_fmea_table


st.set_page_config(
    page_title="Six Sigma SPC Dashboard - Mercedes-Benz", layout="wide"
)


FOOTER_TEXT = (
    "Built with Streamlit | Data: Mercedes-Benz Greener Manufacturing (Kaggle) "
    "| Six Sigma DMAIC methodology"
)


def render_footer():
    st.markdown("---")
    st.caption(FOOTER_TEXT)


@st.cache_data
def _load():
    return load_mercedes()


@st.cache_data
def _summary(df, USL=130.0, LSL=75.0):
    y = get_measurement(df).to_numpy()
    cap = capability_analysis(y, USL, LSL, target=100.0)
    grouped = get_ooc_by_config(df, USL=USL, LSL=LSL)
    top_cfg = grouped.iloc[0]["X0"] if len(grouped) else "n/a"
    viol = int(grouped["ooc_count"].sum())
    fmea = build_fmea_table(df, USL=USL, LSL=LSL)
    max_rpn = int(fmea["RPN"].max())
    return {
        "Cpk": cap["Cpk"], "sigma": cap["sigma_level"],
        "violations": viol, "top_cfg": top_cfg, "max_rpn": max_rpn,
    }


WECO_DESCRIPTIONS = {
    1: "1 point beyond 3σ — strong evidence of a special cause.",
    2: "2 of 3 consecutive points beyond 2σ on the same side — small-shift signal.",
    3: "4 of 5 consecutive points beyond 1σ on the same side — moderate shift.",
    4: "8 consecutive points on the same side of the centerline — mean shift.",
}
NELSON_DESCRIPTIONS = {
    5: "6 consecutive points all increasing or all decreasing — trend.",
    6: "14 consecutive points alternating up/down — over-control or sawtooth.",
    7: "15 consecutive points within 1σ — stratification or reduced variation.",
    8: "8 consecutive points beyond 1σ on both sides — mixture of two processes.",
}


def home_page(df):
    st.title("Six Sigma SPC Dashboard - Mercedes-Benz Test Facility")

    st.subheader("Project Story")
    st.write(
        "Mercedes-Benz test facility runs every assembled car through a test "
        "bench before shipping. Test time (y, seconds) is a critical-to-quality "
        "metric with a **target of 100s** and spec limits 75-130s: too slow "
        "means line stoppage and customer wait; too fast may signal skipped "
        "test steps. We analyzed 4,209 test cycles across 47 car configurations "
        "using SPC and DMAIC methodology to surface stability issues and "
        "prioritize improvement targets."
    )

    st.subheader("Key Findings")
    st.error(
        "- **Process Cpk = 0.675** (target ≥ 1.33) — process is **NOT capable**\n"
        "- Running at **3.52 sigma level** with **31,814 DPMO**\n"
        "- Configuration **X0=az** is the vital few: **25.7% OOC rate** vs 3.0% baseline\n"
        "- **One configuration drives 35%** of all out-of-spec tests\n"
        "- **Shapiro-Wilk p < 0.0001**: data is non-normal — use Pp/Ppk over Cp/Cpk"
    )

    s = _summary(df, 130.0, 75.0)
    st.subheader("Live Summary Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cpk", f"{s['Cpk']:.3f}")
    c2.metric("Sigma level", f"{s['sigma']:.2f}")
    c3.metric("Spec violations", f"{s['violations']:,}")
    c4.metric("Top defect config", f"X0={s['top_cfg']}")
    c5.metric("Max RPN", f"{s['max_rpn']}")

    st.subheader("DMAIC Roadmap")
    phases = [
        ("Define",
         "Project charter, CTQ = test-bench time within 75-130s, target 100s."),
        ("Measure",
         "I-MR / Xbar-R / Xbar-S / EWMA / CUSUM on `y`; capability indices."),
        ("Analyze",
         "Pareto by X0; 6M fishbone; WECO + Nelson rules to surface signals."),
        ("Improve",
         "FMEA RPN ranking; before/after simulation; mitigation focus."),
        ("Control",
         "Locked spec limits, dashboard-driven rule monitoring."),
    ]
    cols = st.columns(5)
    for col, (name, body) in zip(cols, phases):
        with col:
            st.markdown(f"### {name}")
            st.write(body)


def _rule_badges(label, results, descriptions):
    cols = st.columns(len(results))
    for col, (rule_id, idxs) in zip(cols, results.items()):
        passed = len(idxs) == 0
        if passed:
            status = "✅ Pass"
            detail = ""
        else:
            shown = idxs[:5]
            more = f" (+{len(idxs) - 5} more)" if len(idxs) > 5 else ""
            status = f"❌ {len(idxs)} violation(s)"
            detail = f"\n\nFirst points: `{shown}`{more}"
        col.markdown(
            f"**{label} {rule_id}**  \n{status}{detail}\n\n"
            f"_{descriptions[rule_id]}_"
        )


def spc_page(df):
    st.title("SPC Control Charts")

    cfg_options = ["(all)"] + sorted(df["X0"].unique().tolist())
    c1, c2 = st.columns(2)
    cfg = c1.selectbox("Configuration filter (X0)", cfg_options)
    chart_type = c2.selectbox(
        "Chart type", ["I-MR", "Xbar-R", "Xbar-S", "EWMA", "CUSUM"]
    )

    if cfg != "(all)":
        y_all = df.loc[df["X0"] == cfg, "y"].to_numpy()
    else:
        y_all = df["y"].to_numpy()

    if len(y_all) < 10:
        st.warning("Not enough data for this configuration.")
        return

    max_n = min(2000, len(y_all))
    sample_n = st.slider("Sample size to plot (head)", 50, max_n, min(500, max_n), step=50)
    y = y_all[:sample_n]
    st.caption(f"Plotting first {len(y)} of {len(y_all)} observations.")

    if chart_type == "I-MR":
        fig, params = imr_chart(y)
    elif chart_type == "Xbar-R":
        fig, params = xbar_r_chart(y, subgroup_size=5)
    elif chart_type == "Xbar-S":
        fig, params = xbar_s_chart(y, subgroup_size=5)
    elif chart_type == "EWMA":
        fig, params = ewma_chart(y, lambda_=0.2)
    else:
        fig, params = cusum_chart(y)
    st.pyplot(fig)

    st.subheader("Run-Rule Summary")
    mean = params.get("mean", float(np.mean(y)))
    sigma = params.get("sigma", float(np.std(y, ddof=1)))
    weco = weco_rules(y, mean, sigma)
    nelson = nelson_rules(y, mean, sigma)
    st.markdown("**Western Electric (WECO) - Rules 1-4**")
    _rule_badges("WECO", weco, WECO_DESCRIPTIONS)
    st.markdown("**Nelson - Rules 5-8**")
    _rule_badges("Nelson", nelson, NELSON_DESCRIPTIONS)

    with st.expander("What this means"):
        st.markdown(
            "Run rules turn an otherwise quiet control chart into an early-warning "
            "system. Each rule encodes a pattern that is unlikely under a stable, "
            "in-control process:\n\n"
            "**WECO (Western Electric)**\n"
            "- *Rule 1* catches single extreme points — almost always a real disturbance.\n"
            "- *Rule 2* catches small mean shifts before Rule 1 triggers.\n"
            "- *Rule 3* catches moderate sustained shifts.\n"
            "- *Rule 4* catches a step change to one side of the mean.\n\n"
            "**Nelson** (extends WECO with pattern signals)\n"
            "- *Rule 5* catches monotonic drift (tool wear, thermal).\n"
            "- *Rule 6* catches over-control / sawtooth from operator tampering.\n"
            "- *Rule 7* catches stratification — variation lower than expected, often a measurement issue.\n"
            "- *Rule 8* catches mixtures — two distinct populations on the same chart."
        )


def _verdict(cpk):
    if np.isnan(cpk):
        return None
    if cpk >= 1.33:
        return ("success", f"✅ Process Capable (Cpk = {cpk:.3f} ≥ 1.33)")
    if cpk >= 1.0:
        return ("warning", f"⚠️ Marginally Capable (1.00 ≤ Cpk = {cpk:.3f} < 1.33)")
    return ("error", f"🛑 NOT Capable - Improvement Required (Cpk = {cpk:.3f} < 1.00)")


def capability_page(df):
    st.title("Process Capability")
    y = df["y"].to_numpy()
    c1, c2, c3 = st.columns(3)
    usl = c1.slider("USL (s)", 100.0, 200.0, 130.0, 0.5)
    lsl = c2.slider("LSL (s)", 30.0, 100.0, 75.0, 0.5)
    target = c3.slider("Target (s)", lsl, usl, 100.0, 0.5)

    idx = capability_analysis(y, usl, lsl, target=target)

    verdict = _verdict(idx["Cpk"])
    if verdict is not None:
        level, msg = verdict
        getattr(st, level)(msg)

    cols = st.columns(7)
    cols[0].metric("Cp", f"{idx['Cp']:.3f}")
    cols[1].metric("Cpk", f"{idx['Cpk']:.3f}")
    cols[2].metric("Pp", f"{idx['Pp']:.3f}")
    cols[3].metric("Ppk", f"{idx['Ppk']:.3f}")
    cols[4].metric("Cpm", f"{idx['Cpm']:.3f}")
    cols[5].metric("Sigma", f"{idx['sigma_level']:.2f}")
    cols[6].metric("DPMO", f"{idx['dpmo']:.0f}")

    if not np.isnan(idx["normality_p"]) and idx["normality_p"] < 0.05:
        st.warning(
            f"Shapiro-Wilk p={idx['normality_p']:.4f} < 0.05 - data is not normal; "
            "capability indices may be misleading. Consider Box-Cox or non-parametric."
        )
    else:
        st.info(f"Shapiro-Wilk p={idx['normality_p']:.4f}.")

    fig = capability_histogram(y, usl, lsl, target=target)
    st.pyplot(fig)

    with st.expander("Interpretation guide"):
        st.markdown(
            "- **Cp / Pp** >= 1.33 acceptable, >= 1.67 capable, >= 2.0 world-class.\n"
            "- **Cpk / Ppk** account for centering; Cpk < Cp means process is off-target.\n"
            "- **Cpm** penalizes deviation from target.\n"
            "- **Sigma level** ~= 3 * Cpk + 1.5 (assumes 1.5 sigma drift).\n"
            "- **DPMO** parts-per-million expected outside spec under normal model."
        )


def main():
    df = _load()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "SPC Control Charts", "Process Capability",
         "Pareto and RCA", "FMEA"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Rows: {len(df):,} | Configs: {df['X0'].nunique()} | "
        f"y range: {df['y'].min():.1f}-{df['y'].max():.1f}s"
    )

    if page == "Home":
        home_page(df)
    elif page == "SPC Control Charts":
        spc_page(df)
    elif page == "Process Capability":
        capability_page(df)
    elif page == "Pareto and RCA":
        render_pareto(USL=130.0, LSL=75.0)
    elif page == "FMEA":
        render_fmea(USL=130.0, LSL=75.0)

    render_footer()


if __name__ == "__main__":
    main()
