# app.py
import os
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="BEAD Multi-Strategy Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("BEAD Multi-Strategy Explorer")
st.caption("Compare 3 strategies merged into one view. Auto-wrapping facets and auto height per row.")

# =========================
# Files & strategies
# =========================
DEFAULT_FILES = {
    "Fixed cost with weight": "bead_fixedcost_combinations_weight_results.csv",
    "No-weight (coverage first)": "summary_cover_first.csv",
    "No-weight (fiber first)": "summary_fiber_first_noweight.csv",
}
RATIO_COLS  = ["fiber_ratio","fw_ratio","sat_ratio"]
BUDGET_COLS = ["total_spent","remaining_funds"]  # 'funds' 있으면 활용

# =========================
# Helpers
# =========================
def multiselect_with_select_all(label: str, options, key: str, default_n: int = 6):
    """Multiselect with All / Clear buttons (English)."""
    if key not in st.session_state:
        st.session_state[key] = options[:min(default_n, len(options))]
    cols = st.columns([4, 0.9, 0.9])
    with cols[0]:
        current = st.multiselect(
            label, options=options,
            default=st.session_state[key], key=f"{key}_ms"
        )
    with cols[1]:
        if st.button("All", key=f"{key}_all"):
            current = list(options)
    with cols[2]:
        if st.button("Clear", key=f"{key}_none"):
            current = []
    st.session_state[key] = current
    return current

def try_read_csv(obj_or_path):
    try:
        return pd.read_csv(obj_or_path)
    except Exception:
        return None

def load_strategy_df(label: str, file_obj, fallback_path: str):
    """Load CSV for one strategy, normalize columns, add 'strategy' column."""
    if file_obj is not None:
        df = try_read_csv(file_obj)
    else:
        df = try_read_csv(fallback_path if os.path.exists(fallback_path) else None)
    if df is None:
        return None, f"[{label}] file not found or failed to read."

    df = df.copy()
    # normalize keys
    if "State" in df.columns and "state" not in df.columns:
        df = df.rename(columns={"State":"state"})
    if "Scenario" in df.columns and "scenario" not in df.columns:
        df = df.rename(columns={"Scenario":"scenario"})

    if "remaining_funds" not in df.columns and "funds" in df.columns and "total_spent" in df.columns:
        with pd.option_context("mode.use_inf_as_na", True):
            df["remaining_funds"] = pd.to_numeric(df["funds"], errors="coerce") - pd.to_numeric(df["total_spent"], errors="coerce")

    # numeric casting
    for c in RATIO_COLS + BUDGET_COLS + ["funds","fiber_served","fw_served","sat_served"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # must have keys
    for c in ["state","scenario"]:
        if c not in df.columns:
            return None, f"[{label}] missing column: '{c}'"

    df["state"] = df["state"].astype(str)
    df["scenario"] = df["scenario"].astype(str)
    df["strategy"] = label
    return df, None

def auto_wrap_count(n_states: int) -> int:
    """Heuristic: choose facet_col_wrap to keep each panel reasonably wide."""
    if n_states <= 1:  return 1
    if n_states <= 4:  return 2
    if n_states <= 9:  return 3
    if n_states <= 16: return 4
    if n_states <= 25: return 5
    if n_states <= 36: return 6
    if n_states <= 49: return 7
    return 8  # up to 50

def compute_rows(sel_states: int, sel_strategies: int, facet_by_state: bool, facet_by_strategy_row: bool, wrap: int) -> int:
    """How many facet rows will be produced → to scale height automatically."""
    rows_states = math.ceil(sel_states / wrap) if facet_by_state and wrap > 0 else 1
    rows_strat  = sel_strategies if facet_by_strategy_row else 1
    return rows_states * rows_strat

def plot_safe(px_func, args: dict, height: int):
    """Guarded plotting (never leave fig undefined)."""
    try:
        fig = px_func(**args)
    except Exception as e:
        st.error(f"Plotly failed to render: {e}")
        fig = go.Figure()
    # Keep y-axis nice if ratio
    return fig, height

# =========================
# Sidebar: load files
# =========================
with st.sidebar:
    st.header("Load data")
    up1 = st.file_uploader("Fixed cost with weight (CSV)", type=["csv"])
    up2 = st.file_uploader("No-weight (coverage first) (CSV)", type=["csv"])
    up3 = st.file_uploader("No-weight (fiber first) (CSV)", type=["csv"])

dfs, msgs = [], []
for label, up, path in [
    ("Fixed cost with weight", up1, DEFAULT_FILES["Fixed cost with weight"]),
    ("No-weight (coverage first)", up2, DEFAULT_FILES["No-weight (coverage first)"]),
    ("No-weight (fiber first)", up3, DEFAULT_FILES["No-weight (fiber first)"]),
]:
    df_i, msg = load_strategy_df(label, up, path)
    if df_i is None:
        msgs.append(msg)
    else:
        dfs.append(df_i)

if len(dfs) == 0:
    with st.expander("Diagnostics"):
        st.write("Working directory:", os.getcwd())
        st.write("Files:", os.listdir("."))
        st.warning("\n".join([m for m in msgs if m]))
    st.stop()

data = pd.concat(dfs, ignore_index=True)

# =========================
# Filters
# =========================
with st.sidebar:
    st.header("Filters")

all_states     = sorted(data["state"].dropna().unique())
all_scenarios  = list(pd.unique(data["scenario"]))
all_strategies = list(pd.unique(data["strategy"]))

sel_states    = multiselect_with_select_all("States", all_states,    key="states_select",    default_n=6)
sel_scenarios = multiselect_with_select_all("Scenarios", all_scenarios, key="scenarios_select", default_n=len(all_scenarios))
sel_strategies= multiselect_with_select_all("Strategies", all_strategies, key="strategies_select", default_n=len(all_strategies))

if len(sel_states)==0 or len(sel_scenarios)==0 or len(sel_strategies)==0:
    st.warning("Pick at least one State, Scenario, and Strategy.")
    st.stop()

st.sidebar.header("Layout")
facet_by_state = st.sidebar.checkbox("Facet by state", value=(len(sel_states) > 1))
facet_by_strategy_row = st.sidebar.checkbox("Facet row = strategy", value=True)

# 자동 랩핑
wrap_count = auto_wrap_count(len(sel_states)) if facet_by_state else 0
st.sidebar.caption(f"Auto facet wrap = {wrap_count if wrap_count else 'N/A'}")

# =========================
# Apply filters
# =========================
df = data[
    data["state"].isin(sel_states) &
    data["scenario"].isin(sel_scenarios) &
    data["strategy"].isin(sel_strategies)
].copy()

df["state"]    = pd.Categorical(df["state"], categories=sel_states, ordered=True)
df["scenario"] = pd.Categorical(df["scenario"], categories=sel_scenarios, ordered=True)
df["strategy"] = pd.Categorical(df["strategy"], categories=sel_strategies, ordered=True)
df = df.sort_values(["state","strategy","scenario"])

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Stacked Ratios (Fiber/FW/SAT)", "Budget (Spent vs Remaining)", "Data"])

# =========================
# (1) Ratios (stacked)
# =========================
with tab1:
    st.subheader("Fiber / FW / SAT Ratios (stacked by scenario)")
    miss = [c for c in RATIO_COLS if c not in df.columns]
    if miss:
        st.error(f"Missing columns: {miss}")
    else:
        r = df.melt(
            id_vars=["state","scenario","strategy"],
            value_vars=RATIO_COLS,
            var_name="tech", value_name="ratio"
        ).dropna(subset=["ratio","state","scenario","strategy"])
        if r.empty:
            st.warning("No data to plot for the current filters.")
        else:
            r["tech"] = r["tech"].map({"fiber_ratio":"Fiber","fw_ratio":"FW","sat_ratio":"SAT"})
            r["state"]    = pd.Categorical(r["state"],    categories=sel_states, ordered=True)
            r["scenario"] = pd.Categorical(r["scenario"], categories=sel_scenarios, ordered=True)
            r["strategy"] = pd.Categorical(r["strategy"], categories=sel_strategies, ordered=True)

            facet_args = {}
            if facet_by_strategy_row and facet_by_state:
                facet_args = {"facet_row":"strategy", "facet_col":"state", "facet_col_wrap": wrap_count}
            elif facet_by_strategy_row and not facet_by_state:
                facet_args = {"facet_row":"strategy"}
            elif (not facet_by_strategy_row) and facet_by_state:
                facet_args = {"facet_col":"state", "facet_col_wrap": wrap_count}

            # 높이 자동: 행 수 × 기준높이
            n_rows = compute_rows(len(sel_states), len(sel_strategies), facet_by_state, facet_by_strategy_row, wrap_count)
            base_h = 200  # 한 행 당 높이 (px)
            auto_height = max(base_h, base_h * n_rows)

            args = dict(
                data_frame=r, x="scenario", y="ratio", color="tech",
                barmode="stack",
                category_orders={"scenario": sel_scenarios, "strategy": sel_strategies},
                **facet_args
            )
            fig, _ = plot_safe(px.bar, args, auto_height)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(legend_title_text="Tech", height=auto_height)
            st.plotly_chart(fig, use_container_width=True)

# =========================
# (2) Budget
# =========================
with tab2:
    st.subheader("Budget: Spent vs Remaining by scenario")
    miss = [c for c in BUDGET_COLS if c not in df.columns]
    if miss:
        st.error(f"Missing columns: {miss}")
    else:
        b = df.melt(
            id_vars=["state","scenario","strategy"],
            value_vars=BUDGET_COLS,
            var_name="budget_metric", value_name="amount"
        ).dropna(subset=["amount","state","scenario","strategy"])
        if b.empty:
            st.warning("No budget data to plot for the current filters.")
        else:
            b["budget_metric"] = b["budget_metric"].map({"total_spent":"Spent","remaining_funds":"Remaining"})
            b["state"]    = pd.Categorical(b["state"],    categories=sel_states, ordered=True)
            b["scenario"] = pd.Categorical(b["scenario"], categories=sel_scenarios, ordered=True)
            b["strategy"] = pd.Categorical(b["strategy"], categories=sel_strategies, ordered=True)

            facet_args = {}
            if facet_by_state:
                facet_args = {"facet_col":"state", "facet_col_wrap": wrap_count}

            n_rows = compute_rows(len(sel_states), 1, facet_by_state, False, wrap_count)  # budget은 row=strategy 안 씀
            base_h = 550
            auto_height = max(base_h, base_h * n_rows)

            args2 = dict(
                data_frame=b, x="scenario", y="amount",
                color="strategy", barmode="group",
                pattern_shape="budget_metric", pattern_shape_sequence=["","/"],
                category_orders={"scenario": sel_scenarios, "strategy": sel_strategies},
                **facet_args
            )
            fig2, _ = plot_safe(px.bar, args2, auto_height)
            fig2.update_layout(legend_title_text="Strategy", height=auto_height)
            st.plotly_chart(fig2, use_container_width=True)

            # Optional: utilization
            if "funds" in df.columns and df["funds"].notna().any():
                st.markdown("**Budget Utilization = Spent / Funds**")
                util = df[df["funds"] > 0].copy()
                if util.empty:
                    st.info("No positive `funds` to compute utilization.")
                else:
                    util["utilization"] = util["total_spent"] / util["funds"]
                    facet_u = {}
                    if facet_by_state:
                        facet_u = {"facet_col":"state", "facet_col_wrap": wrap_count}

                    n_rows_u = compute_rows(len(sel_states), 1, facet_by_state, False, wrap_count)
                    auto_h_u = max(base_h, base_h * n_rows_u)

                    args3 = dict(
                        data_frame=util, x="scenario", y="utilization",
                        color="strategy", barmode="group",
                        category_orders={"scenario": sel_scenarios, "strategy": sel_strategies},
                        **facet_u
                    )
                    fig3, _ = plot_safe(px.bar, args3, auto_h_u)
                    fig3.update_yaxes(tickformat=".0%")
                    fig3.update_layout(height=auto_h_u)
                    st.plotly_chart(fig3, use_container_width=True)

# =========================
# (3) Data table
# =========================
with tab3:
    show_cols = ["state","scenario","strategy"] + RATIO_COLS + BUDGET_COLS
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=560)
    st.download_button(
        label="Download filtered (CSV)",
        data=df[show_cols].to_csv(index=False),
        file_name="filtered_merged_strategies.csv",
        mime="text/csv"
    )

# =========================
# Diagnostics (optional)
# =========================
with st.expander("Diagnostics"):
    st.write("Working directory:", os.getcwd())
    st.write("Files:", os.listdir("."))
