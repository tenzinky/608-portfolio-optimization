from pathlib import Path

import pandas as pd
import streamlit as st

from demo_data_module import (
    available_models,
    available_months_for_model,
    benchmark_history_for_model,
    benchmark_summary_row,
    load_portfolio_time_series,
    load_portfolio_vs_spy,
    load_portfolio_vs_spy_summary,
    load_selected_stocks,
    load_walk_forward_metrics,
    model_metrics_row,
    portfolio_history_for_model,
    recommended_stocks_for_month,
    validate_demo_artifacts,
)

try:
    from config import METRICS_DIR
except ImportError:
    METRICS_DIR = Path(__file__).resolve().parent / "artifacts" / "metrics"

OPTIMIZER_RESULTS_PATH = Path(METRICS_DIR) / "optimizer_results.csv"

st.set_page_config(
    page_title="Stock Ranking Demo",
    page_icon=":material/finance:",
    layout="wide",
)

@st.cache_data(show_spinner=False)
def load_demo_data():
    selected_df          = load_selected_stocks()
    metrics_df           = load_walk_forward_metrics()
    portfolio_df         = load_portfolio_time_series()
    benchmark_df         = load_portfolio_vs_spy()
    benchmark_summary_df = load_portfolio_vs_spy_summary()
    return selected_df, metrics_df, portfolio_df, benchmark_df, benchmark_summary_df


@st.cache_data(show_spinner=False)
def load_optimizer_results():
    if not OPTIMIZER_RESULTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(OPTIMIZER_RESULTS_PATH)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    return df


def format_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_float(value, digits: int = 3):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def build_performance_chart_data(portfolio_history, benchmark_history):
    chart_df = portfolio_history[["YearMonth", "cumulative_return"]].rename(
        columns={"cumulative_return": "Portfolio"}
    )
    chart_df = chart_df.set_index("YearMonth")
    if not benchmark_history.empty and "benchmark_cumulative_return" in benchmark_history.columns:
        benchmark_label = (
            benchmark_history["benchmark_symbol"].dropna().iloc[0]
            if benchmark_history["benchmark_symbol"].notna().any()
            else "Benchmark"
        )
        chart_df[benchmark_label] = benchmark_history.set_index("YearMonth")[
            "benchmark_cumulative_return"
        ]
    return chart_df.sort_index()


def compute_optimizer_metrics(df):
    if df.empty or "weighted_return" not in df.columns:
        return {}
    monthly = df.groupby("YearMonth")["weighted_return"].sum()
    cumulative = (1 + monthly).prod() - 1
    sharpe  = monthly.mean() / monthly.std() * (12 ** 0.5) if monthly.std() > 0 else 0
    return {
        "cumulative_return":  cumulative,
        "avg_monthly_return": monthly.mean(),
        "monthly_std":        monthly.std(),
        "sharpe":             sharpe,
        "win_rate":           (monthly > 0).mean(),
        "best_month":         monthly.max(),
        "worst_month":        monthly.min(),
    }


def build_optimizer_chart(df, benchmark_df):
    monthly = df.groupby("YearMonth")["weighted_return"].sum().sort_index()
    cum     = (1 + monthly).cumprod() - 1
    chart   = cum.rename("Optimized Portfolio").to_frame()

    if not benchmark_df.empty and "benchmark_cumulative_return" in benchmark_df.columns:
        spy = (
            benchmark_df[benchmark_df["benchmark_cumulative_return"].notna()]
            .drop_duplicates("YearMonth")
            .set_index("YearMonth")["benchmark_cumulative_return"]
            .rename("SPY")
            .sort_index()
        )
        chart = chart.join(spy, how="left")

    return chart.sort_index()


HERO_STYLE = """
<style>
.app-shell {
    background: linear-gradient(135deg, #f4efe4 0%, #f9fbff 45%, #e8f2ea 100%);
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    border: 1px solid rgba(32, 62, 54, 0.10);
    margin-bottom: 1.2rem;
}
.eyebrow {
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #49685b;
    font-size: 0.80rem;
    margin-bottom: 0.35rem;
}
.hero-title {
    color: #153128;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.hero-copy {
    color: #304a41;
    font-size: 1rem;
    max-width: 58rem;
}
</style>
"""


def render_ml_tab(selected_df, metrics_df, portfolio_df, benchmark_df, benchmark_summary_df):
    models = available_models(selected_df)
    if not models:
        st.error("No walk-forward recommendation data found in `selected_stock.csv`.")
        st.stop()

    sidebar    = st.sidebar
    model_name = sidebar.selectbox("Model", models)
    months     = available_months_for_model(selected_df, model_name)
    default_month_index = len(months) - 1 if months else 0
    target_month = sidebar.selectbox(
        "Target Month",
        months,
        index=default_month_index,
        format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m"),
    )

    recommendations  = recommended_stocks_for_month(selected_df, model_name, target_month)
    metrics_row      = model_metrics_row(metrics_df, model_name)
    portfolio_history = portfolio_history_for_model(portfolio_df, model_name)
    benchmark_history = benchmark_history_for_model(benchmark_df, model_name)
    benchmark_summary = benchmark_summary_row(benchmark_summary_df, model_name)

    left_col, right_col = st.columns([1.5, 1], gap="large")

    with left_col:
        st.subheader("Recommended Top-Rank Stocks")
        st.caption(f"Model: {model_name} | Target month: {target_month.strftime('%Y-%m')}")
        if recommendations.empty:
            st.warning("No top-rank recommendations found for the selected month.")
        else:
            display_df = recommendations[["Ticker", "score", "rank"]].copy()
            display_df["score"] = display_df["score"].map(lambda x: f"{x:.4f}")
            display_df["rank"]  = display_df["rank"].map(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Historical Portfolio Performance")
        chart_df = build_performance_chart_data(portfolio_history, benchmark_history)
        if chart_df.empty:
            st.info("No portfolio history is available for the selected model.")
        else:
            st.line_chart(chart_df, use_container_width=True, height=360)

        if not portfolio_history.empty:
            monthly_view = portfolio_history[["YearMonth", "monthly_return", "picks"]].copy()
            monthly_view["YearMonth"]     = monthly_view["YearMonth"].dt.strftime("%Y-%m")
            monthly_view["monthly_return"] = monthly_view["monthly_return"].map(format_pct)
            st.subheader("Monthly Portfolio History")
            st.dataframe(monthly_view, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Evaluation Metrics")
        c1, c2 = st.columns(2)
        c1.metric("ROC AUC",   format_float(metrics_row.get("roc_auc"), 3))
        c2.metric("F1 Score",  format_float(metrics_row.get("f1"), 3))

        c3, c4 = st.columns(2)
        c3.metric("Cumulative Return", format_pct(metrics_row.get("cumulative_return")))
        c4.metric("Sharpe Ratio",      format_float(metrics_row.get("annualized_sharpe"), 2))

        c5, c6 = st.columns(2)
        c5.metric("Avg Monthly Return", format_pct(metrics_row.get("avg_monthly_return")))
        c6.metric("Win Rate",           format_pct(metrics_row.get("monthly_win_rate")))

        summary_table = pd.DataFrame(
            [
                ("Accuracy",              format_pct(metrics_row.get("accuracy"))),
                ("Balanced Accuracy",     format_pct(metrics_row.get("balanced_accuracy"))),
                ("Precision",             format_pct(metrics_row.get("precision"))),
                ("Recall",               format_pct(metrics_row.get("recall"))),
                ("Top20 Avg Return",      format_pct(metrics_row.get("top20_avg_return"))),
                ("Best Month",            format_pct(metrics_row.get("best_month_return"))),
                ("Worst Month",           format_pct(metrics_row.get("worst_month_return"))),
                ("Average Picks / Month", format_float(metrics_row.get("avg_monthly_picks"), 1)),
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        st.subheader("Classification Report")
        st.code(metrics_row.get("classification_report", "N/A"))

        if benchmark_summary is not None:
            st.subheader("Benchmark Comparison")
            benchmark_label = benchmark_summary.get("benchmark_symbol", "Benchmark")
            benchmark_table = pd.DataFrame(
                [
                    (f"{benchmark_label} cumulative return",      format_pct(benchmark_summary.get("benchmark_cumulative_return"))),
                    ("Portfolio excess return",                    format_pct(benchmark_summary.get("excess_cumulative_return"))),
                    (f"Average monthly {benchmark_label} return", format_pct(benchmark_summary.get("avg_monthly_benchmark_return"))),
                    ("Average monthly excess return",             format_pct(benchmark_summary.get("avg_monthly_excess_return"))),
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(benchmark_table, use_container_width=True, hide_index=True)


def render_optimizertab(opt_df, benchmark_df):
    if opt_df.empty:
        st.warning(
            "If no optimizer results found, run `python optimizer.py` first to generate them."
        )
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optimizer Controls")

    models  = sorted(opt_df["model_name"].unique())
    lambdas = sorted(opt_df["lambda"].unique())

    opt_model  = st.sidebar.selectbox("Optimizer Model",  models,  key="opt_model")
    opt_lambda = st.sidebar.selectbox("Lambda",       lambdas, key="opt_lambda")

    filtered = opt_df[
        (opt_df["model_name"] == opt_model) &
        (opt_df["lambda"]     == opt_lambda)
    ]

    m = compute_optimizer_metrics(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative Return",  format_pct(m.get("cumulative_return")))
    c2.metric("Sharpe Ratio",       format_float(m.get("sharpe"), 2))
    c3.metric("Avg Monthly Return", format_pct(m.get("avg_monthly_return")))
    c4.metric("Win Rate",           format_pct(m.get("win_rate")))

    c5, c6 = st.columns(2)
    c5.metric("Best Month",  format_pct(m.get("best_month")))
    c6.metric("Worst Month", format_pct(m.get("worst_month")))

    st.markdown("---")
    left_col, right_col = st.columns([1.5, 1], gap="large")

    with left_col:
        
        st.subheader("Cumulative Portfolio Return vs SPY")
        chart_data = build_optimizer_chart(filtered, benchmark_df)
        st.line_chart(chart_data, use_container_width=True, height=320)

       
        if not benchmark_df.empty and "benchmark_cumulative_return" in benchmark_df.columns:
            spy_cum = benchmark_df["benchmark_cumulative_return"].dropna().iloc[-1] if not benchmark_df["benchmark_cumulative_return"].dropna().empty else None
            opt_cum = m.get("cumulative_return")
            excess  = (opt_cum - spy_cum) if (opt_cum is not None and spy_cum is not None) else None
            st.subheader("Benchmark Comparison")
            bm_table = pd.DataFrame(
                [
                    ("SPY Cumulative Return",       format_pct(spy_cum)),
                    ("Optimizer Cumulative Return", format_pct(opt_cum)),
                    ("Return vs SPY",        format_pct(excess)),
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(bm_table, use_container_width=True, hide_index=True)

        
        st.subheader("Monthly Portfolio History")
        monthly = (
            filtered.groupby("YearMonth")["weighted_return"]
            .sum()
            .sort_index()
            .reset_index()
        )
        monthly["YearMonth"]      = monthly["YearMonth"].dt.strftime("%Y-%m")
        monthly["weighted_return"] = monthly["weighted_return"].map(format_pct)
        monthly.columns            = ["Month", "Portfolio Return"]
        st.dataframe(monthly, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Lambda Sweep — Risk-Return Tradeoff")
        st.caption(f"Model: {opt_model} | All lambda values compared")

        rows = []
        for lam in lambdas:
            lam_df = opt_df[
                (opt_df["model_name"] == opt_model) &
                (opt_df["lambda"]     == lam)
            ]
            lm = compute_optimizer_metrics(lam_df)
            rows.append({
                "λ":               lam,
                "Cum Return":      format_pct(lm.get("cumulative_return")),
                "Sharpe":          format_float(lm.get("sharpe"), 2),
                "Avg Monthly":     format_pct(lm.get("avg_monthly_return")),
                "Win Rate":        format_pct(lm.get("win_rate")),
                "Monthly Std Dev": format_pct(lm.get("monthly_std")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Top Weighted Stocks (Latest Month)")
        latest_month = filtered["YearMonth"].max()
        latest_df    = filtered[filtered["YearMonth"] == latest_month].copy()
        latest_df    = latest_df.nlargest(10, "weight")[["Ticker", "score", "risk_vol_3m", "weight"]]
        latest_df["score"]       = latest_df["score"].map(lambda x: f"{x:.4f}")
        latest_df["risk_vol_3m"] = latest_df["risk_vol_3m"].map(lambda x: f"{x:.4f}")
        latest_df["weight"]      = latest_df["weight"].map(format_pct)
        st.dataframe(latest_df, use_container_width=True, hide_index=True)

       
        st.subheader(f"All Models at λ = {opt_lambda}")
        model_rows = []
        for model in models:
            model_lam_df = opt_df[
                (opt_df["model_name"] == model) &
                (opt_df["lambda"]     == opt_lambda)
            ]
            mm = compute_optimizer_metrics(model_lam_df)
            model_rows.append({
                "Model":       model,
                "Cum Return":  format_pct(mm.get("cumulative_return")),
                "Sharpe":      format_float(mm.get("sharpe"), 2),
                "Win Rate":    format_pct(mm.get("win_rate")),
            })
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)



def main():
    missing_artifacts = validate_demo_artifacts()
    if missing_artifacts:
        st.error("Required demo artifacts are missing. Run `python main.py` first.")
        st.code("\n".join(missing_artifacts))
        st.stop()

    selected_df, metrics_df, portfolio_df, benchmark_df, benchmark_summary_df = load_demo_data()
    opt_df = load_optimizer_results()

    st.markdown(HERO_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-shell">
            <div class="eyebrow">Walk-Forward Stock Selection Demo</div>
            <div class="hero-title">Model-driven monthly stock recommendations</div>
            <div class="hero-copy">
                Select a walk-forward model and target month to inspect the top-ranked stock picks,
                historical portfolio behavior, and evaluation metrics produced by the offline pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Controls")

    tab1, tab2 = st.tabs(["📈 ML Stock Selection", "⚖️ Portfolio Optimization"])

    with tab1:
        render_ml_tab(
            selected_df, metrics_df, portfolio_df, benchmark_df, benchmark_summary_df
        )

    with tab2:
        render_optimizertab(opt_df, benchmark_df)


if __name__ == "__main__":
    main()
