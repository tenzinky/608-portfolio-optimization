from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"
VALIDATION_DIR = BASE_DIR / "artifacts" / "validation"

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

ML_FILE = DATA_DIR / "sp100_monthly_ml_dataset.csv"
MODEL_METRICS_FILE = METRICS_DIR / "model_metrics_summary.csv"
WALK_FORWARD_METRICS_FILE = METRICS_DIR / "walk_forward_metrics_summary.csv"
PORTFOLIO_TS_FILE = METRICS_DIR / "portfolio_time_series.csv"
SPY_SUMMARY_FILE = METRICS_DIR / "portfolio_vs_spy_summary.csv"


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def num(x, digits=4):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def main():
    lines = []

    lines.append("=" * 80)
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 80)

    # ------------------------------------------------------------------
    # 1. Data validation
    # ------------------------------------------------------------------
    if ML_FILE.exists():
        ml_df = pd.read_csv(ML_FILE)
        lines.append("\n[1] Data Validation")
        lines.append(f"ML dataset file: {ML_FILE}")
        lines.append(f"Dataset shape: {ml_df.shape[0]} rows, {ml_df.shape[1]} columns")

        if "label_top20" in ml_df.columns:
            pos_rate = ml_df["label_top20"].mean()
            lines.append(f"Overall positive label rate: {pct(pos_rate)}")

        if "YearMonth" in ml_df.columns:
            months = pd.to_datetime(ml_df["YearMonth"])
            lines.append(f"Date range: {months.min().strftime('%Y-%m')} to {months.max().strftime('%Y-%m')}")
            lines.append(f"Number of unique months: {months.nunique()}")

        if "Ticker" in ml_df.columns:
            lines.append(f"Number of unique tickers: {ml_df['Ticker'].nunique()}")
    else:
        lines.append("\n[1] Data Validation")
        lines.append(f"ML dataset not found: {ML_FILE}")

    # ------------------------------------------------------------------
    # 2. Standard + walk-forward metrics
    # ------------------------------------------------------------------
    if MODEL_METRICS_FILE.exists():
        metrics_df = pd.read_csv(MODEL_METRICS_FILE)

        keep_cols = [
            "evaluation_name",
            "model_name",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "top20_avg_return",
            "overall_avg_return",
            "avg_monthly_return",
            "monthly_win_rate",
            "cumulative_return",
            "annualized_sharpe",
            "avg_monthly_picks",
        ]
        keep_cols = [c for c in keep_cols if c in metrics_df.columns]

        summary_df = metrics_df[keep_cols].copy()

        if {"top20_avg_return", "overall_avg_return"}.issubset(summary_df.columns):
            summary_df["return_lift"] = summary_df["top20_avg_return"] - summary_df["overall_avg_return"]

        summary_df.to_csv(VALIDATION_DIR / "validation_model_metrics_summary.csv", index=False)

        lines.append("\n[2] Model Validation Summary")
        lines.append(summary_df.to_string(index=False))
    else:
        lines.append("\n[2] Model Validation Summary")
        lines.append(f"Model metrics file not found: {MODEL_METRICS_FILE}")

    # ------------------------------------------------------------------
    # 3. Walk-forward metrics only
    # ------------------------------------------------------------------
    if WALK_FORWARD_METRICS_FILE.exists():
        wf_df = pd.read_csv(WALK_FORWARD_METRICS_FILE)

        wf_keep = [
            "model_name",
            "num_folds",
            "test_months",
            "threshold_strategy",
            "avg_decision_threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "actual_positive_rate",
            "predicted_positive_rate",
            "top20_avg_return",
            "overall_avg_return",
            "avg_monthly_return",
            "monthly_win_rate",
            "cumulative_return",
            "annualized_sharpe",
            "best_month_return",
            "worst_month_return",
            "avg_monthly_picks",
        ]
        wf_keep = [c for c in wf_keep if c in wf_df.columns]

        wf_summary = wf_df[wf_keep].copy()

        if {"top20_avg_return", "overall_avg_return"}.issubset(wf_summary.columns):
            wf_summary["return_lift"] = wf_summary["top20_avg_return"] - wf_summary["overall_avg_return"]

        wf_summary.to_csv(VALIDATION_DIR / "validation_walk_forward_summary.csv", index=False)

        lines.append("\n[3] Walk-Forward Validation Summary")
        lines.append(wf_summary.to_string(index=False))
    else:
        lines.append("\n[3] Walk-Forward Validation Summary")
        lines.append(f"Walk-forward metrics file not found: {WALK_FORWARD_METRICS_FILE}")

    # ------------------------------------------------------------------
    # 4. Portfolio time series check
    # ------------------------------------------------------------------
    if PORTFOLIO_TS_FILE.exists():
        port_df = pd.read_csv(PORTFOLIO_TS_FILE)
        lines.append("\n[4] Portfolio Time Series Check")
        lines.append(f"Portfolio time series rows: {len(port_df)}")

        if "evaluation_name" in port_df.columns:
            wf_port = port_df[port_df["evaluation_name"] == "walk_forward"].copy()
        else:
            wf_port = port_df.copy()

        if not wf_port.empty:
            grouped = (
                wf_port.groupby("model_name")
                .agg(
                    months=("YearMonth", "nunique"),
                    avg_monthly_return=("monthly_return", "mean"),
                    final_cumulative_return=("cumulative_return", "last"),
                    avg_picks=("picks", "mean"),
                )
                .reset_index()
            )
            grouped.to_csv(VALIDATION_DIR / "validation_portfolio_time_series_check.csv", index=False)
            lines.append(grouped.to_string(index=False))
    else:
        lines.append("\n[4] Portfolio Time Series Check")
        lines.append(f"Portfolio time series file not found: {PORTFOLIO_TS_FILE}")

    # ------------------------------------------------------------------
    # 5. SPY benchmark summary
    # ------------------------------------------------------------------
    if SPY_SUMMARY_FILE.exists():
        spy_df = pd.read_csv(SPY_SUMMARY_FILE)
        spy_df.to_csv(VALIDATION_DIR / "validation_spy_benchmark_summary.csv", index=False)

        lines.append("\n[5] SPY Benchmark Summary")
        lines.append(spy_df.to_string(index=False))
    else:
        lines.append("\n[5] SPY Benchmark Summary")
        lines.append(f"SPY benchmark summary file not found: {SPY_SUMMARY_FILE}")

    # ------------------------------------------------------------------
    # Save text summary
    # ------------------------------------------------------------------
    output_path = VALIDATION_DIR / "validation_summary.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nSaved validation outputs to:")
    print(VALIDATION_DIR)


if __name__ == "__main__":
    main()