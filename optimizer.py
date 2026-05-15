
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE  = "artifacts/metrics/optimizer_input.csv"
OUTPUT_FILE = "artifacts/metrics/optimizer_results.csv"

# lambda sweep: shows risk-return tradeoff across different penalty strengths
# low lambda  = aggressive (chases returns, ignores risk)
# high lambda = conservative (avoids risk, lower returns)
LAMBDA_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]

# max weight any single stock can receive
MAX_WEIGHT = 0.10

# risk column used to build the diagonal variance matrix
RISK_COLUMN = "risk_vol_3m"

def build_diagonal_sigma(risks: np.ndarray) -> np.ndarray:
    risks_clipped = np.maximum(risks, 1e-8)
    return np.diag(risks_clipped ** 2 + 1e-8)

#optimizer for one month
def optimize_month(scores, sigma, lam, max_weight=MAX_WEIGHT):
    n = len(scores)
    w = cp.Variable(n)

    portfolio_return = scores @ w
    portfolio_risk   = cp.quad_form(w, sigma)

    objective = cp.Minimize(-portfolio_return + lam * portfolio_risk)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None
    return w.value


def run_one(df, model, months, lam):
    model_df = df[df["model_name"] == model]
    results  = []
    solved = skipped = infeasible = 0

    for month in months:
        month_df = model_df[model_df["YearMonth"] == month].copy()
        month_df = month_df.dropna(subset=["score", RISK_COLUMN])

        if len(month_df) * MAX_WEIGHT < 1:
            infeasible += 1
            skipped    += 1
            continue

        if len(month_df) < 2:
            skipped += 1
            continue

        s     = month_df["score"].values
        r     = month_df[RISK_COLUMN].values
        sigma = build_diagonal_sigma(r)

        weights = optimize_month(s, sigma, lam)

        if weights is None:
            skipped += 1
            continue

        month_df               = month_df.copy()
        month_df["weight"]     = weights
        month_df["lambda"]     = lam
        month_df["risk_col"]   = RISK_COLUMN
        month_df["max_weight"] = MAX_WEIGHT
        results.append(month_df)
        solved += 1

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

#Sharpe=(mu_p-rf)/sigma_p where rf is assumed to be 0 for simpilified backtest
def compute_metrics(df):
    if "weighted_return" not in df.columns or df.empty:
        return {}
    ret    = df.groupby("YearMonth")["weighted_return"].sum()
    cum    = (1 + ret).prod() - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0
    return {
        "Avg Monthly Return": f"{ret.mean():.2%}",
        "Cumulative Return":  f"{cum:.2%}",
        "Sharpe Ratio":       f"{sharpe:.2f}",
        "Win Rate":           f"{(ret > 0).mean():.1%}",
        "Monthly Std Dev":    f"{ret.std():.2%}",
    }


def run_optimizer(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df           = pd.read_csv(input_file)
    models       = df["model_name"].unique()
    months       = sorted(df["YearMonth"].unique())
    all_results  = []
    summary_rows = []

    for model in models:
        for lam in LAMBDA_VALUES:
            result = run_one(df, model, months, lam)
            if result.empty:
                continue

            if "future_return_1m" in result.columns:
                result["weighted_return"] = (
                    result["weight"] * result["future_return_1m"]
                )

            all_results.append(result)
            metrics = compute_metrics(result)
            summary_rows.append({
                "Model":  model,
                "Lambda": lam,
                **metrics,
            })


    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    if not summary_rows:
        return

    summary_df = pd.DataFrame(summary_rows)

    for model, grp in summary_df.groupby("Model"):
        print(f"\n  {model}")
        print(grp[["Lambda", "Avg Monthly Return", "Cumulative Return",
                    "Sharpe Ratio", "Win Rate", "Monthly Std Dev"]]
              .to_string(index=False))

    print(" Summary of all combinations:")
    print(summary_df.to_string(index=False))

    summary_df["_cum_float"] = (
        summary_df["Cumulative Return"]
        .str.replace("%", "").astype(float)
    )
    best = summary_df.loc[summary_df["_cum_float"].idxmax()]

    print("Best combination based on cumulative return:")
    print(f"  Model          : {best['Model']}")
    print(f"  Lambda         : {best['Lambda']}")
    print(f"  Risk Column    : {RISK_COLUMN}")
    print(f"  Cumul Return     : {best['Cumulative Return']}")
    print(f"  Sharpe Ratio   : {best['Sharpe Ratio']}")
    print(f"  Avg Monthly    : {best['Avg Monthly Return']}")
    print(f"  Win Rate       : {best['Win Rate']}")
    print(f"  Monthly Std Dev: {best['Monthly Std Dev']}")
    print()

if __name__ == "__main__":
    run_optimizer()
