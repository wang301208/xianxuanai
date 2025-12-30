import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

def load_latest_reports(reports_root: Path) -> Dict[str, Dict[str, float]]:
    """Load success percentage from latest report of each algorithm.

    Returns a nested dict mapping problem -> {algorithm: success_%}.
    """
    problem_data: Dict[str, Dict[str, float]] = {}
    for algo_dir in reports_root.iterdir():
        if not algo_dir.is_dir():
            continue
        runs = [p for p in algo_dir.iterdir() if p.is_dir()]
        if not runs:
            continue
        latest = max(runs)
        report_file = latest / "report.json"
        if not report_file.exists():
            continue
        with open(report_file) as f:
            report = json.load(f)
        tests = report.get("tests", {})
        for prob, info in tests.items():
            metrics = info.get("metrics", {})
            success = metrics.get("success_%")
            if success is None:
                continue
            problem_data.setdefault(prob, {})[algo_dir.name] = success
    return problem_data


def build_dataframe(problem_data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Build a DataFrame from the nested dictionary."""
    return pd.DataFrame(problem_data).T.sort_index()

def wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Wilcoxon signed-rank tests and effect sizes.

    Returns a DataFrame with columns: algo1, algo2, p_value, effect_size.
    """
    results: List[Tuple[str, str, float, float]] = []
    algos = df.columns
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            x = df[a]
            y = df[b]
            mask = x.notna() & y.notna()
            x_clean = x[mask]
            y_clean = y[mask]
            if len(x_clean) == 0:
                continue
            if np.allclose(x_clean, y_clean):
                p = 1.0
                effect = 0.0
            else:
                stat, p = stats.wilcoxon(x_clean, y_clean, zero_method="wilcox", alternative="two-sided")
                n = len(x_clean)
                mean_rank = n * (n + 1) / 4
                sd_rank = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                if sd_rank == 0:
                    effect = 0.0
                else:
                    z = (stat - mean_rank) / sd_rank
                    effect = z / np.sqrt(n)
            results.append((a, b, p, abs(effect)))
    return pd.DataFrame(results, columns=["algo1", "algo2", "p_value", "effect_size"])



def friedman_nemenyi(df: pd.DataFrame) -> Tuple[float, pd.DataFrame, float]:
    """Run Friedman test and Nemenyi post-hoc.

    Returns chi2, Nemenyi DataFrame (algo1, algo2, p_value, effect_size) and Kendall's W.
    """
    clean_df = df.dropna(axis=0, how="any")
    if clean_df.empty:
        return np.nan, pd.DataFrame(columns=["algo1", "algo2", "p_value", "effect_size"]), np.nan
    algos = clean_df.columns
    samples = [clean_df[col].values for col in algos]
    chi2, p = stats.friedmanchisquare(*samples)
    n = len(clean_df)
    k = len(algos)
    kendall_w = chi2 / (n * (k - 1))
    nemenyi = sp.posthoc_nemenyi_friedman(clean_df)
    records: List[Tuple[str, str, float, float]] = []
    ranks = clean_df.rank(axis=1)
    mean_ranks = ranks.mean(axis=0)
    se = np.sqrt(k * (k + 1) / (6 * n))
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            p_val = nemenyi.loc[a, b]
            effect = abs(mean_ranks[a] - mean_ranks[b]) / se
            records.append((a, b, p_val, effect))
    nemenyi_df = pd.DataFrame(records, columns=["algo1", "algo2", "p_value", "effect_size"])
    return chi2, nemenyi_df, kendall_w

def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate combined summary of Wilcoxon and Nemenyi tests."""
    wilcox = wilcoxon_tests(df)
    chi2, nemenyi_df, kendall_w = friedman_nemenyi(df)
    wilcox["test"] = "wilcoxon"
    nemenyi_df["test"] = "nemenyi"
    summary = pd.concat([wilcox, nemenyi_df], ignore_index=True)
    summary["kendall_w"] = kendall_w
    return summary


if __name__ == "__main__":
    reports_path = Path("benchmark/reports")
    data = load_latest_reports(reports_path)
    df = build_dataframe(data)
    summary = generate_summary(df)
    out_path = Path("analysis/stat_summary.csv")
    summary.to_csv(out_path, index=False)
    print(summary)
    print(f"Saved summary to {out_path}")
