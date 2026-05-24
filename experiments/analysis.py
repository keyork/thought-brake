"""Analyse experiment results and produce tables + figures.

Usage:
    uv run python experiments/analysis.py --input experiments/results/riddles.jsonl
    uv run python experiments/analysis.py --input experiments/results/ --output report/
"""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != script_dir]
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.config import DEFAULT_ENCODING, REPORT_DIR, RESULTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> pd.DataFrame:
    if path.is_dir():
        dfs = [
            pd.read_json(p, encoding=DEFAULT_ENCODING, lines=True)
            for p in sorted(path.glob("*.jsonl"))
        ]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_json(path, encoding=DEFAULT_ENCODING, lines=True)

    if "schema_version" in df.columns and df["schema_version"].notna().any():
        latest_schema = int(df["schema_version"].dropna().max())
        df = df[df["schema_version"] == latest_schema].copy()
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, budget) aggregate: savings_rate and quality_score."""
    if "detector" not in df.columns:
        df = df.copy()
        df["detector"] = "budget"

    cols = ["question_id", "category", "reasoning_chars", "quality_score"]
    baseline = df[df["budget"] == 0][cols]
    baseline = (
        baseline.groupby(["question_id", "category"], as_index=False)
        .agg(
            baseline_chars=("reasoning_chars", "mean"),
            baseline_quality=("quality_score", "mean"),
        )
    )

    treated = df[df["budget"] > 0].copy()
    merged = treated.merge(baseline, on=["question_id", "category"], how="left")

    merged["savings_rate"] = (
        (merged["baseline_chars"] - merged["reasoning_chars"])
        / merged["baseline_chars"].clip(lower=1)
    ).clip(lower=0, upper=1)

    merged["quality_retention"] = (
        merged["quality_score"] / merged["baseline_quality"].clip(lower=1e-6)
    ).clip(upper=1.0)

    summary = (
        merged.groupby(["category", "difficulty", "detector", "budget"])
        .agg(
            n=("question_id", "count"),
            truncation_rate=("truncated", "mean"),
            avg_savings_rate=("savings_rate", "mean"),
            avg_reasoning_chars=("reasoning_chars", "mean"),
            avg_baseline_chars=("baseline_chars", "mean"),
            avg_quality_score=("quality_score", "mean"),
            avg_quality_retention=("quality_retention", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def make_markdown_table(summary: pd.DataFrame) -> str:
    cols = [
        "category", "difficulty", "detector", "budget",
        "n", "truncation_rate",
        "avg_savings_rate", "avg_quality_score", "avg_quality_retention",
        "avg_latency_ms",
    ]
    sub = summary[cols].copy()
    sub.columns = [
        "Category", "Difficulty", "Detector", "Budget",
        "N", "Trunc%",
        "TokenSave%", "Quality", "QualityRetention",
        "Latency(ms)",
    ]
    sub["Trunc%"] = (sub["Trunc%"] * 100).round(1).astype(str) + "%"
    sub["TokenSave%"] = (sub["TokenSave%"] * 100).round(1).astype(str) + "%"
    sub["QualityRetention"] = (sub["QualityRetention"] * 100).round(1).astype(str) + "%"
    return sub.to_markdown(index=False)


def plot_pareto(summary: pd.DataFrame, output_dir: Path) -> None:
    """Token savings vs quality retention per budget, one line per (category, difficulty)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    group_cols = ["category", "difficulty", "detector"] if "detector" in summary.columns else [
        "category",
        "difficulty",
    ]
    for key, grp in summary.groupby(group_cols):
        cat = key[0]
        diff = key[1]
        detector = key[2] if len(key) > 2 else "budget"
        grp = grp.sort_values("budget")
        ax.plot(
            grp["avg_savings_rate"] * 100,
            grp["avg_quality_retention"] * 100,
            marker="o",
            label=f"{cat}/{diff}/{detector}",
        )
        for _, row in grp.iterrows():
            ax.annotate(
                str(int(row["budget"])),
                (row["avg_savings_rate"] * 100, row["avg_quality_retention"] * 100),
                fontsize=7,
                textcoords="offset points",
                xytext=(4, 4),
            )

    ax.axhline(95, color="gray", linestyle="--", linewidth=0.8, label="95% quality threshold")
    ax.set_xlabel("Token Savings Rate (%)")
    ax.set_ylabel("Quality Retention (%)")
    ax.set_title("Early Stopping: Token Savings vs Quality Retention")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pareto.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'pareto.png'}")


def plot_reasoning_chars(summary: pd.DataFrame, output_dir: Path) -> None:
    """Average reasoning chars by budget and category."""
    cats = summary["category"].unique()
    fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), 4), squeeze=False)

    for ax, cat in zip(axes[0], cats):
        sub = summary[summary["category"] == cat]
        group_cols = ["difficulty", "detector"] if "detector" in sub.columns else ["difficulty"]
        for key, grp in sub.groupby(group_cols):
            label = "/".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
            ax.plot(grp["budget"], grp["avg_reasoning_chars"], marker="o", label=label)
        ax.set_title(cat)
        ax.set_xlabel("soft_budget (chars)")
        ax.set_ylabel("Avg reasoning chars")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "reasoning_chars.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'reasoning_chars.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyse thought-brake experiment results")
    p.add_argument("--input", default=str(RESULTS_DIR), help="JSONL file or directory")
    p.add_argument("--output", default=str(REPORT_DIR), help="Output directory for report")
    args = p.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    df = load_results(input_path)
    print(f"Loaded {len(df)} records from {input_path}")

    # Drop unevaluated records
    df = df[df["quality_score"] >= 0]
    if df.empty:
        print("No evaluated records found (quality_score == -1). Run with evaluation enabled.")
        return

    summary = compute_metrics(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", encoding=DEFAULT_ENCODING, index=False)
    print(f"Saved {output_dir / 'summary.csv'}")

    md = make_markdown_table(summary)
    (output_dir / "table.md").write_text(md, encoding=DEFAULT_ENCODING)
    print(f"Saved {output_dir / 'table.md'}")
    print("\n" + md)

    plot_pareto(summary, output_dir)
    plot_reasoning_chars(summary, output_dir)


if __name__ == "__main__":
    main()
