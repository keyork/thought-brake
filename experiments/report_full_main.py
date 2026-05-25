"""Focused report for the main full-run experiment.

This report intentionally reads only the three full-run files. It avoids the
generic directory-level analysis because older JSONL files in experiments/results
can pollute the aggregate.
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

from experiments.config import DEFAULT_ENCODING  # noqa: E402

DEFAULT_INPUTS = [
    Path("experiments/results/full_budget.jsonl"),
    Path("experiments/results/full_compression.jsonl"),
    Path("experiments/results/full_keyword.jsonl"),
]
DEFAULT_OUTPUT = Path("experiments/report/full_main_token")

DETECTOR_COLORS = {
    "budget": "#4C78A8",
    "compression": "#F58518",
    "keyword": "#54A24B",
}
OVERALL_QUALITY_TOLERANCE = 0.01
DATASET_QUALITY_TOLERANCE = 0.03


def _load_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        detector = path.stem.replace("full_", "")
        frame = pd.read_json(path, encoding=DEFAULT_ENCODING, lines=True)
        frame["run_detector"] = detector
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["quality_score"] >= 0].copy()
    df["dataset"] = df["category"].where(df["category"].isin(["riddle", "math"]), "mmlu")
    df["config"] = df["run_detector"] + "@" + df["budget"].astype(str)
    return df


def _with_baseline(df: pd.DataFrame) -> pd.DataFrame:
    if "total_tokens" not in df.columns or "estimated_total_tokens" not in df.columns:
        raise ValueError("Results must include total_tokens and estimated_total_tokens")
    df = df.copy()
    df["cost_tokens"] = df["total_tokens"].combine_first(df["estimated_total_tokens"])
    df["cost_token_metric"] = "estimated_total_tokens"
    df.loc[df["total_tokens"].notna(), "cost_token_metric"] = "total_tokens"

    baseline = df[df["budget"] == 0][
        [
            "run_detector",
            "question_id",
            "category",
            "dataset",
            "reasoning_chars",
            "cost_tokens",
            "quality_score",
            "latency_ms",
        ]
    ].rename(
        columns={
            "reasoning_chars": "baseline_chars",
            "cost_tokens": "baseline_cost_tokens",
            "quality_score": "baseline_quality",
            "latency_ms": "baseline_latency_ms",
        }
    )

    treated = df[df["budget"] > 0].copy()
    merged = treated.merge(
        baseline,
        on=["run_detector", "question_id", "category", "dataset"],
        how="left",
    )
    merged["savings_rate"] = (
        (merged["baseline_chars"] - merged["reasoning_chars"])
        / merged["baseline_chars"].clip(lower=1)
    ).clip(lower=0, upper=1)
    merged["cost_savings_rate"] = (
        (merged["baseline_cost_tokens"] - merged["cost_tokens"])
        / merged["baseline_cost_tokens"].clip(lower=1)
    ).clip(lower=0, upper=1)
    merged["quality_retention"] = (
        merged["quality_score"] / merged["baseline_quality"].clip(lower=1e-6)
    ).clip(upper=1.0)
    merged["lost_baseline_correct"] = (
        (merged["baseline_quality"] >= 1.0) & (merged["quality_score"] < 1.0)
    )
    merged["fixed_baseline_wrong"] = (
        (merged["baseline_quality"] < 1.0) & (merged["quality_score"] >= 1.0)
    )
    return merged


def _summary(treated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_summary = (
        treated.groupby(["dataset", "run_detector", "budget", "config"])
        .agg(
            n=("question_id", "count"),
            quality=("quality_score", "mean"),
            baseline_quality=("baseline_quality", "mean"),
            retention=("quality_retention", "mean"),
            savings=("savings_rate", "mean"),
            reasoning_chars=("reasoning_chars", "mean"),
            baseline_chars=("baseline_chars", "mean"),
            cost_tokens=("cost_tokens", "mean"),
            baseline_cost_tokens=("baseline_cost_tokens", "mean"),
            cost_savings=("cost_savings_rate", "mean"),
            truncation_rate=("truncated", "mean"),
            phase2_fail_rate=("phase2_failed", "mean"),
            lost_baseline_correct=("lost_baseline_correct", "sum"),
            fixed_baseline_wrong=("fixed_baseline_wrong", "sum"),
            latency_ms=("latency_ms", "mean"),
            baseline_latency_ms=("baseline_latency_ms", "mean"),
        )
        .reset_index()
    )
    overall = (
        treated.groupby(["run_detector", "budget", "config"])
        .agg(
            n=("question_id", "count"),
            quality=("quality_score", "mean"),
            baseline_quality=("baseline_quality", "mean"),
            retention=("quality_retention", "mean"),
            savings=("savings_rate", "mean"),
            reasoning_chars=("reasoning_chars", "mean"),
            baseline_chars=("baseline_chars", "mean"),
            cost_tokens=("cost_tokens", "mean"),
            baseline_cost_tokens=("baseline_cost_tokens", "mean"),
            cost_savings=("cost_savings_rate", "mean"),
            truncation_rate=("truncated", "mean"),
            phase2_fail_rate=("phase2_failed", "mean"),
            lost_baseline_correct=("lost_baseline_correct", "sum"),
            fixed_baseline_wrong=("fixed_baseline_wrong", "sum"),
            latency_ms=("latency_ms", "mean"),
            baseline_latency_ms=("baseline_latency_ms", "mean"),
        )
        .reset_index()
    )
    return dataset_summary, overall


def _select_best_tradeoff(summary: pd.DataFrame, tolerance: float) -> pd.Series:
    """Pick the highest-savings config close to the best observed quality."""
    quality_floor = summary["quality"].max() - tolerance
    candidates = summary[summary["quality"] >= quality_floor]
    return candidates.sort_values(["cost_savings", "quality"], ascending=False).iloc[0]


def _dataset_recommendations(dataset_summary: pd.DataFrame) -> dict[str, str]:
    recommendations = {}
    for dataset, group in dataset_summary.groupby("dataset"):
        recommendations[dataset] = _select_best_tradeoff(group, DATASET_QUALITY_TOLERANCE)[
            "config"
        ]
    return recommendations


def _plot_overall_tradeoff(overall: pd.DataFrame, output: Path, recommended_config: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.axhspan(90, 100, color="#E8F2E8", alpha=0.8, zorder=0)
    ax.axvspan(25, 100, color="#E8F2E8", alpha=0.35, zorder=0)
    ax.axhline(90, color="#6B7280", linestyle="--", linewidth=1)
    ax.axvline(25, color="#6B7280", linestyle="--", linewidth=1)
    recommended_row = overall[overall["config"] == recommended_config].iloc[0]

    label_offsets = {
        "keyword@1000": (-28, 8),
        "compression@1000": (-30, -10),
        "budget@1000": (8, 6),
        "compression@300": (8, 6),
        "budget@300": (8, 6),
        "keyword@300": (8, 6),
    }
    for _, row in overall.iterrows():
        detector = row["run_detector"]
        budget = int(row["budget"])
        marker = "o" if budget == 300 else "s"
        size = 260 if row["config"] == recommended_config else 150
        edge = "#111827" if row["config"] == recommended_config else "white"
        ax.scatter(
            row["cost_savings"] * 100,
            row["quality"] * 100,
            s=size,
            marker=marker,
            color=DETECTOR_COLORS[detector],
            edgecolor=edge,
            linewidth=1.8,
            zorder=3,
        )
        offset = label_offsets.get(row["config"], (7, 5))
        ax.annotate(
            row["config"],
            (row["cost_savings"] * 100, row["quality"] * 100),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )

    ax.annotate(
        f"Best default: {recommended_config}\nnear-best quality + higher savings",
        xy=(recommended_row["cost_savings"] * 100, recommended_row["quality"] * 100),
        xytext=(36, 94.8),
        arrowprops={"arrowstyle": "->", "color": "#111827", "linewidth": 1.3},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#111827"},
    )
    ax.set_xlim(0, 45)
    ax.set_ylim(84, 96)
    ax.set_xlabel("Total token savings (%)")
    ax.set_ylabel("Answer quality (%)")
    ax.set_title("Overall tradeoff: choose the upper-right point")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "overall_tradeoff.png", dpi=180)
    plt.close(fig)


def _plot_dataset_matrix(
    dataset_summary: pd.DataFrame, output: Path, recommendations: dict[str, str]
) -> None:
    configs = [
        "budget@300",
        "budget@1000",
        "compression@300",
        "compression@1000",
        "keyword@300",
        "keyword@1000",
    ]
    datasets = ["math", "mmlu", "riddle"]
    pivot = dataset_summary.pivot(index="dataset", columns="config", values="quality").reindex(
        index=datasets, columns=configs
    )
    savings = dataset_summary.pivot(index="dataset", columns="config", values="savings").reindex(
        index=datasets, columns=configs
    )
    cost_savings = dataset_summary.pivot(
        index="dataset", columns="config", values="cost_savings"
    ).reindex(index=datasets, columns=configs)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    image = ax.imshow(pivot.to_numpy() * 100, cmap="RdYlGn", vmin=80, vmax=100, aspect="auto")
    fig.colorbar(image, ax=ax, label="Quality (%)")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=35, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title("Dataset decision matrix: quality color, reasoning/cost savings in each cell")

    for i, dataset in enumerate(datasets):
        for j, config in enumerate(configs):
            quality = pivot.loc[dataset, config]
            saving = savings.loc[dataset, config]
            cost_saving = cost_savings.loc[dataset, config]
            text = f"Q {quality * 100:.0f}%\nR {saving * 100:.0f}%\nT {cost_saving * 100:.0f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="#111827")
            if recommendations.get(dataset) == config:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#111827",
                    linewidth=2.5,
                )
                ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(output / "dataset_decision_matrix.png", dpi=180)
    plt.close(fig)


def _plot_loss_vs_savings(overall: pd.DataFrame, output: Path) -> None:
    ordered = overall.sort_values(["budget", "run_detector"]).copy()
    labels = ordered["config"].tolist()
    x = range(len(ordered))

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    bars = ax1.bar(
        x,
        ordered["lost_baseline_correct"],
        color=[DETECTOR_COLORS[d] for d in ordered["run_detector"]],
        alpha=0.85,
    )
    ax1.set_ylabel("Baseline-correct answers lost (count)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title("Quality cost: compare lost correct answers against token savings")
    ax1.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, ordered["lost_baseline_correct"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = ax1.twinx()
    ax2.plot(
        list(x),
        ordered["cost_savings"] * 100,
        color="#111827",
        marker="D",
        linewidth=2,
        label="Savings",
    )
    ax2.set_ylabel("Total token savings (%)")
    ax2.set_ylim(0, 85)

    fig.tight_layout()
    fig.savefig(output / "loss_vs_savings.png", dpi=180)
    plt.close(fig)


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _make_report(
    dataset_summary: pd.DataFrame,
    overall: pd.DataFrame,
    treated: pd.DataFrame,
    output: Path,
    recommended_config: str,
    recommendations: dict[str, str],
) -> str:
    overall_table = overall[
        [
            "config",
            "n",
            "quality",
            "baseline_quality",
            "savings",
            "cost_savings",
            "truncation_rate",
            "lost_baseline_correct",
            "fixed_baseline_wrong",
            "phase2_fail_rate",
        ]
    ].copy()
    for col in [
        "quality",
        "baseline_quality",
        "savings",
        "cost_savings",
        "truncation_rate",
        "phase2_fail_rate",
    ]:
        overall_table[col] = overall_table[col].map(_format_pct)
    overall_table = overall_table.rename(
        columns={
            "config": "Config",
            "n": "N",
            "quality": "Quality",
            "baseline_quality": "Baseline",
            "savings": "Savings",
            "cost_savings": "TokenSavings",
            "truncation_rate": "Trunc",
            "lost_baseline_correct": "Lost",
            "fixed_baseline_wrong": "Fixed",
            "phase2_fail_rate": "Phase2Fail",
        }
    )

    dataset_table = dataset_summary[
        [
            "dataset",
            "config",
            "n",
            "quality",
            "savings",
            "cost_savings",
            "lost_baseline_correct",
        ]
    ].copy()
    dataset_table = dataset_table.sort_values(["dataset", "quality", "savings"], ascending=False)
    for col in ["quality", "savings", "cost_savings"]:
        dataset_table[col] = dataset_table[col].map(_format_pct)
    dataset_table = dataset_table.rename(
        columns={
            "dataset": "Dataset",
            "config": "Config",
            "n": "N",
            "quality": "Quality",
            "savings": "Savings",
            "cost_savings": "TokenSavings",
            "lost_baseline_correct": "Lost",
        }
    )

    recommendation_rows = []
    why = {
        "math": (
            "Keeps quality within the selected tolerance while giving materially better "
            "total-token savings than conservative 1000-token variants."
        ),
        "mmlu": "Best quality and strong total-token savings on the current run.",
        "riddle": (
            "Best observed quality on a small sample; treat this as conservative until "
            "the riddle set is larger."
        ),
    }
    for dataset, config in recommendations.items():
        row = dataset_summary[
            (dataset_summary["dataset"] == dataset) & (dataset_summary["config"] == config)
        ].iloc[0]
        recommendation_rows.append(
            {
                "Dataset": dataset,
                "Recommended": config,
                "Quality": _format_pct(row["quality"]),
                "Savings": _format_pct(row["savings"]),
                "TokenSavings": _format_pct(row["cost_savings"]),
                "Why": why.get(dataset, "Best current quality/savings tradeoff."),
            }
        )
    recommendation_table = pd.DataFrame(recommendation_rows)
    recommended_row = overall[overall["config"] == recommended_config].iloc[0]
    token_source = treated["cost_token_metric"].value_counts().to_dict()
    exact_count = int(token_source.get("total_tokens", 0))
    estimated_count = int(token_source.get("estimated_total_tokens", 0))
    total_count = exact_count + estimated_count

    report = f"""# Full Main Experiment Report

Input files:

- `experiments/results/full_budget.jsonl`
- `experiments/results/full_compression.jsonl`
- `experiments/results/full_keyword.jsonl`

This report is intentionally scoped to the three full-run files above. The older
generic `full_main` directory-level report can mix in previous experiment JSONL
files and should not be used for the final conclusion.

## Conclusion

Use `{recommended_config}` as the current default policy.

It is the best overall tradeoff on the current run: quality is
{_format_pct(recommended_row["quality"])} and total-token savings is
{_format_pct(recommended_row["cost_savings"])}. The report keeps both reasoning
savings and total-token savings: reasoning savings explains the mechanism, while
total-token savings is the cost metric.

For a first router:

{recommendation_table.to_markdown(index=False)}

## Visuals

1. `overall_tradeoff.png` answers: which global policy is the best default?
2. `dataset_decision_matrix.png` answers: which policy should each dataset route to?
3. `loss_vs_savings.png` answers: how much quality damage buys each savings level?

## Overall Metrics

{overall_table.to_markdown(index=False)}

`Lost` means baseline was correct but early-stop answer was not fully correct.
`Fixed` means baseline was wrong but the early-stop answer became correct.
`TokenSavings` uses API `total_tokens` when present and falls back to
`estimated_total_tokens` when streaming usage is unavailable.
In this run, {exact_count}/{total_count} treated rows used API `total_tokens`;
{estimated_count}/{total_count} treated rows used `estimated_total_tokens`.

## Dataset Metrics

{dataset_table.to_markdown(index=False)}
"""
    (output / "report.md").write_text(report, encoding=DEFAULT_ENCODING)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused report for full main experiment")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="Full-run JSONL files to analyze",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output report directory")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = _load_results([Path(p) for p in args.inputs])
    treated = _with_baseline(df)
    dataset_summary, overall = _summary(treated)
    recommended_config = _select_best_tradeoff(overall, OVERALL_QUALITY_TOLERANCE)["config"]
    recommendations = _dataset_recommendations(dataset_summary)

    dataset_summary.to_csv(output / "dataset_summary.csv", index=False, encoding=DEFAULT_ENCODING)
    overall.to_csv(output / "overall_summary.csv", index=False, encoding=DEFAULT_ENCODING)

    _plot_overall_tradeoff(overall, output, recommended_config)
    _plot_dataset_matrix(dataset_summary, output, recommendations)
    _plot_loss_vs_savings(overall, output)
    _make_report(dataset_summary, overall, treated, output, recommended_config, recommendations)

    print(f"Saved focused report to {output}")


if __name__ == "__main__":
    main()
