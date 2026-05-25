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
POLICY_COLORS = {
    "Default": "#111827",
    "Conservative": "#2F855A",
    "Balanced-aggressive": "#B45309",
    "Aggressive": "#B91C1C",
}
CONFIG_ORDER = [
    "budget@300",
    "budget@1000",
    "compression@300",
    "compression@1000",
    "keyword@300",
    "keyword@1000",
]
DATASETS = ["math", "mmlu", "riddle"]
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


def _policy_roles(overall: pd.DataFrame, recommended_config: str) -> dict[str, str]:
    roles = {
        "Default": recommended_config,
        "Conservative": overall.sort_values(
            ["quality", "cost_savings"], ascending=False
        ).iloc[0]["config"],
        "Aggressive": overall.sort_values(
            ["cost_savings", "quality"], ascending=False
        ).iloc[0]["config"],
    }
    recommended_quality = float(
        overall[overall["config"] == recommended_config].iloc[0]["quality"]
    )
    balanced_candidates = overall[
        (overall["quality"] >= recommended_quality - 0.02)
        & (overall["config"] != roles["Default"])
        & (overall["config"] != roles["Conservative"])
    ]
    if not balanced_candidates.empty:
        roles["Balanced-aggressive"] = balanced_candidates.sort_values(
            ["cost_savings", "quality"], ascending=False
        ).iloc[0]["config"]
    return roles


def _pareto_frontier(overall: pd.DataFrame) -> pd.DataFrame:
    points = []
    for _, row in overall.iterrows():
        dominated = False
        for _, other in overall.iterrows():
            if other["config"] == row["config"]:
                continue
            if (
                other["quality"] >= row["quality"]
                and other["cost_savings"] >= row["cost_savings"]
                and (
                    other["quality"] > row["quality"]
                    or other["cost_savings"] > row["cost_savings"]
                )
            ):
                dominated = True
                break
        if not dominated:
            points.append(row)
    return pd.DataFrame(points).sort_values("cost_savings")


def _plot_overall_tradeoff(
    overall: pd.DataFrame,
    output: Path,
    recommended_config: str,
    roles: dict[str, str],
) -> None:
    fig, (ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(14.4, 6.2),
        gridspec_kw={"width_ratios": [3.4, 1.9]},
    )
    max_quality = float(overall["quality"].max())
    quality_floor = (max_quality - OVERALL_QUALITY_TOLERANCE) * 100
    baseline_quality = float(overall["baseline_quality"].mean()) * 100
    ax.axhspan(
        quality_floor,
        100,
        color="#E8F2E8",
        alpha=0.7,
        zorder=0,
    )
    ax.axhline(
        quality_floor,
        color="#6B7280",
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(20, color="#6B7280", linestyle="--", linewidth=1)
    recommended_row = overall[overall["config"] == recommended_config].iloc[0]
    frontier = _pareto_frontier(overall)
    if len(frontier) > 1:
        ax.plot(
            frontier["cost_savings"] * 100,
            frontier["quality"] * 100,
            color="#111827",
            linewidth=1.4,
            alpha=0.65,
            zorder=2,
            label="Pareto frontier",
        )
    ax.scatter(
        0,
        baseline_quality,
        s=170,
        marker="*",
        color="#6B7280",
        edgecolor="#111827",
        linewidth=1,
        zorder=4,
        label="No-stop baseline",
    )
    ax.annotate(
        f"No-stop baseline\nQ {baseline_quality:.1f}% | T 0.0%",
        (0, baseline_quality),
        xytext=(10, -8),
        textcoords="offset points",
        fontsize=9,
        color="#374151",
    )

    label_offsets = {
        "keyword@1000": (-48, 8),
        "compression@1000": (-42, -18),
        "budget@1000": (8, -14),
        "compression@300": (9, 7),
        "budget@300": (8, 7),
        "keyword@300": (8, -18),
    }
    role_by_config = {config: role for role, config in roles.items()}
    for _, row in overall.iterrows():
        detector = row["run_detector"]
        budget = int(row["budget"])
        marker = "o" if budget == 300 else "s"
        role = role_by_config.get(row["config"])
        size = 240 if role else 150
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
            weight="bold" if role else "normal",
        )

    ax.annotate(
        f"Default: {recommended_config}\nnear-best quality + usable savings",
        xy=(recommended_row["cost_savings"] * 100, recommended_row["quality"] * 100),
        xytext=(27.5, 94.5),
        arrowprops={"arrowstyle": "->", "color": "#111827", "linewidth": 1.3},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#111827"},
    )
    ax.text(
        1,
        quality_floor + 0.15,
        "near-best quality band",
        fontsize=8.5,
        color="#374151",
    )
    ax.set_xlim(-1.5, 46)
    ax.set_ylim(87, max(98.2, baseline_quality + 0.8))
    ax.set_xlabel("Total token savings (%)")
    ax.set_ylabel("Answer quality (%)")
    ax.set_title("Overall tradeoff: Pareto frontier and default choice")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    table_ax.axis("off")
    role_order = ["Default", "Conservative", "Balanced-aggressive", "Aggressive"]
    role_rows = []
    role_labels = {
        "Default": "Default",
        "Conservative": "Conserve",
        "Balanced-aggressive": "Balanced",
        "Aggressive": "Aggressive",
    }
    for role in role_order:
        config = roles.get(role)
        if config is None:
            continue
        row = overall[overall["config"] == config].iloc[0]
        role_rows.append(
            [
                role_labels[role],
                config,
                f"{row['quality'] * 100:.1f}%",
                f"{row['cost_savings'] * 100:.1f}%",
                str(int(row["lost_baseline_correct"])),
            ]
        )
    table = table_ax.table(
        cellText=role_rows,
        colLabels=["Role", "Policy", "Q", "T", "Lost"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.22, 0.36, 0.14, 0.15, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row_idx == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(weight="bold")
        elif col_idx == 0:
            cell.set_text_props(weight="bold")
    table_ax.set_title("Policy reading guide", fontsize=11, pad=10)
    table_ax.text(
        0,
        0.08,
        "Q = answer quality\nT = total-token savings\nLost = baseline-correct answers lost",
        fontsize=8.5,
        color="#374151",
        transform=table_ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(output / "overall_tradeoff.png", dpi=180)
    plt.close(fig)


def _plot_strategy_map(overall: pd.DataFrame, output: Path, roles: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_facecolor("#FAFAFA")
    max_quality = float(overall["quality"].max())
    quality_floor = (max_quality - OVERALL_QUALITY_TOLERANCE) * 100
    baseline_quality = float(overall["baseline_quality"].mean()) * 100
    ax.axhspan(quality_floor, 100, color="#E8F2E8", alpha=0.75)
    ax.axvspan(25, 50, color="#FFF7E6", alpha=0.7)
    ax.axhline(
        quality_floor,
        color="#6B7280",
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(25, color="#6B7280", linestyle="--", linewidth=1)

    role_by_config = {config: role for role, config in roles.items()}
    ax.scatter(
        0,
        baseline_quality,
        s=210,
        marker="*",
        color="#6B7280",
        edgecolor="#111827",
        linewidth=1,
        zorder=4,
    )
    ax.annotate(
        f"No-stop baseline\nQ {baseline_quality:.1f}% | T 0.0%",
        (0, baseline_quality),
        xytext=(10, -12),
        textcoords="offset points",
        fontsize=9,
        color="#374151",
    )
    for _, row in overall.iterrows():
        role = role_by_config.get(row["config"])
        detector = row["run_detector"]
        ax.scatter(
            row["cost_savings"] * 100,
            row["quality"] * 100,
            s=310 if role else 140,
            color=POLICY_COLORS.get(role or "", DETECTOR_COLORS[detector]),
            edgecolor="#111827" if role else "white",
            linewidth=2 if role else 1,
            zorder=3 if role else 2,
        )
        label = row["config"]
        if role:
            label = f"{role}\n{row['config']}"
        ax.annotate(
            label,
            (row["cost_savings"] * 100, row["quality"] * 100),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=9,
            weight="bold" if role else "normal",
        )

    ax.text(26, 94.8, "High savings region", fontsize=10, color="#92400E")
    ax.text(
        1,
        quality_floor + 0.15,
        "Near-best quality band",
        fontsize=10,
        color="#166534",
    )
    ax.text(
        6,
        87.7,
        "Read this as a policy map:\n"
        "default balances quality and savings;\n"
        "aggressive maximizes savings;\n"
        "conservative maximizes quality.",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#D1D5DB"},
    )
    ax.set_xlim(-1.5, 46)
    ax.set_ylim(87, max(98.2, baseline_quality + 0.8))
    ax.set_xlabel("Total token savings (%)")
    ax.set_ylabel("Answer quality (%)")
    ax.set_title("Strategy map: choose policy by quality tolerance")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "strategy_map.png", dpi=180)
    plt.close(fig)


def _plot_dataset_matrix(
    dataset_summary: pd.DataFrame, output: Path, recommendations: dict[str, str]
) -> None:
    configs = CONFIG_ORDER
    datasets = DATASETS
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
    lost = dataset_summary.pivot(
        index="dataset", columns="config", values="lost_baseline_correct"
    ).reindex(index=datasets, columns=configs)
    ax.set_title("Dataset decision matrix: quality color, R/T savings and lost answers")

    for i, dataset in enumerate(datasets):
        for j, config in enumerate(configs):
            quality = pivot.loc[dataset, config]
            saving = savings.loc[dataset, config]
            cost_saving = cost_savings.loc[dataset, config]
            lost_count = lost.loc[dataset, config]
            text = (
                f"Q {quality * 100:.0f}%\n"
                f"R {saving * 100:.0f}% | T {cost_saving * 100:.0f}%\n"
                f"Lost {int(lost_count)}"
            )
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
    ax2.set_ylim(0, max(50, float(ordered["cost_savings"].max() * 100) + 8))
    for idx, row in enumerate(ordered.itertuples()):
        ax2.text(
            idx,
            row.cost_savings * 100 + 2.0,
            f"{row.cost_savings * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
        )

    fig.tight_layout()
    fig.savefig(output / "loss_vs_savings.png", dpi=180)
    plt.close(fig)


def _plot_quality_gap(overall: pd.DataFrame, output: Path) -> None:
    ordered = overall.set_index("config").loc[CONFIG_ORDER].reset_index()
    baseline_quality = float(overall["baseline_quality"].mean()) * 100
    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    x = range(len(ordered))
    colors = [DETECTOR_COLORS[d] for d in ordered["run_detector"]]
    bars = ax1.bar(x, ordered["quality"] * 100, color=colors, alpha=0.82)
    ax1.axhline(
        baseline_quality,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label=f"No-stop baseline {baseline_quality:.1f}%",
    )
    ax1.set_ylim(84, 99)
    ax1.set_ylabel("Answer quality (%)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(ordered["config"], rotation=30, ha="right")
    ax1.set_title("Quality drop from the no-stop baseline")
    ax1.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, ordered.itertuples()):
        gap = baseline_quality - row.quality * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            row.quality * 100 + 0.35,
            f"-{gap:.1f}pp",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax2 = ax1.twinx()
    ax2.plot(
        list(x),
        ordered["cost_savings"] * 100,
        color="#111827",
        marker="D",
        linewidth=2,
        label="Total-token savings",
    )
    ax2.set_ylabel("Total-token savings (%)")
    ax2.set_ylim(0, max(50, float(ordered["cost_savings"].max() * 100) + 8))
    ax1.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "quality_gap_from_baseline.png", dpi=180)
    plt.close(fig)


def _plot_savings_decomposition(overall: pd.DataFrame, output: Path) -> None:
    ordered = overall.set_index("config").loc[CONFIG_ORDER].reset_index()
    y = range(len(ordered))
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax.barh(
        y,
        ordered["savings"] * 100,
        color="#CBD5E1",
        label="Reasoning savings",
    )
    ax.barh(
        y,
        ordered["cost_savings"] * 100,
        color=[DETECTOR_COLORS[d] for d in ordered["run_detector"]],
        label="Total-token savings",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(ordered["config"])
    ax.invert_yaxis()
    ax.set_xlim(0, 85)
    ax.set_xlabel("Savings (%)")
    ax.set_title("Mechanism vs bill: reasoning savings does not convert 1:1 to tokens")
    ax.grid(axis="x", alpha=0.25)
    for idx, row in enumerate(ordered.itertuples()):
        tax = (row.savings - row.cost_savings) * 100
        ax.text(row.savings * 100 + 1.0, idx + 0.13, f"R {row.savings * 100:.0f}%", fontsize=8)
        ax.text(
            row.cost_savings * 100 + 1.0,
            idx - 0.14,
            f"T {row.cost_savings * 100:.0f}% | conversion gap {tax:.0f}pp",
            fontsize=8,
            color="#111827",
        )
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "savings_decomposition.png", dpi=180)
    plt.close(fig)


def _plot_token_breakdown(treated: pd.DataFrame, overall: pd.DataFrame, output: Path) -> None:
    tokens = treated.copy()
    tokens["phase1_cost_tokens"] = tokens["phase1_total_tokens"].combine_first(
        tokens["estimated_phase1_total_tokens"]
    )
    tokens["phase2_cost_tokens"] = tokens["phase2_total_tokens"].combine_first(
        tokens["estimated_phase2_total_tokens"]
    )
    tokens["phase2_cost_tokens"] = tokens["phase2_cost_tokens"].fillna(0)
    breakdown = (
        tokens.groupby("config")
        .agg(
            phase1=("phase1_cost_tokens", "mean"),
            phase2=("phase2_cost_tokens", "mean"),
            baseline=("baseline_cost_tokens", "mean"),
        )
        .reindex(CONFIG_ORDER)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    x = range(len(breakdown))
    ax.bar(x, breakdown["phase1"], color="#93C5FD", label="Phase 1")
    ax.bar(
        x,
        breakdown["phase2"],
        bottom=breakdown["phase1"],
        color="#FBBF24",
        label="Phase 2 recovery",
    )
    baseline_mean = float(overall["baseline_cost_tokens"].mean())
    ax.axhline(
        baseline_mean,
        color="#111827",
        linestyle="--",
        linewidth=1.4,
    )
    ax.text(
        len(breakdown) - 1.25,
        baseline_mean + baseline_mean * 0.01,
        f"No-stop baseline mean {baseline_mean:.0f}",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.8},
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(breakdown["config"], rotation=30, ha="right")
    ax.set_ylabel("Mean total tokens per question")
    ax.set_title("Token accounting: Phase 2 recovery is the main savings offset")
    ax.grid(axis="y", alpha=0.25)
    for idx, row in enumerate(breakdown.itertuples()):
        total = row.phase1 + row.phase2
        ax.text(idx, total + baseline_mean * 0.015, f"{total:.0f}", ha="center", fontsize=8.5)
        if row.phase2 > 0:
            ax.text(
                idx,
                row.phase1 + row.phase2 / 2,
                f"P2 {row.phase2:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111827",
            )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, 0.9),
        frameon=True,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout()
    fig.savefig(output / "token_breakdown.png", dpi=180)
    plt.close(fig)


def _plot_dataset_tradeoffs(
    dataset_summary: pd.DataFrame,
    output: Path,
    recommendations: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    label_offsets = {
        "math": {
            "budget@300": (8, 6),
            "budget@1000": (8, -4),
            "compression@300": (8, 5),
            "compression@1000": (8, 6),
            "keyword@300": (8, -4),
            "keyword@1000": (8, 6),
        },
        "mmlu": {
            "budget@300": (8, -8),
            "budget@1000": (8, 6),
            "compression@300": (-62, -14),
            "compression@1000": (-82, 8),
            "keyword@300": (8, 6),
            "keyword@1000": (8, 6),
        },
        "riddle": {
            "budget@300": (8, 6),
            "budget@1000": (8, 6),
            "compression@300": (8, 6),
            "compression@1000": (8, -8),
            "keyword@300": (8, 6),
            "keyword@1000": (8, 6),
        },
    }
    for ax, dataset in zip(axes, DATASETS):
        group = dataset_summary[dataset_summary["dataset"] == dataset]
        baseline_quality = float(group["baseline_quality"].mean()) * 100
        ax.scatter(
            0,
            baseline_quality,
            s=150,
            marker="*",
            color="#6B7280",
            edgecolor="#111827",
            linewidth=1,
            zorder=4,
        )
        baseline_offset = (6, -12) if baseline_quality > 98 else (6, -6)
        ax.annotate(
            "baseline",
            (0, baseline_quality),
            xytext=baseline_offset,
            textcoords="offset points",
        )
        for _, row in group.iterrows():
            config = row["config"]
            recommended = recommendations.get(dataset) == config
            ax.scatter(
                row["cost_savings"] * 100,
                row["quality"] * 100,
                s=180 if recommended else 95,
                color=DETECTOR_COLORS[row["run_detector"]],
                edgecolor="#111827" if recommended else "white",
                linewidth=2 if recommended else 1,
            )
            ax.annotate(
                config,
                (row["cost_savings"] * 100, row["quality"] * 100),
                xytext=label_offsets[dataset].get(config, (6, 5)),
                textcoords="offset points",
                fontsize=8,
                weight="bold" if recommended else "normal",
            )
        ax.set_title(f"{dataset}: recommended {recommendations[dataset]}")
        ax.set_xlabel("Total-token savings (%)")
        ax.set_xlim(-2, 58)
        ax.set_ylim(82, 100.5)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Answer quality (%)")
    fig.suptitle("Dataset-specific tradeoff: one default is not optimal for every task")
    fig.tight_layout()
    fig.savefig(output / "dataset_tradeoffs.png", dpi=180)
    plt.close(fig)


def _plot_detector_profiles(overall: pd.DataFrame, output: Path) -> None:
    fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True)
    for detector, group in overall.groupby("run_detector"):
        group = group.sort_values("budget")
        ax_q.plot(
            group["budget"],
            group["quality"] * 100,
            marker="o",
            linewidth=2,
            color=DETECTOR_COLORS[detector],
            label=detector,
        )
        ax_t.plot(
            group["budget"],
            group["cost_savings"] * 100,
            marker="o",
            linewidth=2,
            color=DETECTOR_COLORS[detector],
            label=detector,
        )
    ax_q.set_title("Quality by fallback budget")
    ax_q.set_ylabel("Answer quality (%)")
    ax_q.set_ylim(87, 96)
    ax_t.set_title("Total-token savings by fallback budget")
    ax_t.set_ylabel("Total-token savings (%)")
    ax_t.set_ylim(0, 48)
    for ax in (ax_q, ax_t):
        ax.set_xlabel("Fallback budget")
        ax.set_xticks([300, 1000])
        ax.grid(True, alpha=0.25)
    ax_q.legend(frameon=False)
    fig.suptitle("Detector profiles: 300 is aggressive, 1000 is conservative")
    fig.tight_layout()
    fig.savefig(output / "detector_profiles.png", dpi=180)
    plt.close(fig)


def _plot_outcome_flow(treated: pd.DataFrame, output: Path) -> None:
    rows = []
    for config, group in treated.groupby("config"):
        rows.append(
            {
                "config": config,
                "kept_correct": int(
                    ((group["baseline_quality"] >= 1.0) & (group["quality_score"] >= 1.0)).sum()
                ),
                "lost": int(group["lost_baseline_correct"].sum()),
                "fixed": int(group["fixed_baseline_wrong"].sum()),
                "still_wrong": int(
                    ((group["baseline_quality"] < 1.0) & (group["quality_score"] < 1.0)).sum()
                ),
            }
        )
    outcome = pd.DataFrame(rows).set_index("config").loc[CONFIG_ORDER]
    colors = {
        "kept_correct": "#86EFAC",
        "lost": "#FCA5A5",
        "fixed": "#93C5FD",
        "still_wrong": "#D1D5DB",
    }
    labels = {
        "kept_correct": "Kept correct",
        "lost": "Lost",
        "fixed": "Fixed",
        "still_wrong": "Still wrong",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bottom = [0] * len(outcome)
    x = range(len(outcome))
    for col in ["kept_correct", "lost", "fixed", "still_wrong"]:
        values = outcome[col].tolist()
        ax.bar(x, values, bottom=bottom, color=colors[col], label=labels[col])
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(outcome.index, rotation=30, ha="right")
    ax.set_ylabel("Questions")
    ax.set_title("Outcome flow: most answers stay correct, but lost cases drive quality cost")
    ax.grid(axis="y", alpha=0.25)
    for idx, row in enumerate(outcome.itertuples()):
        ax.text(idx, row.kept_correct + row.lost / 2, f"Lost {row.lost}", ha="center", fontsize=8)
        ax.text(
            idx,
            row.kept_correct + row.lost + row.fixed / 2,
            f"Fixed {row.fixed}",
            ha="center",
            fontsize=8,
        )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    fig.savefig(output / "outcome_flow.png", dpi=180)
    plt.close(fig)


def _plot_token_estimate_calibration(treated: pd.DataFrame, output: Path) -> None:
    exact = treated[
        treated["phase1_total_tokens"].notna()
        & treated["estimated_phase1_total_tokens"].notna()
        & (treated["phase1_total_tokens"] > 0)
    ].copy()
    if exact.empty:
        return
    calibration = _calibrate_phase1_estimates(treated)
    fig, ax = plt.subplots(figsize=(6.3, 6.0))
    for detector, group in exact.groupby("run_detector"):
        ax.scatter(
            group["phase1_total_tokens"],
            group["estimated_phase1_total_tokens"],
            s=35,
            alpha=0.65,
            color=DETECTOR_COLORS[detector],
            label=detector,
        )
    max_token = float(
        max(exact["phase1_total_tokens"].max(), exact["estimated_phase1_total_tokens"].max())
    )
    ax.plot([0, max_token], [0, max_token], color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("API phase-1 total tokens")
    ax.set_ylabel("Estimated phase-1 total tokens")
    ax.set_title("Token-estimate calibration for rows with API usage")
    ax.text(
        0.05,
        0.95,
        f"n={calibration['n']}\nMAE={_format_pct(float(calibration['mean_abs_error']))}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#D1D5DB"},
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "token_estimate_calibration.png", dpi=180)
    plt.close(fig)


def _plot_latency_change(overall: pd.DataFrame, output: Path) -> None:
    ordered = overall.set_index("config").loc[CONFIG_ORDER].reset_index()
    ordered["latency_change"] = (
        (ordered["latency_ms"] - ordered["baseline_latency_ms"])
        / ordered["baseline_latency_ms"].clip(lower=1)
    )
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    x = range(len(ordered))
    colors = ["#DC2626" if value > 0 else "#16A34A" for value in ordered["latency_change"]]
    bars = ax.bar(x, ordered["latency_change"] * 100, color=colors, alpha=0.82)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["config"], rotation=30, ha="right")
    ax.set_ylabel("Latency change vs no-stop baseline (%)")
    ax.set_title("Latency side metric: current run is faster, but cost remains the main claim")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ordered["latency_change"]):
        va = "bottom" if value >= 0 else "top"
        offset = 0.8 if value >= 0 else -0.8
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 100 + offset,
            f"{value * 100:+.0f}%",
            ha="center",
            va=va,
            fontsize=8.5,
        )
    fig.tight_layout()
    fig.savefig(output / "latency_change.png", dpi=180)
    plt.close(fig)


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _recommendation_reason(
    dataset: str, row: pd.Series, group: pd.DataFrame, tolerance: float
) -> str:
    best_quality = float(group["quality"].max())
    quality_gap = best_quality - float(row["quality"])
    if quality_gap <= 1e-9:
        return "Best observed quality with the current dataset slice."
    return (
        f"Within {_format_pct(tolerance)} of the best observed quality for {dataset} "
        "while preserving materially higher total-token savings."
    )


def _calibrate_phase1_estimates(treated: pd.DataFrame) -> dict[str, float | int]:
    exact = treated[
        treated["phase1_total_tokens"].notna()
        & treated["estimated_phase1_total_tokens"].notna()
        & (treated["phase1_total_tokens"] > 0)
    ].copy()
    if exact.empty:
        return {
            "n": 0,
            "mean_error": 0.0,
            "median_error": 0.0,
            "mean_abs_error": 0.0,
        }

    relative_error = (
        exact["estimated_phase1_total_tokens"] - exact["phase1_total_tokens"]
    ) / exact["phase1_total_tokens"]
    return {
        "n": int(len(exact)),
        "mean_error": float(relative_error.mean()),
        "median_error": float(relative_error.median()),
        "mean_abs_error": float(relative_error.abs().mean()),
    }


def _make_report(
    dataset_summary: pd.DataFrame,
    overall: pd.DataFrame,
    treated: pd.DataFrame,
    output: Path,
    recommended_config: str,
    recommendations: dict[str, str],
    roles: dict[str, str],
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
    for dataset, config in recommendations.items():
        group = dataset_summary[dataset_summary["dataset"] == dataset]
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
                "Why": _recommendation_reason(
                    dataset, row, group, DATASET_QUALITY_TOLERANCE
                ),
            }
        )
    recommendation_table = pd.DataFrame(recommendation_rows)
    role_table = pd.DataFrame(
        [
            {
                "Role": role,
                "Config": config,
                "Quality": _format_pct(
                    overall[overall["config"] == config].iloc[0]["quality"]
                ),
                "TokenSavings": _format_pct(
                    overall[overall["config"] == config].iloc[0]["cost_savings"]
                ),
                "Lost": int(
                    overall[overall["config"] == config].iloc[0][
                        "lost_baseline_correct"
                    ]
                ),
            }
            for role, config in roles.items()
        ]
    )
    recommended_row = overall[overall["config"] == recommended_config].iloc[0]
    token_source = treated["cost_token_metric"].value_counts().to_dict()
    exact_count = int(token_source.get("total_tokens", 0))
    estimated_count = int(token_source.get("estimated_total_tokens", 0))
    total_count = exact_count + estimated_count
    phase1_api_count = int(treated["phase1_total_tokens"].notna().sum())
    phase2_rows = treated[treated["phase2_used"]]
    phase2_api_count = int(phase2_rows["phase2_total_tokens"].notna().sum())
    phase2_total = int(len(phase2_rows))
    calibration = _calibrate_phase1_estimates(treated)

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

Storyline evidence:

1. `quality_gap_from_baseline.png`: no-stop baseline quality is the reference;
   every early-stop policy buys token savings by accepting some quality drop.
2. `savings_decomposition.png`: reasoning savings is much larger than final
   total-token savings because recovery, prompts, and token-estimation effects
   create a conversion gap.
3. `token_breakdown.png`: Phase 1 shrinks under interruption, while Phase 2 is
   the recovery cost that prevents reasoning savings from translating 1:1 into
   billed-token savings.
4. `overall_tradeoff.png`: the global decision is a Pareto tradeoff, with
   `compression@1000` as the current default.
5. `strategy_map.png`: users can choose conservative, default, balanced, or
   aggressive policies according to quality tolerance.
6. `dataset_tradeoffs.png`: dataset-specific routing is justified; the best
   policy differs across math, mmlu, and riddle.
7. `dataset_decision_matrix.png`: the router table shows quality, reasoning
   savings, total-token savings, and lost answers for every dataset/policy cell.
8. `outcome_flow.png`: `Lost` and `Fixed` explain where the quality delta comes
   from, rather than hiding it in aggregate accuracy.
9. `loss_vs_savings.png`: aggressive savings increase the count of
   baseline-correct answers lost.
10. `detector_profiles.png`: 300-token fallback settings are aggressive, while
    1000-token settings are conservative.
11. `token_estimate_calibration.png`: current total-token savings should be
    treated with calibration uncertainty because interrupted Phase 1 often lacks
    provider usage.
12. `latency_change.png`: latency is currently favorable, but it remains a
    secondary metric until we run a dedicated latency-controlled experiment.

In the tradeoff plots, the gray star marks the no-stop baseline: it has zero
token savings and serves only as the quality reference point, not as an
early-stop policy candidate.

Policy roles used in the strategy map:

{role_table.to_markdown(index=False)}

## Overall Metrics

{overall_table.to_markdown(index=False)}

`Lost` means baseline was correct but early-stop answer was not fully correct.
`Fixed` means baseline was wrong but the early-stop answer became correct.
`TokenSavings` uses API `total_tokens` when present and falls back to
`estimated_total_tokens` when streaming usage is unavailable.
In this run, {exact_count}/{total_count} treated rows used API `total_tokens`;
{estimated_count}/{total_count} treated rows used `estimated_total_tokens`.

Phase 1 API usage is available for {phase1_api_count}/{total_count} treated rows.
The missing Phase 1 rows are expected: early stopping closes the stream before the
provider can emit the final streaming usage chunk. Phase 2 API usage is available
for {phase2_api_count}/{phase2_total} rows where Phase 2 was used.

On rows where Phase 1 has both API usage and local estimates
(`n={calibration["n"]}`), local `estimated_phase1_total_tokens` has mean error
{_format_pct(float(calibration["mean_error"]))}, median error
{_format_pct(float(calibration["median_error"]))}, and mean absolute error
{_format_pct(float(calibration["mean_abs_error"]))}. Treat total-token savings as
directionally useful until interrupted Phase 1 cost is calibrated with a better
provider tokenizer or a dedicated calibration run.

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
    roles = _policy_roles(overall, recommended_config)

    dataset_summary.to_csv(output / "dataset_summary.csv", index=False, encoding=DEFAULT_ENCODING)
    overall.to_csv(output / "overall_summary.csv", index=False, encoding=DEFAULT_ENCODING)

    _plot_overall_tradeoff(overall, output, recommended_config, roles)
    _plot_strategy_map(overall, output, roles)
    _plot_quality_gap(overall, output)
    _plot_savings_decomposition(overall, output)
    _plot_token_breakdown(treated, overall, output)
    _plot_dataset_tradeoffs(dataset_summary, output, recommendations)
    _plot_detector_profiles(overall, output)
    _plot_outcome_flow(treated, output)
    _plot_token_estimate_calibration(treated, output)
    _plot_latency_change(overall, output)
    _plot_dataset_matrix(dataset_summary, output, recommendations)
    _plot_loss_vs_savings(overall, output)
    _make_report(
        dataset_summary,
        overall,
        treated,
        output,
        recommended_config,
        recommendations,
        roles,
    )

    print(f"Saved focused report to {output}")


if __name__ == "__main__":
    main()
