#!/usr/bin/env python3
"""Generate publication-ready RAGAS score plots for benchmark CSV outputs.

This module reads a RAGAS results CSV and creates two figures:
1) Per-question multi-panel figure (4 metrics across all questions)
2) Cohort summary figure (text_only vs image_text means with std bars)

Default behavior:
- Auto-select the latest ragas_*.csv in this directory
- Enforce strict 50-row expectation (25 text_only + 25 image_text)
- Export PNG plots to this directory

Examples:
    python graph.py
    python graph.py --csv ragas_20260302_194107.csv
    python graph.py --allow-partial
    python graph.py --out-dir ./plots
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "matplotlib is required for plotting. Install it in your environment, "
        "for example: uv pip install matplotlib"
    ) from exc


METRICS: List[str] = [
    "context_recall",
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
]

REQUIRED_COLUMNS: List[str] = ["id", "category", *METRICS]
TARGET_COLUMNS: Dict[str, str] = {metric: f"{metric}_target" for metric in METRICS}
CATEGORY_ORDER: List[str] = ["text_only", "image_text"]
EXPECTED_COUNTS: Dict[str, int] = {"text_only": 25, "image_text": 25}
EXPECTED_TOTAL: int = 50
METRIC_YMIN: Dict[str, float] = {
    "context_recall": 0.70,
    "faithfulness": 0.80,
    "factual_correctness": 0.50,
    "answer_relevancy": 0.70,
}


@dataclass(frozen=True)
class Record:
    qid: str
    category: str
    question: str
    scores: Dict[str, float]
    targets: Dict[str, float]


def _score_from_row(row: Dict[str, str], column: str, allow_nan: bool = True) -> float:
    raw = (row.get(column) or "").strip()
    if raw == "":
        if allow_nan:
            return float("nan")
        raise ValueError(f"Missing value for column '{column}'")
    value = float(raw)
    if math.isnan(value):
        if allow_nan:
            return value
        raise ValueError(f"Invalid score in '{column}': {raw}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Invalid score in '{column}': {raw}")
    return value


def _parse_records(csv_path: Path) -> List[Record]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        records: List[Record] = []
        for idx, row in enumerate(reader, start=2):
            category = (row.get("category") or "").strip()
            if category not in CATEGORY_ORDER:
                raise ValueError(
                    f"Row {idx}: unsupported category '{category}'. "
                    f"Expected one of: {', '.join(CATEGORY_ORDER)}"
                )

            qid = (row.get("id") or "").strip()
            if not qid:
                raise ValueError(f"Row {idx}: missing id")

            question = (row.get("question") or "").strip()

            scores = {metric: _score_from_row(row, metric, allow_nan=True) for metric in METRICS}

            targets: Dict[str, float] = {}
            for metric, target_col in TARGET_COLUMNS.items():
                if target_col in reader.fieldnames and (row.get(target_col) or "").strip():
                    targets[metric] = _score_from_row(row, target_col, allow_nan=False)

            records.append(
                Record(
                    qid=qid,
                    category=category,
                    question=question,
                    scores=scores,
                    targets=targets,
                )
            )

    if not records:
        raise ValueError("CSV has no data rows")

    return records


def _id_sort_key(record: Record) -> Tuple[int, int, str]:
    # Primary by category order, secondary by numeric suffix in id, fallback by id string.
    category_rank = CATEGORY_ORDER.index(record.category)
    match = re.search(r"(\d+)$", record.qid)
    qnum = int(match.group(1)) if match else 10**9
    return (category_rank, qnum, record.qid)


def _validate_counts(records: Iterable[Record], allow_partial: bool) -> Dict[str, int]:
    counts: Dict[str, int] = {key: 0 for key in CATEGORY_ORDER}
    for rec in records:
        counts[rec.category] += 1

    missing_categories = [cat for cat, count in counts.items() if count == 0]
    if missing_categories:
        raise ValueError(
            "Missing category rows for: " + ", ".join(missing_categories)
        )

    total = sum(counts.values())
    if not allow_partial:
        if total != EXPECTED_TOTAL:
            raise ValueError(
                f"Strict mode expects {EXPECTED_TOTAL} rows, found {total}. "
                "Use --allow-partial to plot non-final benchmark outputs."
            )

        mismatched = [
            f"{cat}={counts[cat]} (expected {expected})"
            for cat, expected in EXPECTED_COUNTS.items()
            if counts[cat] != expected
        ]
        if mismatched:
            raise ValueError(
                "Strict mode cohort count mismatch: " + "; ".join(mismatched)
            )

    return counts


def _resolve_target(records: List[Record], metric: str) -> float | None:
    values = [r.targets[metric] for r in records if metric in r.targets and not math.isnan(r.targets[metric])]
    if not values:
        return None
    return mean(values)


def _finite(values: Iterable[float]) -> List[float]:
    return [v for v in values if not math.isnan(v)]


def _latest_ragas_csv(base_dir: Path) -> Path:
    candidates = sorted(base_dir.glob("ragas_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No ragas_*.csv files found in {base_dir}")
    return candidates[-1]


def _plot_per_question(
    records: List[Record],
    csv_stem: str,
    out_dir: Path,
    counts: Dict[str, int],
    dpi: int,
) -> Path:
    color_map = {"text_only": "#1f77b4", "image_text": "#ff7f0e"}
    marker_map = {"text_only": "o", "image_text": "s"}

    x_values = list(range(1, len(records) + 1))
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 12), sharex=True)

    for axis, metric in zip(axes, METRICS):
        y_min = METRIC_YMIN.get(metric, 0.0)
        for category in CATEGORY_ORDER:
            idxs = [i for i, rec in enumerate(records) if rec.category == category]
            xs = [x_values[i] for i in idxs]
            ys = [records[i].scores[metric] for i in idxs]
            # Add a soft underlay so near-flat segments remain clearly connected.
            axis.plot(
                xs,
                ys,
                color="#ffffff",
                linewidth=4.0,
                alpha=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2,
            )
            axis.plot(
                xs,
                ys,
                color=color_map[category],
                marker=marker_map[category],
                markeredgecolor="#ffffff",
                markeredgewidth=0.8,
                markersize=5,
                linewidth=2.2,
                label=category,
                alpha=0.98,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )

        target = _resolve_target(records, metric)
        if target is not None:
            axis.axhline(
                y=target,
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )
            axis.text(
                0.995,
                max(y_min + 0.02, target - 0.06),
                f"target={target:.2f}",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="top",
                fontsize=8,
                color="#444444",
            )

        axis.set_ylim(y_min, 1.0)
        axis.set_xlim(0.8, len(records) + 0.2)
        axis.set_ylabel(metric, fontsize=10)
        axis.set_axisbelow(True)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    boundary = counts["text_only"] + 0.5
    for axis in axes:
        axis.axvline(boundary, color="#777777", linestyle="-.", linewidth=1.0, alpha=0.8)

    axes[0].text(
        counts["text_only"] / 2,
        1.03,
        "text_only",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=axes[0].transData,
    )
    axes[0].text(
        counts["text_only"] + (counts["image_text"] / 2),
        1.03,
        "image_text",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=axes[0].transData,
    )

    axes[-1].set_xlabel("Question index (ordered by dataset and id)")
    if len(records) == EXPECTED_TOTAL:
        axes[-1].set_xticks(list(range(2, EXPECTED_TOTAL + 1, 2)))
    else:
        tick_step = 2 if len(records) <= 60 else 5
        axes[-1].set_xticks(list(range(1, len(records) + 1, tick_step)))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "RAGAS Scores by Question",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.95))

    png_path = out_dir / "scores_by_question.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return png_path


def _plot_cohort_summary(
    records: List[Record],
    csv_stem: str,
    out_dir: Path,
    dpi: int,
) -> Path:
    color_map = {"text_only": "#1f77b4", "image_text": "#ff7f0e"}

    by_category: Dict[str, List[Record]] = {
        cat: [rec for rec in records if rec.category == cat] for cat in CATEGORY_ORDER
    }

    means: Dict[str, List[float]] = {cat: [] for cat in CATEGORY_ORDER}
    stds: Dict[str, List[float]] = {cat: [] for cat in CATEGORY_ORDER}
    targets: List[float | None] = []

    for metric in METRICS:
        for category in CATEGORY_ORDER:
            vals = _finite([r.scores[metric] for r in by_category[category]])
            if not vals:
                means[category].append(float("nan"))
                stds[category].append(float("nan"))
            else:
                means[category].append(mean(vals))
                stds[category].append(stdev(vals) if len(vals) > 1 else 0.0)
        targets.append(_resolve_target(records, metric))

    x = list(range(len(METRICS)))
    width = 0.35

    fig, axis = plt.subplots(figsize=(12, 6))

    axis.bar(
        [xi - width / 2 for xi in x],
        means["text_only"],
        width=width,
        yerr=stds["text_only"],
        capsize=3,
        color=color_map["text_only"],
        alpha=0.9,
        label="text_only",
    )
    axis.bar(
        [xi + width / 2 for xi in x],
        means["image_text"],
        width=width,
        yerr=stds["image_text"],
        capsize=3,
        color=color_map["image_text"],
        alpha=0.9,
        label="image_text",
    )

    target_xs = [xi for xi, t in enumerate(targets) if t is not None]
    target_vals = [targets[xi] for xi in target_xs]
    if target_xs:
        axis.plot(
            target_xs,
            target_vals,
            linestyle="",
            marker="D",
            markersize=6,
            color="#333333",
            label="target",
        )

    axis.set_xticks(x)
    axis.set_xticklabels(METRICS, rotation=0)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("RAGAS Score Summary by Dataset")
    axis.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.94))

    png_path = out_dir / "score_summary.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return png_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot RAGAS results for text_only and image_text cohorts."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a specific ragas_*.csv file. Defaults to latest in this directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plot files. Defaults to this script directory.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow plotting datasets that are not full 50-row final runs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv).resolve() if args.csv else _latest_ragas_csv(script_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _parse_records(csv_path)
    records.sort(key=_id_sort_key)
    counts = _validate_counts(records, allow_partial=args.allow_partial)

    missing_details: List[Tuple[str, str, str, str]] = []
    for rec in records:
        for metric in METRICS:
            if math.isnan(rec.scores[metric]):
                missing_details.append((rec.qid, rec.category, metric, rec.question))

    missing_by_metric = {
        metric: sum(1 for rec in records if math.isnan(rec.scores[metric]))
        for metric in METRICS
    }

    q_plot_png = _plot_per_question(
        records=records,
        csv_stem=csv_path.stem,
        out_dir=out_dir,
        counts=counts,
        dpi=args.dpi,
    )
    s_plot_png = _plot_cohort_summary(
        records=records,
        csv_stem=csv_path.stem,
        out_dir=out_dir,
        dpi=args.dpi,
    )

    print("[Graph] Input CSV:", csv_path)
    print("[Graph] Total rows:", len(records))
    print(
        "[Graph] Cohorts:",
        ", ".join(f"{key}={counts[key]}" for key in CATEGORY_ORDER),
    )
    print("[Graph] Outputs:")
    print("  -", q_plot_png)
    print("  -", s_plot_png)
    print(
        "[Graph] Missing score cells:",
        ", ".join(f"{metric}={missing_by_metric[metric]}" for metric in METRICS),
    )
    if missing_details:
        print("[Graph] Missing score detail list:")
        for idx, (qid, category, metric, question) in enumerate(missing_details, start=1):
            print(
                f"  {idx}. id={qid} | category={category} | metric={metric} | current_value=nan"
            )
            print(f"     question={question}")
    else:
        print("[Graph] Missing score detail list: none")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command-line diagnostics
        print(f"[Graph] Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
