"""Bounded-memory exploratory analysis of the Alfa Connectome Parquet data.

Each Parquet row group is one sequence. The script samples whole sequences
throughout each split, so it never loads the full 35 GB dataset into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
TARGETS = ("t0", "t1")


@dataclass
class NumericStats:
    count: int = 0
    nonfinite: int = 0
    zero: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        self.nonfinite += int((~finite).sum())
        good = values[finite].astype(np.float64, copy=False)
        if not good.size:
            return
        self.count += len(good)
        self.zero += int(np.count_nonzero(good == 0))
        self.total += float(good.sum())
        self.total_sq += float(np.dot(good, good))
        self.minimum = min(self.minimum, float(good.min()))
        self.maximum = max(self.maximum, float(good.max()))

    def result(self) -> dict:
        mean = self.total / self.count if self.count else None
        std = (
            math.sqrt(max(0.0, self.total_sq / self.count - mean * mean))
            if self.count
            else None
        )
        return {
            "count": self.count,
            "nonfinite": self.nonfinite,
            "zero_fraction": self.zero / self.count if self.count else None,
            "mean": mean,
            "std": std,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
        }


@dataclass
class PearsonStats:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite].astype(np.float64, copy=False)
        y = y[finite].astype(np.float64, copy=False)
        self.count += len(x)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_xx += float(np.dot(x, x))
        self.sum_yy += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))

    def result(self) -> float | None:
        if self.count < 2:
            return None
        covariance = self.sum_xy - self.sum_x * self.sum_y / self.count
        variance_x = self.sum_xx - self.sum_x**2 / self.count
        variance_y = self.sum_yy - self.sum_y**2 / self.count
        if variance_x <= 0 or variance_y <= 0:
            return None
        return max(-1.0, min(1.0, covariance / math.sqrt(variance_x * variance_y)))


def sample_group_indices(total: int, requested: int) -> list[int]:
    if total == 0 or requested < 1:
        return []
    return sorted(set(np.linspace(0, total - 1, min(total, requested), dtype=int)))


def column(table, name: str) -> np.ndarray:
    return table.column(name).to_numpy(zero_copy_only=False)


def scan_split(path: Path, requested_groups: int, mask_path: Path | None = None) -> dict:
    parquet = pq.ParquetFile(path)
    mask_file = pq.ParquetFile(mask_path) if mask_path else None
    metadata = parquet.metadata
    names = parquet.schema_arrow.names
    features = [name for name in names if name not in
                {"seq_ix", "step_in_seq", "need_prediction", "is_scored", *TARGETS}]
    indices = sample_group_indices(metadata.num_row_groups, requested_groups)
    feature_stats = {name: NumericStats() for name in features}
    target_stats = {name: NumericStats() for name in TARGETS}
    feature_corr = {target: {name: PearsonStats() for name in features} for target in TARGETS}
    target_corr = PearsonStats()
    target_lag1 = {name: PearsonStats() for name in TARGETS}
    target_values = {name: [] for name in TARGETS}
    sampled_rows = sampled_prediction_rows = sampled_scored_rows = 0
    bad_sequence_groups = bad_step_groups = mask_mismatch_groups = 0
    sampled_sequence_ids = []

    for index in indices:
        table = parquet.read_row_group(index)
        rows = table.num_rows
        seq = column(table, "seq_ix")
        steps = column(table, "step_in_seq")
        need = column(table, "need_prediction").astype(bool, copy=False)
        sampled_rows += rows
        sampled_prediction_rows += int(need.sum())
        sampled_sequence_ids.append(int(seq[0]))
        if len(np.unique(seq)) != 1:
            bad_sequence_groups += 1
        if rows != 20_000 or not np.array_equal(steps, np.arange(rows)):
            bad_step_groups += 1

        if "is_scored" in names:
            scored = column(table, "is_scored").astype(bool, copy=False)
            if mask_file:
                mask = mask_file.read_row_group(index, columns=["is_scored"])
                if not np.array_equal(scored, column(mask, "is_scored")):
                    mask_mismatch_groups += 1
            selected = need & scored
            sampled_scored_rows += int(selected.sum())
        else:
            selected = need

        y = {name: column(table, name) for name in TARGETS}
        target_corr.update(y["t0"][selected], y["t1"][selected])
        for name in TARGETS:
            target_stats[name].update(y[name][selected])
            target_values[name].append(y[name][selected])
            target_lag1[name].update(y[name][99:-1], y[name][100:])

        for name in features:
            x = column(table, name)
            feature_stats[name].update(x)
            for target in TARGETS:
                feature_corr[target][name].update(x[selected], y[target][selected])

        print(f"{path.name}: group {index + 1}/{metadata.num_row_groups}", flush=True)

    target_results = {}
    for name in TARGETS:
        values = np.concatenate(target_values[name]) if target_values[name] else np.array([])
        values = values[np.isfinite(values)]
        target_results[name] = {
            **target_stats[name].result(),
            "quantiles": dict(zip(
                ("p01", "p05", "p50", "p95", "p99"),
                np.quantile(values, (0.01, 0.05, 0.5, 0.95, 0.99)).tolist()
            )) if len(values) else {},
            "outside_score_clip_fraction": float(np.mean(np.abs(values) > 2)) if len(values) else None,
            "lag1_correlation": target_lag1[name].result(),
        }

    return {
        "file": str(path.relative_to(ROOT)),
        "file_bytes": path.stat().st_size,
        "total_rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "columns": names,
        "feature_count": len(features),
        "non_20000_row_groups": sum(
            metadata.row_group(i).num_rows != 20_000 for i in range(metadata.num_row_groups)
        ),
        "sampled_group_indices": indices,
        "sampled_sequence_ids": sampled_sequence_ids,
        "sampled_rows": sampled_rows,
        "sampled_prediction_rows": sampled_prediction_rows,
        "sampled_scored_rows": sampled_scored_rows if "is_scored" in names else None,
        "bad_sequence_groups": bad_sequence_groups,
        "bad_step_groups": bad_step_groups,
        "mask_mismatch_groups": mask_mismatch_groups if mask_file else None,
        "targets": target_results,
        "target_correlation": target_corr.result(),
        "features": {name: stats.result() for name, stats in feature_stats.items()},
        "feature_target_correlations": {
            target: {name: pair.result() for name, pair in pairs.items()}
            for target, pairs in feature_corr.items()
        },
    }


def format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def markdown_report(train: dict, valid: dict, mask: dict) -> str:
    lines = [
        "# Alfa Connectome: initial EDA",
        "",
        "Generated by `python eda.py`. Numeric distributions and correlations below use",
        "evenly spaced, complete sequence samples. File sizes, row counts, and row-group",
        "counts come from Parquet metadata for the full files.",
        "",
        "## Dataset inventory",
        "",
        "| File | Size (GiB) | Rows | Sequences | Columns |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in (train, valid, mask):
        lines.append(
            f"| `{item['file']}` | {item['file_bytes'] / 2**30:.2f} | "
            f"{item['total_rows']:,} | {item['row_groups']:,} | "
            f"{len(item['columns'])} |"
        )
    lines += [
        "",
        "## Sample and sequence checks",
        "",
        "| Split | Sampled sequences | Sampled rows | Prediction rows | Scored rows | Bad sequence/step groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in (train, valid):
        lines.append(
            f"| {Path(item['file']).stem} | {len(item['sampled_group_indices'])} | "
            f"{item['sampled_rows']:,} | {item['sampled_prediction_rows']:,} | "
            f"{format_number(item['sampled_scored_rows'])} | "
            f"{item['bad_sequence_groups']}/{item['bad_step_groups']} |"
        )
    lines += [
        "",
        f"All row groups with a non-20,000 row count: train {train['non_20000_row_groups']}, "
        f"validation {valid['non_20000_row_groups']}, mask {mask['non_20000_row_groups']}.",
        f"Sampled validation mask mismatches: {valid['mask_mismatch_groups']}.",
        "",
        "## Targets on prediction rows",
        "",
        "Validation target statistics use only scored prediction rows. Train statistics",
        "use all prediction rows in the sample.",
        "",
        "| Split | Target | Mean | Std | p01 | Median | p99 | Outside [-2, 2] | Lag-1 corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in (train, valid):
        for target in TARGETS:
            stats = item["targets"][target]
            q = stats["quantiles"]
            lines.append(
                f"| {Path(item['file']).stem} | {target} | "
                f"{format_number(stats['mean'])} | {format_number(stats['std'])} | "
                f"{format_number(q.get('p01'))} | {format_number(q.get('p50'))} | "
                f"{format_number(q.get('p99'))} | "
                f"{format_number(stats['outside_score_clip_fraction'])} | "
                f"{format_number(stats['lag1_correlation'])} |"
            )
    lines += [
        "",
        f"Sampled `t0`/`t1` Pearson correlation: train "
        f"{format_number(train['target_correlation'])}; validation "
        f"{format_number(valid['target_correlation'])}.",
        "",
        "## Strongest single-feature associations",
        "",
        "These are same-row Pearson correlations on the sampled prediction rows.",
        "They are descriptive and do not establish predictive performance.",
        "",
        "| Split | Target | Feature | Correlation |",
        "|---|---|---|---:|",
    ]
    for item in (train, valid):
        for target in TARGETS:
            ranked = sorted(
                ((name, value) for name, value in
                 item["feature_target_correlations"][target].items() if value is not None),
                key=lambda pair: abs(pair[1]), reverse=True,
            )[:8]
            for name, value in ranked:
                lines.append(f"| {Path(item['file']).stem} | {target} | `{name}` | {value:.4f} |")
    lines += [
        "",
        "## Feature quality and split shift",
        "",
        "The full feature summary is in `reports/feature_stats.csv`. The largest",
        "sampled train-to-validation mean shifts, divided by the train standard",
        "deviation, are:",
        "",
        "| Feature | Train mean | Validation mean | Standardized shift |",
        "|---|---:|---:|---:|",
    ]
    shifts = []
    for name, stats in train["features"].items():
        other = valid["features"][name]
        if stats["std"] and stats["mean"] is not None and other["mean"] is not None:
            shifts.append((name, stats["mean"], other["mean"],
                           (other["mean"] - stats["mean"]) / stats["std"]))
    for name, train_mean, valid_mean, shift in sorted(shifts, key=lambda x: abs(x[3]), reverse=True)[:10]:
        lines.append(f"| `{name}` | {train_mean:.4f} | {valid_mean:.4f} | {shift:+.4f} |")
    for item in (train, valid):
        nonfinite = sum(stats["nonfinite"] for stats in item["features"].values())
        lines.append(f"\n{Path(item['file']).stem} sampled feature nonfinite cells: {nonfinite:,}.")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "Sequence IDs are shuffled identifiers and are not time features. The sample",
        "spans row groups, but it is still a sample; split differences and feature",
        "correlations should be checked on held-out sequences before modeling.",
        "The validation score uses Global Weighted Pearson on `need_prediction`",
        "and `is_scored`, not these unweighted descriptive correlations.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-groups", type=int, default=64)
    parser.add_argument("--valid-groups", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    mask_file = pq.ParquetFile(ROOT / "datasets" / "valid_mask.parquet")
    mask_meta = mask_file.metadata
    mask = {
        "file": "datasets/valid_mask.parquet",
        "file_bytes": (ROOT / "datasets" / "valid_mask.parquet").stat().st_size,
        "total_rows": mask_meta.num_rows,
        "row_groups": mask_meta.num_row_groups,
        "columns": mask_file.schema_arrow.names,
        "non_20000_row_groups": sum(
            mask_meta.row_group(i).num_rows != 20_000 for i in range(mask_meta.num_row_groups)
        ),
    }
    train = scan_split(ROOT / "datasets" / "train.parquet", args.train_groups)
    valid = scan_split(ROOT / "datasets" / "valid.parquet", args.valid_groups,
                       ROOT / "datasets" / "valid_mask.parquet")
    if train["feature_count"] != 112 or valid["feature_count"] != 112:
        raise ValueError("Expected 112 feature columns in both data splits")
    if valid["total_rows"] != mask["total_rows"] or valid["row_groups"] != mask["row_groups"]:
        raise ValueError("Validation data and mask metadata disagree")

    summary = {"train": train, "valid": valid, "valid_mask": mask}
    (output / "eda.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "eda.md").write_text(markdown_report(train, valid, mask), encoding="utf-8")
    with (output / "feature_stats.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("split", "feature", "count", "nonfinite",
                                                  "zero_fraction", "mean", "std", "min", "max"))
        writer.writeheader()
        for item in (train, valid):
            for name, stats in item["features"].items():
                writer.writerow({"split": Path(item["file"]).stem, "feature": name, **stats})
    print(f"Wrote EDA to {output}")


if __name__ == "__main__":
    main()
