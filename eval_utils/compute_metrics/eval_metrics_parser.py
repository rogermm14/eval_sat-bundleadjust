#!/usr/bin/env python3
"""
Summarize eval2 outputs for one eval2 result directory.

For each AOI folder under the input directory, this script writes:

    <eval2_dir>/<AOI>/global_summary.txt

It then aggregates all AOI summaries and writes:

    <eval2_dir>/global_summary_all_aois.txt

The parser supports both regular eval2 outputs and least-similar outputs.
It computes primary held-out metrics from heldout_errors.csv when available,
secondary pairwise 3D metrics from per_track_3d_height_consistency.csv when
available, and DINO robustness summaries from the robustness CSVs.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METHODS: Sequence[Tuple[str, str, str]] = (
    ("raw", "raw_rpcs", "Raw RPCs"),
    ("ames", "ames_rpcs", "AMES"),
    ("satba", "satba_rpcs", "sat-bundleadjust"),
    ("my", "my_rpcs", "My RPCs"),
)

HELDOUT_METRICS: Sequence[str] = (
    "mean_error_px",
    "median_error_px",
    "rmse_error_px",
    "p90_error_px",
    "p95_error_px",
    "max_error_px",
    "frac_below_0p5_px",
    "frac_below_1px",
    "frac_below_2px",
)

PAIRWISE_TRACK_METRICS: Sequence[str] = (
    "dist3d_to_center_m_median",
    "dist3d_to_center_m_p90",
    "horizontal_dist_to_center_m_median",
    "horizontal_dist_to_center_m_p90",
    "alt_residual_m_mad",
    "alt_residual_m_p90",
    "pair_reproj_rmse_px_median",
    "pair_reproj_rmse_px_p90",
)

ROBUSTNESS_METRICS: Sequence[str] = (
    "mean_error_px",
    "median_error_px",
    "rmse_error_px",
    "p90_error_px",
    "p95_error_px",
    "frac_below_0p5_px",
    "frac_below_1px",
    "frac_below_2px",
)

def difficulty_metric_harder_at_higher(metric: Optional[str]) -> Optional[bool]:
    if not metric:
        return None
    metric = str(metric).strip()
    if "outlier" in metric:
        return True
    if "similarity" in metric:
        return False
    return None


def difficulty_metric_description(metric: Optional[str]) -> str:
    if not metric:
        return "unknown"

    mapping = {
        "heldout_image_outlier_score": "held-out image outlier score (higher = harder / more globally unusual within the AOI)",
        "holdout_to_train_min_similarity": "held-out to train minimum similarity (lower = harder)",
        "holdout_to_train_mean_similarity": "held-out to train mean similarity (lower = harder)",
        "holdout_to_train_max_similarity": "held-out to train maximum similarity (lower = harder)",
        "track_min_pair_similarity": "track minimum pair similarity (lower = harder)",
        "track_mean_pair_similarity": "track mean pair similarity (lower = harder)",
        "track_max_pair_similarity": "track maximum pair similarity (lower = harder)",
    }
    return mapping.get(metric, metric)


def infer_hardest_bin_idx(
    difficulty_metric: Optional[str],
    rows: Sequence[dict],
) -> Optional[int]:
    bin_indices = sorted(
        {
            to_int(row.get("similarity_bin_idx"), -1)
            for row in rows
            if to_int(row.get("similarity_bin_idx"), -1) >= 0
        }
    )
    if not bin_indices:
        return None

    harder_at_higher = difficulty_metric_harder_at_higher(difficulty_metric)
    if harder_at_higher is None:
        return None
    return bin_indices[-1] if harder_at_higher else bin_indices[0]


def read_csv_dicts(path: Path) -> List[dict]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def fmt_float(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    errors = finite_array(errors)

    if errors.size == 0:
        return {metric: math.nan for metric in HELDOUT_METRICS}

    return {
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "rmse_error_px": float(np.sqrt(np.mean(errors * errors))),
        "p90_error_px": float(np.percentile(errors, 90)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(np.max(errors)),
        "frac_below_0p5_px": float(np.mean(errors < 0.5)),
        "frac_below_1px": float(np.mean(errors < 1.0)),
        "frac_below_2px": float(np.mean(errors < 2.0)),
    }


def load_heldout_summary(method_dir: Path) -> Optional[dict]:
    rows = read_csv_dicts(method_dir / "heldout_errors.csv")
    if not rows:
        return None

    errors = finite_array(to_float(row.get("heldout_error_px")) for row in rows)
    out = {
        "n_rows": int(len(rows)),
        "n_valid": int(errors.size),
        "n_success": int(sum(str(row.get("success", "")).lower() == "true" for row in rows)),
    }
    out.update(summarize_errors(errors))
    return out


def load_pairwise_summary(aoi_dir: Path, method_dir_name: str) -> Optional[dict]:
    path = aoi_dir / "pairwise_3d" / method_dir_name / "per_track_3d_height_consistency.csv"
    rows = read_csv_dicts(path)
    if not rows:
        return None

    valid_rows = [
        row for row in rows
        if row.get("status") == "ok" and to_int(row.get("n_valid_pairwise_points")) > 0
    ]

    out = {
        "n_track_rows": int(len(rows)),
        "n_valid_track_rows": int(len(valid_rows)),
    }

    for metric in PAIRWISE_TRACK_METRICS:
        values = finite_array(to_float(row.get(metric)) for row in valid_rows)
        out[f"{metric}_mean_over_tracks"] = float(np.mean(values)) if values.size else math.nan
        out[f"{metric}_median_over_tracks"] = float(np.median(values)) if values.size else math.nan
        out[f"{metric}_p90_over_tracks"] = float(np.percentile(values, 90)) if values.size else math.nan

    return out


def load_robustness_rows(aoi_dir: Path) -> List[dict]:
    return read_csv_dicts(aoi_dir / "dino_robustness" / "summary_by_method_and_dino_bin.csv")


def load_difficulty_metric(aoi_dir: Path) -> Optional[str]:
    rows = read_csv_dicts(aoi_dir / "dino_robustness" / "heldout_errors_with_dino_difficulty.csv")
    if not rows:
        return None
    for row in rows:
        metric = row.get("difficulty_metric")
        if metric:
            return str(metric)
    return None


def discover_aoi_dirs(eval2_dir: Path) -> List[Path]:
    out = []
    for child in sorted(eval2_dir.iterdir()):
        if not child.is_dir():
            continue
        if any((child / method_dir).exists() for _, method_dir, _ in METHODS):
            out.append(child)
    return out


def summarize_aoi(aoi_dir: Path) -> dict:
    heldout = {}
    pairwise = {}

    for method_key, method_dir_name, _ in METHODS:
        method_dir = aoi_dir / method_dir_name
        heldout_summary = load_heldout_summary(method_dir)
        if heldout_summary is not None:
            heldout[method_key] = heldout_summary

        pairwise_summary = load_pairwise_summary(aoi_dir, method_dir_name)
        if pairwise_summary is not None:
            pairwise[method_key] = pairwise_summary

    robustness_rows = load_robustness_rows(aoi_dir)
    difficulty_metric = load_difficulty_metric(aoi_dir)
    hardest_bin_idx = infer_hardest_bin_idx(difficulty_metric, robustness_rows)

    summary = {
        "aoi": aoi_dir.name,
        "aoi_dir": aoi_dir,
        "heldout": heldout,
        "pairwise": pairwise,
        "robustness_rows": robustness_rows,
        "difficulty_metric": difficulty_metric,
        "hardest_bin_idx": hardest_bin_idx,
    }

    write_aoi_summary(aoi_dir / "global_summary.txt", summary)
    return summary


def write_aoi_summary(path: Path, summary: dict) -> None:
    lines: List[str] = []
    lines.append(f"Global summary for {summary['aoi']}")
    lines.append("=" * (19 + len(summary["aoi"])))
    lines.append("")
    lines.append(f"AOI directory: {summary['aoi_dir']}")
    if summary.get("difficulty_metric"):
        lines.append(f"Robustness difficulty metric: {summary['difficulty_metric']}")
        lines.append(
            "Robustness metric meaning: "
            + difficulty_metric_description(summary["difficulty_metric"])
        )
        if summary.get("hardest_bin_idx") is not None:
            lines.append(
                "Hardest within-AOI robustness bin: "
                f"{summary['hardest_bin_idx']}"
            )
    if summary.get("robustness_rows"):
        lines.append(
            "Robustness bins are computed within this AOI; bin indices are relative difficulty ranks for this AOI."
        )
    lines.append("")

    lines.append("Primary held-out reprojection error")
    lines.append("------------------------------------")
    if not summary["heldout"]:
        lines.append("No held-out metrics found.")
    else:
        header = (
            "method,n_rows,n_valid,n_success,mean,median,rmse,p90,p95,max,"
            "frac<0.5,frac<1,frac<2"
        )
        lines.append(header)
        for method_key, _, label in METHODS:
            row = summary["heldout"].get(method_key)
            if row is None:
                continue
            lines.append(
                ",".join(
                    [
                        label,
                        str(row["n_rows"]),
                        str(row["n_valid"]),
                        str(row["n_success"]),
                        fmt_float(row["mean_error_px"]),
                        fmt_float(row["median_error_px"]),
                        fmt_float(row["rmse_error_px"]),
                        fmt_float(row["p90_error_px"]),
                        fmt_float(row["p95_error_px"]),
                        fmt_float(row["max_error_px"]),
                        fmt_float(row["frac_below_0p5_px"]),
                        fmt_float(row["frac_below_1px"]),
                        fmt_float(row["frac_below_2px"]),
                    ]
                )
            )
    lines.append("")

    lines.append("Secondary pairwise 3D / height consistency")
    lines.append("-------------------------------------------")
    if not summary["pairwise"]:
        lines.append("No pairwise 3D metrics found.")
    else:
        header = (
            "method,n_valid_tracks,dist3d_median_med,dist3d_p90_med,"
            "height_mad_med,height_p90_med,pair_reproj_median_med,pair_reproj_p90_med"
        )
        lines.append(header)
        for method_key, _, label in METHODS:
            row = summary["pairwise"].get(method_key)
            if row is None:
                continue
            lines.append(
                ",".join(
                    [
                        label,
                        str(row["n_valid_track_rows"]),
                        fmt_float(row["dist3d_to_center_m_median_median_over_tracks"]),
                        fmt_float(row["dist3d_to_center_m_p90_median_over_tracks"]),
                        fmt_float(row["alt_residual_m_mad_median_over_tracks"]),
                        fmt_float(row["alt_residual_m_p90_median_over_tracks"]),
                        fmt_float(row["pair_reproj_rmse_px_median_median_over_tracks"]),
                        fmt_float(row["pair_reproj_rmse_px_p90_median_over_tracks"]),
                    ]
                )
            )
    lines.append("")

    lines.append("Robustness by DINO difficulty bin")
    lines.append("---------------------------------")
    robustness_rows = summary["robustness_rows"]
    if not robustness_rows:
        lines.append("No DINO robustness CSV found.")
    else:
        lines.append("method,bin,n,bin_value_mean,median_error,p90_error,frac<1,frac<2")
        for row in robustness_rows:
            lines.append(
                ",".join(
                    [
                        row.get("method", ""),
                        row.get("similarity_bin_idx", ""),
                        row.get("n", ""),
                        fmt_float(to_float(row.get("bin_similarity_mean"))),
                        fmt_float(to_float(row.get("median_error_px"))),
                        fmt_float(to_float(row.get("p90_error_px"))),
                        fmt_float(to_float(row.get("frac_below_1px"))),
                        fmt_float(to_float(row.get("frac_below_2px"))),
                    ]
                )
            )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_dict(rows: List[dict], keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for key in keys:
        vals = finite_array(to_float(row.get(key)) for row in rows)
        out[key] = float(np.mean(vals)) if vals.size else math.nan
    return out


def aggregate_summaries(eval2_dir: Path, summaries: List[dict]) -> None:
    lines: List[str] = []
    lines.append(f"Global summary across AOIs for {eval2_dir.name}")
    lines.append("=" * (33 + len(eval2_dir.name)))
    lines.append("")
    lines.append(f"Input directory: {eval2_dir}")
    lines.append(f"AOIs included: {len(summaries)}")
    lines.append("AOI list: " + ", ".join(summary["aoi"] for summary in summaries))
    difficulty_metrics = sorted(
        {summary.get("difficulty_metric") for summary in summaries if summary.get("difficulty_metric")}
    )
    if difficulty_metrics:
        metric_text = difficulty_metrics[0] if len(difficulty_metrics) == 1 else ", ".join(difficulty_metrics)
        lines.append(f"Robustness difficulty metric(s): {metric_text}")
        if len(difficulty_metrics) == 1:
            lines.append(
                "Robustness metric meaning: "
                + difficulty_metric_description(difficulty_metrics[0])
            )
    lines.append(
        "Robustness bins are computed separately within each AOI, so bin 0/1/... are within-AOI difficulty ranks rather than global absolute ranges."
    )
    if len(difficulty_metrics) == 1:
        hardest_direction = difficulty_metric_harder_at_higher(difficulty_metrics[0])
        if hardest_direction is True:
            lines.append("For this metric, the highest bin index is the hardest within-AOI bin.")
        elif hardest_direction is False:
            lines.append("For this metric, the lowest bin index is the hardest within-AOI bin.")
    lines.append("")

    lines.append("Average primary held-out reprojection metrics over AOIs")
    lines.append("--------------------------------------------------------")
    lines.append("method,n_aois,mean,median,rmse,p90,p95,max,frac<0.5,frac<1,frac<2")
    for method_key, _, label in METHODS:
        rows = [
            summary["heldout"][method_key]
            for summary in summaries
            if method_key in summary["heldout"]
        ]
        avg = mean_dict(rows, HELDOUT_METRICS)
        lines.append(
            ",".join(
                [
                    label,
                    str(len(rows)),
                    fmt_float(avg["mean_error_px"]),
                    fmt_float(avg["median_error_px"]),
                    fmt_float(avg["rmse_error_px"]),
                    fmt_float(avg["p90_error_px"]),
                    fmt_float(avg["p95_error_px"]),
                    fmt_float(avg["max_error_px"]),
                    fmt_float(avg["frac_below_0p5_px"]),
                    fmt_float(avg["frac_below_1px"]),
                    fmt_float(avg["frac_below_2px"]),
                ]
            )
        )
    lines.append("")

    lines.append("Average secondary pairwise 3D / height metrics over AOIs")
    lines.append("---------------------------------------------------------")
    lines.append(
        "method,n_aois,dist3d_median_med,height_mad_med,"
        "pair_reproj_median_med,dist3d_p90_med,height_p90_med"
    )
    for method_key, _, label in METHODS:
        rows = [
            summary["pairwise"][method_key]
            for summary in summaries
            if method_key in summary["pairwise"]
        ]
        keys = [
            "dist3d_to_center_m_median_median_over_tracks",
            "alt_residual_m_mad_median_over_tracks",
            "pair_reproj_rmse_px_median_median_over_tracks",
            "dist3d_to_center_m_p90_median_over_tracks",
            "alt_residual_m_p90_median_over_tracks",
        ]
        avg = mean_dict(rows, keys)
        lines.append(
            ",".join(
                [
                    label,
                    str(len(rows)),
                    fmt_float(avg[keys[0]]),
                    fmt_float(avg[keys[1]]),
                    fmt_float(avg[keys[2]]),
                    fmt_float(avg[keys[3]]),
                    fmt_float(avg[keys[4]]),
                ]
            )
        )
    lines.append("")

    lines.append("Average robustness metrics by method and bin over AOIs")
    lines.append("------------------------------------------------------")
    robust_groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for summary in summaries:
        for row in summary["robustness_rows"]:
            robust_groups[(row.get("method", ""), to_int(row.get("similarity_bin_idx"), -1))].append(row)

    if not robust_groups:
        lines.append("No DINO robustness rows found.")
    else:
        lines.append("method,bin,n_aois,mean_error,median_error,p90_error,frac<1,frac<2")
        for method, bin_idx in sorted(robust_groups):
            rows = robust_groups[(method, bin_idx)]
            avg = mean_dict(rows, ROBUSTNESS_METRICS)
            lines.append(
                ",".join(
                    [
                        method,
                        str(bin_idx),
                        str(len(rows)),
                        fmt_float(avg["mean_error_px"]),
                        fmt_float(avg["median_error_px"]),
                        fmt_float(avg["p90_error_px"]),
                        fmt_float(avg["frac_below_1px"]),
                        fmt_float(avg["frac_below_2px"]),
                    ]
                )
            )
    lines.append("")

    lines.append("Average robustness metrics in the hardest within-AOI bin over AOIs")
    lines.append("------------------------------------------------------------------")
    hardest_groups: Dict[str, List[dict]] = defaultdict(list)
    for summary in summaries:
        hardest_bin_idx = summary.get("hardest_bin_idx")
        if hardest_bin_idx is None:
            continue
        for row in summary["robustness_rows"]:
            if to_int(row.get("similarity_bin_idx"), -1) == hardest_bin_idx:
                hardest_groups[row.get("method", "")].append(row)

    if not hardest_groups:
        lines.append("No hardest-bin robustness rows found.")
    else:
        lines.append("method,n_aois,hardest_bin_mean_error,hardest_bin_median_error,hardest_bin_p90_error,hardest_bin_frac<1,hardest_bin_frac<2")
        for method, _, label in METHODS:
            rows = hardest_groups.get(method, [])
            avg = mean_dict(rows, ROBUSTNESS_METRICS)
            lines.append(
                ",".join(
                    [
                        label,
                        str(len(rows)),
                        fmt_float(avg["mean_error_px"]),
                        fmt_float(avg["median_error_px"]),
                        fmt_float(avg["p90_error_px"]),
                        fmt_float(avg["frac_below_1px"]),
                        fmt_float(avg["frac_below_2px"]),
                    ]
                )
            )
    lines.append("")

    (eval2_dir / "global_summary_all_aois.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-AOI and all-AOI summaries for an eval2 output directory."
    )
    parser.add_argument(
        "eval2_dir",
        type=Path,
        help="Input eval2 result directory, e.g. eval2/eval2_least_similar_K5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval2_dir = args.eval2_dir.resolve()

    if not eval2_dir.exists() or not eval2_dir.is_dir():
        raise FileNotFoundError(f"Input eval2 directory does not exist: {eval2_dir}")

    aoi_dirs = discover_aoi_dirs(eval2_dir)
    if not aoi_dirs:
        raise RuntimeError(f"No AOI folders found under {eval2_dir}")

    summaries = []
    for aoi_dir in aoi_dirs:
        summary = summarize_aoi(aoi_dir)
        summaries.append(summary)
        print(f"Wrote {aoi_dir / 'global_summary.txt'}")

    aggregate_summaries(eval2_dir, summaries)
    print(f"Wrote {eval2_dir / 'global_summary_all_aois.txt'}")


if __name__ == "__main__":
    main()
