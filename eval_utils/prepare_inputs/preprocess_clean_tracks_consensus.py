#!/usr/bin/env python3
"""
Clean external feature tracks using held-out consensus across raw / AMES / satBA RPCs.

For each observation in each track:
1. Hold that observation out.
2. Optimize a 3D point from the remaining observations for each method.
3. Measure held-out reprojection error under each method.
4. Keep or remove the observation based on a consensus rule across methods.

Outputs:
    output_dir/
    ├── cleaned_C.npy
    ├── cleaned_C_v2.npy                  (if source C_v2 was available)
    ├── observation_consensus_errors.csv
    ├── removed_observations.csv
    ├── removed_tracks.csv
    ├── kept_tracks.csv
    └── summary.txt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import timeit
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


METHOD_SPECS: Sequence[Tuple[str, str]] = (
    ("raw", "raw_eval_dir"),
    ("ames", "ames_eval_dir"),
    ("satba", "satba_eval_dir"),
)


def write_csv_dicts(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def import_helper_module(helper_script_dir: Path):
    helper_path = Path(helper_script_dir) / "evaluate_heldout_fixed_rpcs.py"

    if not helper_path.exists():
        raise FileNotFoundError(f"Missing helper script: {helper_path}")

    spec = importlib.util.spec_from_file_location(
        "evaluate_heldout_fixed_rpcs",
        str(helper_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper script from {helper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_heldout_fixed_rpcs"] = module
    spec.loader.exec_module(module)
    return module


def load_selected_pairs(eval_dir: Path) -> List[Tuple[int, int]]:
    candidates = [
        eval_dir / "selected_pairs.npy",
        eval_dir / "selected_least_similar_pairs.npy",
    ]

    for path in candidates:
        if path.exists():
            arr = np.load(path)
            return [(int(i), int(j)) for i, j in np.asarray(arr, dtype=np.int64)]

    raise FileNotFoundError(
        f"Could not find selected pair file in {eval_dir}. "
        "Expected selected_pairs.npy or selected_least_similar_pairs.npy."
    )


def arrays_equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, equal_nan=True))


def validate_and_load_reference_track_state(
    method_eval_dirs: Dict[str, Path],
    reference_method: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[int, int]], Path]:
    if reference_method not in method_eval_dirs:
        raise ValueError(
            f"Unknown reference method {reference_method!r}. "
            f"Expected one of {sorted(method_eval_dirs.keys())}."
        )

    reference_dir = method_eval_dirs[reference_method]

    C = np.asarray(np.load(reference_dir / "C.npy"), dtype=np.float64)
    C_v2_path = reference_dir / "C_v2.npy"
    C_v2 = np.asarray(np.load(C_v2_path), dtype=np.float64) if C_v2_path.exists() else None
    selected_pairs = load_selected_pairs(reference_dir)

    for method_name, eval_dir in method_eval_dirs.items():
        current_C = np.asarray(np.load(eval_dir / "C.npy"), dtype=np.float64)
        if current_C.shape[0] != C.shape[0]:
            raise ValueError(
                f"Image-count mismatch between reference {reference_method} and "
                f"{method_name}: {C.shape[0]} rows vs {current_C.shape[0]} rows."
            )

        current_pairs = load_selected_pairs(eval_dir)
        if len(current_pairs) != len(selected_pairs):
            raise ValueError(
                f"Selected pair-count mismatch between reference {reference_method} "
                f"and {method_name}: {len(selected_pairs)} vs {len(current_pairs)}."
            )

    return C, C_v2, selected_pairs, reference_dir


def finite_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.median(arr))


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def summarize_kept_counts(mask: np.ndarray) -> Tuple[int, int]:
    kept = int(np.sum(mask))
    removed = int(mask.size - kept)
    return kept, removed


def proxy_reprojection_error(
    helper,
    point_lon_lat_alt: np.ndarray,
    observation: Tuple[int, float, float],
    rpcs: List,
) -> Tuple[float, str]:
    image_idx, observed_col, observed_row = observation

    if not np.all(np.isfinite(point_lon_lat_alt)):
        return np.nan, "invalid_full_track_point"

    try:
        projected_col, projected_row = helper.project_rpc_safe(
            rpcs[int(image_idx)],
            float(point_lon_lat_alt[0]),
            float(point_lon_lat_alt[1]),
            float(point_lon_lat_alt[2]),
        )
    except Exception:
        return np.nan, "projection_failed"

    dx = float(projected_col) - float(observed_col)
    dy = float(projected_row) - float(observed_row)
    err = float(np.sqrt(dx * dx + dy * dy))
    return err, "proxy_projected"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_eval_dir", type=Path, required=True)
    parser.add_argument("--ames_eval_dir", type=Path, required=True)
    parser.add_argument("--satba_eval_dir", type=Path, required=True)
    parser.add_argument("--raw_rpc_dir", type=Path, required=True)
    parser.add_argument("--ames_rpc_dir", type=Path, required=True)
    parser.add_argument("--satba_rpc_dir", type=Path, required=True)
    parser.add_argument("--matches_dir", type=Path, required=True)
    parser.add_argument("--sat_bundleadjust_repo", type=Path, required=True)
    parser.add_argument("--helper_script_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_method",
        type=str,
        default="raw",
        choices=["raw", "ames", "satba"],
        help=(
            "Method whose C.npy and selected pairs define the reference external "
            "track set to clean. All methods are still used for consensus scoring."
        ),
    )
    parser.add_argument("--max_observation_error_px", type=float, default=3.0)
    parser.add_argument(
        "--consensus_rule",
        type=str,
        default="at_least_two",
        choices=["all_three", "at_least_two", "median"],
        help=(
            "Consensus rule for keeping an observation. "
            "'all_three': all methods <= threshold; "
            "'at_least_two': at least two methods <= threshold; "
            "'median': median method error <= threshold."
        ),
    )
    parser.add_argument(
        "--min_methods_with_valid_error",
        type=int,
        default=2,
        help="Minimum number of methods that must produce finite held-out errors.",
    )
    parser.add_argument(
        "--min_track_length_after_cleaning",
        type=int,
        default=3,
        help="Tracks shorter than this after observation pruning are removed.",
    )
    parser.add_argument(
        "--max_removed_fraction_per_track",
        type=float,
        default=0.33,
        help="Remove tracks if more than this fraction of observations are pruned.",
    )
    parser.add_argument("--lon_lat_scale_deg", type=float, default=1e-5)
    parser.add_argument("--alt_scale_m", type=float, default=10.0)
    parser.add_argument("--max_lon_lat_step_deg", type=float, default=0.01)
    parser.add_argument("--max_alt_step_m", type=float, default=1000.0)
    parser.add_argument("--loss", type=str, default="linear")
    parser.add_argument("--f_scale", type=float, default=1.0)
    parser.add_argument("--max_nfev", type=int, default=30)
    parser.add_argument("--projection_failure_penalty_px", type=float, default=1e6)
    parser.add_argument("--verbose_triangulation", action="store_true")
    parser.add_argument(
        "--cleaning_mode",
        type=str,
        default="fast_proxy",
        choices=["fast_proxy", "slow_strict_loo"],
        help=(
            "Cleaning mode. "
            "'fast_proxy' uses one full-track 3D point per method and scores each "
            "observation by reprojection error, which is much faster. "
            "'slow_strict_loo' runs full per-observation strict leave-one-out."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    helper = import_helper_module(args.helper_script_dir)
    (
        cam_utils,
        geo_utils,
        ft_match,
        select_optimal_pairs_to_match,
        init_pts3d,
    ) = helper.import_sat_bundleadjust(args.sat_bundleadjust_repo)

    method_eval_dirs = {
        "raw": args.raw_eval_dir.resolve(),
        "ames": args.ames_eval_dir.resolve(),
        "satba": args.satba_eval_dir.resolve(),
    }
    method_rpc_dirs = {
        "raw": args.raw_rpc_dir.resolve(),
        "ames": args.ames_rpc_dir.resolve(),
        "satba": args.satba_rpc_dir.resolve(),
    }

    C, C_v2, selected_pairs, reference_eval_dir = validate_and_load_reference_track_state(
        method_eval_dirs=method_eval_dirs,
        reference_method=args.reference_method,
    )
    image_paths = helper.load_image_paths(args.matches_dir)

    method_rpcs: Dict[str, List] = {}
    method_init_points: Dict[str, np.ndarray] = {}

    print("Loading RPCs and full-track initial points for each method...")
    for method_name in method_eval_dirs:
        rpcs = helper.load_rpcs_for_images(method_rpc_dirs[method_name], image_paths)
        method_rpcs[method_name] = rpcs
        method_init_points[method_name] = helper.triangulate_C_to_lon_lat_alt(
            C=C,
            rpcs=rpcs,
            pairs_to_triangulate=selected_pairs,
            init_pts3d=init_pts3d,
            geo_utils=geo_utils,
            verbose=args.verbose_triangulation,
        )

    n_tracks = C.shape[1]
    n_images = C.shape[0] // 2

    cleaned_C = np.full_like(C, np.nan)
    cleaned_C_v2 = np.full_like(C_v2, np.nan) if C_v2 is not None else None

    observation_rows: List[dict] = []
    removed_observation_rows: List[dict] = []
    removed_track_rows: List[dict] = []
    kept_track_rows: List[dict] = []

    t0 = timeit.default_timer()

    for track_idx in range(n_tracks):
        obs = helper.observations_from_C_column(C, track_idx)
        track_length = len(obs)
        keep_mask = np.zeros(track_length, dtype=bool)

        for holdout_local_idx, holdout_obs in enumerate(obs):
            train_obs = [
                current_obs
                for k, current_obs in enumerate(obs)
                if k != holdout_local_idx
            ]

            row = {
                "track_idx": int(track_idx),
                "track_length": int(track_length),
                "holdout_local_idx": int(holdout_local_idx),
                "holdout_image_idx": int(holdout_obs[0]),
                "holdout_col": float(holdout_obs[1]),
                "holdout_row": float(holdout_obs[2]),
            }

            method_errors = []
            valid_method_count = 0
            passing_methods = 0

            for method_name in ("raw", "ames", "satba"):
                if args.cleaning_mode == "fast_proxy":
                    point = method_init_points[method_name][track_idx]
                    err, status = proxy_reprojection_error(
                        helper=helper,
                        point_lon_lat_alt=point,
                        observation=holdout_obs,
                        rpcs=method_rpcs[method_name],
                    )
                else:
                    if len(train_obs) < 2:
                        row[f"{method_name}_heldout_error_px"] = np.nan
                        row[f"{method_name}_status"] = "insufficient_train_observations"
                        continue

                    point0 = helper.initial_point_for_holdout(
                        C=C,
                        track_idx=track_idx,
                        train_obs=train_obs,
                        full_track_initial_points_lon_lat_alt=method_init_points[method_name],
                        rpcs=method_rpcs[method_name],
                        selected_pairs=selected_pairs,
                        init_pts3d=init_pts3d,
                        geo_utils=geo_utils,
                        strict_loo_initialization=True,
                        verbose_triangulation=args.verbose_triangulation,
                    )

                    refined_point, opt_stats = helper.optimize_point_from_training_observations(
                        point0_lon_lat_alt=point0,
                        train_obs=train_obs,
                        rpcs=method_rpcs[method_name],
                        lon_lat_scale_deg=args.lon_lat_scale_deg,
                        alt_scale_m=args.alt_scale_m,
                        max_lon_lat_step_deg=args.max_lon_lat_step_deg,
                        max_alt_step_m=args.max_alt_step_m,
                        loss=args.loss,
                        f_scale=args.f_scale,
                        max_nfev=args.max_nfev,
                        projection_failure_penalty_px=args.projection_failure_penalty_px,
                    )

                    heldout_row = helper.build_heldout_row(
                        track_idx=track_idx,
                        track_length=track_length,
                        holdout_obs=holdout_obs,
                        train_obs=train_obs,
                        point0=point0,
                        refined_point=refined_point,
                        opt_stats=opt_stats,
                        rpcs=method_rpcs[method_name],
                    )

                    err = (
                        float(heldout_row["heldout_error_px"])
                        if np.isfinite(heldout_row["heldout_error_px"])
                        else np.nan
                    )
                    status = heldout_row["status"]

                row[f"{method_name}_heldout_error_px"] = err
                row[f"{method_name}_status"] = status
                method_errors.append(err)

                if np.isfinite(err):
                    valid_method_count += 1
                    if err <= args.max_observation_error_px:
                        passing_methods += 1

            median_error = finite_median(method_errors)
            mean_error = finite_mean(method_errors)

            if valid_method_count < args.min_methods_with_valid_error:
                keep = False
                decision_reason = "insufficient_valid_methods"
            elif args.consensus_rule == "all_three":
                keep = passing_methods == 3
                decision_reason = "all_three_below_threshold" if keep else "failed_all_three_rule"
            elif args.consensus_rule == "at_least_two":
                keep = passing_methods >= 2
                decision_reason = "at_least_two_below_threshold" if keep else "failed_at_least_two_rule"
            else:
                keep = bool(np.isfinite(median_error) and median_error <= args.max_observation_error_px)
                decision_reason = "median_below_threshold" if keep else "failed_median_rule"

            keep_mask[holdout_local_idx] = keep

            row["n_methods_with_valid_error"] = int(valid_method_count)
            row["n_methods_passing_threshold"] = int(passing_methods)
            row["consensus_mean_error_px"] = mean_error
            row["consensus_median_error_px"] = median_error
            row["keep_observation"] = bool(keep)
            row["decision_reason"] = decision_reason
            observation_rows.append(row)

            if not keep:
                removed_observation_rows.append(row)

        kept_count, removed_count = summarize_kept_counts(keep_mask)
        removed_fraction = float(removed_count / track_length) if track_length > 0 else 1.0

        if kept_count < args.min_track_length_after_cleaning:
            removed_track_rows.append(
                {
                    "track_idx": int(track_idx),
                    "original_track_length": int(track_length),
                    "kept_observations": int(kept_count),
                    "removed_observations": int(removed_count),
                    "removed_fraction": removed_fraction,
                    "reason": "too_short_after_cleaning",
                }
            )
            continue

        if removed_fraction > args.max_removed_fraction_per_track:
            removed_track_rows.append(
                {
                    "track_idx": int(track_idx),
                    "original_track_length": int(track_length),
                    "kept_observations": int(kept_count),
                    "removed_observations": int(removed_count),
                    "removed_fraction": removed_fraction,
                    "reason": "removed_fraction_above_limit",
                }
            )
            continue

        kept_track_rows.append(
            {
                "track_idx": int(track_idx),
                "original_track_length": int(track_length),
                "kept_observations": int(kept_count),
                "removed_observations": int(removed_count),
                "removed_fraction": removed_fraction,
            }
        )

        kept_obs = [obs[k] for k in range(track_length) if keep_mask[k]]
        for image_idx, col, row in kept_obs:
            cleaned_C[2 * image_idx, track_idx] = col
            cleaned_C[2 * image_idx + 1, track_idx] = row
            if cleaned_C_v2 is not None and C_v2 is not None:
                cleaned_C_v2[image_idx, track_idx] = C_v2[image_idx, track_idx]

        if track_idx % 100 == 0:
            elapsed = timeit.default_timer() - t0
            print(
                f"[consensus-clean] track {track_idx}/{n_tracks}, "
                f"kept_tracks={len(kept_track_rows)}, removed_tracks={len(removed_track_rows)}, "
                f"obs_rows={len(observation_rows)}, elapsed={elapsed:.1f}s"
            )

    keep_track_mask = np.any(np.isfinite(cleaned_C[::2]), axis=0)
    cleaned_C = cleaned_C[:, keep_track_mask]
    if cleaned_C_v2 is not None:
        cleaned_C_v2 = cleaned_C_v2[:, keep_track_mask]

    np.save(output_dir / "cleaned_C.npy", cleaned_C)
    if cleaned_C_v2 is not None:
        np.save(output_dir / "cleaned_C_v2.npy", cleaned_C_v2)

    observation_fieldnames = [
        "track_idx",
        "track_length",
        "holdout_local_idx",
        "holdout_image_idx",
        "holdout_col",
        "holdout_row",
        "raw_heldout_error_px",
        "ames_heldout_error_px",
        "satba_heldout_error_px",
        "raw_status",
        "ames_status",
        "satba_status",
        "n_methods_with_valid_error",
        "n_methods_passing_threshold",
        "consensus_mean_error_px",
        "consensus_median_error_px",
        "keep_observation",
        "decision_reason",
    ]
    track_fieldnames = [
        "track_idx",
        "original_track_length",
        "kept_observations",
        "removed_observations",
        "removed_fraction",
        "reason",
    ]
    kept_track_fieldnames = [
        "track_idx",
        "original_track_length",
        "kept_observations",
        "removed_observations",
        "removed_fraction",
    ]

    write_csv_dicts(output_dir / "observation_consensus_errors.csv", observation_rows, observation_fieldnames)
    write_csv_dicts(output_dir / "removed_observations.csv", removed_observation_rows, observation_fieldnames)
    write_csv_dicts(output_dir / "removed_tracks.csv", removed_track_rows, track_fieldnames)
    write_csv_dicts(output_dir / "kept_tracks.csv", kept_track_rows, kept_track_fieldnames)

    original_obs = int(np.sum(np.isfinite(C[::2])))
    cleaned_obs = int(np.sum(np.isfinite(cleaned_C[::2])))
    original_tracks = int(C.shape[1])
    cleaned_tracks = int(cleaned_C.shape[1])

    lines = [
        "Consensus external-track cleaning summary",
        "========================================",
        "",
        f"reference_eval_dir: {reference_eval_dir}",
        f"matches_dir: {args.matches_dir.resolve()}",
        f"sat_bundleadjust_repo: {args.sat_bundleadjust_repo.resolve()}",
        "",
        "Method eval dirs:",
        f"  raw:   {method_eval_dirs['raw']}",
        f"  ames:  {method_eval_dirs['ames']}",
        f"  satba: {method_eval_dirs['satba']}",
        "",
        "Cleaning settings:",
        f"  reference_method: {args.reference_method}",
        f"  cleaning_mode: {args.cleaning_mode}",
        f"  max_observation_error_px: {args.max_observation_error_px}",
        f"  consensus_rule: {args.consensus_rule}",
        f"  min_methods_with_valid_error: {args.min_methods_with_valid_error}",
        f"  min_track_length_after_cleaning: {args.min_track_length_after_cleaning}",
        f"  max_removed_fraction_per_track: {args.max_removed_fraction_per_track}",
        f"  strict_loo_initialization: {args.cleaning_mode == 'slow_strict_loo'}",
        f"  full_track_proxy_points: {args.cleaning_mode == 'fast_proxy'}",
        "",
        "Counts:",
        f"  original tracks: {original_tracks}",
        f"  cleaned tracks: {cleaned_tracks}",
        f"  removed tracks: {original_tracks - cleaned_tracks}",
        f"  original observations: {original_obs}",
        f"  cleaned observations: {cleaned_obs}",
        f"  removed observations: {original_obs - cleaned_obs}",
        "",
        "Outputs:",
        f"  {output_dir / 'cleaned_C.npy'}",
        f"  {output_dir / 'observation_consensus_errors.csv'}",
        f"  {output_dir / 'removed_observations.csv'}",
        f"  {output_dir / 'removed_tracks.csv'}",
        f"  {output_dir / 'kept_tracks.csv'}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print((output_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
