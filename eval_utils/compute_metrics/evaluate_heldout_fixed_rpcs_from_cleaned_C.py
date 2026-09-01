#!/usr/bin/env python3
"""
Held-out evaluation for fixed RPCs using a precomputed cleaned C matrix.

This script reuses the held-out evaluation logic from evaluate_heldout_fixed_rpcs.py
but skips feature-track construction. Instead, it loads:

  - cleaned_C.npy from a cleaned-track preprocessing output directory
  - selected_pairs.npy (or selected_least_similar_pairs.npy) from a reference
    eval directory such as eval2_least_similar_K5/<AOI>/raw_rpcs

The output format is intentionally compatible with the downstream eval2 scripts
such as analyze_heldout_by_dino_similarity.py and
evaluate_pairwise_3d_height_consistency.py.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


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


def load_selected_pairs(source_dir: Path) -> Tuple[List[Tuple[int, int]], str]:
    candidates = [
        ("selected_pairs.npy", "selected_pairs"),
        ("selected_least_similar_pairs.npy", "selected_least_similar_pairs"),
    ]

    for filename, label in candidates:
        path = source_dir / filename
        if path.exists():
            arr = np.asarray(np.load(path), dtype=np.int64)
            pairs = [(int(i), int(j)) for i, j in arr]
            return pairs, label

    raise FileNotFoundError(
        f"Could not find selected pair file in {source_dir}. "
        "Expected selected_pairs.npy or selected_least_similar_pairs.npy."
    )


def write_pairs_txt(path: Path, pairs: List[Tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, j in pairs:
            f.write(f"{int(i)} {int(j)}\n")


def copy_optional_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_summary_txt(
    path: Path,
    rows: List[dict],
    n_images: int,
    n_tracks: int,
    selected_pairs_count: int,
    min_train_observations: int,
    strict_loo_initialization: bool,
    all_holdouts: bool,
    holdout_seed: int,
    cleaned_tracks_dir: Path,
    selected_pairs_source_dir: Path,
) -> None:
    valid_err = []
    for row in rows:
        err = row.get("heldout_error_px", np.nan)
        if np.isfinite(err):
            valid_err.append(float(err))

    valid_err = np.asarray(valid_err, dtype=np.float64)
    n_success = sum(1 for row in rows if row["success"])
    n_valid = int(valid_err.size)

    lines = []
    lines.append("Held-out fixed-RPC evaluation summary using cleaned external tracks")
    lines.append("==============================================================")
    lines.append("")
    lines.append("Metric interpretation:")
    lines.append("  The cleaned external track matrix is treated as fixed input.")
    lines.append("  For each held-out image observation, one 3D point is optimized using")
    lines.append("  the other observations in that track. RPCs are fixed.")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  cleaned_tracks_dir: {cleaned_tracks_dir}")
    lines.append(f"  selected_pairs_source_dir: {selected_pairs_source_dir}")
    lines.append(f"  images: {n_images}")
    lines.append(f"  tracks after filtering: {n_tracks}")
    lines.append("")
    lines.append("Pair selection:")
    lines.append(f"  selected pairs used: {selected_pairs_count}")
    lines.append("")
    lines.append("Track / held-out settings:")
    lines.append(f"  min_train_observations: {min_train_observations}")
    lines.append(f"  required total track length: >= {min_train_observations + 1}")
    lines.append(f"  all_holdouts: {all_holdouts}")
    lines.append(f"  holdout_seed: {holdout_seed}")
    lines.append(f"  strict_loo_initialization: {strict_loo_initialization}")
    lines.append("")
    lines.append("Held-out optimization:")
    lines.append(f"  held-out rows attempted: {len(rows)}")
    lines.append(f"  optimizer successes: {n_success}")
    lines.append(f"  valid held-out projections: {n_valid}")
    lines.append("")

    if valid_err.size == 0:
        lines.append("No valid held-out errors.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("Global held-out reprojection error, pixels:")
    lines.append(f"  mean:   {np.mean(valid_err):.6f}")
    lines.append(f"  median: {np.median(valid_err):.6f}")
    lines.append(f"  rmse:   {np.sqrt(np.mean(valid_err * valid_err)):.6f}")
    lines.append(f"  p90:    {np.percentile(valid_err, 90):.6f}")
    lines.append(f"  p95:    {np.percentile(valid_err, 95):.6f}")
    lines.append(f"  max:    {np.max(valid_err):.6f}")
    lines.append(f"  frac < 0.5 px: {np.mean(valid_err < 0.5):.6f}")
    lines.append(f"  frac < 1.0 px: {np.mean(valid_err < 1.0):.6f}")
    lines.append(f"  frac < 2.0 px: {np.mean(valid_err < 2.0):.6f}")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_tracks_dir", type=Path, required=True)
    parser.add_argument("--selected_pairs_source_dir", type=Path, required=True)
    parser.add_argument("--matches_dir", type=Path, required=True)
    parser.add_argument("--rpc_dir", type=Path, required=True)
    parser.add_argument("--sat_bundleadjust_repo", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--helper_script_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--min_train_observations", type=int, default=3)
    parser.add_argument("--min_track_length", type=int, default=None)
    parser.add_argument("--all_holdouts", action="store_true")
    parser.add_argument("--holdout_seed", type=int, default=0)
    parser.add_argument("--verbose_triangulation", action="store_true")
    parser.add_argument("--lon_lat_scale_deg", type=float, default=1e-5)
    parser.add_argument("--alt_scale_m", type=float, default=10.0)
    parser.add_argument("--max_lon_lat_step_deg", type=float, default=0.01)
    parser.add_argument("--max_alt_step_m", type=float, default=1000.0)
    parser.add_argument(
        "--loss",
        type=str,
        default="linear",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
    )
    parser.add_argument("--f_scale", type=float, default=1.0)
    parser.add_argument("--max_nfev", type=int, default=30)
    parser.add_argument("--projection_failure_penalty_px", type=float, default=1e6)
    parser.add_argument("--strict_loo_initialization", action="store_true")
    parser.add_argument("--no_log_redirect", action="store_true")
    return parser.parse_args()


def run_main(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.min_track_length is not None:
        min_train_observations = int(args.min_track_length)
        print(
            "[warning] --min_track_length is deprecated. "
            "Using it as --min_train_observations."
        )
    else:
        min_train_observations = int(args.min_train_observations)

    helper = import_helper_module(args.helper_script_dir)
    (
        cam_utils,
        geo_utils,
        ft_match,
        select_optimal_pairs_to_match,
        init_pts3d,
    ) = helper.import_sat_bundleadjust(args.sat_bundleadjust_repo)

    print("Running held-out fixed-RPC evaluation from cleaned C matrix")
    print("----------------------------------------------------------")
    print(f"cleaned_tracks_dir: {args.cleaned_tracks_dir}")
    print(f"selected_pairs_source_dir: {args.selected_pairs_source_dir}")
    print(f"matches_dir: {args.matches_dir}")
    print(f"rpc_dir: {args.rpc_dir}")
    print(f"output_dir: {output_dir}")
    print("")
    print("Configuration:")
    print("  RPCs fixed: True")
    print("  optimized variable per test: one lon/lat/alt point")
    print("  input track source: cleaned_C.npy")
    print(f"  min_train_observations: {min_train_observations}")
    print(f"  all_holdouts: {args.all_holdouts}")
    print(f"  holdout_seed: {args.holdout_seed}")
    print(f"  strict_loo_initialization: {args.strict_loo_initialization}")
    print("----------------------------------------------------------")
    print("")

    image_paths = helper.load_image_paths(args.matches_dir)
    rpcs = helper.load_rpcs_for_images(args.rpc_dir, image_paths)

    cleaned_C_path = args.cleaned_tracks_dir / "cleaned_C.npy"
    if not cleaned_C_path.exists():
        raise FileNotFoundError(f"Missing cleaned_C.npy: {cleaned_C_path}")

    C = np.asarray(np.load(cleaned_C_path), dtype=np.float64)
    if C.shape[0] != 2 * len(image_paths):
        raise ValueError(
            f"Image count mismatch: cleaned_C.npy has {C.shape[0] // 2} images, "
            f"but image_paths.txt has {len(image_paths)}."
        )

    cleaned_C_v2_path = args.cleaned_tracks_dir / "cleaned_C_v2.npy"
    C_v2 = np.asarray(np.load(cleaned_C_v2_path), dtype=np.float64) if cleaned_C_v2_path.exists() else None

    selected_pairs, selected_pairs_label = load_selected_pairs(args.selected_pairs_source_dir)

    min_total_length = min_train_observations + 1
    if C_v2 is None:
        C_v2 = np.full((C.shape[0] // 2, C.shape[1]), np.nan, dtype=np.float64)

    # Remove tracks that are too short for held-out evaluation.
    C, C_v2 = helper.filter_C_by_length_range(
        C=C,
        C_v2=C_v2,
        min_total_length=min_total_length,
        max_total_length=None,
    )

    # Remove tracks that no longer contain any valid selected pair after cleaning.
    tracks_to_preserve = helper.filter_C_using_pairs_to_triangulate_local(
        C=C,
        pairs_to_triangulate=selected_pairs,
    )
    C = C[:, tracks_to_preserve]
    C_v2 = C_v2[:, tracks_to_preserve]

    if C.shape[1] == 0:
        raise RuntimeError("No tracks left after cleaned-track filtering.")

    print(f"C shape: {C.shape}")
    print(f"C_v2 shape: {C_v2.shape}")
    helper.print_track_length_stats(C)

    np.save(output_dir / "C.npy", C)
    np.save(output_dir / "C_v2.npy", C_v2)
    np.save(output_dir / "selected_pairs.npy", np.asarray(selected_pairs, dtype=np.int64))
    write_pairs_txt(output_dir / "selected_pairs.txt", selected_pairs)

    if selected_pairs_label == "selected_least_similar_pairs":
        np.save(output_dir / "selected_least_similar_pairs.npy", np.asarray(selected_pairs, dtype=np.int64))
        write_pairs_txt(output_dir / "selected_least_similar_pairs.txt", selected_pairs)

    copy_optional_file(
        args.selected_pairs_source_dir / "dino_pair_scores.csv",
        output_dir / "dino_pair_scores.csv",
    )

    holdout_local_indices = helper.choose_holdout_indices_for_tracks(
        C=C,
        seed=args.holdout_seed,
    )
    np.save(output_dir / "holdout_local_indices.npy", holdout_local_indices)
    helper.write_holdout_selection_csv(output_dir / "holdout_selection.csv", C, holdout_local_indices)

    print("\nComputing full-track initial points for optimizer initialization...")
    full_track_initial_points_lon_lat_alt = helper.triangulate_C_to_lon_lat_alt(
        C=C,
        rpcs=rpcs,
        pairs_to_triangulate=selected_pairs,
        init_pts3d=init_pts3d,
        geo_utils=geo_utils,
        verbose=args.verbose_triangulation,
    )
    np.save(output_dir / "full_track_initial_points_lon_lat_alt.npy", full_track_initial_points_lon_lat_alt)

    if args.all_holdouts:
        heldout_rows = helper.run_heldout_evaluation_all_holdouts(
            C=C,
            rpcs=rpcs,
            selected_pairs=selected_pairs,
            init_pts3d=init_pts3d,
            geo_utils=geo_utils,
            min_train_observations=min_train_observations,
            full_track_initial_points_lon_lat_alt=full_track_initial_points_lon_lat_alt,
            strict_loo_initialization=args.strict_loo_initialization,
            verbose_triangulation=args.verbose_triangulation,
            lon_lat_scale_deg=args.lon_lat_scale_deg,
            alt_scale_m=args.alt_scale_m,
            max_lon_lat_step_deg=args.max_lon_lat_step_deg,
            max_alt_step_m=args.max_alt_step_m,
            loss=args.loss,
            f_scale=args.f_scale,
            max_nfev=args.max_nfev,
            projection_failure_penalty_px=args.projection_failure_penalty_px,
        )
    else:
        heldout_rows = helper.run_heldout_evaluation_one_per_track(
            C=C,
            rpcs=rpcs,
            selected_pairs=selected_pairs,
            init_pts3d=init_pts3d,
            geo_utils=geo_utils,
            min_train_observations=min_train_observations,
            full_track_initial_points_lon_lat_alt=full_track_initial_points_lon_lat_alt,
            strict_loo_initialization=args.strict_loo_initialization,
            verbose_triangulation=args.verbose_triangulation,
            lon_lat_scale_deg=args.lon_lat_scale_deg,
            alt_scale_m=args.alt_scale_m,
            max_lon_lat_step_deg=args.max_lon_lat_step_deg,
            max_alt_step_m=args.max_alt_step_m,
            loss=args.loss,
            f_scale=args.f_scale,
            max_nfev=args.max_nfev,
            projection_failure_penalty_px=args.projection_failure_penalty_px,
            holdout_local_indices=holdout_local_indices,
        )

    helper.write_heldout_errors_csv(output_dir / "heldout_errors.csv", heldout_rows)
    np.save(output_dir / "heldout_errors.npy", helper.rows_to_numpy(heldout_rows))
    helper.write_per_camera_heldout_stats_csv(
        output_dir / "per_camera_heldout_error_stats.csv",
        heldout_rows,
        image_paths,
    )
    helper.write_heldout_error_by_track_length_csv(
        output_dir / "heldout_error_by_track_length.csv",
        heldout_rows,
    )
    write_summary_txt(
        output_dir / "summary.txt",
        heldout_rows,
        n_images=len(image_paths),
        n_tracks=C.shape[1],
        selected_pairs_count=len(selected_pairs),
        min_train_observations=min_train_observations,
        strict_loo_initialization=args.strict_loo_initialization,
        all_holdouts=args.all_holdouts,
        holdout_seed=args.holdout_seed,
        cleaned_tracks_dir=args.cleaned_tracks_dir.resolve(),
        selected_pairs_source_dir=args.selected_pairs_source_dir.resolve(),
    )

    print("\nSaved outputs:")
    print(f"  {output_dir / 'bundle_adjust.log'}")
    print(f"  {output_dir / 'summary.txt'}")
    print(f"  {output_dir / 'heldout_errors.csv'}")
    print(f"  {output_dir / 'heldout_errors.npy'}")
    print(f"  {output_dir / 'per_camera_heldout_error_stats.csv'}")
    print(f"  {output_dir / 'heldout_error_by_track_length.csv'}")
    print(f"  {output_dir / 'C.npy'}")
    print(f"  {output_dir / 'C_v2.npy'}")

    print("\nSummary:")
    print((output_dir / "summary.txt").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "bundle_adjust.log"

    if args.no_log_redirect:
        run_main(args)
        return

    print("Running held-out fixed-RPC evaluation from cleaned C matrix ...")
    print(f"Path to log file: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            run_main(args)

    print("... done !")
    print(f"Path to output files: {output_dir}")
    print(f"Path to log file: {log_path}")


if __name__ == "__main__":
    main()
