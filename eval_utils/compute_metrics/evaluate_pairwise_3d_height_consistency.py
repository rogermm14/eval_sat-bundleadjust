#!/usr/bin/env python3
"""
Pairwise 3D and height consistency evaluation for fixed RPC sets.

Save as:

    evaluate_pairwise_3d_height_consistency.py

Purpose:

    Compute a secondary geometric consistency metric using independent external
    feature tracks.

    For each feature track:
        1. Find all image observations in the track.
        2. For every valid image pair in the track:
              triangulate a 3D point using only that pair.
        3. Measure dispersion among all pairwise 3D points:
              - 3D ECEF distance to robust center
              - horizontal distance to robust center
              - height / altitude dispersion
        4. Optionally project each pairwise 3D point back into the two images
           used to triangulate it and record pair reprojection RMSE.

Interpretation:

    If corrected RPCs are mutually consistent, then different stereo pairs
    observing the same external track should triangulate similar 3D points.

    This is especially useful for RPC satellite imagery because height
    inconsistency often reveals residual camera model inconsistency.

Expected inputs:

    output_dir from evaluate_heldout_fixed_rpcs.py:

        eval_output_dir/
        ├── C.npy
        ├── selected_pairs.npy
        └── image_paths.txt may not exist here

    matches_dir from the original feature matching directory:

        matches_dir/
        └── image_paths.txt

    RPC directory for the method being evaluated:

        --rpc_dir /path/to/raw_or_ames_or_satba_rpcs

    sat-bundleadjust repository:

        --sat_bundleadjust_repo /path/to/sat-bundleadjust

Example:

    python evaluate_pairwise_3d_height_consistency.py \
        --eval_output_dir /path/to/ames_eval \
        --matches_dir /path/to/superpoint_lightglue_matches \
        --rpc_dir /path/to/ames_rpcs \
        --sat_bundleadjust_repo /home/roger/sat-bundleadjust \
        --output_dir /path/to/ames_pairwise_3d \
        --min_track_length 3 \
        --max_pairs_per_track 100

Outputs:

    output_dir/
    ├── pairwise_3d_points.csv
    ├── per_track_3d_height_consistency.csv
    ├── summary_by_track_length.csv
    └── summary.txt

Notes:

    This script imports helper functions from evaluate_heldout_fixed_rpcs.py.
    Put this script in the same directory as evaluate_heldout_fixed_rpcs.py,
    or pass --helper_script_dir pointing to that directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import timeit
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rpcm


# -----------------------------------------------------------------------------
# Import helper script
# -----------------------------------------------------------------------------


def import_helpers(helper_script_dir: Path):
    helper_path = Path(helper_script_dir) / "evaluate_heldout_fixed_rpcs.py"

    if not helper_path.exists():
        raise FileNotFoundError(
            f"Could not find evaluate_heldout_fixed_rpcs.py at {helper_path}"
        )

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


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------


def write_csv_dicts(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# -----------------------------------------------------------------------------
# WGS84 / ECEF helpers
# -----------------------------------------------------------------------------


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lon_lat_alt_to_ecef(lon_deg: float, lat_deg: float, alt_m: float) -> np.ndarray:
    lon = math.radians(float(lon_deg))
    lat = math.radians(float(lat_deg))
    alt = float(alt_m)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat

    return np.asarray([x, y, z], dtype=np.float64)


def robust_center_xyz(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    valid = np.all(np.isfinite(points_xyz), axis=1)

    if not np.any(valid):
        return np.full(3, np.nan, dtype=np.float64)

    return np.median(points_xyz[valid], axis=0)


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def finite_stats(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_mad": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_mad": mad(values),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


# -----------------------------------------------------------------------------
# Track and pair helpers
# -----------------------------------------------------------------------------


def observations_from_C_column(C: np.ndarray, track_idx: int) -> List[Tuple[int, float, float]]:
    n_images = C.shape[0] // 2
    obs: List[Tuple[int, float, float]] = []

    for image_idx in range(n_images):
        col = C[2 * image_idx, track_idx]
        row = C[2 * image_idx + 1, track_idx]

        if np.isfinite(col) and np.isfinite(row):
            obs.append((image_idx, float(col), float(row)))

    return obs


def all_pairs_from_observations(obs: List[Tuple[int, float, float]]) -> List[Tuple[int, int]]:
    image_indices = sorted(int(o[0]) for o in obs)
    return [(i, j) for i, j in combinations(image_indices, 2)]


def filter_pairs_by_allowed_pairs(
    pairs: List[Tuple[int, int]],
    allowed_pairs: Optional[List[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    if allowed_pairs is None:
        return pairs

    allowed_set = set((min(int(i), int(j)), max(int(i), int(j))) for i, j in allowed_pairs)

    return [
        (i, j)
        for i, j in pairs
        if (min(int(i), int(j)), max(int(i), int(j))) in allowed_set
    ]


def subsample_pairs_deterministic(
    pairs: List[Tuple[int, int]],
    max_pairs_per_track: Optional[int],
    seed: int,
    track_idx: int,
) -> List[Tuple[int, int]]:
    if max_pairs_per_track is None:
        return pairs

    if len(pairs) <= max_pairs_per_track:
        return pairs

    rng = np.random.default_rng(seed + int(track_idx))
    idx = rng.choice(len(pairs), size=max_pairs_per_track, replace=False)
    idx = sorted(int(i) for i in idx)

    return [pairs[i] for i in idx]


def make_two_observation_C(
    n_images: int,
    obs_by_image: dict,
    i: int,
    j: int,
) -> np.ndarray:
    C_pair = np.full((2 * n_images, 1), np.nan, dtype=np.float64)

    col_i, row_i = obs_by_image[int(i)]
    col_j, row_j = obs_by_image[int(j)]

    C_pair[2 * int(i), 0] = float(col_i)
    C_pair[2 * int(i) + 1, 0] = float(row_i)

    C_pair[2 * int(j), 0] = float(col_j)
    C_pair[2 * int(j) + 1, 0] = float(row_j)

    return C_pair


def project_rpc_safe(
    rpc: rpcm.RPCModel,
    lon: float,
    lat: float,
    alt: float,
) -> Tuple[float, float]:
    col, row = rpc.projection(lon, lat, alt)

    col = float(np.asarray(col))
    row = float(np.asarray(row))

    if not np.isfinite(col) or not np.isfinite(row):
        raise ValueError("RPC projection returned non-finite coordinates")

    return col, row


# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------


def triangulate_one_pair_lon_lat_alt(
    C_pair: np.ndarray,
    rpcs: List[rpcm.RPCModel],
    pair: Tuple[int, int],
    init_pts3d,
    geo_utils,
    helper_module,
    verbose: bool,
) -> np.ndarray:
    pts = helper_module.triangulate_C_to_lon_lat_alt(
        C=C_pair,
        rpcs=rpcs,
        pairs_to_triangulate=[pair],
        init_pts3d=init_pts3d,
        geo_utils=geo_utils,
        verbose=verbose,
    )

    pts = np.asarray(pts, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[0] != 1 or pts.shape[1] != 3:
        raise RuntimeError(f"Unexpected triangulation output shape: {pts.shape}")

    return pts[0]


def pair_reprojection_rmse_px(
    point_lon_lat_alt: np.ndarray,
    pair: Tuple[int, int],
    obs_by_image: dict,
    rpcs: List[rpcm.RPCModel],
) -> Tuple[float, float, float, float, float]:
    lon, lat, alt = map(float, point_lon_lat_alt)

    residuals = []
    projected = {}

    for image_idx in pair:
        obs_col, obs_row = obs_by_image[int(image_idx)]

        pred_col, pred_row = project_rpc_safe(
            rpcs[int(image_idx)],
            lon,
            lat,
            alt,
        )

        dx = pred_col - float(obs_col)
        dy = pred_row - float(obs_row)

        residuals.extend([dx, dy])
        projected[int(image_idx)] = (pred_col, pred_row, dx, dy)

    residuals = np.asarray(residuals, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))

    i, j = pair
    err_i = math.sqrt(projected[i][2] ** 2 + projected[i][3] ** 2)
    err_j = math.sqrt(projected[j][2] ** 2 + projected[j][3] ** 2)

    return (
        rmse,
        float(err_i),
        float(err_j),
        float(projected[i][2]),
        float(projected[j][2]),
    )


def evaluate_tracks_pairwise_consistency(
    C: np.ndarray,
    rpcs: List[rpcm.RPCModel],
    allowed_pairs: Optional[List[Tuple[int, int]]],
    init_pts3d,
    geo_utils,
    helper_module,
    min_track_length: int,
    max_track_length: Optional[int],
    max_pairs_per_track: Optional[int],
    pair_subsample_seed: int,
    use_allowed_pairs_only: bool,
    verbose_triangulation: bool,
) -> Tuple[List[dict], List[dict]]:
    n_images = C.shape[0] // 2
    n_tracks = C.shape[1]

    pair_rows: List[dict] = []
    track_rows: List[dict] = []

    t0 = timeit.default_timer()

    for track_idx in range(n_tracks):
        obs = observations_from_C_column(C, track_idx)
        track_length = len(obs)

        if track_length < min_track_length:
            continue

        if max_track_length is not None and track_length > max_track_length:
            continue

        obs_by_image = {int(image_idx): (float(col), float(row)) for image_idx, col, row in obs}

        pairs = all_pairs_from_observations(obs)

        if use_allowed_pairs_only:
            pairs = filter_pairs_by_allowed_pairs(pairs, allowed_pairs)

        pairs = subsample_pairs_deterministic(
            pairs=pairs,
            max_pairs_per_track=max_pairs_per_track,
            seed=pair_subsample_seed,
            track_idx=track_idx,
        )

        if len(pairs) == 0:
            continue

        current_pair_rows = []

        for i, j in pairs:
            pair = (int(i), int(j))

            row = {
                "track_idx": int(track_idx),
                "track_length": int(track_length),
                "image_i": int(i),
                "image_j": int(j),
                "lon": np.nan,
                "lat": np.nan,
                "alt_m": np.nan,
                "ecef_x_m": np.nan,
                "ecef_y_m": np.nan,
                "ecef_z_m": np.nan,
                "pair_reproj_rmse_px": np.nan,
                "pair_reproj_error_i_px": np.nan,
                "pair_reproj_error_j_px": np.nan,
                "status": "not_run",
                "message": "",
            }

            try:
                C_pair = make_two_observation_C(
                    n_images=n_images,
                    obs_by_image=obs_by_image,
                    i=i,
                    j=j,
                )

                point_lla = triangulate_one_pair_lon_lat_alt(
                    C_pair=C_pair,
                    rpcs=rpcs,
                    pair=pair,
                    init_pts3d=init_pts3d,
                    geo_utils=geo_utils,
                    helper_module=helper_module,
                    verbose=verbose_triangulation,
                )

                if not np.all(np.isfinite(point_lla)):
                    row["status"] = "invalid_triangulation"
                    current_pair_rows.append(row)
                    pair_rows.append(row)
                    continue

                lon, lat, alt = map(float, point_lla)
                xyz = lon_lat_alt_to_ecef(lon, lat, alt)

                row["lon"] = lon
                row["lat"] = lat
                row["alt_m"] = alt
                row["ecef_x_m"] = float(xyz[0])
                row["ecef_y_m"] = float(xyz[1])
                row["ecef_z_m"] = float(xyz[2])

                try:
                    rmse, err_i, err_j, _, _ = pair_reprojection_rmse_px(
                        point_lon_lat_alt=point_lla,
                        pair=pair,
                        obs_by_image=obs_by_image,
                        rpcs=rpcs,
                    )

                    row["pair_reproj_rmse_px"] = rmse
                    row["pair_reproj_error_i_px"] = err_i
                    row["pair_reproj_error_j_px"] = err_j

                except Exception as exc:
                    row["message"] = f"projection_failed: {repr(exc)}"

                row["status"] = "ok"

            except Exception as exc:
                row["status"] = "triangulation_failed"
                row["message"] = repr(exc)

            current_pair_rows.append(row)
            pair_rows.append(row)

        valid_rows = [
            row for row in current_pair_rows
            if row["status"] == "ok"
            and np.isfinite(row["ecef_x_m"])
            and np.isfinite(row["ecef_y_m"])
            and np.isfinite(row["ecef_z_m"])
            and np.isfinite(row["alt_m"])
        ]

        if len(valid_rows) == 0:
            track_rows.append(
                {
                    "track_idx": int(track_idx),
                    "track_length": int(track_length),
                    "n_candidate_pairs": int(len(pairs)),
                    "n_valid_pairwise_points": 0,
                    "status": "no_valid_pairwise_points",
                }
            )
            continue

        xyz = np.asarray(
            [
                [row["ecef_x_m"], row["ecef_y_m"], row["ecef_z_m"]]
                for row in valid_rows
            ],
            dtype=np.float64,
        )

        alt = np.asarray([row["alt_m"] for row in valid_rows], dtype=np.float64)
        pair_rmse = np.asarray(
            [row["pair_reproj_rmse_px"] for row in valid_rows],
            dtype=np.float64,
        )

        center = robust_center_xyz(xyz)

        dxyz = xyz - center[None, :]
        dist_3d = np.linalg.norm(dxyz, axis=1)

        # Horizontal distance in ECEF tangent approximation:
        # remove radial component relative to the robust center.
        center_norm = np.linalg.norm(center)

        if np.isfinite(center_norm) and center_norm > 0:
            radial_unit = center / center_norm
            radial_component = dxyz @ radial_unit
            horizontal_vec = dxyz - radial_component[:, None] * radial_unit[None, :]
            horizontal_dist = np.linalg.norm(horizontal_vec, axis=1)
            vertical_like_dist = np.abs(radial_component)
        else:
            horizontal_dist = np.full(dist_3d.shape, np.nan)
            vertical_like_dist = np.full(dist_3d.shape, np.nan)

        track_row = {
            "track_idx": int(track_idx),
            "track_length": int(track_length),
            "n_candidate_pairs": int(len(pairs)),
            "n_valid_pairwise_points": int(len(valid_rows)),
            "center_ecef_x_m": float(center[0]),
            "center_ecef_y_m": float(center[1]),
            "center_ecef_z_m": float(center[2]),
            "status": "ok",
        }

        track_row.update(finite_stats(dist_3d, "dist3d_to_center_m"))
        track_row.update(finite_stats(horizontal_dist, "horizontal_dist_to_center_m"))
        track_row.update(finite_stats(vertical_like_dist, "radial_dist_to_center_m"))
        track_row.update(finite_stats(alt, "alt_m"))
        track_row.update(finite_stats(alt - np.median(alt), "alt_residual_m"))
        track_row.update(finite_stats(pair_rmse, "pair_reproj_rmse_px"))

        track_rows.append(track_row)

        if track_idx % 100 == 0:
            elapsed = timeit.default_timer() - t0
            print(
                f"[pairwise-consistency] track {track_idx}/{n_tracks}, "
                f"pair rows={len(pair_rows)}, track rows={len(track_rows)}, "
                f"elapsed={elapsed:.1f}s"
            )

    return pair_rows, track_rows


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------


def summarize_track_rows_by_length(track_rows: List[dict]) -> List[dict]:
    valid_rows = [
        row for row in track_rows
        if row.get("status") == "ok"
        and int(row.get("n_valid_pairwise_points", 0)) > 0
    ]

    if len(valid_rows) == 0:
        return []

    lengths = sorted(set(int(row["track_length"]) for row in valid_rows))
    out_rows = []

    metrics = [
        "dist3d_to_center_m_median",
        "dist3d_to_center_m_p90",
        "horizontal_dist_to_center_m_median",
        "horizontal_dist_to_center_m_p90",
        "alt_residual_m_mad",
        "alt_residual_m_p90",
        "pair_reproj_rmse_px_median",
        "pair_reproj_rmse_px_p90",
    ]

    for length in lengths:
        sub = [row for row in valid_rows if int(row["track_length"]) == length]

        out = {
            "track_length": int(length),
            "n_tracks": int(len(sub)),
            "mean_valid_pairwise_points": float(
                np.mean([row["n_valid_pairwise_points"] for row in sub])
            ),
        }

        for metric in metrics:
            vals = np.asarray(
                [float(row.get(metric, np.nan)) for row in sub],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]

            out[f"{metric}_mean_over_tracks"] = float(np.mean(vals)) if vals.size else np.nan
            out[f"{metric}_median_over_tracks"] = float(np.median(vals)) if vals.size else np.nan
            out[f"{metric}_p90_over_tracks"] = float(np.percentile(vals, 90)) if vals.size else np.nan

        out_rows.append(out)

    return out_rows


def write_summary_txt(
    path: Path,
    pair_rows: List[dict],
    track_rows: List[dict],
    args: argparse.Namespace,
) -> None:
    valid_track_rows = [
        row for row in track_rows
        if row.get("status") == "ok"
        and int(row.get("n_valid_pairwise_points", 0)) > 0
    ]

    lines = []
    lines.append("Pairwise 3D and height consistency summary")
    lines.append("==========================================")
    lines.append("")
    lines.append("Metric interpretation:")
    lines.append("  For each external feature track, every valid image pair triangulates")
    lines.append("  an independent 3D point. Better corrected RPCs should produce lower")
    lines.append("  dispersion among those pairwise 3D points.")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  eval_output_dir: {args.eval_output_dir}")
    lines.append(f"  matches_dir: {args.matches_dir}")
    lines.append(f"  rpc_dir: {args.rpc_dir}")
    lines.append(f"  sat_bundleadjust_repo: {args.sat_bundleadjust_repo}")
    lines.append("")
    lines.append("Settings:")
    lines.append(f"  min_track_length: {args.min_track_length}")
    lines.append(f"  max_track_length: {args.max_track_length}")
    lines.append(f"  max_pairs_per_track: {args.max_pairs_per_track}")
    lines.append(f"  use_selected_pairs_only: {args.use_selected_pairs_only}")
    lines.append("")
    lines.append("Counts:")
    lines.append(f"  pair rows: {len(pair_rows)}")
    lines.append(f"  track rows: {len(track_rows)}")
    lines.append(f"  valid track rows: {len(valid_track_rows)}")
    lines.append("")

    if len(valid_track_rows) == 0:
        lines.append("No valid track rows.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    headline_metrics = [
        "dist3d_to_center_m_median",
        "dist3d_to_center_m_p90",
        "horizontal_dist_to_center_m_median",
        "horizontal_dist_to_center_m_p90",
        "alt_residual_m_mad",
        "alt_residual_m_p90",
        "pair_reproj_rmse_px_median",
        "pair_reproj_rmse_px_p90",
    ]

    lines.append("Global statistics over per-track metrics:")

    for metric in headline_metrics:
        vals = np.asarray(
            [float(row.get(metric, np.nan)) for row in valid_track_rows],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            lines.append(f"  {metric}: no valid values")
            continue

        lines.append(
            f"  {metric}: "
            f"mean={np.mean(vals):.6f}, "
            f"median={np.median(vals):.6f}, "
            f"p90={np.percentile(vals, 90):.6f}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_output_dir",
        type=Path,
        required=True,
        help="Output directory from evaluate_heldout_fixed_rpcs.py containing C.npy.",
    )

    parser.add_argument(
        "--matches_dir",
        type=Path,
        required=True,
        help="Original matches directory containing image_paths.txt.",
    )

    parser.add_argument(
        "--rpc_dir",
        type=Path,
        required=True,
        help="Directory containing the RPC files for this method.",
    )

    parser.add_argument(
        "--sat_bundleadjust_repo",
        type=Path,
        required=True,
        help="Path to sat-bundleadjust repository root.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where outputs will be written.",
    )

    parser.add_argument(
        "--helper_script_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Directory containing evaluate_heldout_fixed_rpcs.py. "
            "Default: this script's directory."
        ),
    )

    parser.add_argument(
        "--min_track_length",
        type=int,
        default=3,
        help="Minimum number of observations in an external track.",
    )

    parser.add_argument(
        "--max_track_length",
        type=int,
        default=None,
        help="Optional maximum track length.",
    )

    parser.add_argument(
        "--max_pairs_per_track",
        type=int,
        default=None,
        help=(
            "Optional deterministic subsampling of image pairs per track. "
            "Useful for very long tracks."
        ),
    )

    parser.add_argument(
        "--pair_subsample_seed",
        type=int,
        default=0,
        help="Seed used for deterministic pair subsampling.",
    )

    parser.add_argument(
        "--use_selected_pairs_only",
        action="store_true",
        help=(
            "Only triangulate image pairs that are present in selected_pairs.npy "
            "from eval_output_dir. If not set, all image pairs within each track "
            "are used."
        ),
    )

    parser.add_argument(
        "--verbose_triangulation",
        action="store_true",
        help="Print verbose triangulation output from sat-bundleadjust init_pts3d.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    helper = import_helpers(args.helper_script_dir)

    (
        cam_utils,
        geo_utils,
        ft_match,
        select_optimal_pairs_to_match,
        init_pts3d,
    ) = helper.import_sat_bundleadjust(args.sat_bundleadjust_repo)

    C_path = args.eval_output_dir / "C.npy"

    if not C_path.exists():
        raise FileNotFoundError(f"Missing C.npy: {C_path}")

    C = np.load(C_path)
    C = np.asarray(C, dtype=np.float64)

    selected_pairs = None

    if args.use_selected_pairs_only:
        selected_pairs_path = args.eval_output_dir / "selected_pairs.npy"

        if not selected_pairs_path.exists():
            raise FileNotFoundError(
                f"--use_selected_pairs_only was set, but selected_pairs.npy is missing: "
                f"{selected_pairs_path}"
            )

        selected_pairs_arr = np.load(selected_pairs_path)
        selected_pairs = [
            (int(i), int(j))
            for i, j in np.asarray(selected_pairs_arr, dtype=np.int64)
        ]

    image_paths = helper.load_image_paths(args.matches_dir)

    if len(image_paths) != C.shape[0] // 2:
        raise ValueError(
            f"Image count mismatch: image_paths.txt has {len(image_paths)} images, "
            f"C.npy has {C.shape[0] // 2} images."
        )

    rpcs = helper.load_rpcs_for_images(args.rpc_dir, image_paths)

    print("Running pairwise 3D / height consistency evaluation")
    print("----------------------------------------------------")
    print(f"eval_output_dir: {args.eval_output_dir}")
    print(f"matches_dir: {args.matches_dir}")
    print(f"rpc_dir: {args.rpc_dir}")
    print(f"output_dir: {output_dir}")
    print(f"tracks in C: {C.shape[1]}")
    print(f"images: {len(image_paths)}")
    print(f"use_selected_pairs_only: {args.use_selected_pairs_only}")
    print("----------------------------------------------------")

    pair_rows, track_rows = evaluate_tracks_pairwise_consistency(
        C=C,
        rpcs=rpcs,
        allowed_pairs=selected_pairs,
        init_pts3d=init_pts3d,
        geo_utils=geo_utils,
        helper_module=helper,
        min_track_length=args.min_track_length,
        max_track_length=args.max_track_length,
        max_pairs_per_track=args.max_pairs_per_track,
        pair_subsample_seed=args.pair_subsample_seed,
        use_allowed_pairs_only=args.use_selected_pairs_only,
        verbose_triangulation=args.verbose_triangulation,
    )

    pair_fieldnames = [
        "track_idx",
        "track_length",
        "image_i",
        "image_j",
        "lon",
        "lat",
        "alt_m",
        "ecef_x_m",
        "ecef_y_m",
        "ecef_z_m",
        "pair_reproj_rmse_px",
        "pair_reproj_error_i_px",
        "pair_reproj_error_j_px",
        "status",
        "message",
    ]

    track_fieldnames = sorted(
        set(k for row in track_rows for k in row.keys())
    )

    write_csv_dicts(
        output_dir / "pairwise_3d_points.csv",
        pair_rows,
        pair_fieldnames,
    )

    write_csv_dicts(
        output_dir / "per_track_3d_height_consistency.csv",
        track_rows,
        track_fieldnames,
    )

    summary_by_length_rows = summarize_track_rows_by_length(track_rows)

    summary_by_length_fieldnames = sorted(
        set(k for row in summary_by_length_rows for k in row.keys())
    ) if summary_by_length_rows else ["track_length", "n_tracks"]

    write_csv_dicts(
        output_dir / "summary_by_track_length.csv",
        summary_by_length_rows,
        summary_by_length_fieldnames,
    )

    write_summary_txt(
        output_dir / "summary.txt",
        pair_rows,
        track_rows,
        args,
    )

    print("\nSaved outputs:")
    print(f"  {output_dir / 'pairwise_3d_points.csv'}")
    print(f"  {output_dir / 'per_track_3d_height_consistency.csv'}")
    print(f"  {output_dir / 'summary_by_track_length.csv'}")
    print(f"  {output_dir / 'summary.txt'}")

    print("\nSummary:")
    print((output_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
