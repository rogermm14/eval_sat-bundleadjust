#!/usr/bin/env python3
"""
Prepare input-only similar-pair selection and feature-track matrices.

This script intentionally does not compute held-out errors. It writes:
  - selected least/most-similar image pairs once, under imagepair_similarities
  - C.npy / C_v2.npy per feature-matcher directory
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np


def import_least_similar_helper(script_dir: Path):
    helper_path = script_dir / "evaluate_heldout_fixed_rpcs_least_similar_pairs.py"
    if not helper_path.exists():
        raise FileNotFoundError(f"Missing helper script: {helper_path}")

    spec = importlib.util.spec_from_file_location("least_similar_helper", str(helper_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helper script: {helper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["least_similar_helper"] = module
    spec.loader.exec_module(module)
    return module


def all_image_pairs(n_images: int) -> List[Tuple[int, int]]:
    return [(i, j) for i, j in combinations(range(n_images), 2)]


def write_pairs_txt(path: Path, pairs: List[Tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, j in pairs:
            f.write(f"{int(i)} {int(j)}\n")


def write_summary(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mode_label(mode: str) -> str:
    if mode not in {"least", "most"}:
        raise ValueError(f"Unsupported pair_selection_mode: {mode}")
    return f"{mode}_similar"


def copy_selected_pairs_to_track_dir(pair_dir: Path, track_dir: Path, mode: str) -> None:
    label = mode_label(mode)
    arr = np.load(pair_dir / f"selected_{label}_pairs.npy")
    np.save(track_dir / f"selected_{label}_pairs.npy", arr)
    np.save(track_dir / "selected_pairs.npy", arr)
    write_pairs_txt(track_dir / f"selected_{label}_pairs.txt", [(int(i), int(j)) for i, j in arr])
    write_pairs_txt(track_dir / "selected_pairs.txt", [(int(i), int(j)) for i, j in arr])


def load_or_compute_selected_pairs(helper, args, image_paths, rpcs, satba_imports):
    pair_dir = args.pair_output_dir.resolve()
    pair_dir.mkdir(parents=True, exist_ok=True)
    label = mode_label(args.pair_selection_mode)

    selected_path = pair_dir / f"selected_{label}_pairs.npy"
    candidate_path = pair_dir / "candidate_pairs.npy"

    if selected_path.exists() and candidate_path.exists() and not args.force_pair_selection:
        selected_pairs = [(int(i), int(j)) for i, j in np.asarray(np.load(selected_path), dtype=np.int64)]
        candidate_pairs = [(int(i), int(j)) for i, j in np.asarray(np.load(candidate_path), dtype=np.int64)]
        print(f"Using existing selected pairs: {selected_path}")
        return selected_pairs, candidate_pairs

    cam_utils, geo_utils, ft_match, ft_pair_ranking, _, _ = satba_imports
    images = helper.make_satellite_images(image_paths, rpcs, cam_utils)
    helper.initialize_image_geometry(images)

    available_pairs = all_image_pairs(len(image_paths))
    selected_pairs, candidate_pairs = helper.select_similar_pairs_with_satba_dino(
        images=images,
        available_pairs=available_pairs,
        ft_match=ft_match,
        geo_utils=geo_utils,
        ft_pair_ranking=ft_pair_ranking,
        K=args.least_pair_ranking_K,
        mode=args.pair_selection_mode,
        filter_pairs=(not args.disable_pair_geometry_filter),
        scores_csv_path=pair_dir / "dino_pair_scores.csv",
    )

    np.save(pair_dir / "available_pairs.npy", np.asarray(available_pairs, dtype=np.int64))
    np.save(candidate_path, np.asarray(candidate_pairs, dtype=np.int64))
    np.save(selected_path, np.asarray(selected_pairs, dtype=np.int64))
    np.save(pair_dir / "selected_pairs.npy", np.asarray(selected_pairs, dtype=np.int64))
    write_pairs_txt(pair_dir / "available_pairs.txt", available_pairs)
    write_pairs_txt(pair_dir / "candidate_pairs.txt", candidate_pairs)
    write_pairs_txt(pair_dir / f"selected_{label}_pairs.txt", selected_pairs)
    write_pairs_txt(pair_dir / "selected_pairs.txt", selected_pairs)

    write_summary(
        pair_dir / "summary.txt",
        [
            f"{label.replace('_', '-').title()} image-pair selection",
            "====================================",
            f"images: {len(image_paths)}",
            f"available pairs: {len(available_pairs)}",
            f"candidate pairs: {len(candidate_pairs)}",
            f"selected pairs: {len(selected_pairs)}",
            f"pair_selection_mode: {args.pair_selection_mode}",
            f"pair_ranking_K: {args.least_pair_ranking_K}",
            f"geometry_filter: {not args.disable_pair_geometry_filter}",
        ],
    )

    return selected_pairs, candidate_pairs


def build_tracks_for_matches_dir(helper, args, matches_dir: Path, output_dir: Path, selected_pairs):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = helper.load_image_paths(matches_dir)
    feature_paths = helper.get_feature_paths(matches_dir, image_paths)
    pairwise_matches_all = helper.load_pairwise_matches(matches_dir)

    pairwise_matches = helper.filter_pairwise_matches_to_pairs(
        pairwise_matches_all,
        selected_pairs,
    )

    if pairwise_matches.shape[0] == 0:
        raise RuntimeError(f"No pairwise matches remain after selected-pair filtering: {matches_dir}")

    C, C_v2 = helper.feature_tracks_from_pairwise_matches_variable_length(
        feature_paths=feature_paths,
        pairwise_matches=pairwise_matches,
        pairs_to_triangulate=selected_pairs,
        reject_conflicted_tracks=(not args.keep_conflicted_tracks),
    )

    C, C_v2 = helper.filter_C_by_length_range(
        C=C,
        C_v2=C_v2,
        min_total_length=args.min_track_observations,
        max_total_length=args.FT_max_length,
    )

    if C.shape[1] == 0:
        raise RuntimeError(f"No tracks left after filtering: {matches_dir}")

    np.save(output_dir / "C.npy", C)
    np.save(output_dir / "C_v2.npy", C_v2)
    np.save(output_dir / "pairwise_matches_used.npy", pairwise_matches)
    copy_selected_pairs_to_track_dir(args.pair_output_dir, output_dir, args.pair_selection_mode)

    track_lengths = np.sum(np.isfinite(C[::2, :]), axis=0)
    with (output_dir / "track_length_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in [
            ("n_tracks", C.shape[1]),
            ("min", np.min(track_lengths)),
            ("mean", np.mean(track_lengths)),
            ("median", np.median(track_lengths)),
            ("max", np.max(track_lengths)),
        ]:
            writer.writerow({"metric": metric, "value": float(value)})

    write_summary(
        output_dir / "summary.txt",
        [
            "Least-similar feature-track matrix",
            "==================================",
            f"matches_dir: {matches_dir}",
            f"selected_pair_dir: {args.pair_output_dir}",
            f"C shape: {C.shape}",
            f"C_v2 shape: {C_v2.shape}",
            f"pairwise matches used: {pairwise_matches.shape[0]}",
            f"min_track_observations: {args.min_track_observations}",
            f"FT_max_length: {args.FT_max_length}",
            f"pair_selection_mode: {args.pair_selection_mode}",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_matches_dir", type=Path, required=True)
    parser.add_argument("--rpc_dir", type=Path, required=True)
    parser.add_argument("--sat_bundleadjust_repo", type=Path, required=True)
    parser.add_argument("--pair_output_dir", type=Path, required=True)
    parser.add_argument("--track_output", action="append", default=[], help="feature_name:matches_dir:output_dir")
    parser.add_argument("--least_pair_ranking_K", type=int, default=5)
    parser.add_argument("--pair_selection_mode", choices=["least", "most"], default="least")
    parser.add_argument("--min_track_observations", type=int, default=4)
    parser.add_argument("--FT_max_length", type=int, default=None)
    parser.add_argument("--disable_pair_geometry_filter", action="store_true")
    parser.add_argument("--keep_conflicted_tracks", action="store_true")
    parser.add_argument("--force_pair_selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    helper = import_least_similar_helper(script_dir)

    satba_imports = helper.import_sat_bundleadjust(args.sat_bundleadjust_repo)
    image_paths = helper.load_image_paths(args.reference_matches_dir)
    rpcs = helper.load_rpcs_for_images(args.rpc_dir, image_paths)

    selected_pairs, _ = load_or_compute_selected_pairs(
        helper=helper,
        args=args,
        image_paths=image_paths,
        rpcs=rpcs,
        satba_imports=satba_imports,
    )

    if not args.track_output:
        raise ValueError("At least one --track_output feature_name:matches_dir:output_dir is required.")

    for spec in args.track_output:
        try:
            feature_name, matches_dir, output_dir = spec.split(":", 2)
        except ValueError as exc:
            raise ValueError(f"Invalid --track_output spec: {spec}") from exc

        print(f"\nBuilding least-similar tracks for {feature_name}")
        build_tracks_for_matches_dir(
            helper=helper,
            args=args,
            matches_dir=Path(matches_dir),
            output_dir=Path(output_dir),
            selected_pairs=selected_pairs,
        )


if __name__ == "__main__":
    main()
