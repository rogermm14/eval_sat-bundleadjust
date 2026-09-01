#!/usr/bin/env python3
"""
Utilities for visualizing SuperPoint-LightGlue feature tracks.

This file builds feature tracks directly from:

    matches_dir/
        image_paths.txt
        pairwise_matches.npy
        features/<image_stem>.npy

It does not require C.npy or C_v2.npy to already exist on disk.

Feature array convention:
    features[image_idx][:, 0] = col / x
    features[image_idx][:, 1] = row / y

Pairwise matches convention:
    pairwise_matches.npy has shape M x 4:

        column 0 = keypoint index in image im_i
        column 1 = keypoint index in image im_j
        column 2 = im_i
        column 3 = im_j

Correspondence matrix C convention:
    C[2 * image_idx,     track_idx] = col / x
    C[2 * image_idx + 1, track_idx] = row / y

Correspondence matrix C_v2 convention:
    C_v2[image_idx, track_idx] = keypoint index in that image
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =============================================================================
# Loading SuperPoint / LightGlue outputs
# =============================================================================


def load_image_paths(matches_dir: Path) -> List[Path]:
    """
    Load image paths from:

        matches_dir/image_paths.txt
    """
    matches_dir = Path(matches_dir)
    image_paths_txt = matches_dir / "image_paths.txt"

    if not image_paths_txt.exists():
        raise FileNotFoundError(f"Missing image_paths.txt: {image_paths_txt}")

    with image_paths_txt.open("r", encoding="utf-8") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]

    if len(image_paths) == 0:
        raise RuntimeError(f"No image paths found in {image_paths_txt}")

    return image_paths


def get_feature_paths(matches_dir: Path, image_paths: List[Path]) -> List[Path]:
    """
    Return one feature file path per image.

    Expected layout:

        matches_dir/features/<image_stem>.npy
    """
    matches_dir = Path(matches_dir)
    features_dir = matches_dir / "features"

    if not features_dir.exists():
        raise FileNotFoundError(f"Missing features directory: {features_dir}")

    feature_paths: List[Path] = []
    missing: List[Path] = []

    for image_path in image_paths:
        feature_path = features_dir / f"{image_path.stem}.npy"

        if not feature_path.exists():
            missing.append(feature_path)

        feature_paths.append(feature_path)

    if missing:
        msg = "\n".join(str(p) for p in missing[:30])
        raise FileNotFoundError(
            "Some feature files are missing. Expected feature files named from "
            f"image basenames.\nFirst missing files:\n{msg}"
        )

    return feature_paths


def load_pairwise_matches(matches_dir: Path) -> np.ndarray:
    """
    Load pairwise_matches.npy from matches_dir.
    """
    matches_dir = Path(matches_dir)
    path = matches_dir / "pairwise_matches.npy"

    if not path.exists():
        raise FileNotFoundError(f"Missing pairwise_matches.npy: {path}")

    pairwise_matches = np.load(path)

    if pairwise_matches.ndim != 2 or pairwise_matches.shape[1] != 4:
        raise ValueError(
            f"Expected pairwise_matches shape Mx4, got {pairwise_matches.shape}"
        )

    return pairwise_matches.astype(np.int64)


def all_pairs_from_pairwise_matches(pairwise_matches: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return all image pairs that have at least one match.
    """
    pairwise_matches = np.asarray(pairwise_matches)

    if pairwise_matches.shape[0] == 0:
        return []

    pairs = pairwise_matches[:, 2:4].astype(int)
    pairs = np.unique(pairs, axis=0)

    normalized_pairs = []
    for i, j in pairs:
        i = int(i)
        j = int(j)
        normalized_pairs.append((min(i, j), max(i, j)))

    return sorted(set(normalized_pairs))


# =============================================================================
# Building C and C_v2
# =============================================================================


def filter_C_using_pairs_to_triangulate_local(
    C: np.ndarray,
    pairs_to_triangulate: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Keep only tracks that contain at least one image pair listed in pairs_to_triangulate.

    Args:
        C:
            2 * n_images x n_tracks correspondence matrix.

        pairs_to_triangulate:
            List of valid image-index pairs.

    Returns:
        1D array of preserved track indices.
    """
    visible = np.isfinite(C[::2, :])
    pairs_to_triangulate_set = {
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in pairs_to_triangulate
    }

    columns_to_preserve = []

    for track_idx in range(C.shape[1]):
        image_indices = np.where(visible[:, track_idx])[0]

        all_pairs_current_track = set(
            (int(i), int(j))
            for i in image_indices
            for j in image_indices
            if i < j
        )

        has_valid_pair = len(pairs_to_triangulate_set & all_pairs_current_track) > 0
        columns_to_preserve.append(has_valid_pair)

    return np.where(columns_to_preserve)[0]


def feature_tracks_from_pairwise_matches_variable_length(
    feature_paths: List[Path],
    pairwise_matches: np.ndarray,
    pairs_to_triangulate: List[Tuple[int, int]],
    remove_conflicted_tracks: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build correspondence matrices C and C_v2 from variable-length feature arrays.

    This supports SuperPoint feature arrays where each image can have a different
    number of keypoints.

    Args:
        feature_paths:
            List of per-image .npy feature arrays.
            Each feature array must have at least:
                column 0 = col / x
                column 1 = row / y

        pairwise_matches:
            M x 4 int array:
                column 0 = keypoint index in image im_i
                column 1 = keypoint index in image im_j
                column 2 = im_i
                column 3 = im_j

        pairs_to_triangulate:
            Image-index pairs used to keep valid tracks.

        remove_conflicted_tracks:
            If True, remove tracks that contain more than one keypoint in the
            same image.

        verbose:
            If True, print progress.

    Returns:
        C:
            2 * n_images x n_tracks matrix.

        C_v2:
            n_images x n_tracks matrix of keypoint indices.
    """
    feature_paths = [Path(p) for p in feature_paths]
    pairwise_matches = np.asarray(pairwise_matches, dtype=np.int64)

    if pairwise_matches.ndim != 2 or pairwise_matches.shape[1] != 4:
        raise ValueError(
            f"Expected pairwise_matches shape Mx4, got {pairwise_matches.shape}"
        )

    n_cams = len(feature_paths)

    features: List[np.ndarray] = []
    feature_ids: List[np.ndarray] = []

    global_id_start = 0

    if verbose:
        print("Loading feature arrays and assigning global keypoint IDs...")

    for im_idx, feature_path in enumerate(feature_paths):
        features_i = np.load(feature_path, mmap_mode="r")

        if features_i.ndim != 2 or features_i.shape[1] < 2:
            raise ValueError(
                f"Feature file {feature_path} must be NxD with D >= 2. "
                f"Got shape {features_i.shape}"
            )

        features.append(features_i)

        ids_i = np.arange(
            global_id_start,
            global_id_start + features_i.shape[0],
            dtype=np.int64,
        )
        feature_ids.append(ids_i)

        global_id_start += features_i.shape[0]

        if verbose:
            print(
                f"  image {im_idx:04d}: {feature_path.name}, "
                f"{features_i.shape[0]} keypoints"
            )

    n_global_keypoints = global_id_start

    if n_global_keypoints == 0:
        return (
            np.zeros((2 * n_cams, 0), dtype=np.float64),
            np.zeros((n_cams, 0), dtype=np.float64),
        )

    parent = np.arange(n_global_keypoints, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_a] = root_b

    if verbose:
        print("Merging pairwise matches into feature tracks...")

    for row_idx in range(pairwise_matches.shape[0]):
        kp_i, kp_j, im_i, im_j = pairwise_matches[row_idx]

        im_i = int(im_i)
        im_j = int(im_j)
        kp_i = int(kp_i)
        kp_j = int(kp_j)

        if im_i < 0 or im_i >= n_cams or im_j < 0 or im_j >= n_cams:
            raise IndexError(
                f"Invalid image index in pairwise_matches row {row_idx}: "
                f"im_i={im_i}, im_j={im_j}, n_cams={n_cams}"
            )

        if kp_i < 0 or kp_i >= features[im_i].shape[0]:
            raise IndexError(
                f"Invalid keypoint index in row {row_idx}: "
                f"kp_i={kp_i}, image {im_i} has {features[im_i].shape[0]} keypoints"
            )

        if kp_j < 0 or kp_j >= features[im_j].shape[0]:
            raise IndexError(
                f"Invalid keypoint index in row {row_idx}: "
                f"kp_j={kp_j}, image {im_j} has {features[im_j].shape[0]} keypoints"
            )

        global_i = int(feature_ids[im_i][kp_i])
        global_j = int(feature_ids[im_j][kp_j])

        union(global_i, global_j)

    if verbose:
        print("Compressing feature-track graph...")

    parents = np.array([find(i) for i in range(n_global_keypoints)], dtype=np.int64)

    unique_parents, inverse, counts = np.unique(
        parents,
        return_inverse=True,
        return_counts=True,
    )

    valid_global_mask = counts[inverse] > 1
    valid_parent_ids = np.unique(parents[valid_global_mask])

    parent_to_track = {
        int(parent_id): track_idx
        for track_idx, parent_id in enumerate(valid_parent_ids)
    }

    n_tracks = len(parent_to_track)

    if verbose:
        print(f"Initial number of tracks with >=2 keypoints: {n_tracks}")

    C = np.full((2 * n_cams, n_tracks), np.nan, dtype=np.float64)
    C_v2 = np.full((n_cams, n_tracks), np.nan, dtype=np.float64)

    conflicting_track_indices = set()
    same_camera_conflict_count = 0

    if verbose:
        print("Filling C and C_v2...")

    for row_idx in range(pairwise_matches.shape[0]):
        kp_i, kp_j, im_i, im_j = pairwise_matches[row_idx]

        im_i = int(im_i)
        im_j = int(im_j)
        kp_i = int(kp_i)
        kp_j = int(kp_j)

        global_i = int(feature_ids[im_i][kp_i])
        global_j = int(feature_ids[im_j][kp_j])

        parent_i = int(parents[global_i])
        parent_j = int(parents[global_j])

        if parent_i != parent_j:
            continue

        if parent_i not in parent_to_track:
            continue

        track_idx = parent_to_track[parent_i]

        if np.isnan(C_v2[im_i, track_idx]):
            C[2 * im_i, track_idx] = float(features[im_i][kp_i, 0])
            C[2 * im_i + 1, track_idx] = float(features[im_i][kp_i, 1])
            C_v2[im_i, track_idx] = float(kp_i)
        elif int(C_v2[im_i, track_idx]) != kp_i:
            same_camera_conflict_count += 1
            conflicting_track_indices.add(track_idx)

        if np.isnan(C_v2[im_j, track_idx]):
            C[2 * im_j, track_idx] = float(features[im_j][kp_j, 0])
            C[2 * im_j + 1, track_idx] = float(features[im_j][kp_j, 1])
            C_v2[im_j, track_idx] = float(kp_j)
        elif int(C_v2[im_j, track_idx]) != kp_j:
            same_camera_conflict_count += 1
            conflicting_track_indices.add(track_idx)

    if verbose:
        print(f"Same-camera keypoint conflicts: {same_camera_conflict_count}")
        print(f"Tracks with same-camera conflicts: {len(conflicting_track_indices)}")

    if remove_conflicted_tracks and len(conflicting_track_indices) > 0:
        keep_mask = np.ones(C.shape[1], dtype=bool)
        keep_mask[list(conflicting_track_indices)] = False

        C = C[:, keep_mask]
        C_v2 = C_v2[:, keep_mask]

        if verbose:
            print(f"C.shape after removing conflicted tracks: {C.shape}")

    if verbose:
        print(f"C.shape before pair-preservation check: {C.shape}")

    tracks_to_preserve = filter_C_using_pairs_to_triangulate_local(
        C=C,
        pairs_to_triangulate=pairs_to_triangulate,
    )

    C = C[:, tracks_to_preserve]
    C_v2 = C_v2[:, tracks_to_preserve]

    if verbose:
        print(f"C.shape after pair-preservation check: {C.shape}")

    return C, C_v2


def build_C_and_C_v2_from_matches_dir(
    matches_dir: Path,
    pairs_to_triangulate: Optional[List[Tuple[int, int]]] = None,
    remove_conflicted_tracks: bool = True,
    verbose: bool = True,
) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    """
    Convenience helper for notebooks.

    Loads:
        image_paths.txt
        features/*.npy
        pairwise_matches.npy

    Builds:
        C
        C_v2

    Returns:
        image_paths, C, C_v2
    """
    matches_dir = Path(matches_dir)

    image_paths = load_image_paths(matches_dir)
    feature_paths = get_feature_paths(matches_dir, image_paths)
    pairwise_matches = load_pairwise_matches(matches_dir)

    if pairs_to_triangulate is None:
        pairs_to_triangulate = all_pairs_from_pairwise_matches(pairwise_matches)

    if len(pairs_to_triangulate) == 0:
        raise RuntimeError("No image pairs found in pairwise_matches.npy")

    if verbose:
        print(f"Images: {len(image_paths)}")
        print(f"Pairwise matches: {pairwise_matches.shape}")
        print(f"Pairs used for track filtering: {len(pairs_to_triangulate)}")

    C, C_v2 = feature_tracks_from_pairwise_matches_variable_length(
        feature_paths=feature_paths,
        pairwise_matches=pairwise_matches,
        pairs_to_triangulate=pairs_to_triangulate,
        remove_conflicted_tracks=remove_conflicted_tracks,
        verbose=verbose,
    )

    C = np.asarray(C, dtype=np.float64)
    C_v2 = np.asarray(C_v2, dtype=np.float64)

    if C.shape[0] != 2 * len(image_paths):
        raise ValueError(
            f"C has {C.shape[0]} rows, but image_paths has {len(image_paths)} images. "
            f"Expected C.shape[0] == {2 * len(image_paths)}."
        )

    if C_v2.shape[0] != len(image_paths):
        raise ValueError(
            f"C_v2 has {C_v2.shape[0]} rows, but image_paths has "
            f"{len(image_paths)} images."
        )

    return image_paths, C, C_v2


# =============================================================================
# Track selection and observation extraction
# =============================================================================


def compute_visibility_and_track_lengths(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute visibility matrix and track lengths.

    Returns:
        visible:
            n_images x n_tracks boolean matrix.

        track_lengths:
            n_tracks integer array.
    """
    if C.ndim != 2 or C.shape[0] % 2 != 0:
        raise ValueError(f"C must have shape 2*n_images x n_tracks. Got {C.shape}")

    visible = np.isfinite(C[::2, :]) & np.isfinite(C[1::2, :])
    track_lengths = visible.sum(axis=0)

    return visible, track_lengths


def print_track_length_stats(C: np.ndarray) -> None:
    """
    Print distribution of number of image observations per track.
    """
    _, track_lengths = compute_visibility_and_track_lengths(C)

    if track_lengths.size == 0:
        print("No tracks found.")
        return

    print("\nTrack length statistics:")
    print(f"  tracks total: {track_lengths.shape[0]}")
    print(f"  min:          {np.min(track_lengths)}")
    print(f"  mean:         {np.mean(track_lengths):.3f}")
    print(f"  median:       {np.median(track_lengths):.3f}")
    print(f"  p90:          {np.percentile(track_lengths, 90):.3f}")
    print(f"  p95:          {np.percentile(track_lengths, 95):.3f}")
    print(f"  max:          {np.max(track_lengths)}")

    unique_lengths, counts = np.unique(track_lengths, return_counts=True)
    for length, count in zip(unique_lengths, counts):
        print(f"  tracks with length {int(length)}: {int(count)}")


def select_track_by_visibility_rank(C: np.ndarray, rank: int = 0) -> int:
    """
    Select a track by descending visibility count.

    rank = 0 gives the most-observed track.
    rank = 1 gives the second-most-observed track.
    """
    _, track_lengths = compute_visibility_and_track_lengths(C)
    n_tracks = C.shape[1]

    if n_tracks == 0:
        raise RuntimeError("No tracks found in C.")

    if rank < 0 or rank >= n_tracks:
        raise ValueError(f"rank={rank} is out of range. Number of tracks: {n_tracks}")

    sorted_track_indices = np.argsort(-track_lengths)
    return int(sorted_track_indices[rank])


def select_track_by_exact_length(
    C: np.ndarray,
    target_track_length: int,
    rank: int = 0,
) -> int:
    """
    Select the rank-th track observed in exactly target_track_length images.

    rank = 0 gives the first track with that exact length.
    rank = 1 gives the second track with that exact length.
    """
    _, track_lengths = compute_visibility_and_track_lengths(C)

    candidate_track_indices = np.where(track_lengths == target_track_length)[0]

    if len(candidate_track_indices) == 0:
        unique_lengths, counts = np.unique(track_lengths, return_counts=True)
        available = dict(zip(unique_lengths.astype(int), counts.astype(int)))

        raise RuntimeError(
            f"No tracks observed in exactly {target_track_length} images. "
            f"Available track length counts: {available}"
        )

    if rank < 0 or rank >= len(candidate_track_indices):
        raise ValueError(
            f"rank={rank} is out of range. "
            f"There are only {len(candidate_track_indices)} tracks observed in "
            f"exactly {target_track_length} images."
        )

    return int(candidate_track_indices[rank])


def get_track_observations(
    C: np.ndarray,
    C_v2: np.ndarray,
    image_paths: List[Path],
    track_idx: int,
) -> List[Dict[str, object]]:
    """
    Extract all observations for one feature track.

    Returns a list of dictionaries with:
        image_idx
        image_path
        col
        row
        keypoint_idx
    """
    n_images = len(image_paths)
    n_tracks = C.shape[1]

    if C.shape[0] != 2 * n_images:
        raise ValueError(
            f"C.shape[0]={C.shape[0]} but expected 2 * len(image_paths)={2 * n_images}"
        )

    if C_v2.shape[0] != n_images:
        raise ValueError(
            f"C_v2.shape[0]={C_v2.shape[0]} but expected len(image_paths)={n_images}"
        )

    if track_idx < 0 or track_idx >= n_tracks:
        raise ValueError(f"track_idx={track_idx} out of range. n_tracks={n_tracks}")

    observations: List[Dict[str, object]] = []

    for image_idx, image_path in enumerate(image_paths):
        col = C[2 * image_idx, track_idx]
        row = C[2 * image_idx + 1, track_idx]

        if np.isfinite(col) and np.isfinite(row):
            kp_idx = C_v2[image_idx, track_idx]
            kp_idx = int(kp_idx) if np.isfinite(kp_idx) else None

            observations.append(
                {
                    "image_idx": int(image_idx),
                    "image_path": Path(image_path),
                    "col": float(col),
                    "row": float(row),
                    "keypoint_idx": kp_idx,
                }
            )

    return observations


def print_track_observations(observations: List[Dict[str, object]]) -> None:
    """
    Print observations in a compact format.
    """
    for obs in observations:
        print(
            f"image_idx={int(obs['image_idx']):03d}, "
            f"kp_idx={obs['keypoint_idx']}, "
            f"col={float(obs['col']):.2f}, "
            f"row={float(obs['row']):.2f}, "
            f"name={Path(obs['image_path']).name}"
        )


# =============================================================================
# Image display and plotting
# =============================================================================


def read_image_for_display(path: Path) -> np.ndarray:
    """
    Read an image and normalize it for matplotlib display.

    Handles:
        grayscale
        RGB
        uint8
        uint16
        float
    """
    img = np.array(Image.open(path))

    if img.ndim == 2:
        img = img.astype(np.float32)
        lo, hi = np.nanpercentile(img, [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        return img

    if img.ndim == 3:
        original_dtype = img.dtype
        img = img[:, :, :3].astype(np.float32)

        if original_dtype == np.uint8:
            img = img / 255.0
        else:
            lo, hi = np.nanpercentile(img, [1, 99])
            img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

        return img

    raise ValueError(f"Unsupported image shape for {path}: {img.shape}")


def extract_crop(
    img: np.ndarray,
    col: float,
    row: float,
    crop_radius: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Extract a crop around (col, row).

    Returns:
        crop
        cx: point x coordinate in crop frame
        cy: point y coordinate in crop frame
    """
    h, w = img.shape[:2]

    x0 = int(max(0, np.floor(col - crop_radius)))
    x1 = int(min(w, np.ceil(col + crop_radius)))
    y0 = int(max(0, np.floor(row - crop_radius)))
    y1 = int(min(h, np.ceil(row + crop_radius)))

    crop = img[y0:y1, x0:x1]

    cx = float(col - x0)
    cy = float(row - y0)

    return crop, cx, cy


def plot_feature_track_overlay(
    observations: List[Dict[str, object]],
    track_idx: Optional[int] = None,
    crop_radius: int = 80,
    max_images: Optional[int] = None,
    point_size: int = 80,
    figsize_per_panel: float = 4.0,
) -> None:
    """
    Show one crop per image around the feature-track observation.
    """
    if len(observations) == 0:
        print("No observations to plot.")
        return

    if max_images is not None:
        observations = observations[:max_images]

    n = len(observations)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel * n, figsize_per_panel))

    if n == 1:
        axes = [axes]

    for ax, obs in zip(axes, observations):
        image_path = Path(obs["image_path"])
        img = read_image_for_display(image_path)

        col = float(obs["col"])
        row = float(obs["row"])

        crop, cx, cy = extract_crop(
            img=img,
            col=col,
            row=row,
            crop_radius=crop_radius,
        )

        ax.imshow(crop, cmap="gray" if crop.ndim == 2 else None)

        ax.scatter(
            [cx],
            [cy],
            s=point_size,
            marker="x",
            color="red",
            linewidths=2.5,
            zorder=3,
        )

        ax.scatter(
            [cx],
            [cy],
            s=2 * point_size,
            facecolors="none",
            edgecolors="white",
            linewidths=2.0,
            zorder=2,
        )

        ax.set_title(
            f"img {int(obs['image_idx'])}\n"
            f"{image_path.name}\n"
            f"kp {obs['keypoint_idx']}"
        )

        ax.set_xlim(0, crop.shape[1])
        ax.set_ylim(crop.shape[0], 0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    title = "SuperPoint-LightGlue feature track"
    if track_idx is not None:
        title += f" {track_idx}"

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
