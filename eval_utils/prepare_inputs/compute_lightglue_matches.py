#!/usr/bin/env python3
"""
Generate local features and all-pairs LightGlue matches for a directory of TIFF images.

Outputs:
    output_dir/pairwise_matches.npy
        int64 array of shape M x 4.

        Each row:
            column 0 = keypoint index in image 1, indexing the feature array for im1_index
            column 1 = keypoint index in image 2, indexing the feature array for im2_index
            column 2 = im1_index
            column 3 = im2_index

    output_dir/features/<image_basename>.npy
        float32 array of shape Ni x D_total.

        Default row layout:
            column 0 = col / x
            column 1 = row / y
            column 2 = keypoint score
            columns 3: = local descriptor

    output_dir/image_paths.txt
        One image path per line, in the exact order used for feature extraction
        and for im1_index / im2_index in pairwise_matches.

Example:
    python compute_lightglue_matches.py \
        --image_dir /path/to/tifs \
        --output_dir lightglue_output \
        --feature_type superpoint \
        --max_keypoints 4096
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import numpy as np
import torch
import cv2
import rasterio
from rasterio.enums import Resampling

from lightglue import ALIKED, LightGlue, SuperPoint
from lightglue.utils import rbd

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}


def list_tif_images(image_dir: Path) -> List[Path]:
    image_paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix in TIFF_EXTENSIONS
    ]
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"No TIFF images found in: {image_dir}")

    return image_paths


def make_unique_feature_path(features_dir: Path, image_path: Path, used_names: set[str]) -> Path:
    """
    Create a unique feature filename based on the image basename.

    Example:
        image.tif -> image.npy

    If duplicate basenames exist:
        image.npy
        image__1.npy
        image__2.npy
    """
    stem = image_path.stem
    candidate = f"{stem}.npy"

    if candidate not in used_names:
        used_names.add(candidate)
        return features_dir / candidate

    counter = 1
    while True:
        candidate = f"{stem}__{counter}.npy"
        if candidate not in used_names:
            used_names.add(candidate)
            return features_dir / candidate
        counter += 1


def normalize_raster_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a Rasterio image array to float32 values in [0, 1].

    Rasterio returns bands x rows x cols. LightGlue expects a torch tensor with
    channels x rows x cols and floating point values normalized to [0, 1].
    """
    image = np.asarray(image)

    if image.ndim != 3:
        raise RuntimeError(f"Expected Rasterio image with shape CxHxW, got {image.shape}")

    if image.shape[0] > 3:
        image = image[:3]

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        image = image.astype(np.float32) / float(info.max)
    else:
        image = image.astype(np.float32)
        finite = np.isfinite(image)
        if np.any(finite):
            min_value = float(np.min(image[finite]))
            max_value = float(np.max(image[finite]))
            if max_value > min_value:
                image = (image - min_value) / (max_value - min_value)
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def load_geotiff_for_lightglue(image_path: Path, resize: int | None) -> torch.Tensor:
    """
    Read a GeoTIFF with Rasterio and return a LightGlue-compatible image tensor.
    """
    with rasterio.open(image_path) as src:
        height = src.height
        width = src.width

        if resize is not None and resize > 0:
            scale = resize / float(max(height, width))
            out_height = max(1, int(round(height * scale)))
            out_width = max(1, int(round(width * scale)))
            image = src.read(
                out_shape=(src.count, out_height, out_width),
                resampling=Resampling.bilinear,
            )
        else:
            image = src.read()

    image = normalize_raster_image(image)

    return torch.from_numpy(image)


def to_numpy_features(feats: Dict[str, torch.Tensor]) -> np.ndarray:
    """
    Convert a LightGlue feature dictionary into the requested per-image array.

    Output row layout:
        col 0: x / column
        col 1: y / row
        col 2: keypoint score
        col 3 onward: local descriptor
    """
    keypoints = feats["keypoints"].detach().cpu().numpy().astype(np.float32)
    descriptors = feats["descriptors"].detach().cpu().numpy().astype(np.float32)

    if "keypoint_scores" in feats:
        scores = feats["keypoint_scores"].detach().cpu().numpy().astype(np.float32)
    elif "scores" in feats:
        scores = feats["scores"].detach().cpu().numpy().astype(np.float32)
    else:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)

    scores = scores.reshape(-1, 1)

    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise RuntimeError(f"Unexpected keypoints shape: {keypoints.shape}")

    if descriptors.ndim != 2:
        raise RuntimeError(f"Unexpected descriptors shape: {descriptors.shape}")

    if descriptors.shape[0] != keypoints.shape[0]:
        raise RuntimeError(
            f"Descriptor/keypoint count mismatch: "
            f"{descriptors.shape[0]} descriptors vs {keypoints.shape[0]} keypoints"
        )

    return np.concatenate([keypoints, scores, descriptors], axis=1)

def filter_matches_fundamental_ransac(
    matches: np.ndarray,
    features0: np.ndarray,
    features1: np.ndarray,
    threshold_px: float = 0.3,
    confidence: float = 0.999,
    max_iters: int = 10000,
    min_matches: int = 8,
) -> np.ndarray:
    """
    Filter LightGlue matches using a fundamental matrix estimated with RANSAC.

    Safe version: if OpenCV fails for a pair, return zero matches instead of
    crashing the full AOI run.
    """
    matches = np.asarray(matches, dtype=np.int64)

    if matches.ndim != 2 or matches.shape[1] != 2:
        print(f"[warning] Invalid matches shape for RANSAC: {matches.shape}")
        return np.zeros((0, 2), dtype=np.int64)

    if matches.shape[0] < min_matches:
        return np.zeros((0, 2), dtype=np.int64)

    # Remove invalid keypoint indices before indexing features.
    valid_idx = (
        (matches[:, 0] >= 0)
        & (matches[:, 0] < features0.shape[0])
        & (matches[:, 1] >= 0)
        & (matches[:, 1] < features1.shape[0])
    )

    if not np.all(valid_idx):
        n_bad = int(np.sum(~valid_idx))
        print(f"[warning] Dropping {n_bad} matches with invalid keypoint indices")
        matches = matches[valid_idx]

    if matches.shape[0] < min_matches:
        return np.zeros((0, 2), dtype=np.int64)

    pts0 = features0[matches[:, 0], 0:2].astype(np.float64)
    pts1 = features1[matches[:, 1], 0:2].astype(np.float64)

    # Remove NaN/Inf coordinates.
    finite = (
        np.isfinite(pts0[:, 0])
        & np.isfinite(pts0[:, 1])
        & np.isfinite(pts1[:, 0])
        & np.isfinite(pts1[:, 1])
    )

    if not np.all(finite):
        n_bad = int(np.sum(~finite))
        print(f"[warning] Dropping {n_bad} matches with non-finite coordinates")
        matches = matches[finite]
        pts0 = pts0[finite]
        pts1 = pts1[finite]

    if matches.shape[0] < min_matches:
        return np.zeros((0, 2), dtype=np.int64)

    # OpenCV is happier with contiguous float32/float64 Nx2 arrays.
    pts0 = np.ascontiguousarray(pts0, dtype=np.float64)
    pts1 = np.ascontiguousarray(pts1, dtype=np.float64)

    try:
        F, inlier_mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=threshold_px,
            confidence=confidence,
            maxIters=max_iters,
        )
    except cv2.error as exc:
        print(
            "[warning] cv2.findFundamentalMat failed; "
            f"returning 0 inliers. Error: {exc}"
        )
        return np.zeros((0, 2), dtype=np.int64)
    except Exception as exc:
        print(
            "[warning] findFundamentalMat failed unexpectedly; "
            f"returning 0 inliers. Error: {repr(exc)}"
        )
        return np.zeros((0, 2), dtype=np.int64)

    if F is None or inlier_mask is None:
        return np.zeros((0, 2), dtype=np.int64)

    inlier_mask = np.asarray(inlier_mask).reshape(-1).astype(bool)

    if inlier_mask.shape[0] != matches.shape[0]:
        print(
            "[warning] RANSAC inlier mask length mismatch: "
            f"mask={inlier_mask.shape[0]}, matches={matches.shape[0]}; "
            "returning 0 inliers"
        )
        return np.zeros((0, 2), dtype=np.int64)

    return matches[inlier_mask].astype(np.int64)

def extract_lightglue_features(
    image_paths: List[Path],
    extractor,
    device: torch.device,
    resize: int | None,
    features_dir: Path,
) -> Tuple[List[Dict[str, torch.Tensor]], List[np.ndarray]]:
    """
    Extract local features once per image.

    Saves:
        features_dir/<image_basename>.npy

    Returns:
        lightglue_features:
            Batched LightGlue feature dictionaries.
            These are used directly by LightGlue matching.

        output_features:
            Unbatched NumPy arrays.
            These are saved to disk and used for RANSAC filtering.
    """
    lightglue_features: List[Dict[str, torch.Tensor]] = []
    output_features: List[np.ndarray] = []
    used_feature_names: set[str] = set()

    for idx, image_path in enumerate(image_paths):
        image = load_geotiff_for_lightglue(image_path, resize=resize).to(device)

        with torch.inference_mode():
            feats_batched = extractor.extract(image)

        # IMPORTANT:
        # Keep the batched version for LightGlue.
        # LightGlue expects keypoints with shape 1 x N x 2.
        lightglue_features.append(feats_batched)

        # Use an unbatched copy only for saving and RANSAC.
        feats_unbatched = rbd(feats_batched)

        feature_array = to_numpy_features(feats_unbatched)
        feature_path = make_unique_feature_path(features_dir, image_path, used_feature_names)

        np.save(feature_path, feature_array)
        output_features.append(feature_array)

        print(
            f"[features] {idx:04d} {image_path.name}: "
            f"{feature_array.shape[0]} keypoints -> {feature_path.name}"
        )

    return lightglue_features, output_features


def match_all_pairs(
    lightglue_features: List[Dict[str, torch.Tensor]],
    output_features: List[np.ndarray],
    matcher: LightGlue,
    log_label: str,
    fundamental_threshold_px: float = 0.3,
    min_raw_matches: int = 8,
    min_inlier_matches: int = 8,
) -> np.ndarray:
    """
    Match all image pairs with LightGlue, then filter each pair with
    fundamental-matrix RANSAC.

    Returns:
        pairwise_matches: M x 4 int64 array
            [kp_index_image1, kp_index_image2, im1_index, im2_index]
    """
    num_images = len(lightglue_features)
    all_rows: List[np.ndarray] = []

    total_pairs = num_images * (num_images - 1) // 2
    pair_counter = 0

    for im1_index in range(num_images):
        for im2_index in range(im1_index + 1, num_images):
            pair_counter += 1

            feats0 = lightglue_features[im1_index]
            feats1 = lightglue_features[im2_index]

            try:
                with torch.inference_mode():
                    pred = matcher({"image0": feats0, "image1": feats1})

                pred = rbd(pred)

                if "matches" not in pred:
                    print(
                        f"[{log_label}] pair {pair_counter:04d}/{total_pairs}: "
                        f"{im1_index}-{im2_index}: no 'matches' key -> skipped"
                    )
                    continue

                raw_matches_tensor = pred["matches"]

                if raw_matches_tensor is None:
                    print(
                        f"[{log_label}] pair {pair_counter:04d}/{total_pairs}: "
                        f"{im1_index}-{im2_index}: matches is None -> skipped"
                    )
                    continue

                raw_matches = raw_matches_tensor.detach().cpu().numpy().astype(np.int64)

                if raw_matches.size == 0:
                    raw_matches = np.zeros((0, 2), dtype=np.int64)

                if raw_matches.ndim != 2 or raw_matches.shape[1] != 2:
                    print(
                        f"[{log_label}] pair {pair_counter:04d}/{total_pairs}: "
                        f"{im1_index}-{im2_index}: unexpected matches shape "
                        f"{raw_matches.shape} -> skipped"
                    )
                    continue

            except Exception as exc:
                print(
                    f"[{log_label}] pair {pair_counter:04d}/{total_pairs}: "
                    f"{im1_index}-{im2_index}: matcher failed with {repr(exc)} -> skipped"
                )
                continue

            if raw_matches.shape[0] < min_raw_matches:
                print(
                    f"[{log_label}] pair {pair_counter:04d}/{total_pairs}: "
                    f"{im1_index}-{im2_index}: "
                    f"{raw_matches.shape[0]} raw, 0 F-inliers"
                )
                continue

            filtered_matches = filter_matches_fundamental_ransac(
                matches=raw_matches,
                features0=output_features[im1_index],
                features1=output_features[im2_index],
                threshold_px=fundamental_threshold_px,
            )

            if filtered_matches.shape[0] < min_inlier_matches:
                print(
                    f"[{log_label}] pair {pair_counter:04d}/{total_pairs}: "
                    f"{im1_index}-{im2_index}: "
                    f"{raw_matches.shape[0]} raw, "
                    f"{filtered_matches.shape[0]} F-inliers -> skipped"
                )
                continue

            im_cols = np.empty((filtered_matches.shape[0], 2), dtype=np.int64)
            im_cols[:, 0] = im1_index
            im_cols[:, 1] = im2_index

            rows = np.concatenate([filtered_matches, im_cols], axis=1)
            all_rows.append(rows)

            inlier_ratio = filtered_matches.shape[0] / raw_matches.shape[0]

            print(
                f"[{log_label}] pair {pair_counter:04d}/{total_pairs}: "
                f"{im1_index}-{im2_index}: "
                f"{raw_matches.shape[0]} raw, "
                f"{filtered_matches.shape[0]} F-inliers, "
                f"inlier ratio={inlier_ratio:.3f}"
            )

    if len(all_rows) == 0:
        return np.zeros((0, 4), dtype=np.int64)

    return np.vstack(all_rows).astype(np.int64)

def save_image_paths(image_paths: List[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for image_path in image_paths:
            f.write(str(image_path.resolve()) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing input .tif/.tiff images.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where outputs will be written.",
    )

    parser.add_argument(
        "--feature_type",
        type=str,
        default="superpoint",
        choices=["superpoint", "aliked"],
        help="Local feature extractor to use with LightGlue.",
    )

    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=4096,
        help="Maximum keypoints per image. Use -1 for no explicit cap.",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help=(
            "Resize largest image dimension before feature extraction. "
            "Omit this argument to keep original resolution."
        ),
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=0.3,
        help="LightGlue match filtering threshold.",
    )

    parser.add_argument(
        "--flash",
        action="store_true",
        help="Enable FlashAttention if available.",
    )
    
    parser.add_argument(
        "--fundamental_threshold_px",
        type=float,
        default=0.3,
        help="Fundamental matrix RANSAC inlier threshold in pixels.",
    )

    parser.add_argument(
        "--min_raw_matches",
        type=int,
        default=8,
        help="Minimum raw LightGlue matches required before RANSAC.",
    )

    parser.add_argument(
        "--min_inlier_matches",
        type=int,
        default=8,
        help="Minimum fundamental-matrix inlier matches required to keep a pair.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    features_dir = output_dir / "features"

    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise RuntimeError(f"Image directory does not exist: {image_dir}")

    image_paths = list_tif_images(image_dir)

    image_paths_txt = output_dir / "image_paths.txt"
    pairwise_matches_path = output_dir / "pairwise_matches.npy"

    save_image_paths(image_paths, image_paths_txt)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    print(f"Device: {device}")
    print(f"Found {len(image_paths)} TIFF images")
    print(f"Image order saved to: {image_paths_txt}")

    max_num_keypoints = None if args.max_keypoints < 0 else args.max_keypoints

    if args.feature_type == "superpoint":
        extractor = SuperPoint(
            max_num_keypoints=max_num_keypoints,
        ).eval().to(device)
    elif args.feature_type == "aliked":
        extractor = ALIKED(
            max_num_keypoints=max_num_keypoints,
        ).eval().to(device)
    else:
        raise ValueError(f"Unsupported feature_type: {args.feature_type}")

    matcher = LightGlue(
        features=args.feature_type,
        n_layers=9,
        filter_threshold=args.filter_threshold,
        depth_confidence=-1,
        width_confidence=-1,
        flash=args.flash,
    ).eval().to(device)

    lightglue_features, output_features = extract_lightglue_features(
        image_paths=image_paths,
        extractor=extractor,
        device=device,
        resize=args.resize,
        features_dir=features_dir,
    )

    pairwise_matches = match_all_pairs(
        lightglue_features=lightglue_features,
        output_features=output_features,
        matcher=matcher,
        log_label=f"{args.feature_type}+lightglue matches",
        fundamental_threshold_px=args.fundamental_threshold_px,
        min_raw_matches=args.min_raw_matches,
        min_inlier_matches=args.min_inlier_matches,
    )

    np.save(pairwise_matches_path, pairwise_matches)

    print("\nSaved outputs:")
    print(f"  Pairwise matches: {pairwise_matches_path}")
    print(f"  Image paths:      {image_paths_txt}")
    print(f"  Feature arrays:   {features_dir}")
    print(f"\nPairwise matches shape: {pairwise_matches.shape}")


if __name__ == "__main__":
    main()
