#!/usr/bin/env python3
"""
Refine LightGlue feature coordinates with Keypt2Subpx.

The input and output directory layouts match compute_lightglue_matches.py:

    matches_dir/
      image_paths.txt
      pairwise_matches.npy              # kp_i, kp_j, im_i, im_j
      features/<image_stem>.npy         # x, y, score, descriptor...

    output_dir/
      image_paths.txt
      pairwise_matches.npy              # copied unchanged
      features/<image_stem>.npy         # x, y refined, score, descriptor...

Keypt2Subpx is pairwise. A keypoint may receive several pair-specific refined
positions, one from each match edge. This script averages those positions per
(image, keypoint) so the existing feature-track builder can continue consuming
one coordinate per keypoint.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

torch = None
load_geotiff_for_lightglue = None


DETECTOR_FOR_FEATURE_TYPE = {
    "superpoint": "splg",
    "aliked": "aliked",
}


def load_image_paths(matches_dir: Path) -> List[Path]:
    path = matches_dir / "image_paths.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing image_paths.txt: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


def feature_paths_for(matches_dir: Path, image_paths: List[Path]) -> List[Path]:
    features_dir = matches_dir / "features"
    paths = [features_dir / f"{p.stem}.npy" for p in image_paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing feature files. First missing file: " + str(missing[0])
        )
    return paths


def load_features(matches_dir: Path, image_paths: List[Path]) -> List[np.ndarray]:
    return [
        np.asarray(np.load(path), dtype=np.float32)
        for path in feature_paths_for(matches_dir, image_paths)
    ]


def grouped_pair_matches(pairwise_matches: np.ndarray) -> Iterable[Tuple[int, int, np.ndarray]]:
    pairs = pairwise_matches[:, 2:4].astype(np.int64)
    unique_pairs = np.unique(pairs, axis=0)
    for im_i, im_j in unique_pairs:
        mask = (pairs[:, 0] == im_i) & (pairs[:, 1] == im_j)
        yield int(im_i), int(im_j), pairwise_matches[mask]


def sparse_score_map(
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    height: int,
    width: int,
    radius: int,
) -> torch.Tensor:
    score = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    offsets = []
    sigma = max(float(radius) / 2.0, 1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            w = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
            offsets.append((dx, dy, float(w)))

    for (x, y), s in zip(keypoints_xy, scores):
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        for dx, dy, w in offsets:
            xx = cx + dx
            yy = cy + dy
            if 0 <= xx < width and 0 <= yy < height:
                value = float(s) * w
                score[yy, xx] += value
                weight[yy, xx] += w

    valid = weight > 0
    score[valid] /= weight[valid]
    return torch.from_numpy(score[None, :, :])


def normalize_descriptors(desc: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(desc, dtype=np.float32))
    return torch.nn.functional.normalize(tensor, p=2, dim=1)


def refine_pair_in_batches(
    model,
    image_i: torch.Tensor,
    image_j: torch.Tensor,
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    features_i: np.ndarray,
    features_j: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    refined_i = np.empty((rows.shape[0], 2), dtype=np.float32)
    refined_j = np.empty((rows.shape[0], 2), dtype=np.float32)

    kp_i_all = rows[:, 0].astype(np.int64)
    kp_j_all = rows[:, 1].astype(np.int64)

    image_i = image_i.to(device)
    image_j = image_j.to(device)
    score_i = score_i.to(device)
    score_j = score_j.to(device)

    for start in range(0, rows.shape[0], batch_size):
        end = min(start + batch_size, rows.shape[0])
        kp_i = kp_i_all[start:end]
        kp_j = kp_j_all[start:end]

        keypt_i = torch.from_numpy(features_i[kp_i, 0:2]).to(device)
        keypt_j = torch.from_numpy(features_j[kp_j, 0:2]).to(device)
        desc_i = normalize_descriptors(features_i[kp_i, 3:]).to(device)
        desc_j = normalize_descriptors(features_j[kp_j, 3:]).to(device)

        with torch.inference_mode():
            out_i, out_j = model(
                keypt_i,
                keypt_j,
                image_i,
                image_j,
                desc_i,
                desc_j,
                score_i,
                score_j,
            )

        refined_i[start:end] = out_i.detach().cpu().numpy().astype(np.float32)
        refined_j[start:end] = out_j.detach().cpu().numpy().astype(np.float32)

    return refined_i, refined_j


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--feature_type", choices=sorted(DETECTOR_FOR_FEATURE_TYPE), required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--score_radius", type=int, default=3)
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--torch_hub_repo", default="KimSinjeong/keypt2subpx")
    parser.add_argument("--torch_hub_source", default="github", choices=["github", "local"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global torch
    global load_geotiff_for_lightglue
    try:
        import torch as torch_module
    except ImportError as exc:
        raise ImportError(
            "Keypt2Subpx refinement requires PyTorch. Install torch in the "
            "environment used to run this script."
        ) from exc

    try:
        from compute_lightglue_matches import load_geotiff_for_lightglue as load_image_fn
    except ImportError as exc:
        raise ImportError(
            "Could not import compute_lightglue_matches dependencies. This "
            "refinement script should run in the same environment used for "
            "LightGlue feature extraction."
        ) from exc

    torch = torch_module
    load_geotiff_for_lightglue = load_image_fn

    matches_dir = args.matches_dir.resolve()
    output_dir = args.output_dir.resolve()
    features_out_dir = output_dir / "features"

    pairwise_matches_path = matches_dir / "pairwise_matches.npy"
    if not pairwise_matches_path.exists():
        raise FileNotFoundError(f"Missing pairwise_matches.npy: {pairwise_matches_path}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}. "
            "Pass --force to overwrite refined feature files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    features_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    detector = DETECTOR_FOR_FEATURE_TYPE[args.feature_type]

    print("Loading Keypt2Subpx model")
    print(f"  repo/source: {args.torch_hub_repo} ({args.torch_hub_source})")
    print(f"  detector:    {detector}")
    print(f"  pretrained:  {not args.no_pretrained}")
    print(f"  device:      {device}")
    model = torch.hub.load(
        args.torch_hub_repo,
        "Keypt2Subpx",
        pretrained=not args.no_pretrained,
        detector=detector,
        source=args.torch_hub_source,
    )
    model = model.eval().to(device)

    image_paths = load_image_paths(matches_dir)
    features = load_features(matches_dir, image_paths)
    pairwise_matches = np.asarray(np.load(pairwise_matches_path), dtype=np.int64)
    if pairwise_matches.ndim != 2 or pairwise_matches.shape[1] != 4:
        raise ValueError(f"Expected pairwise_matches.npy to have shape Mx4, got {pairwise_matches.shape}")

    images: Dict[int, torch.Tensor] = {}
    scores: Dict[int, torch.Tensor] = {}
    refinements: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

    def get_image_and_score(image_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if image_idx not in images:
            image = load_geotiff_for_lightglue(image_paths[image_idx], resize=args.resize)
            _, height, width = image.shape
            feat = features[image_idx]
            score = sparse_score_map(
                keypoints_xy=feat[:, 0:2],
                scores=feat[:, 2],
                height=height,
                width=width,
                radius=args.score_radius,
            )
            images[image_idx] = image
            scores[image_idx] = score
        return images[image_idx], scores[image_idx]

    for pair_idx, (im_i, im_j, rows) in enumerate(grouped_pair_matches(pairwise_matches), start=1):
        print(f"[keypt2subpx] pair {pair_idx:04d}: {im_i}-{im_j}, matches={rows.shape[0]}")
        image_i, score_i = get_image_and_score(im_i)
        image_j, score_j = get_image_and_score(im_j)
        refined_i, refined_j = refine_pair_in_batches(
            model=model,
            image_i=image_i,
            image_j=image_j,
            score_i=score_i,
            score_j=score_j,
            features_i=features[im_i],
            features_j=features[im_j],
            rows=rows,
            batch_size=args.batch_size,
            device=device,
        )

        for local_idx, row in enumerate(rows):
            refinements[(im_i, int(row[0]))].append(refined_i[local_idx])
            refinements[(im_j, int(row[1]))].append(refined_j[local_idx])

    refined_features = [feat.copy() for feat in features]
    refined_count = 0
    displacements = []
    for (image_idx, keypoint_idx), coords in refinements.items():
        coords_arr = np.asarray(coords, dtype=np.float32)
        old_xy = refined_features[image_idx][keypoint_idx, 0:2].copy()
        new_xy = np.mean(coords_arr, axis=0)
        refined_features[image_idx][keypoint_idx, 0:2] = new_xy
        refined_count += 1
        displacements.append(float(np.linalg.norm(new_xy - old_xy)))

    for image_path, feat in zip(image_paths, refined_features):
        np.save(features_out_dir / f"{image_path.stem}.npy", feat.astype(np.float32, copy=False))

    shutil.copy2(matches_dir / "image_paths.txt", output_dir / "image_paths.txt")
    shutil.copy2(pairwise_matches_path, output_dir / "pairwise_matches.npy")

    displacement_arr = np.asarray(displacements, dtype=np.float64)
    summary_lines = [
        "Keypt2Subpx-refined LightGlue matches",
        "====================================",
        f"source_matches_dir: {matches_dir}",
        f"feature_type: {args.feature_type}",
        f"detector: {detector}",
        f"pairwise_matches: {pairwise_matches.shape[0]}",
        f"refined_unique_keypoints: {refined_count}",
        f"batch_size: {args.batch_size}",
        f"score_radius: {args.score_radius}",
        f"resize: {args.resize}",
    ]
    if displacement_arr.size:
        summary_lines.extend(
            [
                f"mean_displacement_px: {np.mean(displacement_arr):.6f}",
                f"median_displacement_px: {np.median(displacement_arr):.6f}",
                f"p95_displacement_px: {np.percentile(displacement_arr, 95):.6f}",
                f"max_displacement_px: {np.max(displacement_arr):.6f}",
            ]
        )
    (output_dir / "keypt2subpx_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nSaved Keypt2Subpx-refined matches:")
    print(f"  {output_dir / 'image_paths.txt'}")
    print(f"  {output_dir / 'pairwise_matches.npy'}")
    print(f"  {features_out_dir}")
    print(f"  {output_dir / 'keypt2subpx_summary.txt'}")


if __name__ == "__main__":
    main()
