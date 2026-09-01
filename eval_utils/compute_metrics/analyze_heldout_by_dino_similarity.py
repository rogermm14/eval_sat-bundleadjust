#!/usr/bin/env python3
"""
Analyze external held-out RPC validation errors stratified by image similarity.

Save as:

    analyze_heldout_by_dino_similarity.py

Purpose:

    Post-process outputs from evaluate_heldout_fixed_rpcs.py and compute:

        held-out reprojection error
        stratified by DINO / image-similarity difficulty.

Supported similarity sources, in priority order:

    1. --dino_similarity_matrix
        Existing NxN cosine similarity matrix.

    2. --dino_embeddings
        Existing NxD embeddings. Cosine similarity is computed internally.

    3. --pairwise_similarity_csv
        Pairwise CSV with image index columns and a similarity column, e.g.

            image_1_index,image_2_index,dinov2,...

    4. --compute_dino_embeddings
        Compute DINOv2 embeddings directly from image_paths.txt.

Example using your pairwise CSV:

    python analyze_heldout_by_dino_similarity.py \
      --method raw:/path/to/raw_rpcs_eval \
      --method ames:/path/to/ames_rpcs_eval \
      --method satba:/path/to/satba_rpcs_eval \
      --pairwise_similarity_csv /path/to/pairwise_image_similarities.csv \
      --pairwise_similarity_column dinov2 \
      --output_dir /path/to/dino_robustness \
      --difficulty_metric holdout_to_train_min_similarity \
      --n_quantile_bins 5 \
      --paired_baseline_method ames \
      --paired_comparison_method satba

Example computing DINOv2 embeddings internally:

    python analyze_heldout_by_dino_similarity.py \
      --method raw:/path/to/raw_rpcs_eval \
      --method ames:/path/to/ames_rpcs_eval \
      --method satba:/path/to/satba_rpcs_eval \
      --matches_dir /path/to/superpoint_lightglue_matching/AOI \
      --compute_dino_embeddings \
      --save_dino_embeddings /path/to/dino_embeddings.npy \
      --save_dino_similarity_matrix /path/to/dino_similarity_matrix.npy \
      --output_dir /path/to/dino_robustness \
      --difficulty_metric holdout_to_train_min_similarity \
      --n_quantile_bins 5 \
      --paired_baseline_method ames \
      --paired_comparison_method satba

Outputs:

    output_dir/
    ├── image_outlier_scores.csv
    ├── image_outlier_scores.txt
    ├── heldout_errors_with_dino_difficulty.csv
    ├── summary_by_method_and_dino_bin.csv
    ├── paired_delta_by_dino_bin.csv
    └── summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Basic CSV helpers
# -----------------------------------------------------------------------------


def read_csv_dicts(path: Path) -> List[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_dicts(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def to_int(value, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def to_float(value, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Image path loading
# -----------------------------------------------------------------------------


def load_image_paths_txt(image_paths_txt: Path) -> List[Path]:
    image_paths_txt = Path(image_paths_txt)

    if not image_paths_txt.exists():
        raise FileNotFoundError(f"Missing image_paths.txt: {image_paths_txt}")

    with image_paths_txt.open("r", encoding="utf-8") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]

    if len(image_paths) == 0:
        raise RuntimeError(f"No image paths found in {image_paths_txt}")

    return image_paths


def resolve_image_paths_txt(
    matches_dir: Optional[Path],
    image_paths_txt: Optional[Path],
) -> Path:
    if image_paths_txt is not None:
        return Path(image_paths_txt)

    if matches_dir is not None:
        return Path(matches_dir) / "image_paths.txt"

    raise ValueError(
        "Need --image_paths_txt or --matches_dir when computing DINO embeddings."
    )


def maybe_load_image_paths(
    matches_dir: Optional[Path],
    image_paths_txt: Optional[Path],
    n_images_hint: Optional[int],
) -> Optional[List[Path]]:
    try:
        path = resolve_image_paths_txt(matches_dir=matches_dir, image_paths_txt=image_paths_txt)
    except Exception:
        return None

    image_paths = load_image_paths_txt(path)

    if n_images_hint is not None and len(image_paths) != n_images_hint:
        raise ValueError(
            f"Image count mismatch: image_paths.txt has {len(image_paths)} images, "
            f"but expected {n_images_hint} images."
        )

    return image_paths


# -----------------------------------------------------------------------------
# DINO embedding computation
# -----------------------------------------------------------------------------


def read_rgb_image_for_dino(path: Path):
    """
    Read an image as RGB PIL Image.

    Tries PIL first. If that fails, tries rasterio for GeoTIFF-like inputs.
    """
    from PIL import Image

    path = Path(path)

    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as pil_exc:
        try:
            import rasterio
            from PIL import Image

            with rasterio.open(path) as src:
                if src.count >= 3:
                    arr = src.read([1, 2, 3])
                elif src.count == 1:
                    band = src.read(1)
                    arr = np.stack([band, band, band], axis=0)
                else:
                    raise RuntimeError(f"Unsupported band count: {src.count}")

            arr = np.asarray(arr, dtype=np.float32)

            # Convert C,H,W to H,W,C.
            arr = np.transpose(arr, (1, 2, 0))

            # Robust contrast scaling for satellite images.
            out = np.zeros_like(arr, dtype=np.uint8)

            for c in range(3):
                band = arr[:, :, c]
                valid = np.isfinite(band)

                if not np.any(valid):
                    continue

                lo, hi = np.percentile(band[valid], [2, 98])

                if hi <= lo:
                    hi = lo + 1.0

                scaled = (band - lo) / (hi - lo)
                scaled = np.clip(scaled, 0.0, 1.0)
                out[:, :, c] = np.asarray(255.0 * scaled, dtype=np.uint8)

            return Image.fromarray(out, mode="RGB")

        except Exception as rasterio_exc:
            raise RuntimeError(
                f"Could not read image as PIL or rasterio: {path}\n"
                f"PIL error: {repr(pil_exc)}\n"
                f"rasterio error: {repr(rasterio_exc)}"
            )


def make_dino_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def compute_dino_embeddings_from_images(
    image_paths: List[Path],
    model_name: str,
    image_size: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
    except Exception as exc:
        raise ImportError(
            "PyTorch is required to compute DINO embeddings. "
            "Install torch and torchvision, or provide --dino_embeddings, "
            "--dino_similarity_matrix, or --pairwise_similarity_csv instead."
        ) from exc

    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "torchvision is required to compute DINO embeddings."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    device_obj = torch.device(device)

    print(f"Loading DINOv2 model through torch.hub: {model_name}")
    print("Note: the first run may require internet access unless the model is cached.")

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device_obj)

    transform = make_dino_transform(image_size)

    embeddings = []

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []

            for p in batch_paths:
                img = read_rgb_image_for_dino(p)
                x = transform(img)
                batch_tensors.append(x)

            batch = torch.stack(batch_tensors, dim=0).to(device_obj)

            feats = model(batch)

            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats:
                    feats = feats["x_norm_clstoken"]
                elif "x_prenorm" in feats:
                    feats = feats["x_prenorm"]
                else:
                    raise RuntimeError(f"Unexpected DINO output keys: {feats.keys()}")

            feats = feats.detach().cpu().numpy().astype(np.float64)
            embeddings.append(feats)

            print(
                f"Computed DINO embeddings for "
                f"{min(start + batch_size, len(image_paths))}/{len(image_paths)} images"
            )

    embeddings = np.vstack(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms

    return embeddings


def cosine_similarity_matrix_from_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings NxD, got shape {embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    E = embeddings / norms

    S = E @ E.T
    S = np.clip(S, -1.0, 1.0)

    return S


# -----------------------------------------------------------------------------
# Similarity matrix loading
# -----------------------------------------------------------------------------


def similarity_matrix_from_pairwise_csv(
    csv_path: Path,
    similarity_column: str,
    n_images: Optional[int],
    image_1_index_column: str,
    image_2_index_column: str,
) -> np.ndarray:
    rows = read_csv_dicts(csv_path)

    if len(rows) == 0:
        raise RuntimeError(f"No rows in pairwise similarity CSV: {csv_path}")

    required = [image_1_index_column, image_2_index_column, similarity_column]

    missing = [c for c in required if c not in rows[0]]

    if missing:
        available = sorted(rows[0].keys())
        raise ValueError(
            f"Missing columns in pairwise similarity CSV: {missing}\n"
            f"Available columns: {available}"
        )

    max_idx = -1

    parsed_rows = []

    for row in rows:
        i = to_int(row.get(image_1_index_column))
        j = to_int(row.get(image_2_index_column))
        sim = to_float(row.get(similarity_column))

        if i < 0 or j < 0 or not np.isfinite(sim):
            continue

        parsed_rows.append((i, j, sim))
        max_idx = max(max_idx, i, j)

    if len(parsed_rows) == 0:
        raise RuntimeError(
            f"No valid rows found in {csv_path} using column {similarity_column}"
        )

    if n_images is None:
        n_images = max_idx + 1

    S = np.full((n_images, n_images), np.nan, dtype=np.float64)
    np.fill_diagonal(S, 1.0)

    for i, j, sim in parsed_rows:
        if i >= n_images or j >= n_images:
            raise ValueError(
                f"Pairwise CSV has image index ({i}, {j}), but n_images={n_images}"
            )

        S[i, j] = sim
        S[j, i] = sim

    n_missing = int(np.sum(~np.isfinite(S)))

    if n_missing > 0:
        print(
            f"[warning] Similarity matrix has {n_missing} missing entries. "
            "Rows involving missing similarities will get NaN difficulty values."
        )

    return S


def load_or_compute_similarity_matrix(args: argparse.Namespace, n_images_hint: Optional[int]) -> np.ndarray:
    provided_sources = [
        args.dino_similarity_matrix is not None,
        args.dino_embeddings is not None,
        args.pairwise_similarity_csv is not None,
        args.compute_dino_embeddings,
    ]

    if sum(bool(x) for x in provided_sources) != 1:
        raise ValueError(
            "Choose exactly one similarity source:\n"
            "  --dino_similarity_matrix\n"
            "  --dino_embeddings\n"
            "  --pairwise_similarity_csv\n"
            "  --compute_dino_embeddings"
        )

    if args.dino_similarity_matrix is not None:
        S = np.load(args.dino_similarity_matrix)
        S = np.asarray(S, dtype=np.float64)

        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(
                f"Expected square similarity matrix, got shape {S.shape}"
            )

        return S

    if args.dino_embeddings is not None:
        embeddings = np.load(args.dino_embeddings)
        embeddings = np.asarray(embeddings, dtype=np.float64)
        S = cosine_similarity_matrix_from_embeddings(embeddings)
        return S

    if args.pairwise_similarity_csv is not None:
        S = similarity_matrix_from_pairwise_csv(
            csv_path=args.pairwise_similarity_csv,
            similarity_column=args.pairwise_similarity_column,
            n_images=n_images_hint,
            image_1_index_column=args.image_1_index_column,
            image_2_index_column=args.image_2_index_column,
        )
        return S

    if args.compute_dino_embeddings:
        image_paths_txt = resolve_image_paths_txt(
            matches_dir=args.matches_dir,
            image_paths_txt=args.image_paths_txt,
        )

        image_paths = load_image_paths_txt(image_paths_txt)

        if n_images_hint is not None and len(image_paths) != n_images_hint:
            raise ValueError(
                f"Image count mismatch: image_paths.txt has {len(image_paths)} images, "
                f"but method C.npy implies {n_images_hint} images."
            )

        embeddings = compute_dino_embeddings_from_images(
            image_paths=image_paths,
            model_name=args.dino_model_name,
            image_size=args.dino_image_size,
            batch_size=args.dino_batch_size,
            device=args.dino_device,
        )

        if args.save_dino_embeddings is not None:
            args.save_dino_embeddings.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_dino_embeddings, embeddings)
            print(f"Saved DINO embeddings: {args.save_dino_embeddings}")

        S = cosine_similarity_matrix_from_embeddings(embeddings)

        if args.save_dino_similarity_matrix is not None:
            args.save_dino_similarity_matrix.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_dino_similarity_matrix, S)
            print(f"Saved DINO similarity matrix: {args.save_dino_similarity_matrix}")

        return S

    raise RuntimeError("Internal error: no similarity source selected.")


# -----------------------------------------------------------------------------
# Track observation helpers
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


def pairwise_similarities(image_indices: List[int], S: np.ndarray) -> np.ndarray:
    values = []

    for i, j in combinations(sorted(set(image_indices)), 2):
        values.append(float(S[int(i), int(j)]))

    return np.asarray(values, dtype=np.float64)


def compute_similarity_difficulty_for_row(
    C: np.ndarray,
    S: np.ndarray,
    image_score_rows_by_index: Dict[int, dict],
    row: dict,
) -> dict:
    track_idx = to_int(row.get("track_idx"))
    holdout_image_idx = to_int(row.get("holdout_image_idx"))

    out = {
        "track_image_count": np.nan,
        "holdout_to_train_min_similarity": np.nan,
        "holdout_to_train_mean_similarity": np.nan,
        "holdout_to_train_max_similarity": np.nan,
        "track_min_pair_similarity": np.nan,
        "track_mean_pair_similarity": np.nan,
        "track_max_pair_similarity": np.nan,
        "heldout_image_mean_similarity_to_others": np.nan,
        "heldout_image_median_similarity_to_others": np.nan,
        "heldout_image_max_similarity_to_others": np.nan,
        "heldout_image_topk_mean_similarity": np.nan,
        "heldout_image_outlier_score": np.nan,
    }

    if track_idx < 0 or track_idx >= C.shape[1]:
        return out

    obs = observations_from_C_column(C, track_idx)

    if len(obs) == 0:
        return out

    image_indices = [int(o[0]) for o in obs]
    out["track_image_count"] = int(len(image_indices))

    if holdout_image_idx not in image_indices:
        return out

    image_row = image_score_rows_by_index.get(holdout_image_idx)
    if image_row is not None:
        out["heldout_image_mean_similarity_to_others"] = to_float(
            image_row.get("mean_similarity_to_others")
        )
        out["heldout_image_median_similarity_to_others"] = to_float(
            image_row.get("median_similarity_to_others")
        )
        out["heldout_image_max_similarity_to_others"] = to_float(
            image_row.get("max_similarity_to_others")
        )
        out["heldout_image_topk_mean_similarity"] = to_float(
            image_row.get("topk_mean_similarity")
        )
        out["heldout_image_outlier_score"] = to_float(
            image_row.get("outlier_score")
        )

    train_indices = [i for i in image_indices if i != holdout_image_idx]

    if len(train_indices) > 0:
        hvals = np.asarray(
            [float(S[holdout_image_idx, j]) for j in train_indices],
            dtype=np.float64,
        )
        hvals = hvals[np.isfinite(hvals)]

        if hvals.size > 0:
            out["holdout_to_train_min_similarity"] = float(np.min(hvals))
            out["holdout_to_train_mean_similarity"] = float(np.mean(hvals))
            out["holdout_to_train_max_similarity"] = float(np.max(hvals))

    pvals = pairwise_similarities(image_indices, S)
    pvals = pvals[np.isfinite(pvals)]

    if pvals.size > 0:
        out["track_min_pair_similarity"] = float(np.min(pvals))
        out["track_mean_pair_similarity"] = float(np.mean(pvals))
        out["track_max_pair_similarity"] = float(np.max(pvals))

    return out


# -----------------------------------------------------------------------------
# Binning and statistics
# -----------------------------------------------------------------------------


def make_quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        raise ValueError("No valid values available for quantile binning.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(valid, quantiles)

    unique_edges = [float(edges[0])]

    for x in edges[1:]:
        if float(x) > unique_edges[-1]:
            unique_edges.append(float(x))

    edges = np.asarray(unique_edges, dtype=np.float64)

    if edges.size < 2:
        raise ValueError(
            "Could not form at least one non-empty bin. "
            "All similarity values may be identical."
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def assign_bin(value: float, edges: np.ndarray) -> int:
    if not np.isfinite(value):
        return -1

    return int(np.searchsorted(edges, value, side="right") - 1)


def bin_label(bin_idx: int, edges: np.ndarray) -> str:
    if bin_idx < 0:
        return "invalid"

    lo = edges[bin_idx]
    hi = edges[bin_idx + 1]

    if np.isneginf(lo):
        return f"bin_{bin_idx:02d}_lowest_value_to_{hi:.6f}"

    if np.isposinf(hi):
        return f"bin_{bin_idx:02d}_{lo:.6f}_to_highest_value"

    return f"bin_{bin_idx:02d}_{lo:.6f}_to_{hi:.6f}"


def finite_array(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def error_stats(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]

    if errors.size == 0:
        return {
            "n": 0,
            "mean_error_px": np.nan,
            "median_error_px": np.nan,
            "rmse_error_px": np.nan,
            "p75_error_px": np.nan,
            "p90_error_px": np.nan,
            "p95_error_px": np.nan,
            "max_error_px": np.nan,
            "frac_below_0p5_px": np.nan,
            "frac_below_1px": np.nan,
            "frac_below_2px": np.nan,
        }

    return {
        "n": int(errors.size),
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "rmse_error_px": float(np.sqrt(np.mean(errors * errors))),
        "p75_error_px": float(np.percentile(errors, 75)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(np.max(errors)),
        "frac_below_0p5_px": float(np.mean(errors < 0.5)),
        "frac_below_1px": float(np.mean(errors < 1.0)),
        "frac_below_2px": float(np.mean(errors < 2.0)),
    }


def topk_finite(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return values

    k = max(1, int(k))
    k = min(k, int(values.size))
    idx = np.argsort(values)[-k:]
    return values[idx]


def compute_image_outlier_rows(
    S: np.ndarray,
    image_paths: Optional[List[Path]],
    top_k: int,
) -> List[dict]:
    n_images = S.shape[0]
    rows: List[dict] = []

    for image_idx in range(n_images):
        sims = np.asarray(S[image_idx], dtype=np.float64).copy()
        if image_idx < sims.size:
            sims[image_idx] = np.nan

        finite = sims[np.isfinite(sims)]
        topk = topk_finite(sims, top_k)

        basename = (
            image_paths[image_idx].name
            if image_paths is not None and image_idx < len(image_paths)
            else f"image_{image_idx:04d}"
        )

        row = {
            "image_idx": int(image_idx),
            "image_basename": basename,
            "n_valid_similarities": int(finite.size),
            "mean_similarity_to_others": float(np.mean(finite)) if finite.size else np.nan,
            "median_similarity_to_others": float(np.median(finite)) if finite.size else np.nan,
            "max_similarity_to_others": float(np.max(finite)) if finite.size else np.nan,
            "topk_used": int(topk.size),
            "topk_mean_similarity": float(np.mean(topk)) if topk.size else np.nan,
            "outlier_score": (
                float(1.0 - np.mean(topk))
                if topk.size and np.isfinite(np.mean(topk))
                else np.nan
            ),
        }
        rows.append(row)

    return rows


def write_image_outlier_scores_txt(path: Path, rows: List[dict], top_k: int) -> None:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -to_float(row.get("outlier_score"), default=-np.inf),
            to_int(row.get("image_idx"), default=10**9),
        ),
    )

    lines = []
    lines.append("Image-level DINO outlier scores")
    lines.append("===============================")
    lines.append("")
    lines.append(
        f"Higher outlier_score means the image is less similar to its top-{int(top_k)} "
        "nearest DINO neighbors within the AOI."
    )
    lines.append("")
    lines.append(
        "image_idx,image_basename,outlier_score,topk_mean_similarity,"
        "mean_similarity_to_others,median_similarity_to_others,max_similarity_to_others"
    )

    for row in sorted_rows:
        lines.append(
            ",".join(
                [
                    str(to_int(row.get("image_idx"))),
                    str(row.get("image_basename", "")),
                    f"{to_float(row.get('outlier_score')):.6f}" if np.isfinite(to_float(row.get("outlier_score"))) else "nan",
                    f"{to_float(row.get('topk_mean_similarity')):.6f}" if np.isfinite(to_float(row.get("topk_mean_similarity"))) else "nan",
                    f"{to_float(row.get('mean_similarity_to_others')):.6f}" if np.isfinite(to_float(row.get("mean_similarity_to_others"))) else "nan",
                    f"{to_float(row.get('median_similarity_to_others')):.6f}" if np.isfinite(to_float(row.get("median_similarity_to_others"))) else "nan",
                    f"{to_float(row.get('max_similarity_to_others')):.6f}" if np.isfinite(to_float(row.get("max_similarity_to_others"))) else "nan",
                ]
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_by_method_and_bin(rows: List[dict], edges: np.ndarray) -> List[dict]:
    methods = sorted(set(row["method"] for row in rows))
    bin_indices = sorted(
        set(
            to_int(row["similarity_bin_idx"])
            for row in rows
            if to_int(row["similarity_bin_idx"]) >= 0
        )
    )

    out_rows = []

    for method in methods:
        for b in bin_indices:
            sub = [
                row for row in rows
                if row["method"] == method
                and to_int(row["similarity_bin_idx"]) == b
            ]

            errors = finite_array([to_float(row.get("heldout_error_px")) for row in sub])
            sims = finite_array([to_float(row.get("difficulty_similarity")) for row in sub])

            stats = error_stats(errors)

            out = {
                "method": method,
                "similarity_bin_idx": b,
                "similarity_bin_label": bin_label(b, edges),
                "bin_similarity_min": float(np.min(sims)) if sims.size else np.nan,
                "bin_similarity_mean": float(np.mean(sims)) if sims.size else np.nan,
                "bin_similarity_max": float(np.max(sims)) if sims.size else np.nan,
            }

            out.update(stats)
            out_rows.append(out)

    return out_rows


def paired_delta_summary(
    rows: List[dict],
    baseline_method: str,
    comparison_method: str,
    edges: np.ndarray,
) -> List[dict]:
    """
    delta = error_baseline - error_comparison

    Positive delta means comparison_method has lower error.
    """
    key_to_row: Dict[Tuple[str, int, int], dict] = {}

    for row in rows:
        method = row["method"]
        track_idx = to_int(row.get("track_idx"))
        holdout_image_idx = to_int(row.get("holdout_image_idx"))

        if track_idx < 0 or holdout_image_idx < 0:
            continue

        key_to_row[(method, track_idx, holdout_image_idx)] = row

    paired = []

    for key, base_row in key_to_row.items():
        method, track_idx, holdout_image_idx = key

        if method != baseline_method:
            continue

        comp_row = key_to_row.get((comparison_method, track_idx, holdout_image_idx))

        if comp_row is None:
            continue

        e_base = to_float(base_row.get("heldout_error_px"))
        e_comp = to_float(comp_row.get("heldout_error_px"))

        if not np.isfinite(e_base) or not np.isfinite(e_comp):
            continue

        b = to_int(base_row.get("similarity_bin_idx"))

        paired.append(
            {
                "track_idx": track_idx,
                "holdout_image_idx": holdout_image_idx,
                "similarity_bin_idx": b,
                "delta_error_px": e_base - e_comp,
                "baseline_error_px": e_base,
                "comparison_error_px": e_comp,
            }
        )

    out_rows = []

    for b in sorted(set(row["similarity_bin_idx"] for row in paired if row["similarity_bin_idx"] >= 0)):
        sub = [row for row in paired if row["similarity_bin_idx"] == b]
        deltas = finite_array([row["delta_error_px"] for row in sub])

        if deltas.size == 0:
            continue

        out_rows.append(
            {
                "baseline_method": baseline_method,
                "comparison_method": comparison_method,
                "similarity_bin_idx": b,
                "similarity_bin_label": bin_label(b, edges),
                "n_paired": int(deltas.size),
                "median_delta_error_px": float(np.median(deltas)),
                "mean_delta_error_px": float(np.mean(deltas)),
                "p10_delta_error_px": float(np.percentile(deltas, 10)),
                "p90_delta_error_px": float(np.percentile(deltas, 90)),
                "fraction_comparison_better": float(np.mean(deltas > 0.0)),
                "fraction_comparison_worse": float(np.mean(deltas < 0.0)),
                "fraction_equal": float(np.mean(deltas == 0.0)),
            }
        )

    return out_rows


# -----------------------------------------------------------------------------
# Method parsing
# -----------------------------------------------------------------------------


def parse_method_arg(value: str) -> Tuple[str, Path]:
    if ":" not in value:
        raise ValueError(
            "Each --method must have format name:/path/to/eval_output_dir"
        )

    name, path = value.split(":", 1)
    name = name.strip()

    if not name:
        raise ValueError(f"Invalid method name in --method {value!r}")

    return name, Path(path)


def infer_n_images_from_first_method(method_specs: List[Tuple[str, Path]]) -> int:
    if len(method_specs) == 0:
        raise ValueError("No --method entries provided.")

    method_name, method_dir = method_specs[0]
    C_path = method_dir / "C.npy"

    if not C_path.exists():
        raise FileNotFoundError(f"Missing C.npy for method {method_name}: {C_path}")

    C = np.load(C_path)

    if C.ndim != 2 or C.shape[0] % 2 != 0:
        raise ValueError(f"Invalid C.npy shape for method {method_name}: {C.shape}")

    return C.shape[0] // 2


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help=(
            "Method and output directory from evaluate_heldout_fixed_rpcs.py. "
            "Format: name:/path/to/output_dir. "
            "Use multiple times, e.g. --method raw:/x/raw --method ames:/x/ames."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where analysis outputs will be written.",
    )

    # Similarity source options.
    parser.add_argument(
        "--dino_similarity_matrix",
        type=Path,
        default=None,
        help="Existing NxN DINO/image similarity matrix.",
    )

    parser.add_argument(
        "--dino_embeddings",
        type=Path,
        default=None,
        help="Existing NxD DINO embeddings. Cosine similarity matrix will be computed.",
    )

    parser.add_argument(
        "--pairwise_similarity_csv",
        type=Path,
        default=None,
        help=(
            "Pairwise similarity CSV. Expected columns include image indices and "
            "a similarity column such as dinov2."
        ),
    )

    parser.add_argument(
        "--pairwise_similarity_column",
        type=str,
        default="dinov2",
        help="Similarity column to use from --pairwise_similarity_csv.",
    )

    parser.add_argument(
        "--image_1_index_column",
        type=str,
        default="image_1_index",
        help="First image index column in --pairwise_similarity_csv.",
    )

    parser.add_argument(
        "--image_2_index_column",
        type=str,
        default="image_2_index",
        help="Second image index column in --pairwise_similarity_csv.",
    )

    parser.add_argument(
        "--compute_dino_embeddings",
        action="store_true",
        help="Compute DINOv2 embeddings internally from image_paths.txt.",
    )

    parser.add_argument(
        "--matches_dir",
        type=Path,
        default=None,
        help="Directory containing image_paths.txt. Used with --compute_dino_embeddings.",
    )

    parser.add_argument(
        "--image_paths_txt",
        type=Path,
        default=None,
        help="Explicit path to image_paths.txt. Used with --compute_dino_embeddings.",
    )

    parser.add_argument(
        "--save_dino_embeddings",
        type=Path,
        default=None,
        help="Optional path to save computed DINO embeddings.",
    )

    parser.add_argument(
        "--save_dino_similarity_matrix",
        type=Path,
        default=None,
        help="Optional path to save computed DINO similarity matrix.",
    )

    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="dinov2_vits14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
        help="DINOv2 model name for --compute_dino_embeddings.",
    )

    parser.add_argument(
        "--dino_image_size",
        type=int,
        default=518,
        help="Input crop size for DINOv2.",
    )

    parser.add_argument(
        "--dino_batch_size",
        type=int,
        default=8,
        help="Batch size for DINO embedding computation.",
    )

    parser.add_argument(
        "--dino_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for DINO embedding computation.",
    )

    # Analysis options.
    parser.add_argument(
        "--difficulty_metric",
        type=str,
        default="holdout_to_train_min_similarity",
        choices=[
            "holdout_to_train_min_similarity",
            "holdout_to_train_mean_similarity",
            "holdout_to_train_max_similarity",
            "track_min_pair_similarity",
            "track_mean_pair_similarity",
            "track_max_pair_similarity",
            "heldout_image_mean_similarity_to_others",
            "heldout_image_median_similarity_to_others",
            "heldout_image_max_similarity_to_others",
            "heldout_image_topk_mean_similarity",
            "heldout_image_outlier_score",
        ],
        help=(
            "Difficulty metric used to bin held-out rows. "
            "For similarity metrics, lower means harder. "
            "For heldout_image_outlier_score, higher means harder."
        ),
    )

    parser.add_argument(
        "--n_quantile_bins",
        type=int,
        default=5,
        help="Number of quantile bins for the selected difficulty metric.",
    )

    parser.add_argument(
        "--image_outlier_top_k",
        type=int,
        default=5,
        help="Top-k neighbors used for image-level DINO outlier scoring.",
    )

    parser.add_argument(
        "--paired_baseline_method",
        type=str,
        default="ames",
        help="Baseline method for paired deltas. Default: ames.",
    )

    parser.add_argument(
        "--paired_comparison_method",
        type=str,
        default="satba",
        help="Comparison method for paired deltas. Default: satba.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    method_specs = [parse_method_arg(v) for v in args.method]
    n_images = infer_n_images_from_first_method(method_specs)

    S = load_or_compute_similarity_matrix(args, n_images_hint=n_images)
    image_paths = maybe_load_image_paths(
        matches_dir=args.matches_dir,
        image_paths_txt=args.image_paths_txt,
        n_images_hint=n_images,
    )

    if S.shape[0] != n_images or S.shape[1] != n_images:
        raise ValueError(
            f"Similarity matrix shape {S.shape} does not match inferred image count {n_images}."
        )

    image_outlier_rows = compute_image_outlier_rows(
        S=S,
        image_paths=image_paths,
        top_k=args.image_outlier_top_k,
    )
    image_score_rows_by_index = {
        int(row["image_idx"]): row
        for row in image_outlier_rows
    }

    image_outlier_fieldnames = [
        "image_idx",
        "image_basename",
        "n_valid_similarities",
        "mean_similarity_to_others",
        "median_similarity_to_others",
        "max_similarity_to_others",
        "topk_used",
        "topk_mean_similarity",
        "outlier_score",
    ]

    write_csv_dicts(
        output_dir / "image_outlier_scores.csv",
        image_outlier_rows,
        image_outlier_fieldnames,
    )
    write_image_outlier_scores_txt(
        output_dir / "image_outlier_scores.txt",
        image_outlier_rows,
        top_k=args.image_outlier_top_k,
    )

    all_rows: List[dict] = []
    difficulty_values = []

    for method_name, method_dir in method_specs:
        method_dir = method_dir.resolve()

        C_path = method_dir / "C.npy"
        heldout_path = method_dir / "heldout_errors.csv"

        if not C_path.exists():
            raise FileNotFoundError(f"Missing C.npy for method {method_name}: {C_path}")

        if not heldout_path.exists():
            raise FileNotFoundError(
                f"Missing heldout_errors.csv for method {method_name}: {heldout_path}"
            )

        C = np.load(C_path)
        C = np.asarray(C, dtype=np.float64)

        if C.shape[0] // 2 != S.shape[0]:
            raise ValueError(
                f"Image count mismatch for method {method_name}: "
                f"C has {C.shape[0] // 2} images but similarity matrix is {S.shape}"
            )

        rows = read_csv_dicts(heldout_path)

        for row in rows:
            difficulty = compute_similarity_difficulty_for_row(
                C=C,
                S=S,
                image_score_rows_by_index=image_score_rows_by_index,
                row=row,
            )

            out = dict(row)
            out["method"] = method_name
            out["method_output_dir"] = str(method_dir)

            for k, v in difficulty.items():
                out[k] = v

            sim = to_float(out.get(args.difficulty_metric))
            out["difficulty_metric"] = args.difficulty_metric
            out["difficulty_similarity"] = sim

            if np.isfinite(sim):
                difficulty_values.append(sim)

            all_rows.append(out)

    if len(all_rows) == 0:
        raise RuntimeError("No held-out rows found.")

    edges = make_quantile_bins(
        np.asarray(difficulty_values, dtype=np.float64),
        n_bins=args.n_quantile_bins,
    )

    for row in all_rows:
        sim = to_float(row.get("difficulty_similarity"))
        b = assign_bin(sim, edges)

        row["similarity_bin_idx"] = b
        row["similarity_bin_label"] = bin_label(b, edges)

        # Backward-compatible aliases.
        row["dino_bin_idx"] = b
        row["dino_bin_label"] = bin_label(b, edges)

    detail_fieldnames = sorted(set(k for row in all_rows for k in row.keys()))

    write_csv_dicts(
        output_dir / "heldout_errors_with_dino_difficulty.csv",
        all_rows,
        detail_fieldnames,
    )

    summary_rows = summarize_by_method_and_bin(all_rows, edges)

    summary_fieldnames = [
        "method",
        "similarity_bin_idx",
        "similarity_bin_label",
        "bin_similarity_min",
        "bin_similarity_mean",
        "bin_similarity_max",
        "n",
        "mean_error_px",
        "median_error_px",
        "rmse_error_px",
        "p75_error_px",
        "p90_error_px",
        "p95_error_px",
        "max_error_px",
        "frac_below_0p5_px",
        "frac_below_1px",
        "frac_below_2px",
    ]

    write_csv_dicts(
        output_dir / "summary_by_method_and_dino_bin.csv",
        summary_rows,
        summary_fieldnames,
    )

    delta_rows = paired_delta_summary(
        rows=all_rows,
        baseline_method=args.paired_baseline_method,
        comparison_method=args.paired_comparison_method,
        edges=edges,
    )

    delta_fieldnames = [
        "baseline_method",
        "comparison_method",
        "similarity_bin_idx",
        "similarity_bin_label",
        "n_paired",
        "median_delta_error_px",
        "mean_delta_error_px",
        "p10_delta_error_px",
        "p90_delta_error_px",
        "fraction_comparison_better",
        "fraction_comparison_worse",
        "fraction_equal",
    ]

    write_csv_dicts(
        output_dir / "paired_delta_by_dino_bin.csv",
        delta_rows,
        delta_fieldnames,
    )

    source = "unknown"

    if args.dino_similarity_matrix is not None:
        source = f"dino_similarity_matrix: {args.dino_similarity_matrix}"
    elif args.dino_embeddings is not None:
        source = f"dino_embeddings: {args.dino_embeddings}"
    elif args.pairwise_similarity_csv is not None:
        source = (
            f"pairwise_similarity_csv: {args.pairwise_similarity_csv}, "
            f"column: {args.pairwise_similarity_column}"
        )
    elif args.compute_dino_embeddings:
        source = f"computed DINO embeddings with model: {args.dino_model_name}"

    lines = []
    lines.append("Held-out robustness analysis by image similarity")
    lines.append("================================================")
    lines.append("")
    lines.append(f"similarity source: {source}")
    lines.append(f"difficulty_metric: {args.difficulty_metric}")
    lines.append(f"n_quantile_bins requested: {args.n_quantile_bins}")
    lines.append(f"image_outlier_top_k: {args.image_outlier_top_k}")
    lines.append(f"actual bin edges: {edges.tolist()}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  For similarity metrics, lower means harder appearance/season-change case.")
    lines.append("  For heldout_image_outlier_score, higher means the held-out image is more globally unusual in the AOI.")
    lines.append("  For paired deltas, positive delta means the comparison method has lower error.")
    lines.append("")
    if image_paths is not None:
        lines.append(f"image_paths source: {resolve_image_paths_txt(args.matches_dir, args.image_paths_txt)}")
        lines.append("")
    lines.append("Methods:")
    for method_name, method_dir in method_specs:
        lines.append(f"  {method_name}: {method_dir}")
    lines.append("")
    lines.append("Outputs:")
    lines.append("  image_outlier_scores.csv")
    lines.append("  image_outlier_scores.txt")
    lines.append("  heldout_errors_with_dino_difficulty.csv")
    lines.append("  summary_by_method_and_dino_bin.csv")
    lines.append("  paired_delta_by_dino_bin.csv")

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  {output_dir / 'image_outlier_scores.csv'}")
    print(f"  {output_dir / 'image_outlier_scores.txt'}")
    print(f"  {output_dir / 'heldout_errors_with_dino_difficulty.csv'}")
    print(f"  {output_dir / 'summary_by_method_and_dino_bin.csv'}")
    print(f"  {output_dir / 'paired_delta_by_dino_bin.csv'}")
    print(f"  {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
