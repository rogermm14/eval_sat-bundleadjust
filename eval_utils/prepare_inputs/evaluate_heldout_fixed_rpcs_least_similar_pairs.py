#!/usr/bin/env python3
"""
Held-out residual evaluation for fixed RPC sets using least-similar DINO pairs.

Save as:

    evaluate_heldout_fixed_rpcs_least_similar_pairs.py

Purpose:

    Compare raw RPCs, AMES RPCs, and sat-bundleadjust RPCs using the same
    SuperPoint/LightGlue tracks, while selecting the least-similar DINO image
    pairs instead of the most-similar DINO pairs.

Protocol:

    1. Load SuperPoint/LightGlue pairwise matches.
    2. Build sat-bundleadjust SatelliteImage objects from the input RPC set.
    3. Compute candidate pairs using sat-bundleadjust geometry filtering.
    4. Rank candidate pairs by DINO similarity.
    5. Select least-similar pairs with a K-neighbor greedy coverage rule.
    6. Build feature tracks only from those selected pairwise matches.
    7. For each track:
        - choose one held-out observation with --holdout_seed, or all holdouts
          if --all_holdouts is used;
        - optimize one lon/lat/alt point using only training observations;
        - keep RPCs fixed;
        - score the held-out observation.

Interpretation:

    This measures fixed-RPC held-out predictive consistency on difficult,
    visually less-similar image pairs. It is not absolute geolocation accuracy.

Required usage:

    python evaluate_heldout_fixed_rpcs_least_similar_pairs.py \
        --matches_dir /path/to/superpoint_lightglue_matching/AOI \
        --rpc_dir /path/to/rpcs \
        --sat_bundleadjust_repo /home/roger/sat-bundleadjust \
        --output_dir /path/to/output \
        --min_train_observations 3 \
        --least_pair_ranking_K 5 \
        --holdout_seed 0

Optional:

    --FT_max_length 10
    --all_holdouts
    --strict_loo_initialization
    --disable_pair_geometry_filter

Outputs:

    output_dir/
    ├── bundle_adjust.log
    ├── C.npy
    ├── C_v2.npy
    ├── pairwise_matches_used.npy
    ├── available_pairs.npy / .txt
    ├── candidate_pairs.npy / .txt
    ├── selected_least_similar_pairs.npy / .txt
    ├── dino_pair_scores.csv
    ├── holdout_local_indices.npy
    ├── holdout_selection.csv
    ├── full_track_initial_points_lon_lat_alt.npy
    ├── heldout_errors.csv
    ├── heldout_errors.npy
    ├── per_camera_heldout_error_stats.csv
    ├── heldout_error_by_track_length.csv
    ├── summary.txt
    └── rpcs/
        └── copies of input RPCs
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import inspect
import sys
import timeit
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rpcm

try:
    from scipy.optimize import least_squares
except Exception as exc:
    least_squares = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# RPC loading
# -----------------------------------------------------------------------------


def rpc_from_ames_adjusted_xml(xml_path: Path) -> rpcm.RPCModel:
    xml_path = Path(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def find_text(tag: str) -> str:
        elem = root.find(f".//{tag}")
        if elem is None or elem.text is None:
            raise ValueError(f"Missing XML tag <{tag}> in {xml_path}")
        return elem.text.strip()

    def find_float(tag: str) -> float:
        return float(find_text(tag))

    def find_coeffs(tag: str) -> str:
        coeffs = [float(x) for x in find_text(tag).replace(",", " ").split()]
        if len(coeffs) != 20:
            raise ValueError(
                f"Expected 20 coefficients in <{tag}> in {xml_path}, "
                f"got {len(coeffs)}"
            )
        return " ".join(f"{c:.17g}" for c in coeffs)

    rpc_dict = {
        "LINE_OFF": find_float("LINEOFFSET"),
        "SAMP_OFF": find_float("SAMPOFFSET"),
        "LAT_OFF": find_float("LATOFFSET"),
        "LONG_OFF": find_float("LONGOFFSET"),
        "HEIGHT_OFF": find_float("HEIGHTOFFSET"),
        "LINE_SCALE": find_float("LINESCALE"),
        "SAMP_SCALE": find_float("SAMPSCALE"),
        "LAT_SCALE": find_float("LATSCALE"),
        "LONG_SCALE": find_float("LONGSCALE"),
        "HEIGHT_SCALE": find_float("HEIGHTSCALE"),
        "LINE_NUM_COEFF": find_coeffs("LINENUMCOEF"),
        "LINE_DEN_COEFF": find_coeffs("LINEDENCOEF"),
        "SAMP_NUM_COEFF": find_coeffs("SAMPNUMCOEF"),
        "SAMP_DEN_COEFF": find_coeffs("SAMPDENCOEF"),
    }

    try:
        rpc_dict["ERR_BIAS"] = find_float("ERRBIAS")
    except Exception:
        rpc_dict["ERR_BIAS"] = -1.0

    try:
        rpc_dict["ERR_RAND"] = find_float("ERRRAND")
    except Exception:
        rpc_dict["ERR_RAND"] = -1.0

    return rpcm.RPCModel(rpc_dict)


def load_rpc_model_any_format(rpc_path: Path) -> rpcm.RPCModel:
    rpc_path = Path(rpc_path)

    if rpc_path.suffix.lower() == ".xml":
        return rpc_from_ames_adjusted_xml(rpc_path)

    return rpcm.rpc_from_rpc_file(str(rpc_path))


def find_rpc_file_for_image(rpc_dir: Path, image_path: Path) -> Optional[Path]:
    stem = image_path.stem

    candidates = [
        rpc_dir / f"{stem}.rpc",
        rpc_dir / f"{stem}.rpc_adj",
        rpc_dir / f"{stem}.xml",
        rpc_dir / f"{stem}.adjusted_rpc.xml",
        rpc_dir / f"run-{stem}.adjusted_rpc.xml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    patterns = [
        str(rpc_dir / f"*{stem}*.adjusted_rpc.xml"),
        str(rpc_dir / f"*{stem}*.xml"),
    ]

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    matches = sorted(set(matches))

    if len(matches) == 1:
        return Path(matches[0])

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous RPC match for image {image_path.name}. Found:\n"
            + "\n".join(matches[:50])
        )

    return None


def load_rpcs_for_images(rpc_dir: Path, image_paths: List[Path]) -> List[rpcm.RPCModel]:
    rpcs: List[rpcm.RPCModel] = []
    missing: List[Path] = []
    loaded_paths: List[Path] = []

    for image_path in image_paths:
        rpc_path = find_rpc_file_for_image(rpc_dir, image_path)

        if rpc_path is None:
            missing.append(image_path)
            continue

        try:
            rpc = load_rpc_model_any_format(rpc_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load RPC for {image_path.name}\n"
                f"RPC file: {rpc_path}\n"
                f"Error: {exc}"
            ) from exc

        rpcs.append(rpc)
        loaded_paths.append(rpc_path)

    if missing:
        msg = "\n".join(
            f"{p.name} -> expected .rpc, .rpc_adj, .xml, or adjusted_rpc.xml"
            for p in missing[:50]
        )
        raise FileNotFoundError(f"Missing RPCs:\n{msg}")

    print("Loaded RPC files:")
    for image_path, rpc_path in zip(image_paths, loaded_paths):
        print(f"  {image_path.name} -> {rpc_path.name}")

    return rpcs


# -----------------------------------------------------------------------------
# Matches and features
# -----------------------------------------------------------------------------


def load_image_paths(matches_dir: Path) -> List[Path]:
    image_paths_txt = matches_dir / "image_paths.txt"

    if not image_paths_txt.exists():
        raise FileNotFoundError(f"Missing image_paths.txt: {image_paths_txt}")

    with image_paths_txt.open("r", encoding="utf-8") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]

    if len(image_paths) == 0:
        raise RuntimeError(f"No image paths found in {image_paths_txt}")

    return image_paths


def get_feature_paths(matches_dir: Path, image_paths: List[Path]) -> List[str]:
    features_dir = matches_dir / "features"

    if not features_dir.exists():
        raise FileNotFoundError(f"Missing features directory: {features_dir}")

    feature_paths: List[str] = []
    missing: List[Path] = []

    for image_path in image_paths:
        feature_path = features_dir / f"{image_path.stem}.npy"

        if not feature_path.exists():
            missing.append(feature_path)

        feature_paths.append(str(feature_path))

    if missing:
        msg = "\n".join(str(p) for p in missing[:50])
        raise FileNotFoundError(f"Missing feature files:\n{msg}")

    return feature_paths


def load_pairwise_matches(matches_dir: Path) -> np.ndarray:
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
    if pairwise_matches.shape[0] == 0:
        return []

    pairs = pairwise_matches[:, 2:4].astype(int)
    pairs = np.unique(pairs, axis=0)

    normalized = []
    for i, j in pairs:
        i = int(i)
        j = int(j)
        normalized.append((min(i, j), max(i, j)))

    return sorted(set(normalized))


def filter_pairwise_matches_to_pairs(
    pairwise_matches: np.ndarray,
    selected_pairs: List[Tuple[int, int]],
) -> np.ndarray:
    selected_set = set(
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in selected_pairs
    )

    keep = []
    for row in pairwise_matches:
        im_i = int(row[2])
        im_j = int(row[3])
        pair = (min(im_i, im_j), max(im_i, im_j))
        keep.append(pair in selected_set)

    keep = np.asarray(keep, dtype=bool)
    out = pairwise_matches[keep].astype(np.int64)

    print("Filtered pairwise matches using selected pairs:")
    print(f"  before: {pairwise_matches.shape[0]}")
    print(f"  after:  {out.shape[0]}")

    return out


# -----------------------------------------------------------------------------
# sat-bundleadjust imports and pair selection
# -----------------------------------------------------------------------------


def import_sat_bundleadjust(repo_path: Path):
    repo_path = repo_path.resolve()

    if not repo_path.exists():
        raise FileNotFoundError(f"sat-bundleadjust repo not found: {repo_path}")

    sys.path.insert(0, str(repo_path))

    try:
        from bundle_adjust import cam_utils, geo_utils
        from bundle_adjust.feature_tracks import ft_match

        try:
            from bundle_adjust.feature_tracks import ft_pair_ranking
        except Exception:
            # Some public versions use a different filename. The user’s local
            # branch likely has ft_pair_ranking, but keep this fallback.
            ft_pair_ranking = None

        try:
            from bundle_adjust.feature_tracks.ft_pair_ranking import select_optimal_pairs_to_match
        except Exception:
            select_optimal_pairs_to_match = None

        from bundle_adjust.feature_tracks.ft_triangulate import init_pts3d

    except Exception as exc:
        raise ImportError(
            "Could not import sat-bundleadjust components. Make sure "
            f"--sat_bundleadjust_repo points to the repository root: {repo_path}"
        ) from exc

    return (
        cam_utils,
        geo_utils,
        ft_match,
        ft_pair_ranking,
        select_optimal_pairs_to_match,
        init_pts3d,
    )


def make_satellite_images(
    image_paths: List[Path],
    rpcs: List[rpcm.RPCModel],
    cam_utils,
) -> list:
    images = []

    for image_path, rpc in zip(image_paths, rpcs):
        im = cam_utils.SatelliteImage(str(image_path), rpc)
        images.append(im)

    return images


def initialize_image_geometry(images: list) -> None:
    print("Estimating camera centers with sat-bundleadjust SatelliteImage methods...")

    for idx, im in enumerate(images):
        try:
            im.set_camera_center()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to estimate camera center for image {idx}: {im.geotiff_path}\n"
                f"Error: {exc}"
            ) from exc

    print("Estimating footprints for pair selection...")

    for idx, im in enumerate(images):
        try:
            lon = im.rpc.lon_offset
            lat = im.rpc.lat_offset

            try:
                import srtm4

                alt = float(srtm4.srtm4(lon, lat))
            except Exception:
                alt = float(im.rpc.alt_offset)

            im.set_footprint(alt=alt)

        except Exception as exc:
            print(
                f"[warning] Could not set footprint for image {idx}: "
                f"{im.geotiff_path}. Error: {exc}"
            )


def compute_candidate_pairs(
    images: list,
    available_pairs: List[Tuple[int, int]],
    ft_match,
    geo_utils,
    filter_pairs: bool,
) -> List[Tuple[int, int]]:
    available_pairs = sorted(
        set((min(int(i), int(j)), max(int(i), int(j))) for i, j in available_pairs)
    )

    utm_poly = lambda im: {
        "geojson": geo_utils.utm_geojson_from_lonlat_geojson(im.lonlat_geojson),
        "z": im.alt,
    }

    footprints = [utm_poly(im) for im in images]
    optical_centers = [im.center for im in images]

    if filter_pairs:
        _, pairs_to_triangulate = ft_match.compute_pairs_to_match(
            available_pairs,
            footprints,
            optical_centers,
        )
    else:
        _, pairs_to_triangulate = ft_match.compute_pairs_to_match(
            available_pairs,
            footprints,
            optical_centers,
            min_overlap=0,
            min_baseline=0,
        )

    candidate_pairs = sorted(
        set((min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs_to_triangulate)
    )

    if len(candidate_pairs) == 0:
        raise RuntimeError("No candidate pairs available after geometry filtering")

    return candidate_pairs


def _normalize_pair_scores(
    scores_obj,
    candidate_pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """
    Convert several possible score-return formats into:

        {(i, j): score}

    Higher score is assumed to mean more similar.
    """
    candidate_pairs = [
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in candidate_pairs
    ]

    if isinstance(scores_obj, dict):
        out = {}
        for k, v in scores_obj.items():
            if isinstance(k, tuple) and len(k) == 2:
                pair = (min(int(k[0]), int(k[1])), max(int(k[0]), int(k[1])))
                out[pair] = float(v)
        return out

    arr = np.asarray(scores_obj)

    # Case: one score per candidate pair.
    if arr.ndim == 1 and arr.shape[0] == len(candidate_pairs):
        return {
            pair: float(score)
            for pair, score in zip(candidate_pairs, arr)
        }

    # Case: Nx3 rows = i, j, score.
    if arr.ndim == 2 and arr.shape[1] >= 3:
        out = {}
        for row in arr:
            i = int(row[0])
            j = int(row[1])
            pair = (min(i, j), max(i, j))
            out[pair] = float(row[2])
        return out

    # Case: square similarity matrix.
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        out = {}
        for i, j in candidate_pairs:
            out[(i, j)] = float(arr[i, j])
        return out

    raise ValueError(
        "Could not normalize DINO pair scores. Unsupported return format: "
        f"type={type(scores_obj)}, shape={getattr(scores_obj, 'shape', None)}"
    )


def _try_call_score_function(
    fn: Callable,
    candidate_pairs: List[Tuple[int, int]],
    images: list,
) -> Optional[Dict[Tuple[int, int], float]]:
    """
    Try to call a score-producing function with several likely signatures.
    """
    candidate_pairs = [
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in candidate_pairs
    ]

    call_patterns = [
        lambda: fn(candidate_pairs, images),
        lambda: fn(images, candidate_pairs),
        lambda: fn(candidate_pairs=candidate_pairs, images=images),
        lambda: fn(pairs=candidate_pairs, images=images),
        lambda: fn(images=images, pairs=candidate_pairs),
    ]

    for call in call_patterns:
        try:
            result = call()
            scores = _normalize_pair_scores(result, candidate_pairs)
            if len(scores) > 0:
                return scores
        except TypeError:
            continue
        except Exception:
            continue

    return None


def compute_dino_pair_scores_from_satba(
    ft_pair_ranking,
    candidate_pairs: List[Tuple[int, int]],
    images: list,
) -> Dict[Tuple[int, int], float]:
    """
    Compute DINO similarity scores for candidate pairs using the public functions
    exposed by the local sat-bundleadjust ft_pair_ranking module.

    Required ft_pair_ranking functions:
        extract_DINO_embeddings(images)
        cosine_similarity_np(a, b)

    Higher score means more visually similar.
    """
    if ft_pair_ranking is None:
        raise RuntimeError(
            "Could not import bundle_adjust.feature_tracks.ft_pair_ranking."
        )

    if not hasattr(ft_pair_ranking, "extract_DINO_embeddings"):
        raise RuntimeError(
            "ft_pair_ranking does not expose extract_DINO_embeddings(images)."
        )

    if not hasattr(ft_pair_ranking, "cosine_similarity_np"):
        raise RuntimeError(
            "ft_pair_ranking does not expose cosine_similarity_np(a, b)."
        )

    print("Computing DINO embeddings with ft_pair_ranking.extract_DINO_embeddings...")
    embeddings = ft_pair_ranking.extract_DINO_embeddings(images)

    if isinstance(embeddings, dict):
        get_embedding = lambda idx: embeddings[idx]
        n_embeddings = len(embeddings)
    else:
        embeddings = list(embeddings)
        get_embedding = lambda idx: embeddings[idx]
        n_embeddings = len(embeddings)

    if n_embeddings < len(images):
        raise RuntimeError(
            f"extract_DINO_embeddings returned {n_embeddings} embeddings "
            f"for {len(images)} images."
        )

    print("Computing pairwise DINO cosine similarities...")
    scores: Dict[Tuple[int, int], float] = {}

    for i, j in candidate_pairs:
        i = int(i)
        j = int(j)

        emb_i = np.asarray(get_embedding(i), dtype=np.float64).reshape(-1)
        emb_j = np.asarray(get_embedding(j), dtype=np.float64).reshape(-1)

        score = ft_pair_ranking.cosine_similarity_np(emb_i, emb_j)

        # cosine_similarity_np may return scalar, array, or 1x1 array.
        score = float(np.asarray(score).squeeze())

        pair = (min(i, j), max(i, j))
        scores[pair] = score

    print(f"Computed DINO similarity scores for {len(scores)} pairs.")

    return scores


def select_least_similar_pairs_from_scores(
    candidate_pairs: List[Tuple[int, int]],
    scores: Dict[Tuple[int, int], float],
    n_images: int,
    K: int,
) -> List[Tuple[int, int]]:
    """
    Select least-similar pairs with K-neighbor coverage.

    Greedy rule:
        - sort candidate pairs by ascending DINO similarity
        - add a pair if at least one endpoint still has degree < K
        - continue until every image reaches degree >= K or pairs run out

    This mirrors the spirit of selecting K best connections per image, but uses
    the least-similar pairs instead of the most-similar pairs.
    """
    candidate_pairs = [
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in candidate_pairs
    ]

    missing = [p for p in candidate_pairs if p not in scores]
    if missing:
        raise RuntimeError(
            f"DINO scores are missing for {len(missing)} candidate pairs. "
            f"First missing pair: {missing[0]}"
        )

    sorted_pairs = sorted(candidate_pairs, key=lambda p: (scores[p], p[0], p[1]))

    degree = np.zeros(n_images, dtype=np.int64)
    selected: List[Tuple[int, int]] = []

    for i, j in sorted_pairs:
        if degree[i] < K or degree[j] < K:
            selected.append((i, j))
            degree[i] += 1
            degree[j] += 1

        if np.all(degree >= K):
            break

    if len(selected) == 0:
        raise RuntimeError("Least-similar pair selection returned zero pairs")

    print("Least-similar pair selection:")
    print(f"  candidate pairs: {len(candidate_pairs)}")
    print(f"  selected pairs:  {len(selected)}")
    print(f"  target K:         {K}")
    print(f"  min degree:       {int(np.min(degree))}")
    print(f"  max degree:       {int(np.max(degree))}")

    return selected


def select_similar_pairs_from_scores(
    candidate_pairs: List[Tuple[int, int]],
    scores: Dict[Tuple[int, int], float],
    n_images: int,
    K: int,
    mode: str,
) -> List[Tuple[int, int]]:
    """
    Select least- or most-similar pairs with K-neighbor coverage.
    """
    if mode == "least":
        return select_least_similar_pairs_from_scores(
            candidate_pairs=candidate_pairs,
            scores=scores,
            n_images=n_images,
            K=K,
        )

    if mode != "most":
        raise ValueError(f"Unsupported similar-pair selection mode: {mode}")

    candidate_pairs = [
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in candidate_pairs
    ]

    missing = [p for p in candidate_pairs if p not in scores]
    if missing:
        raise RuntimeError(
            f"DINO scores are missing for {len(missing)} candidate pairs. "
            f"First missing pair: {missing[0]}"
        )

    sorted_pairs = sorted(candidate_pairs, key=lambda p: (-scores[p], p[0], p[1]))

    degree = np.zeros(n_images, dtype=np.int64)
    selected: List[Tuple[int, int]] = []

    for i, j in sorted_pairs:
        if degree[i] < K or degree[j] < K:
            selected.append((i, j))
            degree[i] += 1
            degree[j] += 1

        if np.all(degree >= K):
            break

    if len(selected) == 0:
        raise RuntimeError("Most-similar pair selection returned zero pairs")

    print("Most-similar pair selection:")
    print(f"  candidate pairs: {len(candidate_pairs)}")
    print(f"  selected pairs:  {len(selected)}")
    print(f"  target K:         {K}")
    print(f"  min degree:       {int(np.min(degree))}")
    print(f"  max degree:       {int(np.max(degree))}")

    return selected


def write_pair_scores_csv(
    path: Path,
    candidate_pairs: List[Tuple[int, int]],
    scores: Dict[Tuple[int, int], float],
    selected_pairs: List[Tuple[int, int]],
    mode: str = "least",
) -> None:
    selected_set = set(selected_pairs)
    selected_column = f"selected_{mode}_similar"
    if mode == "most":
        sort_key = lambda p: (-scores[p], p[0], p[1])
    else:
        sort_key = lambda p: (scores[p], p[0], p[1])

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["im_i", "im_j", "dino_similarity", selected_column]
        if mode != "least":
            fieldnames.append("selected_least_similar")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, j in sorted(candidate_pairs, key=sort_key):
            row = {
                "im_i": int(i),
                "im_j": int(j),
                "dino_similarity": float(scores[(i, j)]),
                selected_column: int((i, j) in selected_set),
            }
            if mode != "least":
                row["selected_least_similar"] = 0
            writer.writerow(row)


def select_least_similar_pairs_with_satba_dino(
    images: list,
    available_pairs: List[Tuple[int, int]],
    ft_match,
    geo_utils,
    ft_pair_ranking,
    K: int,
    filter_pairs: bool,
    scores_csv_path: Path,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Select least-similar DINO pairs.

    Returns:
        selected_pairs, candidate_pairs
    """
    available_pairs = sorted(
        set((min(int(i), int(j)), max(int(i), int(j))) for i, j in available_pairs)
    )

    print("\nSelecting least-similar DINO pairs...")
    print(f"  available pairs from pairwise_matches.npy: {len(available_pairs)}")
    print(f"  least_pair_ranking_K: {K}")
    print(f"  geometry filtering: {filter_pairs}")

    candidate_pairs = compute_candidate_pairs(
        images=images,
        available_pairs=available_pairs,
        ft_match=ft_match,
        geo_utils=geo_utils,
        filter_pairs=filter_pairs,
    )

    print(f"  candidate pairs after geometry filtering: {len(candidate_pairs)}")

    scores = compute_dino_pair_scores_from_satba(
        ft_pair_ranking=ft_pair_ranking,
        candidate_pairs=candidate_pairs,
        images=images,
    )

    selected_pairs = select_least_similar_pairs_from_scores(
        candidate_pairs=candidate_pairs,
        scores=scores,
        n_images=len(images),
        K=K,
    )

    write_pair_scores_csv(
        path=scores_csv_path,
        candidate_pairs=candidate_pairs,
        scores=scores,
        selected_pairs=selected_pairs,
    )

    return selected_pairs, candidate_pairs


def select_similar_pairs_with_satba_dino(
    images: list,
    available_pairs: List[Tuple[int, int]],
    ft_match,
    geo_utils,
    ft_pair_ranking,
    K: int,
    mode: str,
    filter_pairs: bool,
    scores_csv_path: Path,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if mode == "least":
        return select_least_similar_pairs_with_satba_dino(
            images=images,
            available_pairs=available_pairs,
            ft_match=ft_match,
            geo_utils=geo_utils,
            ft_pair_ranking=ft_pair_ranking,
            K=K,
            filter_pairs=filter_pairs,
            scores_csv_path=scores_csv_path,
        )

    if mode != "most":
        raise ValueError(f"Unsupported similar-pair selection mode: {mode}")

    available_pairs = sorted(
        set((min(int(i), int(j)), max(int(i), int(j))) for i, j in available_pairs)
    )

    print("\nSelecting most-similar DINO pairs...")
    print(f"  available pairs from pairwise_matches.npy: {len(available_pairs)}")
    print(f"  pair_ranking_K: {K}")
    print(f"  geometry filtering: {filter_pairs}")

    candidate_pairs = compute_candidate_pairs(
        images=images,
        available_pairs=available_pairs,
        ft_match=ft_match,
        geo_utils=geo_utils,
        filter_pairs=filter_pairs,
    )

    print(f"  candidate pairs after geometry filtering: {len(candidate_pairs)}")

    scores = compute_dino_pair_scores_from_satba(
        ft_pair_ranking=ft_pair_ranking,
        candidate_pairs=candidate_pairs,
        images=images,
    )

    selected_pairs = select_similar_pairs_from_scores(
        candidate_pairs=candidate_pairs,
        scores=scores,
        n_images=len(images),
        K=K,
        mode=mode,
    )

    write_pair_scores_csv(
        path=scores_csv_path,
        candidate_pairs=candidate_pairs,
        scores=scores,
        selected_pairs=selected_pairs,
        mode=mode,
    )

    return selected_pairs, candidate_pairs


# -----------------------------------------------------------------------------
# Feature-track construction
# -----------------------------------------------------------------------------


def filter_C_using_pairs_to_triangulate_local(
    C: np.ndarray,
    pairs_to_triangulate: List[Tuple[int, int]],
) -> np.ndarray:
    columns_to_preserve = []

    visible = ~np.isnan(C[::2])
    pairs_to_triangulate_set = set(tuple(p) for p in pairs_to_triangulate)

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
    feature_paths: List[str],
    pairwise_matches: np.ndarray,
    pairs_to_triangulate: List[Tuple[int, int]],
    reject_conflicted_tracks: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    pairwise_matches = np.asarray(pairwise_matches, dtype=np.int64)

    if pairwise_matches.ndim != 2 or pairwise_matches.shape[1] != 4:
        raise ValueError(
            f"Expected pairwise_matches shape Mx4, got {pairwise_matches.shape}"
        )

    n_cams = len(feature_paths)

    features: List[np.ndarray] = []
    feature_ids: List[np.ndarray] = []

    global_id_start = 0

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

        print(
            f"  image {im_idx:04d}: {Path(feature_path).name}, "
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

    print("Merging pairwise matches into feature tracks...")

    for row_idx in range(pairwise_matches.shape[0]):
        kp_i, kp_j, im_i, im_j = pairwise_matches[row_idx]

        if im_i < 0 or im_i >= n_cams or im_j < 0 or im_j >= n_cams:
            raise IndexError(
                f"Invalid image index in row {row_idx}: "
                f"im_i={im_i}, im_j={im_j}, n_cams={n_cams}"
            )

        if kp_i < 0 or kp_i >= features[im_i].shape[0]:
            raise IndexError(
                f"Invalid kp_i in row {row_idx}: kp_i={kp_i}, "
                f"image {im_i} has {features[im_i].shape[0]} keypoints"
            )

        if kp_j < 0 or kp_j >= features[im_j].shape[0]:
            raise IndexError(
                f"Invalid kp_j in row {row_idx}: kp_j={kp_j}, "
                f"image {im_j} has {features[im_j].shape[0]} keypoints"
            )

        global_i = int(feature_ids[im_i][kp_i])
        global_j = int(feature_ids[im_j][kp_j])

        union(global_i, global_j)

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

    print(f"Initial number of tracks with >=2 keypoints: {n_tracks}")

    C = np.full((2 * n_cams, n_tracks), np.nan, dtype=np.float64)
    C_v2 = np.full((n_cams, n_tracks), np.nan, dtype=np.float64)

    conflicting_track_indices = set()
    conflicting_track_image_pairs = set()
    same_camera_conflict_events = 0

    print("Filling C and C_v2...")

    for row_idx in range(pairwise_matches.shape[0]):
        kp_i, kp_j, im_i, im_j = pairwise_matches[row_idx]

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
        elif int(C_v2[im_i, track_idx]) != int(kp_i):
            same_camera_conflict_events += 1
            conflicting_track_indices.add(track_idx)
            conflicting_track_image_pairs.add((track_idx, int(im_i)))

        if np.isnan(C_v2[im_j, track_idx]):
            C[2 * im_j, track_idx] = float(features[im_j][kp_j, 0])
            C[2 * im_j + 1, track_idx] = float(features[im_j][kp_j, 1])
            C_v2[im_j, track_idx] = float(kp_j)
        elif int(C_v2[im_j, track_idx]) != int(kp_j):
            same_camera_conflict_events += 1
            conflicting_track_indices.add(track_idx)
            conflicting_track_image_pairs.add((track_idx, int(im_j)))

    print(f"Same-camera keypoint conflict events: {same_camera_conflict_events}")
    print(f"Unique track-image conflicts: {len(conflicting_track_image_pairs)}")
    print(f"Tracks with same-camera conflicts: {len(conflicting_track_indices)}")

    if reject_conflicted_tracks and len(conflicting_track_indices) > 0:
        keep_mask = np.ones(C.shape[1], dtype=bool)
        keep_mask[list(conflicting_track_indices)] = False

        C = C[:, keep_mask]
        C_v2 = C_v2[:, keep_mask]

        print(f"C.shape after removing conflicted tracks: {C.shape}")

    tracks_to_preserve = filter_C_using_pairs_to_triangulate_local(
        C=C,
        pairs_to_triangulate=pairs_to_triangulate,
    )

    C = C[:, tracks_to_preserve]
    C_v2 = C_v2[:, tracks_to_preserve]

    print(f"C.shape after selected-pair baseline check: {C.shape}")

    return C, C_v2


def filter_C_by_length_range(
    C: np.ndarray,
    C_v2: np.ndarray,
    min_total_length: int,
    max_total_length: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    track_lengths = np.sum(np.isfinite(C[::2, :]), axis=0)

    keep = track_lengths >= min_total_length

    if max_total_length is not None:
        keep &= track_lengths <= max_total_length

    print("Track length filter:")
    print(f"  min total track length: {min_total_length}")
    print(f"  max total track length: {max_total_length}")
    print(f"  keeping {int(np.sum(keep))} / {C.shape[1]} tracks")

    return C[:, keep], C_v2[:, keep]


def print_track_length_stats(C: np.ndarray) -> None:
    track_lengths = np.sum(np.isfinite(C[::2, :]), axis=0)

    print("\nTrack length statistics:")
    print(f"  tracks total: {track_lengths.shape[0]}")

    if track_lengths.shape[0] == 0:
        return

    print(f"  min:          {np.min(track_lengths)}")
    print(f"  mean:         {np.mean(track_lengths):.3f}")
    print(f"  median:       {np.median(track_lengths):.3f}")
    print(f"  p90:          {np.percentile(track_lengths, 90):.3f}")
    print(f"  p95:          {np.percentile(track_lengths, 95):.3f}")
    print(f"  max:          {np.max(track_lengths)}")

    for length in range(2, int(np.max(track_lengths)) + 1):
        n = int(np.sum(track_lengths == length))
        if n > 0:
            print(f"  tracks with length {length}: {n}")


# -----------------------------------------------------------------------------
# Geometry helpers
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


def ecef_xyz_to_lon_lat_alt(points_xyz: np.ndarray, geo_utils) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    out = np.full((points_xyz.shape[0], 3), np.nan, dtype=np.float64)

    valid = np.all(np.isfinite(points_xyz), axis=1)
    valid &= ~np.all(np.isclose(points_xyz, 0.0), axis=1)

    if not np.any(valid):
        return out

    x = points_xyz[valid, 0]
    y = points_xyz[valid, 1]
    z = points_xyz[valid, 2]

    if hasattr(geo_utils, "ecef_to_latlon_custom"):
        lat, lon, alt = geo_utils.ecef_to_latlon_custom(x, y, z)
    elif hasattr(geo_utils, "ecef_to_latlon"):
        lat, lon, alt = geo_utils.ecef_to_latlon(x, y, z)
    else:
        raise AttributeError(
            "Could not find geo_utils.ecef_to_latlon_custom or geo_utils.ecef_to_latlon"
        )

    out[valid, 0] = lon
    out[valid, 1] = lat
    out[valid, 2] = alt

    return out


def triangulate_C_to_lon_lat_alt(
    C: np.ndarray,
    rpcs: List[rpcm.RPCModel],
    pairs_to_triangulate: List[Tuple[int, int]],
    init_pts3d,
    geo_utils,
    verbose: bool,
) -> np.ndarray:
    pts_ecef = init_pts3d(
        C=C,
        cameras=rpcs,
        cam_model="rpc",
        pairs_to_triangulate=pairs_to_triangulate,
        verbose=verbose,
    )

    pts_ecef = np.asarray(pts_ecef, dtype=np.float64)

    return ecef_xyz_to_lon_lat_alt(pts_ecef, geo_utils)


def pairs_inside_observation_set(
    image_indices: List[int],
    allowed_pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    image_set = set(int(i) for i in image_indices)
    allowed_set = set((min(int(i), int(j)), max(int(i), int(j))) for i, j in allowed_pairs)

    pairs = []

    for i in image_indices:
        for j in image_indices:
            if i < j:
                pair = (int(i), int(j))
                if pair in allowed_set and int(i) in image_set and int(j) in image_set:
                    pairs.append(pair)

    return pairs


# -----------------------------------------------------------------------------
# Hold-out selection
# -----------------------------------------------------------------------------


def choose_holdout_indices_for_tracks(
    C: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_tracks = C.shape[1]

    holdout_local_indices = np.full(n_tracks, -1, dtype=np.int64)

    for track_idx in range(n_tracks):
        obs = observations_from_C_column(C, track_idx)

        if len(obs) == 0:
            continue

        holdout_local_indices[track_idx] = int(rng.integers(0, len(obs)))

    return holdout_local_indices


def write_holdout_selection_csv(
    path: Path,
    C: np.ndarray,
    holdout_local_indices: np.ndarray,
) -> None:
    fieldnames = [
        "track_idx",
        "track_length",
        "holdout_local_idx",
        "holdout_image_idx",
        "holdout_col",
        "holdout_row",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for track_idx in range(C.shape[1]):
            obs = observations_from_C_column(C, track_idx)
            h = int(holdout_local_indices[track_idx])

            if h < 0 or h >= len(obs):
                continue

            image_idx, col, row = obs[h]

            writer.writerow(
                {
                    "track_idx": track_idx,
                    "track_length": len(obs),
                    "holdout_local_idx": h,
                    "holdout_image_idx": image_idx,
                    "holdout_col": col,
                    "holdout_row": row,
                }
            )


# -----------------------------------------------------------------------------
# Held-out optimization
# -----------------------------------------------------------------------------


def residuals_for_point_scaled(
    z: np.ndarray,
    point0_lon_lat_alt: np.ndarray,
    scales_lon_lat_alt: np.ndarray,
    train_obs: List[Tuple[int, float, float]],
    rpcs: List[rpcm.RPCModel],
    projection_failure_penalty_px: float,
) -> np.ndarray:
    point = point0_lon_lat_alt + z * scales_lon_lat_alt
    lon, lat, alt = map(float, point)

    residuals = []

    for image_idx, observed_col, observed_row in train_obs:
        try:
            projected_col, projected_row = project_rpc_safe(
                rpcs[image_idx],
                lon,
                lat,
                alt,
            )
            residuals.append(projected_col - observed_col)
            residuals.append(projected_row - observed_row)
        except Exception:
            residuals.append(projection_failure_penalty_px)
            residuals.append(projection_failure_penalty_px)

    return np.asarray(residuals, dtype=np.float64)


def optimize_point_from_training_observations(
    point0_lon_lat_alt: np.ndarray,
    train_obs: List[Tuple[int, float, float]],
    rpcs: List[rpcm.RPCModel],
    lon_lat_scale_deg: float,
    alt_scale_m: float,
    max_lon_lat_step_deg: float,
    max_alt_step_m: float,
    loss: str,
    f_scale: float,
    max_nfev: int,
    projection_failure_penalty_px: float,
) -> Tuple[np.ndarray, dict]:
    point0 = np.asarray(point0_lon_lat_alt, dtype=np.float64)

    stats = {
        "success": False,
        "status": "not_run",
        "initial_train_rmse_px": np.nan,
        "final_train_rmse_px": np.nan,
        "initial_cost": np.nan,
        "final_cost": np.nan,
        "nfev": 0,
        "message": "",
    }

    if least_squares is None:
        raise ImportError(
            "scipy.optimize.least_squares could not be imported. "
            f"Original import error: {SCIPY_IMPORT_ERROR}"
        )

    if not np.all(np.isfinite(point0)):
        stats["status"] = "invalid_initial_point"
        return point0.copy(), stats

    scales = np.asarray(
        [lon_lat_scale_deg, lon_lat_scale_deg, alt_scale_m],
        dtype=np.float64,
    )

    z0 = np.zeros(3, dtype=np.float64)

    lower = np.asarray(
        [
            -max_lon_lat_step_deg / lon_lat_scale_deg,
            -max_lon_lat_step_deg / lon_lat_scale_deg,
            -max_alt_step_m / alt_scale_m,
        ],
        dtype=np.float64,
    )

    upper = np.asarray(
        [
            max_lon_lat_step_deg / lon_lat_scale_deg,
            max_lon_lat_step_deg / lon_lat_scale_deg,
            max_alt_step_m / alt_scale_m,
        ],
        dtype=np.float64,
    )

    r0 = residuals_for_point_scaled(
        z=z0,
        point0_lon_lat_alt=point0,
        scales_lon_lat_alt=scales,
        train_obs=train_obs,
        rpcs=rpcs,
        projection_failure_penalty_px=projection_failure_penalty_px,
    )

    stats["initial_cost"] = 0.5 * float(np.sum(r0 * r0))
    stats["initial_train_rmse_px"] = float(np.sqrt(np.mean(r0 * r0)))

    try:
        result = least_squares(
            residuals_for_point_scaled,
            z0,
            bounds=(lower, upper),
            args=(
                point0,
                scales,
                train_obs,
                rpcs,
                projection_failure_penalty_px,
            ),
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        refined = point0 + result.x * scales

        r1 = residuals_for_point_scaled(
            z=result.x,
            point0_lon_lat_alt=point0,
            scales_lon_lat_alt=scales,
            train_obs=train_obs,
            rpcs=rpcs,
            projection_failure_penalty_px=projection_failure_penalty_px,
        )

        stats["success"] = bool(result.success)
        stats["status"] = "optimized"
        stats["final_cost"] = 0.5 * float(np.sum(r1 * r1))
        stats["final_train_rmse_px"] = float(np.sqrt(np.mean(r1 * r1)))
        stats["nfev"] = int(result.nfev)
        stats["message"] = str(result.message)

        return refined, stats

    except Exception as exc:
        stats["status"] = "optimization_failed"
        stats["message"] = repr(exc)
        return point0.copy(), stats


def initial_point_for_holdout(
    C: np.ndarray,
    track_idx: int,
    train_obs: List[Tuple[int, float, float]],
    full_track_initial_points_lon_lat_alt: np.ndarray,
    rpcs: List[rpcm.RPCModel],
    selected_pairs: List[Tuple[int, int]],
    init_pts3d,
    geo_utils,
    strict_loo_initialization: bool,
    verbose_triangulation: bool,
) -> np.ndarray:
    if not strict_loo_initialization:
        return full_track_initial_points_lon_lat_alt[track_idx].copy()

    n_images = C.shape[0] // 2
    C_train = np.full((2 * n_images, 1), np.nan, dtype=np.float64)

    train_image_indices = []

    for image_idx, col, row in train_obs:
        C_train[2 * image_idx, 0] = col
        C_train[2 * image_idx + 1, 0] = row
        train_image_indices.append(int(image_idx))

    train_pairs = pairs_inside_observation_set(train_image_indices, selected_pairs)

    if len(train_pairs) == 0:
        return np.full(3, np.nan, dtype=np.float64)

    point_lon_lat_alt = triangulate_C_to_lon_lat_alt(
        C=C_train,
        rpcs=rpcs,
        pairs_to_triangulate=train_pairs,
        init_pts3d=init_pts3d,
        geo_utils=geo_utils,
        verbose=verbose_triangulation,
    )

    return point_lon_lat_alt[0]


def build_heldout_row(
    track_idx: int,
    track_length: int,
    holdout_obs: Tuple[int, float, float],
    train_obs: List[Tuple[int, float, float]],
    point0: np.ndarray,
    refined_point: np.ndarray,
    opt_stats: dict,
    rpcs: List[rpcm.RPCModel],
) -> dict:
    holdout_image_idx, holdout_col, holdout_row = holdout_obs

    row = {
        "track_idx": int(track_idx),
        "track_length": int(track_length),
        "holdout_image_idx": int(holdout_image_idx),
        "n_train_observations": int(len(train_obs)),
        "holdout_col": float(holdout_col),
        "holdout_row": float(holdout_row),
        "projected_col": np.nan,
        "projected_row": np.nan,
        "dx_px": np.nan,
        "dy_px": np.nan,
        "heldout_error_px": np.nan,
        "status": opt_stats["status"],
        "success": bool(opt_stats["success"]),
        "initial_train_rmse_px": opt_stats["initial_train_rmse_px"],
        "final_train_rmse_px": opt_stats["final_train_rmse_px"],
        "initial_cost": opt_stats["initial_cost"],
        "final_cost": opt_stats["final_cost"],
        "nfev": opt_stats["nfev"],
        "lon0": float(point0[0]) if np.all(np.isfinite(point0)) else np.nan,
        "lat0": float(point0[1]) if np.all(np.isfinite(point0)) else np.nan,
        "alt0": float(point0[2]) if np.all(np.isfinite(point0)) else np.nan,
        "lon_refined": float(refined_point[0]) if np.all(np.isfinite(refined_point)) else np.nan,
        "lat_refined": float(refined_point[1]) if np.all(np.isfinite(refined_point)) else np.nan,
        "alt_refined": float(refined_point[2]) if np.all(np.isfinite(refined_point)) else np.nan,
        "message": opt_stats["message"],
    }

    if np.all(np.isfinite(refined_point)):
        try:
            projected_col, projected_row = project_rpc_safe(
                rpcs[holdout_image_idx],
                float(refined_point[0]),
                float(refined_point[1]),
                float(refined_point[2]),
            )

            dx = projected_col - holdout_col
            dy = projected_row - holdout_row
            err = float(np.sqrt(dx * dx + dy * dy))

            row["projected_col"] = float(projected_col)
            row["projected_row"] = float(projected_row)
            row["dx_px"] = float(dx)
            row["dy_px"] = float(dy)
            row["heldout_error_px"] = err

        except Exception as exc:
            row["status"] = "holdout_projection_failed"
            row["message"] = repr(exc)

    return row


def run_heldout_evaluation_one_per_track(
    C: np.ndarray,
    rpcs: List[rpcm.RPCModel],
    selected_pairs: List[Tuple[int, int]],
    init_pts3d,
    geo_utils,
    min_train_observations: int,
    full_track_initial_points_lon_lat_alt: np.ndarray,
    strict_loo_initialization: bool,
    verbose_triangulation: bool,
    lon_lat_scale_deg: float,
    alt_scale_m: float,
    max_lon_lat_step_deg: float,
    max_alt_step_m: float,
    loss: str,
    f_scale: float,
    max_nfev: int,
    projection_failure_penalty_px: float,
    holdout_local_indices: np.ndarray,
) -> List[dict]:
    rows: List[dict] = []

    n_tracks = C.shape[1]

    print("\nRunning one-heldout-per-track evaluation...")
    print(f"  tracks: {n_tracks}")
    print(f"  min_train_observations: {min_train_observations}")
    print(f"  strict_loo_initialization: {strict_loo_initialization}")
    print(f"  loss: {loss}")
    print(f"  f_scale: {f_scale}")
    print(f"  max_nfev: {max_nfev}")

    t0 = timeit.default_timer()

    for track_idx in range(n_tracks):
        obs = observations_from_C_column(C, track_idx)
        track_length = len(obs)

        if track_length < min_train_observations + 1:
            continue

        holdout_local_idx = int(holdout_local_indices[track_idx])

        if holdout_local_idx < 0 or holdout_local_idx >= track_length:
            continue

        holdout_obs = obs[holdout_local_idx]

        train_obs = [
            current_obs
            for k, current_obs in enumerate(obs)
            if k != holdout_local_idx
        ]

        if len(train_obs) < min_train_observations:
            continue

        point0 = initial_point_for_holdout(
            C=C,
            track_idx=track_idx,
            train_obs=train_obs,
            full_track_initial_points_lon_lat_alt=full_track_initial_points_lon_lat_alt,
            rpcs=rpcs,
            selected_pairs=selected_pairs,
            init_pts3d=init_pts3d,
            geo_utils=geo_utils,
            strict_loo_initialization=strict_loo_initialization,
            verbose_triangulation=verbose_triangulation,
        )

        refined_point, opt_stats = optimize_point_from_training_observations(
            point0_lon_lat_alt=point0,
            train_obs=train_obs,
            rpcs=rpcs,
            lon_lat_scale_deg=lon_lat_scale_deg,
            alt_scale_m=alt_scale_m,
            max_lon_lat_step_deg=max_lon_lat_step_deg,
            max_alt_step_m=max_alt_step_m,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            projection_failure_penalty_px=projection_failure_penalty_px,
        )

        row = build_heldout_row(
            track_idx=track_idx,
            track_length=track_length,
            holdout_obs=holdout_obs,
            train_obs=train_obs,
            point0=point0,
            refined_point=refined_point,
            opt_stats=opt_stats,
            rpcs=rpcs,
        )

        rows.append(row)

        if track_idx % 100 == 0:
            elapsed = timeit.default_timer() - t0
            print(
                f"[heldout-one-per-track] track {track_idx}/{n_tracks}, "
                f"rows={len(rows)}, elapsed={elapsed:.1f}s"
            )

    print(f"One-heldout-per-track evaluation complete. Rows: {len(rows)}")

    return rows


def run_heldout_evaluation_all_holdouts(
    C: np.ndarray,
    rpcs: List[rpcm.RPCModel],
    selected_pairs: List[Tuple[int, int]],
    init_pts3d,
    geo_utils,
    min_train_observations: int,
    full_track_initial_points_lon_lat_alt: np.ndarray,
    strict_loo_initialization: bool,
    verbose_triangulation: bool,
    lon_lat_scale_deg: float,
    alt_scale_m: float,
    max_lon_lat_step_deg: float,
    max_alt_step_m: float,
    loss: str,
    f_scale: float,
    max_nfev: int,
    projection_failure_penalty_px: float,
) -> List[dict]:
    rows: List[dict] = []

    n_tracks = C.shape[1]

    print("\nRunning full leave-one-out held-out evaluation...")
    print(f"  tracks: {n_tracks}")
    print(f"  min_train_observations: {min_train_observations}")
    print(f"  strict_loo_initialization: {strict_loo_initialization}")
    print(f"  loss: {loss}")
    print(f"  f_scale: {f_scale}")
    print(f"  max_nfev: {max_nfev}")

    t0 = timeit.default_timer()

    for track_idx in range(n_tracks):
        obs = observations_from_C_column(C, track_idx)
        track_length = len(obs)

        if track_length < min_train_observations + 1:
            continue

        for holdout_local_idx, holdout_obs in enumerate(obs):
            train_obs = [
                current_obs
                for k, current_obs in enumerate(obs)
                if k != holdout_local_idx
            ]

            if len(train_obs) < min_train_observations:
                continue

            point0 = initial_point_for_holdout(
                C=C,
                track_idx=track_idx,
                train_obs=train_obs,
                full_track_initial_points_lon_lat_alt=full_track_initial_points_lon_lat_alt,
                rpcs=rpcs,
                selected_pairs=selected_pairs,
                init_pts3d=init_pts3d,
                geo_utils=geo_utils,
                strict_loo_initialization=strict_loo_initialization,
                verbose_triangulation=verbose_triangulation,
            )

            refined_point, opt_stats = optimize_point_from_training_observations(
                point0_lon_lat_alt=point0,
                train_obs=train_obs,
                rpcs=rpcs,
                lon_lat_scale_deg=lon_lat_scale_deg,
                alt_scale_m=alt_scale_m,
                max_lon_lat_step_deg=max_lon_lat_step_deg,
                max_alt_step_m=max_alt_step_m,
                loss=loss,
                f_scale=f_scale,
                max_nfev=max_nfev,
                projection_failure_penalty_px=projection_failure_penalty_px,
            )

            row = build_heldout_row(
                track_idx=track_idx,
                track_length=track_length,
                holdout_obs=holdout_obs,
                train_obs=train_obs,
                point0=point0,
                refined_point=refined_point,
                opt_stats=opt_stats,
                rpcs=rpcs,
            )

            rows.append(row)

        if track_idx % 100 == 0:
            elapsed = timeit.default_timer() - t0
            print(
                f"[heldout-all] track {track_idx}/{n_tracks}, "
                f"rows={len(rows)}, elapsed={elapsed:.1f}s"
            )

    print(f"Full leave-one-out evaluation complete. Rows: {len(rows)}")

    return rows


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def write_pairs_txt(path: Path, pairs: List[Tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, j in pairs:
            f.write(f"{int(i)} {int(j)}\n")


def save_pairs_and_rpcs(
    output_dir: Path,
    image_paths: List[Path],
    rpcs: List[rpcm.RPCModel],
    available_pairs: List[Tuple[int, int]],
    candidate_pairs: List[Tuple[int, int]],
    selected_pairs: List[Tuple[int, int]],
) -> None:
    rpc_out_dir = output_dir / "rpcs"
    rpc_out_dir.mkdir(parents=True, exist_ok=True)

    for image_path, rpc in zip(image_paths, rpcs):
        out_path = rpc_out_dir / f"{image_path.stem}.rpc"
        rpc.write_to_file(str(out_path))

    np.save(output_dir / "available_pairs.npy", np.asarray(available_pairs, dtype=np.int64))
    np.save(output_dir / "candidate_pairs.npy", np.asarray(candidate_pairs, dtype=np.int64))
    np.save(output_dir / "selected_least_similar_pairs.npy", np.asarray(selected_pairs, dtype=np.int64))
    np.save(output_dir / "selected_pairs.npy", np.asarray(selected_pairs, dtype=np.int64))

    write_pairs_txt(output_dir / "available_pairs.txt", available_pairs)
    write_pairs_txt(output_dir / "candidate_pairs.txt", candidate_pairs)
    write_pairs_txt(output_dir / "selected_least_similar_pairs.txt", selected_pairs)
    write_pairs_txt(output_dir / "selected_pairs.txt", selected_pairs)


def write_heldout_errors_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "track_idx",
        "track_length",
        "holdout_image_idx",
        "n_train_observations",
        "holdout_col",
        "holdout_row",
        "projected_col",
        "projected_row",
        "dx_px",
        "dy_px",
        "heldout_error_px",
        "status",
        "success",
        "initial_train_rmse_px",
        "final_train_rmse_px",
        "initial_cost",
        "final_cost",
        "nfev",
        "lon0",
        "lat0",
        "alt0",
        "lon_refined",
        "lat_refined",
        "alt_refined",
        "message",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def rows_to_numpy(rows: List[dict]) -> np.ndarray:
    out = np.full((len(rows), 18), np.nan, dtype=np.float64)

    for i, row in enumerate(rows):
        out[i, 0] = row["track_idx"]
        out[i, 1] = row["track_length"]
        out[i, 2] = row["holdout_image_idx"]
        out[i, 3] = row["n_train_observations"]
        out[i, 4] = row["holdout_col"]
        out[i, 5] = row["holdout_row"]
        out[i, 6] = row["projected_col"]
        out[i, 7] = row["projected_row"]
        out[i, 8] = row["dx_px"]
        out[i, 9] = row["dy_px"]
        out[i, 10] = row["heldout_error_px"]
        out[i, 11] = 1.0 if row["success"] else 0.0
        out[i, 12] = row["initial_train_rmse_px"]
        out[i, 13] = row["final_train_rmse_px"]
        out[i, 14] = row["nfev"]
        out[i, 15] = row["lon_refined"]
        out[i, 16] = row["lat_refined"]
        out[i, 17] = row["alt_refined"]

    return out


def valid_error_array(rows: List[dict]) -> np.ndarray:
    vals = []

    for row in rows:
        err = row.get("heldout_error_px", np.nan)
        if np.isfinite(err):
            vals.append(float(err))

    return np.asarray(vals, dtype=np.float64)


def write_per_camera_heldout_stats_csv(
    path: Path,
    rows: List[dict],
    image_paths: List[Path],
) -> None:
    fieldnames = [
        "image_idx",
        "image_name",
        "n_heldout_observations",
        "mean_dx_px",
        "mean_dy_px",
        "median_dx_px",
        "median_dy_px",
        "mean_error_px",
        "median_error_px",
        "rmse_error_px",
        "p90_error_px",
        "p95_error_px",
        "max_error_px",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_idx, image_path in enumerate(image_paths):
            sub = [
                row
                for row in rows
                if int(row["holdout_image_idx"]) == image_idx
                and np.isfinite(row["heldout_error_px"])
            ]

            if len(sub) == 0:
                writer.writerow(
                    {
                        "image_idx": image_idx,
                        "image_name": image_path.name,
                        "n_heldout_observations": 0,
                        "mean_dx_px": np.nan,
                        "mean_dy_px": np.nan,
                        "median_dx_px": np.nan,
                        "median_dy_px": np.nan,
                        "mean_error_px": np.nan,
                        "median_error_px": np.nan,
                        "rmse_error_px": np.nan,
                        "p90_error_px": np.nan,
                        "p95_error_px": np.nan,
                        "max_error_px": np.nan,
                    }
                )
                continue

            dx = np.asarray([row["dx_px"] for row in sub], dtype=np.float64)
            dy = np.asarray([row["dy_px"] for row in sub], dtype=np.float64)
            err = np.asarray([row["heldout_error_px"] for row in sub], dtype=np.float64)

            writer.writerow(
                {
                    "image_idx": image_idx,
                    "image_name": image_path.name,
                    "n_heldout_observations": int(err.size),
                    "mean_dx_px": float(np.mean(dx)),
                    "mean_dy_px": float(np.mean(dy)),
                    "median_dx_px": float(np.median(dx)),
                    "median_dy_px": float(np.median(dy)),
                    "mean_error_px": float(np.mean(err)),
                    "median_error_px": float(np.median(err)),
                    "rmse_error_px": float(np.sqrt(np.mean(err * err))),
                    "p90_error_px": float(np.percentile(err, 90)),
                    "p95_error_px": float(np.percentile(err, 95)),
                    "max_error_px": float(np.max(err)),
                }
            )


def write_heldout_error_by_track_length_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "track_length",
        "n_tracks",
        "n_heldout_observations",
        "mean_error_px",
        "median_error_px",
        "rmse_error_px",
        "p90_error_px",
        "p95_error_px",
        "max_error_px",
        "mean_final_train_rmse_px",
        "median_final_train_rmse_px",
    ]

    valid_rows = [
        row for row in rows
        if np.isfinite(row["heldout_error_px"])
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if len(valid_rows) == 0:
            return

        lengths = sorted(set(int(row["track_length"]) for row in valid_rows))

        for length in lengths:
            sub = [row for row in valid_rows if int(row["track_length"]) == length]
            track_ids = sorted(set(int(row["track_idx"]) for row in sub))

            err = np.asarray([row["heldout_error_px"] for row in sub], dtype=np.float64)
            train_rmse = np.asarray(
                [row["final_train_rmse_px"] for row in sub],
                dtype=np.float64,
            )
            train_rmse = train_rmse[np.isfinite(train_rmse)]

            writer.writerow(
                {
                    "track_length": int(length),
                    "n_tracks": int(len(track_ids)),
                    "n_heldout_observations": int(err.size),
                    "mean_error_px": float(np.mean(err)),
                    "median_error_px": float(np.median(err)),
                    "rmse_error_px": float(np.sqrt(np.mean(err * err))),
                    "p90_error_px": float(np.percentile(err, 90)),
                    "p95_error_px": float(np.percentile(err, 95)),
                    "max_error_px": float(np.max(err)),
                    "mean_final_train_rmse_px": (
                        float(np.mean(train_rmse)) if train_rmse.size else np.nan
                    ),
                    "median_final_train_rmse_px": (
                        float(np.median(train_rmse)) if train_rmse.size else np.nan
                    ),
                }
            )


def write_summary_txt(
    path: Path,
    rows: List[dict],
    n_images: int,
    n_tracks: int,
    available_pairs_count: int,
    candidate_pairs_count: int,
    selected_pairs_count: int,
    least_pair_ranking_K: int,
    FT_max_length: Optional[int],
    min_train_observations: int,
    strict_loo_initialization: bool,
    all_holdouts: bool,
    holdout_seed: int,
) -> None:
    valid_err = valid_error_array(rows)

    n_success = sum(1 for row in rows if row["success"])
    n_valid = int(valid_err.size)

    lines = []

    lines.append("Held-out fixed-RPC evaluation summary using least-similar DINO pairs")
    lines.append("====================================================================")
    lines.append("")
    lines.append("Metric interpretation:")
    lines.append("  For each held-out image observation, one 3D point is optimized using")
    lines.append("  the other observations in that track. RPCs are fixed.")
    lines.append("  The held-out observation is not used in the point refinement objective.")
    lines.append("  Pairs are selected from the least-similar DINO-ranked candidate pairs.")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  images: {n_images}")
    lines.append(f"  tracks after filtering: {n_tracks}")
    lines.append("")
    lines.append("Pair selection:")
    lines.append(f"  available pairs from pairwise_matches.npy: {available_pairs_count}")
    lines.append(f"  candidate pairs after geometry filtering: {candidate_pairs_count}")
    lines.append(f"  selected least-similar pairs used: {selected_pairs_count}")
    lines.append(f"  least_pair_ranking_K: {least_pair_ranking_K}")
    lines.append("")
    lines.append("Track / held-out settings:")
    lines.append(f"  min_train_observations: {min_train_observations}")
    lines.append(f"  required total track length: >= {min_train_observations + 1}")
    lines.append(f"  FT_max_length: {FT_max_length}")
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
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("Global held-out reprojection error, pixels:")
    lines.append(f"  mean:   {np.mean(valid_err):.6f}")
    lines.append(f"  median: {np.median(valid_err):.6f}")
    lines.append(f"  rmse:   {np.sqrt(np.mean(valid_err * valid_err)):.6f}")
    lines.append(f"  p90:    {np.percentile(valid_err, 90):.6f}")
    lines.append(f"  p95:    {np.percentile(valid_err, 95):.6f}")
    lines.append(f"  max:    {np.max(valid_err):.6f}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--matches_dir",
        type=Path,
        required=True,
        help="Directory containing image_paths.txt, pairwise_matches.npy, and features/*.npy.",
    )

    parser.add_argument(
        "--rpc_dir",
        type=Path,
        required=True,
        help="Directory containing RPC files.",
    )

    parser.add_argument(
        "--sat_bundleadjust_repo",
        type=Path,
        required=True,
        help="Path to the root of the sat-bundleadjust GitHub repository.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where outputs will be written.",
    )

    parser.add_argument(
        "--min_train_observations",
        type=int,
        default=3,
        help=(
            "Minimum number of observations used to optimize the 3D point after "
            "one observation is held out. Total track length must be at least "
            "min_train_observations + 1."
        ),
    )

    parser.add_argument(
        "--min_track_length",
        type=int,
        default=None,
        help="Deprecated alias. If provided, overrides --min_train_observations.",
    )

    parser.add_argument(
        "--FT_max_length",
        type=int,
        default=None,
        help=(
            "Optional maximum total feature-track length. "
            "If omitted, no maximum track-length filtering is applied."
        ),
    )

    parser.add_argument(
        "--least_pair_ranking_K",
        type=int,
        default=5,
        help="K target degree for least-similar DINO pair selection.",
    )

    parser.add_argument(
        "--disable_pair_geometry_filter",
        action="store_true",
        help=(
            "Disable overlap/baseline filtering before DINO ranking. "
            "Equivalent to compute_pairs_to_match(..., min_overlap=0, min_baseline=0)."
        ),
    )

    parser.add_argument(
        "--keep_conflicted_tracks",
        action="store_true",
        help="Do not reject tracks with duplicate keypoints in the same image.",
    )

    parser.add_argument(
        "--strict_loo_initialization",
        action="store_true",
        help=(
            "Initialize each held-out test using only training observations. "
            "This avoids initialization leakage but is much slower."
        ),
    )

    parser.add_argument(
        "--all_holdouts",
        action="store_true",
        help=(
            "Run full leave-one-out: one optimization per observation. "
            "Default is one seeded held-out observation per track."
        ),
    )

    parser.add_argument(
        "--holdout_seed",
        type=int,
        default=0,
        help="Random seed used to choose one held-out observation per track.",
    )

    parser.add_argument(
        "--verbose_triangulation",
        action="store_true",
        help="Print progress from sat-bundleadjust init_pts3d.",
    )

    parser.add_argument(
        "--lon_lat_scale_deg",
        type=float,
        default=1e-5,
        help="Optimizer scale for lon/lat degrees.",
    )

    parser.add_argument(
        "--alt_scale_m",
        type=float,
        default=10.0,
        help="Optimizer scale for altitude in meters.",
    )

    parser.add_argument(
        "--max_lon_lat_step_deg",
        type=float,
        default=0.01,
        help="Maximum lon/lat step from initialized point during refinement.",
    )

    parser.add_argument(
        "--max_alt_step_m",
        type=float,
        default=1000.0,
        help="Maximum altitude step from initialized point during refinement.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="linear",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="Loss for per-heldout point optimization.",
    )

    parser.add_argument(
        "--f_scale",
        type=float,
        default=1.0,
        help="Robust loss scale in pixels.",
    )

    parser.add_argument(
        "--max_nfev",
        type=int,
        default=30,
        help="Maximum function evaluations per held-out optimization.",
    )

    parser.add_argument(
        "--projection_failure_penalty_px",
        type=float,
        default=1e6,
        help="Penalty residual if an RPC projection fails during optimization.",
    )

    parser.add_argument(
        "--no_log_redirect",
        action="store_true",
        help="Print to console instead of redirecting stdout/stderr to bundle_adjust.log.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


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

    (
        cam_utils,
        geo_utils,
        ft_match,
        ft_pair_ranking,
        select_optimal_pairs_to_match,
        init_pts3d,
    ) = import_sat_bundleadjust(args.sat_bundleadjust_repo)

    print("Running held-out fixed-RPC evaluation with least-similar DINO pairs")
    print("------------------------------------------------------------------")
    print(f"matches_dir: {args.matches_dir}")
    print(f"rpc_dir: {args.rpc_dir}")
    print(f"output_dir: {output_dir}")
    print(f"sat_bundleadjust_repo: {args.sat_bundleadjust_repo}")
    print("")
    print("Configuration:")
    print("  RPCs fixed: True")
    print("  optimized variable per test: one lon/lat/alt point")
    print("  held-out observations excluded from objective: True")
    print("  pair selection: least-similar DINO candidate pairs")
    print(f"  min_train_observations: {min_train_observations}")
    print(f"  FT_max_length: {args.FT_max_length}")
    print(f"  least_pair_ranking_K: {args.least_pair_ranking_K}")
    print(f"  all_holdouts: {args.all_holdouts}")
    print(f"  holdout_seed: {args.holdout_seed}")
    print(f"  strict_loo_initialization: {args.strict_loo_initialization}")
    print("------------------------------------------------------------------")
    print("")

    image_paths = load_image_paths(args.matches_dir)
    feature_paths = get_feature_paths(args.matches_dir, image_paths)
    pairwise_matches_all = load_pairwise_matches(args.matches_dir)

    print(f"Images: {len(image_paths)}")
    print(f"Pairwise matches: {pairwise_matches_all.shape}")

    available_pairs = all_pairs_from_pairwise_matches(pairwise_matches_all)

    if len(available_pairs) == 0:
        raise RuntimeError("No available pairs in pairwise_matches.npy")

    rpcs = load_rpcs_for_images(args.rpc_dir, image_paths)

    images = make_satellite_images(image_paths, rpcs, cam_utils)
    initialize_image_geometry(images)

    selected_pairs, candidate_pairs = select_least_similar_pairs_with_satba_dino(
        images=images,
        available_pairs=available_pairs,
        ft_match=ft_match,
        geo_utils=geo_utils,
        ft_pair_ranking=ft_pair_ranking,
        K=args.least_pair_ranking_K,
        filter_pairs=(not args.disable_pair_geometry_filter),
        scores_csv_path=output_dir / "dino_pair_scores.csv",
    )

    pairwise_matches = filter_pairwise_matches_to_pairs(
        pairwise_matches_all,
        selected_pairs,
    )

    if pairwise_matches.shape[0] == 0:
        raise RuntimeError("No pairwise matches remain after selected-pair filtering")

    C, C_v2 = feature_tracks_from_pairwise_matches_variable_length(
        feature_paths=feature_paths,
        pairwise_matches=pairwise_matches,
        pairs_to_triangulate=selected_pairs,
        reject_conflicted_tracks=(not args.keep_conflicted_tracks),
    )

    min_total_length = min_train_observations + 1

    C, C_v2 = filter_C_by_length_range(
        C=C,
        C_v2=C_v2,
        min_total_length=min_total_length,
        max_total_length=args.FT_max_length,
    )

    print(f"C shape: {C.shape}")
    print(f"C_v2 shape: {C_v2.shape}")
    print_track_length_stats(C)

    if C.shape[1] == 0:
        raise RuntimeError("No tracks left after length filtering")

    np.save(output_dir / "C.npy", C)
    np.save(output_dir / "C_v2.npy", C_v2)
    np.save(output_dir / "pairwise_matches_used.npy", pairwise_matches)

    save_pairs_and_rpcs(
        output_dir=output_dir,
        image_paths=image_paths,
        rpcs=rpcs,
        available_pairs=available_pairs,
        candidate_pairs=candidate_pairs,
        selected_pairs=selected_pairs,
    )

    holdout_local_indices = choose_holdout_indices_for_tracks(
        C=C,
        seed=args.holdout_seed,
    )

    np.save(
        output_dir / "holdout_local_indices.npy",
        holdout_local_indices,
    )

    write_holdout_selection_csv(
        output_dir / "holdout_selection.csv",
        C,
        holdout_local_indices,
    )

    print("\nComputing full-track initial points for optimizer initialization...")
    full_track_initial_points_lon_lat_alt = triangulate_C_to_lon_lat_alt(
        C=C,
        rpcs=rpcs,
        pairs_to_triangulate=selected_pairs,
        init_pts3d=init_pts3d,
        geo_utils=geo_utils,
        verbose=args.verbose_triangulation,
    )

    np.save(
        output_dir / "full_track_initial_points_lon_lat_alt.npy",
        full_track_initial_points_lon_lat_alt,
    )

    if args.all_holdouts:
        heldout_rows = run_heldout_evaluation_all_holdouts(
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
        heldout_rows = run_heldout_evaluation_one_per_track(
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

    write_heldout_errors_csv(
        output_dir / "heldout_errors.csv",
        heldout_rows,
    )

    np.save(
        output_dir / "heldout_errors.npy",
        rows_to_numpy(heldout_rows),
    )

    write_per_camera_heldout_stats_csv(
        output_dir / "per_camera_heldout_error_stats.csv",
        heldout_rows,
        image_paths,
    )

    write_heldout_error_by_track_length_csv(
        output_dir / "heldout_error_by_track_length.csv",
        heldout_rows,
    )

    write_summary_txt(
        output_dir / "summary.txt",
        heldout_rows,
        n_images=len(image_paths),
        n_tracks=C.shape[1],
        available_pairs_count=len(available_pairs),
        candidate_pairs_count=len(candidate_pairs),
        selected_pairs_count=len(selected_pairs),
        least_pair_ranking_K=args.least_pair_ranking_K,
        FT_max_length=args.FT_max_length,
        min_train_observations=min_train_observations,
        strict_loo_initialization=args.strict_loo_initialization,
        all_holdouts=args.all_holdouts,
        holdout_seed=args.holdout_seed,
    )

    print("\nSaved outputs:")
    print(f"  {output_dir / 'bundle_adjust.log'}")
    print(f"  {output_dir / 'summary.txt'}")
    print(f"  {output_dir / 'heldout_errors.csv'}")
    print(f"  {output_dir / 'heldout_errors.npy'}")
    print(f"  {output_dir / 'per_camera_heldout_error_stats.csv'}")
    print(f"  {output_dir / 'heldout_error_by_track_length.csv'}")
    print(f"  {output_dir / 'dino_pair_scores.csv'}")
    print(f"  {output_dir / 'selected_least_similar_pairs.txt'}")
    print(f"  {output_dir / 'holdout_local_indices.npy'}")
    print(f"  {output_dir / 'holdout_selection.csv'}")
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

    print("Running held-out fixed-RPC evaluation with least-similar DINO pairs ...")
    print(f"Path to log file: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            run_main(args)

    print("... done !")
    print(f"Path to output files: {output_dir}")
    print(f"Path to log file: {log_path}")


if __name__ == "__main__":
    main()
