#!/usr/bin/env python3
"""
Compute all-pairs image similarity scores for satellite images.

The intended image order is the same order used by compute_lightglue_matches.py:
    output_dir/image_paths.txt

Each CSV row corresponds to one unordered image pair:
    image_1_index,
    image_2_index,
    image_1_basename,
    image_2_basename,
    superpoint_lightglue_num_matches,
    ssim,
    dinov2,
    dinov3_sat,
    clip_rn50,
    clip_vit_b32,
    clip_vit_l14

All similarity scores are normalized to [0, 1].

Notes
-----
- SSIM is clipped to [0, 1], following ft_pair_classifier.py.
- DINO and CLIP similarities are cosine similarities mapped from [-1, 1] to [0, 1].
- DINOv3 satellite model is gated on Hugging Face. Set HF_TOKEN if required:
      export HF_TOKEN=hf_...
- LightGlue match counts are read from pairwise_matches.npy produced by
  compute_lightglue_matches.py. Its rows are expected to be:
      [kp_index_image1, kp_index_image2, im1_index, im2_index]
- This script assumes the bundle_adjust package is available.
"""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from bundle_adjust import loader


TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}

os.environ["HF_TOKEN"] = "hf_COMPLETE_WITH_YOURS"

# ----------------------------
# Image ordering / pair helpers
# ----------------------------

def list_tif_images(image_dir: Path) -> List[Path]:
    """
    Same ordering policy as compute_lightglue_matches.py:
    sorted TIFF paths from the input directory.
    """
    image_paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix in TIFF_EXTENSIONS
    ]
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"No TIFF images found in: {image_dir}")

    return image_paths


def read_image_paths_txt(image_paths_txt: Path) -> List[Path]:
    """
    Read image paths written by compute_lightglue_matches.py.

    This is the safest way to guarantee identical indexing to the
    pairwise_matches.npy output.
    """
    with image_paths_txt.open("r", encoding="utf-8") as f:
        paths = [Path(line.strip()) for line in f if line.strip()]

    if len(paths) == 0:
        raise RuntimeError(f"No image paths found in: {image_paths_txt}")

    for p in paths:
        if not p.exists():
            raise RuntimeError(f"Image path from image_paths.txt does not exist: {p}")

    return paths


def make_all_pairs(num_images: int) -> List[Tuple[int, int]]:
    """
    Same nested all-pairs order as compute_lightglue_matches.py:
        for i in range(N):
            for j in range(i + 1, N):
    """
    return [
        (i, j)
        for i in range(num_images)
        for j in range(i + 1, num_images)
    ]


# ----------------------------
# SuperPoint + LightGlue counts
# ----------------------------

def load_pairwise_match_counts(
    pairwise_matches_npy: Path | None,
    num_images: int,
) -> Dict[Tuple[int, int], int]:
    """
    Load per-pair SuperPoint+LightGlue match counts from pairwise_matches.npy.

    Expected array layout from compute_lightglue_matches.py:
        shape M x 4
        col 0: keypoint index in image 1
        col 1: keypoint index in image 2
        col 2: im1_index
        col 3: im2_index

    Returns
    -------
    dict[(int, int), int]
        Pair key -> number of saved filtered matches.
        Missing pairs should be interpreted as zero matches.
    """
    counts: Dict[Tuple[int, int], int] = defaultdict(int)

    if pairwise_matches_npy is None:
        return counts

    pairwise_matches_npy = pairwise_matches_npy.resolve()

    if not pairwise_matches_npy.exists():
        raise RuntimeError(f"pairwise_matches.npy does not exist: {pairwise_matches_npy}")

    matches = np.load(pairwise_matches_npy)

    if matches.size == 0:
        return counts

    matches = np.asarray(matches)

    if matches.ndim != 2 or matches.shape[1] != 4:
        raise RuntimeError(
            "Expected pairwise_matches.npy to have shape M x 4, "
            f"got {matches.shape}"
        )

    matches = matches.astype(np.int64, copy=False)

    for row in matches:
        i = int(row[2])
        j = int(row[3])

        if i == j:
            print(f"[warning] Ignoring self-pair in pairwise_matches.npy: ({i}, {j})")
            continue

        if i < 0 or j < 0 or i >= num_images or j >= num_images:
            print(
                "[warning] Ignoring out-of-range pair in pairwise_matches.npy: "
                f"({i}, {j}), num_images={num_images}"
            )
            continue

        # compute_lightglue_matches.py stores i < j, but normalize defensively.
        if i > j:
            i, j = j, i

        counts[(i, j)] += 1

    return dict(counts)


def pairwise_match_count_list(
    pairs: Sequence[Tuple[int, int]],
    match_counts: Dict[Tuple[int, int], int],
) -> List[int]:
    """
    Convert a match-count dictionary into a list aligned with `pairs`.
    Missing pairs get count 0.
    """
    return [int(match_counts.get((i, j), 0)) for i, j in pairs]


# ----------------------------
# Shared image utilities
# ----------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Adapted from ft_pair_classifier.py.

    Converts an array to uint8 for SSIM / PIL compatibility.
    """
    arr = np.asarray(arr)

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr_min = float(np.nanmin(arr))
        arr_max = float(np.nanmax(arr))

        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)

        return (255 * arr).clip(0, 255).astype(np.uint8)

    return np.clip(arr, 0, 255).astype(np.uint8)


def load_satellite_array(image_path: Path) -> np.ndarray:
    """
    Load an image using bundle_adjust.loader when available.

    The user's project commonly uses loader.load_image(image.geotiff_path, offset=image.offset).
    Here we only have paths, so we use the path form.
    """
    try:
        arr = loader.load_image(str(image_path))
    except TypeError:
        arr = loader.load_image(image_path)

    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Keep RGB/RGBA-like arrays. If more than 3 channels exist, keep the first 3.
        if arr.shape[2] >= 3:
            return arr[:, :, :3]
        if arr.shape[2] == 1:
            return arr[:, :, 0]

    raise ValueError(f"Unsupported image shape for {image_path}: {arr.shape}")


def open_rgb_image(image_or_array: Path | str | np.ndarray) -> Image.Image:
    """
    Adapted from ft_pair_classifier.py.

    Accepts:
      - path
      - numpy array

    Returns:
      RGB PIL image.
    """
    if isinstance(image_or_array, np.ndarray):
        arr = _to_uint8(image_or_array)

        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return Image.fromarray(arr[:, :, 0]).convert("RGB")
            if arr.shape[2] in (3, 4):
                return Image.fromarray(arr).convert("RGB")

        raise ValueError(f"Unsupported numpy image shape: {arr.shape}")

    return Image.open(image_or_array).convert("RGB")


def downsample_arrays(
    image_paths: Sequence[Path],
    output_size: Tuple[int, int] = (256, 256),
) -> List[np.ndarray]:
    """
    Adapted from downsample_image_collection in ft_pair_classifier.py.

    Returns same-size uint8 arrays suitable for SSIM and optionally for embeddings.
    """
    h, w = output_size
    downsampled = []

    for image_path in tqdm(image_paths, desc="Loading/downsampling images"):
        arr = load_satellite_array(image_path)
        arr = _to_uint8(arr)
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        downsampled.append(arr)

    return downsampled


# ----------------------------
# SSIM
# ----------------------------

def compute_ssim(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Adapted from compute_SSIM in ft_pair_classifier.py.

    OpenCV QualitySSIM returns values that can be outside [0, 1].
    We clip to [0, 1], as in the original helper.
    """
    if array1.shape[:2] != array2.shape[:2]:
        raise ValueError(
            f"SSIM arrays must have same H/W, got {array1.shape} vs {array2.shape}"
        )

    array1 = _to_uint8(array1)
    array2 = _to_uint8(array2)

    # If channel counts differ, convert both to grayscale.
    if array1.ndim != array2.ndim:
        if array1.ndim == 3:
            array1 = cv2.cvtColor(array1, cv2.COLOR_RGB2GRAY)
        if array2.ndim == 3:
            array2 = cv2.cvtColor(array2, cv2.COLOR_RGB2GRAY)

    ssim_obj = cv2.quality.QualitySSIM_create(array1)
    raw = ssim_obj.compute(array2)

    # raw is often a tuple/list with per-channel values.
    if isinstance(raw, tuple):
        raw_score = float(np.mean(raw[0]))
    else:
        raw_score = float(np.mean(raw))

    return max(0.0, min(1.0, raw_score))


def compute_pairwise_ssim(
    pairs: Sequence[Tuple[int, int]],
    downsampled_images: Sequence[np.ndarray],
) -> List[float]:
    scores = []

    for i, j in tqdm(pairs, desc="Computing SSIM pair scores"):
        score = compute_ssim(downsampled_images[i], downsampled_images[j])
        scores.append(score)

    return scores


# ----------------------------
# Embedding similarity helpers
# ----------------------------

def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Same logic as ft_pair_classifier.py, with float32 conversion.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def cosine_to_01(cosine_score: float) -> float:
    """
    Map cosine similarity from [-1, 1] to [0, 1].
    """
    score = 0.5 * (cosine_score + 1.0)
    return max(0.0, min(1.0, score))


def pairwise_embedding_scores(
    pairs: Sequence[Tuple[int, int]],
    embeddings: Sequence[np.ndarray],
    desc: str,
) -> List[float]:
    scores = []

    for i, j in tqdm(pairs, desc=desc):
        cos = cosine_similarity_np(embeddings[i], embeddings[j])
        scores.append(cosine_to_01(cos))

    return scores


# ----------------------------
# DINOv2 / DINOv3
# ----------------------------

def get_huggingface_token_arg(model_name: str):
    """
    Return the token argument passed to Transformers.

    Environment variables take priority. If none are set and the requested
    model is gated, return True so Transformers uses the cached token from
    `huggingface-cli login`.
    """
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            print(f"Using Hugging Face token from ${env_name}")
            return token

    if "dinov3" in model_name.lower():
        print(
            "No HF_TOKEN/HUGGINGFACE_HUB_TOKEN/HUGGING_FACE_HUB_TOKEN found; "
            "using cached Hugging Face login for gated DINOv3 model."
        )
        return True

    return None


def get_huggingface_env_token_name() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env_name):
            return env_name
    return None


def has_huggingface_cached_token() -> bool:
    try:
        from huggingface_hub import HfFolder
    except Exception:
        return False

    try:
        return bool(HfFolder.get_token())
    except Exception:
        return False


def has_huggingface_auth_for_gated_model() -> bool:
    return get_huggingface_env_token_name() is not None or has_huggingface_cached_token()


def load_dino_model(
    model_name: str,
    device: str | None = None,
    torch_dtype: torch.dtype | None = None,
):
    """
    Adapted from ft_pair_classifier.py.

    For gated DINOv3 satellite model, set:
        export HF_TOKEN=...
    or log in once with:
        huggingface-cli login
    """
    from transformers import AutoImageProcessor, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        if device == "cuda":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    token_arg = get_huggingface_token_arg(model_name)

    try:
        processor = AutoImageProcessor.from_pretrained(model_name, token=token_arg)
        model = AutoModel.from_pretrained(
            model_name,
            token=token_arg,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:
        if "gated repo" in str(exc).lower() or "401 client error" in str(exc).lower():
            raise RuntimeError(
                textwrap.dedent(
                    f"""
                    Could not access gated Hugging Face model: {model_name}

                    The script looked for a token in:
                      - HF_TOKEN
                      - HUGGINGFACE_HUB_TOKEN
                      - HUGGING_FACE_HUB_TOKEN

                    If your token is only stored by the Hugging Face CLI, make sure
                    this shell can see it:
                        huggingface-cli whoami

                    Otherwise export a token with access to {model_name}:
                        export HF_TOKEN=hf_...

                    Original error: {exc}
                    """
                ).strip()
            ) from exc
        raise

    model.to(device)
    model.eval()

    return processor, model, device


def compute_dino_embedding(
    image_or_array: Path | str | np.ndarray,
    processor,
    model,
    device: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Adapted from ft_pair_classifier.py.

    Uses token 0 from last_hidden_state as the global CLS embedding.
    This is also the special/global token used here for DINOv3.
    """
    image = open_rgb_image(image_or_array)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError(
                "DINO model output does not contain last_hidden_state. "
                f"Model output fields: {outputs.keys() if hasattr(outputs, 'keys') else type(outputs)}"
            )

        embedding = outputs.last_hidden_state[:, 0, :]  # CLS/global token

    embedding = embedding.squeeze(0).float()

    if normalize:
        embedding = F.normalize(embedding, dim=0)

    return embedding.cpu().numpy()


def extract_dino_embeddings(
    images_for_embedding: Sequence[np.ndarray],
    model_name: str,
    device: str | None = None,
    normalize: bool = True,
) -> List[np.ndarray]:
    processor, model, device = load_dino_model(
        model_name=model_name,
        device=device,
    )

    embeddings = []

    for img in tqdm(images_for_embedding, desc=f"Computing embeddings: {model_name}"):
        emb = compute_dino_embedding(
            img,
            processor=processor,
            model=model,
            device=device,
            normalize=normalize,
        )
        embeddings.append(emb)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


# ----------------------------
# CLIP
# ----------------------------

def load_clip_model(
    model_name: str,
    device: str | None = None,
):
    """
    Load OpenAI CLIP model.

    Requires:
        pip install git+https://github.com/openai/CLIP.git
    """
    import clip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    return model, preprocess, device


def compute_clip_embedding(
    image_or_array: Path | str | np.ndarray,
    model,
    preprocess,
    device: str,
    normalize: bool = True,
) -> np.ndarray:
    image = open_rgb_image(image_or_array)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model.encode_image(image_tensor)

    embedding = embedding.squeeze(0).float()

    if normalize:
        embedding = F.normalize(embedding, dim=0)

    return embedding.cpu().numpy()


def extract_clip_embeddings(
    images_for_embedding: Sequence[np.ndarray],
    model_name: str,
    device: str | None = None,
    normalize: bool = True,
) -> List[np.ndarray]:
    model, preprocess, device = load_clip_model(
        model_name=model_name,
        device=device,
    )

    embeddings = []

    for img in tqdm(images_for_embedding, desc=f"Computing CLIP embeddings: {model_name}"):
        emb = compute_clip_embedding(
            img,
            model=model,
            preprocess=preprocess,
            device=device,
            normalize=normalize,
        )
        embeddings.append(emb)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


# ----------------------------
# CSV writing
# ----------------------------

def write_similarity_csv(
    output_csv: Path,
    image_paths: Sequence[Path],
    pairs: Sequence[Tuple[int, int]],
    match_count_scores: Sequence[int],
    ssim_scores: Sequence[float],
    dinov2_scores: Sequence[float],
    dinov3_scores: Sequence[float],
    clip_rn50_scores: Sequence[float],
    clip_vit_b32_scores: Sequence[float],
    clip_vit_l14_scores: Sequence[float],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "image_1_index",
        "image_2_index",
        "image_1_basename",
        "image_2_basename",
        "superpoint_lightglue_num_matches",
        "ssim",
        "dinov2",
        "dinov3_sat",
        "clip_rn50",
        "clip_vit_b32",
        "clip_vit_l14",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row_idx, (i, j) in enumerate(pairs):
            writer.writerow([
                i,
                j,
                image_paths[i].name,
                image_paths[j].name,
                int(match_count_scores[row_idx]),
                f"{ssim_scores[row_idx]:.8f}",
                f"{dinov2_scores[row_idx]:.8f}",
                f"{dinov3_scores[row_idx]:.8f}",
                f"{clip_rn50_scores[row_idx]:.8f}",
                f"{clip_vit_b32_scores[row_idx]:.8f}",
                f"{clip_vit_l14_scores[row_idx]:.8f}",
            ])


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--image_paths_txt",
        type=Path,
        help=(
            "Path to image_paths.txt produced by compute_lightglue_matches.py. "
            "Recommended because it guarantees identical image indexing."
        ),
    )

    input_group.add_argument(
        "--image_dir",
        type=Path,
        help=(
            "Directory containing TIFF images. Uses the same sorted TIFF listing "
            "as compute_lightglue_matches.py."
        ),
    )

    parser.add_argument(
        "--pairwise_matches_npy",
        type=Path,
        default=None,
        help=(
            "Optional path to pairwise_matches.npy produced by compute_lightglue_matches.py. "
            "If provided, the CSV will include the number of saved "
            "SuperPoint+LightGlue matches for each image pair. Missing pairs get count 0."
        ),
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, or omit for automatic selection.",
    )

    parser.add_argument(
        "--ssim_size",
        type=int,
        default=256,
        help="Square resize size for SSIM.",
    )

    parser.add_argument(
        "--embedding_size",
        type=int,
        default=224,
        help=(
            "Square resize size before DINO/CLIP embedding extraction. "
            "CLIP preprocessing will still apply its own model-specific transform."
        ),
    )

    parser.add_argument(
        "--dinov2_model",
        type=str,
        default="facebook/dinov2-large",
        help="Hugging Face DINOv2 model name.",
    )

    parser.add_argument(
        "--dinov3_model",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-sat493m",
        help="Hugging Face DINOv3 satellite model name.",
    )

    parser.add_argument(
        "--clip_rn_model",
        type=str,
        default="RN50",
        help="OpenAI CLIP ResNet model name.",
    )

    parser.add_argument(
        "--clip_vit_model",
        type=str,
        default="ViT-B/32",
        help="OpenAI CLIP ViT model name.",
    )

    parser.add_argument(
        "--clip_vit_large_model",
        type=str,
        default="ViT-L/14",
        help="OpenAI CLIP ViT Large model name.",
    )

    parser.add_argument(
        "--skip_ssim",
        action="store_true",
        help="Skip SSIM and write zeros for SSIM.",
    )

    parser.add_argument(
        "--skip_dinov2",
        action="store_true",
        help="Skip DINOv2 and write zeros for DINOv2.",
    )

    parser.add_argument(
        "--skip_dinov3",
        action="store_true",
        help="Skip DINOv3 SAT and write zeros for DINOv3 SAT.",
    )

    parser.add_argument(
        "--skip_clip",
        action="store_true",
        help="Skip all CLIP models and write zeros for CLIP scores.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image_paths_txt is not None:
        image_paths = read_image_paths_txt(args.image_paths_txt.resolve())
    else:
        image_dir = args.image_dir.resolve()
        if not image_dir.exists():
            raise RuntimeError(f"Image directory does not exist: {image_dir}")
        image_paths = list_tif_images(image_dir)

    num_images = len(image_paths)

    if num_images < 2:
        raise RuntimeError("At least two images are required to compute pairwise scores.")

    pairs = make_all_pairs(num_images)

    print(f"Found {num_images} images")
    print(f"Computing {len(pairs)} image pairs")
    print(f"Output CSV: {args.output_csv.resolve()}")

    # Load SuperPoint+LightGlue match counts, if requested.
    match_counts_by_pair = load_pairwise_match_counts(
        pairwise_matches_npy=args.pairwise_matches_npy,
        num_images=num_images,
    )
    match_count_scores = pairwise_match_count_list(
        pairs=pairs,
        match_counts=match_counts_by_pair,
    )

    if args.pairwise_matches_npy is None:
        print("SuperPoint+LightGlue match counts: not provided; writing zeros")
    else:
        nonzero_pairs = sum(1 for c in match_count_scores if c > 0)
        total_matches = int(sum(match_count_scores))
        print(
            "SuperPoint+LightGlue match counts: "
            f"{nonzero_pairs}/{len(pairs)} pairs have matches, "
            f"{total_matches} total matches"
        )

    # Load and downsample once for SSIM.
    if args.skip_ssim:
        ssim_scores = [0.0] * len(pairs)
    else:
        ssim_images = downsample_arrays(
            image_paths,
            output_size=(args.ssim_size, args.ssim_size),
        )
        ssim_scores = compute_pairwise_ssim(pairs, ssim_images)

    # Load and downsample once for all embedding models.
    # Keeping this separate from SSIM allows a different embedding size.
    embedding_images = downsample_arrays(
        image_paths,
        output_size=(args.embedding_size, args.embedding_size),
    )

    if args.skip_dinov2:
        dinov2_scores = [0.0] * len(pairs)
    else:
        dinov2_embeddings = extract_dino_embeddings(
            embedding_images,
            model_name=args.dinov2_model,
            device=args.device,
            normalize=True,
        )
        dinov2_scores = pairwise_embedding_scores(
            pairs,
            dinov2_embeddings,
            desc="Computing DINOv2 pair scores",
        )

    if args.skip_dinov3:
        dinov3_scores = [0.0] * len(pairs)
    elif not has_huggingface_auth_for_gated_model():
        print("WARNING: HF_TOKEN not found. Skipping dinov3_sat computation and writing zeros.")
        dinov3_scores = [0.0] * len(pairs)
    else:
        try:
            dinov3_embeddings = extract_dino_embeddings(
                embedding_images,
                model_name=args.dinov3_model,
                device=args.device,
                normalize=True,
            )
            dinov3_scores = pairwise_embedding_scores(
                pairs,
                dinov3_embeddings,
                desc="Computing DINOv3 SAT pair scores",
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "gated hugging face model" in message or "gated repo" in message or "401 client error" in message:
                print(
                    "WARNING: Could not access DINOv3 SAT gated model. "
                    "Skipping dinov3_sat computation and writing zeros."
                )
                print(f"WARNING: {exc}")
                dinov3_scores = [0.0] * len(pairs)
            else:
                raise

    if args.skip_clip:
        clip_rn50_scores = [0.0] * len(pairs)
        clip_vit_b32_scores = [0.0] * len(pairs)
        clip_vit_l14_scores = [0.0] * len(pairs)
    else:
        clip_rn50_embeddings = extract_clip_embeddings(
            embedding_images,
            model_name=args.clip_rn_model,
            device=args.device,
            normalize=True,
        )
        clip_rn50_scores = pairwise_embedding_scores(
            pairs,
            clip_rn50_embeddings,
            desc="Computing CLIP RN pair scores",
        )

        clip_vit_b32_embeddings = extract_clip_embeddings(
            embedding_images,
            model_name=args.clip_vit_model,
            device=args.device,
            normalize=True,
        )
        clip_vit_b32_scores = pairwise_embedding_scores(
            pairs,
            clip_vit_b32_embeddings,
            desc="Computing CLIP ViT pair scores",
        )

        clip_vit_l14_embeddings = extract_clip_embeddings(
            embedding_images,
            model_name=args.clip_vit_large_model,
            device=args.device,
            normalize=True,
        )
        clip_vit_l14_scores = pairwise_embedding_scores(
            pairs,
            clip_vit_l14_embeddings,
            desc="Computing CLIP ViT Large pair scores",
        )

    write_similarity_csv(
        output_csv=args.output_csv.resolve(),
        image_paths=image_paths,
        pairs=pairs,
        match_count_scores=match_count_scores,
        ssim_scores=ssim_scores,
        dinov2_scores=dinov2_scores,
        dinov3_scores=dinov3_scores,
        clip_rn50_scores=clip_rn50_scores,
        clip_vit_b32_scores=clip_vit_b32_scores,
        clip_vit_l14_scores=clip_vit_l14_scores,
    )

    print("\nSaved:")
    print(f"  {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
