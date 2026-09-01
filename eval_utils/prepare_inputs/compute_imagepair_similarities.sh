#!/usr/bin/env bash
set -euo pipefail

AOI_ID="${AOI_ID:?AOI_ID must be set by the caller}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$EVAL_UTILS_DIR/set_paths.sh"

IMAGE_PATHS_TXT="$(eval_feature_matches_dir superpoint "$AOI_ID")/image_paths.txt"
OUTPUT_DIR="$(eval_imagepair_dir "$AOI_ID")"
OUTPUT_CSV="$OUTPUT_DIR/pairwise_image_similarities.csv"
PAIRWISE_MATCHES_NPY="${PAIRWISE_MATCHES_NPY:-$(eval_feature_matches_dir superpoint "$AOI_ID")/pairwise_matches.npy}"

DEVICE="${DEVICE:-cuda}"
SSIM_SIZE="${SSIM_SIZE:-256}"
EMBEDDING_SIZE="${EMBEDDING_SIZE:-224}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
  elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
  fi
fi

EXTRA_ARGS=()

if [[ "${SKIP_SSIM:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_ssim)
fi

if [[ "${SKIP_DINOV2:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_dinov2)
fi

if [[ "${SKIP_DINOV3:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_dinov3)
fi

if [[ "${SKIP_CLIP:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip_clip)
fi

eval_require_file "$IMAGE_PATHS_TXT"
eval_require_file "$PAIRWISE_MATCHES_NPY"

mkdir -p "$OUTPUT_DIR"

CMD=(
  python "$SCRIPT_DIR/compute_pairwise_image_similarities.py"
  --image_paths_txt "$IMAGE_PATHS_TXT"
  --output_csv "$OUTPUT_CSV"
  --device "$DEVICE"
  --ssim_size "$SSIM_SIZE"
  --embedding_size "$EMBEDDING_SIZE"
  "${EXTRA_ARGS[@]}"
)

if [[ -n "$PAIRWISE_MATCHES_NPY" ]]; then
  CMD+=(--pairwise_matches_npy "$PAIRWISE_MATCHES_NPY")
fi

"${CMD[@]}"

echo "Prepared pairwise image similarities: $OUTPUT_CSV"
