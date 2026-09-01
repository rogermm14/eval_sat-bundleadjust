#!/usr/bin/env bash
set -euo pipefail

AOI_ID="${AOI_ID:?AOI_ID must be set by the caller}"
FEATURE_TYPE="${FEATURE_TYPE:?FEATURE_TYPE must be set by the caller}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$EVAL_UTILS_DIR/set_paths.sh"

IMAGE_DIR="$(eval_image_dir "$AOI_ID")"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"

MAX_KEYPOINTS="${MAX_KEYPOINTS:-4096}"
RESIZE="${RESIZE:-}"
FILTER_THRESHOLD="${FILTER_THRESHOLD:-0.3}"
FUNDAMENTAL_THRESHOLD_PX="${FUNDAMENTAL_THRESHOLD_PX:-0.3}"
MIN_RAW_MATCHES="${MIN_RAW_MATCHES:-8}"
MIN_INLIER_MATCHES="${MIN_INLIER_MATCHES:-8}"
DEVICE_ARGS=()

if [[ "${FORCE_CPU:-false}" == "true" ]]; then
  DEVICE_ARGS+=(--cpu)
fi

if [[ "${FLASH:-false}" == "true" ]]; then
  DEVICE_ARGS+=(--flash)
fi

if [[ -n "$RESIZE" ]]; then
  DEVICE_ARGS+=(--resize "$RESIZE")
fi

eval_require_dir "$IMAGE_DIR"

matches_dir="$(eval_feature_matches_dir "$FEATURE_TYPE" "$AOI_ID")"

mkdir -p "$matches_dir"

python "$SCRIPT_DIR/compute_lightglue_matches.py" \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$matches_dir" \
  --feature_type "$FEATURE_TYPE" \
  --max_keypoints "$MAX_KEYPOINTS" \
  --filter_threshold "$FILTER_THRESHOLD" \
  --fundamental_threshold_px "$FUNDAMENTAL_THRESHOLD_PX" \
  --min_raw_matches "$MIN_RAW_MATCHES" \
  --min_inlier_matches "$MIN_INLIER_MATCHES" \
  "${DEVICE_ARGS[@]}"

echo "Prepared ${FEATURE_TYPE}+LightGlue inputs: $matches_dir"
