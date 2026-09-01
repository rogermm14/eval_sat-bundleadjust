#!/usr/bin/env bash
set -euo pipefail

AOI_ID="${AOI_ID:?AOI_ID must be set by the caller}"
FEATURE_TYPE="${FEATURE_TYPE:?FEATURE_TYPE must be set by the caller}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$EVAL_UTILS_DIR/set_paths.sh"

case "$FEATURE_TYPE" in
  superpoint|aliked)
    ;;
  *)
    echo "FEATURE_TYPE must be either 'superpoint' or 'aliked'. Got: $FEATURE_TYPE"
    exit 1
    ;;
esac

SOURCE_MATCHES_DIR="$(eval_feature_matches_dir "$FEATURE_TYPE" "$AOI_ID")"
OUTPUT_MATCHES_DIR="$(eval_keypt2subpx_matches_dir "$FEATURE_TYPE" "$AOI_ID")"

BATCH_SIZE="${KEYPT2SUBPX_BATCH_SIZE:-2048}"
SCORE_RADIUS="${KEYPT2SUBPX_SCORE_RADIUS:-3}"
RESIZE="${RESIZE:-}"
DEVICE_ARGS=()

if [[ "${FORCE_CPU:-false}" == "true" ]]; then
  DEVICE_ARGS+=(--cpu)
fi

if [[ "${KEYPT2SUBPX_FORCE:-false}" == "true" ]]; then
  DEVICE_ARGS+=(--force)
fi

if [[ -n "$RESIZE" ]]; then
  DEVICE_ARGS+=(--resize "$RESIZE")
fi

eval_require_dir "$SOURCE_MATCHES_DIR"
eval_require_file "$SOURCE_MATCHES_DIR/image_paths.txt"
eval_require_file "$SOURCE_MATCHES_DIR/pairwise_matches.npy"
eval_require_dir "$SOURCE_MATCHES_DIR/features"

mkdir -p "$OUTPUT_MATCHES_DIR"

python "$SCRIPT_DIR/refine_lightglue_matches_keypt2subpx.py" \
  --matches_dir "$SOURCE_MATCHES_DIR" \
  --output_dir "$OUTPUT_MATCHES_DIR" \
  --feature_type "$FEATURE_TYPE" \
  --batch_size "$BATCH_SIZE" \
  --score_radius "$SCORE_RADIUS" \
  "${DEVICE_ARGS[@]}"

echo "Prepared ${FEATURE_TYPE}+LightGlue+Keypt2Subpx inputs: $OUTPUT_MATCHES_DIR"
