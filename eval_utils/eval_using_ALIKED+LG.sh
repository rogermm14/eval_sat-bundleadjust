#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$SCRIPT_DIR"
source "$EVAL_UTILS_DIR/set_paths.sh"

IMAGE_OUTLIER_TOP_K="${IMAGE_OUTLIER_TOP_K:-5}"
KEYPT2SUBPX="${KEYPT2SUBPX:-false}" # true or false
PAIR_SELECTION_MODE="${PAIR_SELECTION_MODE:-least}" #most or least

PAIR_SELECTION_MODE=most
KEYPT2SUBPX=true

case "$PAIR_SELECTION_MODE" in
  least|most)
    ;;
  *)
    echo "PAIR_SELECTION_MODE must be 'least' or 'most'. Got: $PAIR_SELECTION_MODE"
    exit 1
    ;;
esac

if [[ "$KEYPT2SUBPX" == "true" ]]; then
  FEATURE_TYPE_VALUE="aliked_keypt2subpx"
  RUN_NAME="eval_using_ALIKED+LG_keypt2subpx/eval_${PAIR_SELECTION_MODE}_similarK5_clean_tracks_image_outlier_topk${IMAGE_OUTLIER_TOP_K}_thr3_SIFT_Cauchy"
else
  FEATURE_TYPE_VALUE="aliked"
  RUN_NAME="eval_using_ALIKED+LG/eval_${PAIR_SELECTION_MODE}_similarK5_clean_tracks_image_outlier_topk${IMAGE_OUTLIER_TOP_K}_thr3_SIFT_Cauchy"
fi

export FEATURE_TYPE="$FEATURE_TYPE_VALUE"
export PAIR_SELECTION_MODE
export IMAGE_OUTLIER_TOP_K
export BASE_OUT_ROOT="$EVAL_OUTPUTS_ROOT/$RUN_NAME"
export LOG_DIR="$EVAL_LOGS_ROOT/$RUN_NAME"
export RUN_ONE_AOI=false

echo "Running ALIKED+LightGlue evaluation"
echo "  KEYPT2SUBPX: $KEYPT2SUBPX"
echo "  PAIR_SELECTION_MODE: $PAIR_SELECTION_MODE"
echo "  FEATURE_TYPE: $FEATURE_TYPE"
echo "  BASE_OUT_ROOT: $BASE_OUT_ROOT"

bash "$EVAL_UTILS_DIR/eval_least_similarK5_clean_tracks_image_outlier.sh"
