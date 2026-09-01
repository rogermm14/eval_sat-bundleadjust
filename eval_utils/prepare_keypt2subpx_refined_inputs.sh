#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$SCRIPT_DIR"
source "$EVAL_UTILS_DIR/set_paths.sh"

LOG_DIR="${LOG_DIR:-$EVAL_LOGS_ROOT/prepare_keypt2subpx_refined_inputs}"
mkdir -p "$LOG_DIR"

DEFAULT_AOIS=(
  "OMA_042" "OMA_059" "OMA_132" "OMA_134" "OMA_144" "OMA_163"
  "OMA_172" "OMA_176" "OMA_203" "OMA_211" "OMA_230" "OMA_244"
  "OMA_247" "OMA_248" "OMA_251" "OMA_258" "OMA_269" "OMA_278"
  "OMA_281" "OMA_287" "OMA_315" "OMA_329" "OMA_331" "OMA_342"
  "OMA_353" "OMA_355" "OMA_367" "OMA_383" "OMA_389" "OMA_391"
)

if [[ -n "${AOIS:-}" ]]; then
  read -r -a AOI_LIST <<< "$AOIS"
else
  AOI_LIST=("${DEFAULT_AOIS[@]}")
fi

FEATURE_TYPES="${FEATURE_TYPES:-superpoint aliked}"

keypt2subpx_feature_type() {
  local feature_type="$1"
  case "$feature_type" in
    superpoint)
      echo "superpoint_keypt2subpx"
      ;;
    aliked)
      echo "aliked_keypt2subpx"
      ;;
    superpoint_keypt2subpx|aliked_keypt2subpx)
      echo "$feature_type"
      ;;
    *)
      echo "Unsupported FEATURE_TYPE for Keypt2Subpx refinement: $feature_type" >&2
      return 1
      ;;
  esac
}

prepare_one_aoi() {
  local aoi="$1"
  local feature_type
  local refined_feature_types=()
  local refined_feature_type

  export AOI_ID="$aoi"

  for feature_type in $FEATURE_TYPES; do
    case "$feature_type" in
      superpoint_keypt2subpx)
        feature_type="superpoint"
        ;;
      aliked_keypt2subpx)
        feature_type="aliked"
        ;;
    esac

    FEATURE_TYPE="$feature_type" bash "$EVAL_UTILS_DIR/prepare_inputs/prepare_keypt2subpx_refined_matches.sh" || return $?

    refined_feature_type="$(keypt2subpx_feature_type "$feature_type")" || return $?
    refined_feature_types+=("$refined_feature_type")
  done

  local pair_selection_mode

  for pair_selection_mode in least most; do
    echo ""
    echo "Building ${pair_selection_mode}-similar tracks for Keypt2Subpx-refined inputs:"
    echo "  ${refined_feature_types[*]}"

    PAIR_SELECTION_MODE="$pair_selection_mode" \
      FEATURE_TYPES="${refined_feature_types[*]}" \
      bash "$EVAL_UTILS_DIR/prepare_inputs/prepare_least_similar_tracks.sh" || return $?

    echo ""
    echo "Cleaning Keypt2Subpx-refined ${pair_selection_mode}-similar tracks:"
    for refined_feature_type in "${refined_feature_types[@]}"; do
      PAIR_SELECTION_MODE="$pair_selection_mode" \
        FEATURE_TYPE="$refined_feature_type" \
        bash "$EVAL_UTILS_DIR/prepare_inputs/preprocess_clean_tracks_consensus_thr3.sh" || return $?
    done
  done

  echo "Prepared Keypt2Subpx-refined inputs, least/most-similar tracks, and cleaned tracks for $aoi"
}

failed=()

echo "Preparing Keypt2Subpx-refined eval_utils inputs for ${#AOI_LIST[@]} AOIs"
echo "Source feature types: $FEATURE_TYPES"
echo "Also building and cleaning least- and most-similar tracks for the corresponding *_keypt2subpx features"
echo "Logs: $LOG_DIR"

for aoi in "${AOI_LIST[@]}"; do
  log_path="$LOG_DIR/${aoi}.log"

  echo ""
  echo "============================================================"
  echo "Preparing Keypt2Subpx inputs for AOI: $aoi"
  echo "Log: $log_path"
  echo "============================================================"

  if prepare_one_aoi "$aoi" 2>&1 | tee "$log_path"; then
    echo "AOI $aoi Keypt2Subpx inputs prepared."
  else
    status=${PIPESTATUS[0]}
    echo "AOI $aoi failed with exit code $status."
    failed+=("$aoi")
  fi
done

echo ""
echo "============================================================"
echo "Keypt2Subpx input preparation summary"
echo "============================================================"
echo "Total AOIs: ${#AOI_LIST[@]}"
echo "Succeeded: $(( ${#AOI_LIST[@]} - ${#failed[@]} ))"
echo "Failed:    ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
  echo "Failed AOIs: ${failed[*]}"
  exit 1
fi
