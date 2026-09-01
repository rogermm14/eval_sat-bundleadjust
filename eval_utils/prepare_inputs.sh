#!/usr/bin/env bash
set -uo pipefail

#PATH="$CONDA_PREFIX/bin:$PATH"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$SCRIPT_DIR"
source "$EVAL_UTILS_DIR/set_paths.sh"

LOG_DIR="$EVAL_LOGS_ROOT/prepare_inputs"
mkdir -p "$LOG_DIR"

AOI_LIST=(
  "OMA_042" "OMA_059" "OMA_132" "OMA_134" "OMA_144" "OMA_163"
  "OMA_172" "OMA_176" "OMA_203" "OMA_211" "OMA_230" "OMA_244"
  "OMA_247" "OMA_248" "OMA_251" "OMA_258" "OMA_269" "OMA_278"
  "OMA_281" "OMA_287" "OMA_315" "OMA_329" "OMA_331" "OMA_342"
  "OMA_353" "OMA_355" "OMA_367" "OMA_383" "OMA_389" "OMA_391"
)

prepare_one_aoi() {
  local aoi="$1"

  export AOI_ID="$aoi"
  
  # (1) COMPUTE SUPERPOINT AND ALIKED LIGHTGLUE MATCHES
  FEATURE_TYPE=superpoint bash "$EVAL_UTILS_DIR/prepare_inputs/prepare_lightglue_matches.sh"
  FEATURE_TYPE=aliked bash "$EVAL_UTILS_DIR/prepare_inputs/prepare_lightglue_matches.sh"

  # (2) COMPUTE DINOV2 IMAGEPAIR SIMILARITIES
  bash "$EVAL_UTILS_DIR/prepare_inputs/compute_imagepair_similarities.sh"
  
  # (3) BUILD FEATURE TRACKS USING SELECTIONS OF LEAST AND MOST SIMILAR PAIRS
  PAIR_SELECTION_MODE=least bash "$EVAL_UTILS_DIR/prepare_inputs/prepare_least_similar_tracks.sh" || return $?
  PAIR_SELECTION_MODE=most bash "$EVAL_UTILS_DIR/prepare_inputs/prepare_least_similar_tracks.sh" || return $?

  # (4) REMOVE POSSIBLE OUTLIERS FROM THE FEATURE TRACKS VIA CONSENSUS
  PAIR_SELECTION_MODE=least FEATURE_TYPE=superpoint bash "$EVAL_UTILS_DIR/prepare_inputs/preprocess_clean_tracks_consensus_thr3.sh" || return $?
  PAIR_SELECTION_MODE=least FEATURE_TYPE=aliked bash "$EVAL_UTILS_DIR/prepare_inputs/preprocess_clean_tracks_consensus_thr3.sh" || return $?
  PAIR_SELECTION_MODE=most FEATURE_TYPE=superpoint bash "$EVAL_UTILS_DIR/prepare_inputs/preprocess_clean_tracks_consensus_thr3.sh" || return $?
  PAIR_SELECTION_MODE=most FEATURE_TYPE=aliked bash "$EVAL_UTILS_DIR/prepare_inputs/preprocess_clean_tracks_consensus_thr3.sh" || return $?

  echo "Prepared eval_utils inputs for $aoi"
}

failed=()

echo "Preparing eval_utils inputs for ${#AOI_LIST[@]} AOIs"
echo "Logs: $LOG_DIR"

for aoi in "${AOI_LIST[@]}"; do
  log_path="$LOG_DIR/${aoi}.log"

  echo ""
  echo "============================================================"
  echo "Preparing inputs for AOI: $aoi"
  echo "Log: $log_path"
  echo "============================================================"

  if prepare_one_aoi "$aoi" 2>&1 | tee "$log_path"; then
    echo "AOI $aoi inputs prepared."
  else
    status=${PIPESTATUS[0]}
    echo "AOI $aoi failed with exit code $status."
    failed+=("$aoi")
  fi
done

echo ""
echo "============================================================"
echo "Input preparation summary"
echo "============================================================"
echo "Total AOIs: ${#AOI_LIST[@]}"
echo "Succeeded: $(( ${#AOI_LIST[@]} - ${#failed[@]} ))"
echo "Failed:    ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
  echo "Failed AOIs: ${failed[*]}"
  exit 1
fi
