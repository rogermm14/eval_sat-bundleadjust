#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$SCRIPT_DIR"
source "$EVAL_UTILS_DIR/set_paths.sh"

FEATURE_TYPE="${FEATURE_TYPE:-superpoint}"
LEAST_PAIR_RANKING_K="${LEAST_PAIR_RANKING_K:-5}"
PAIR_SELECTION_MODE="${PAIR_SELECTION_MODE:-least}"

case "$PAIR_SELECTION_MODE" in
  least|most)
    ;;
  *)
    echo "PAIR_SELECTION_MODE must be 'least' or 'most'. Got: $PAIR_SELECTION_MODE"
    exit 1
    ;;
esac

if [[ "${RUN_ONE_AOI:-false}" != "true" ]]; then
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

  export IMAGE_OUTLIER_TOP_K="${IMAGE_OUTLIER_TOP_K:-5}"
  if [[ -z "${BASE_OUT_ROOT:-}" ]]; then
    if [[ "$PAIR_SELECTION_MODE" == "least" ]]; then
      export BASE_OUT_ROOT="$(eval_metric_output_root "$IMAGE_OUTLIER_TOP_K")"
    else
      export BASE_OUT_ROOT="$EVAL_OUTPUTS_ROOT/eval_${PAIR_SELECTION_MODE}_similarK5_clean_tracks_image_outlier_topk${IMAGE_OUTLIER_TOP_K}_thr3_SIFT_Cauchy"
    fi
  fi

  LOG_DIR="${LOG_DIR:-$EVAL_LOGS_ROOT/eval_${PAIR_SELECTION_MODE}_similarK5_clean_tracks_image_outlier_topk${IMAGE_OUTLIER_TOP_K}_thr3_SIFT_Cauchy}"
  mkdir -p "$LOG_DIR"

  failed=()

  echo "Running clean eval_utils reproduction for ${#AOI_LIST[@]} AOIs"
  echo "Feature type: $FEATURE_TYPE"
  echo "Pair selection mode: $PAIR_SELECTION_MODE"
  echo "Inputs root: $EVAL_INPUTS_ROOT"
  echo "Output root: $BASE_OUT_ROOT"
  echo "Logs: $LOG_DIR"

  for aoi in "${AOI_LIST[@]}"; do
    log_path="$LOG_DIR/${aoi}.log"

    echo ""
    echo "============================================================"
    echo "AOI: $aoi"
    echo "Log: $log_path"
    echo "============================================================"

    if AOI_ID="$aoi" RUN_ONE_AOI=true bash "$0" 2>&1 | tee "$log_path"; then
      echo "AOI $aoi finished successfully."
    else
      status=${PIPESTATUS[0]}
      echo "AOI $aoi failed with exit code $status."
      failed+=("$aoi")
    fi
  done

  echo ""
  echo "============================================================"
  echo "Evaluation summary"
  echo "============================================================"
  echo "Total AOIs: ${#AOI_LIST[@]}"
  echo "Succeeded: $(( ${#AOI_LIST[@]} - ${#failed[@]} ))"
  echo "Failed:    ${#failed[@]}"

  if (( ${#failed[@]} > 0 )); then
    echo "Failed AOIs: ${failed[*]}"
    exit 1
  fi

  IMAGE_OUTLIER_TOP_K="${IMAGE_OUTLIER_TOP_K:-5}"
  if [[ -z "${BASE_OUT_ROOT:-}" ]]; then
    if [[ "$PAIR_SELECTION_MODE" == "least" ]]; then
      BASE_OUT_ROOT="$(eval_metric_output_root "$IMAGE_OUTLIER_TOP_K")"
    else
      BASE_OUT_ROOT="$EVAL_OUTPUTS_ROOT/eval_${PAIR_SELECTION_MODE}_similarK5_clean_tracks_image_outlier_topk${IMAGE_OUTLIER_TOP_K}_thr3_SIFT_Cauchy"
    fi
  fi
  $EVAL_UTILS_DIR/compute_metrics/eval_metrics_parser.py $BASE_OUT_ROOT

  echo "All AOIs finished successfully."
  exit 0
fi

AOI_ID="${AOI_ID:?AOI_ID must be set when RUN_ONE_AOI=true}"

# Keep eval_utils independent from eval/eval2/old_eval by default.  Evaluation
# inputs and outputs live under the repository-level EVAL root.
MATCHES_DIR="$(eval_feature_matches_dir "$FEATURE_TYPE" "$AOI_ID")"
TRACKS_DIR="$(eval_tracks_dir "$FEATURE_TYPE" "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")"
CLEANED_TRACKS_DIR="$(eval_cleaned_tracks_dir "$FEATURE_TYPE" "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")"
REFERENCE_SELECTED_PAIRS_SOURCE_DIR="$(eval_pair_selection_dir "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")"
IMAGE_OUTLIER_TOP_K="${IMAGE_OUTLIER_TOP_K:-5}"
if [[ -z "${BASE_OUT_ROOT:-}" ]]; then
  if [[ "$PAIR_SELECTION_MODE" == "least" ]]; then
    BASE_OUT_ROOT="$(eval_metric_output_root "$IMAGE_OUTLIER_TOP_K")"
  else
    BASE_OUT_ROOT="$EVAL_OUTPUTS_ROOT/eval_${PAIR_SELECTION_MODE}_similarK5_clean_tracks_image_outlier_topk${IMAGE_OUTLIER_TOP_K}_thr3_SIFT_Cauchy"
  fi
fi
BASE_OUT_DIR=$BASE_OUT_ROOT/$AOI_ID
mkdir -p "$BASE_OUT_DIR"

HELPER_SCRIPT_DIR=$EVAL_UTILS_DIR/compute_metrics

HELDOUT_SCRIPT=$EVAL_UTILS_DIR/compute_metrics/evaluate_heldout_fixed_rpcs_from_cleaned_C.py
ROBUSTNESS_SCRIPT=$EVAL_UTILS_DIR/compute_metrics/analyze_heldout_by_dino_similarity.py
PAIRWISE_3D_SCRIPT=$EVAL_UTILS_DIR/compute_metrics/evaluate_pairwise_3d_height_consistency.py

RAW_RPC_DIR="/home/roger/sat-bundleadjust_github/DATA/data_ready_full/OMA/$AOI_ID/rpcs"
AMES_RPC_DIR="/home/roger/ames_ba/outputs_Cauchy_SIFT/$AOI_ID/adjusted_rpcs"
SATBA_RPC_DIR="/home/roger/sat-bundleadjust_github/LOGS/exhaustive_eval_full/OMA_newnew/$AOI_ID/opencv-flann-baseline/ba_bruteforce/rpcs_adj"
MY_RPC_DIR="/home/roger/sat-bundleadjust_github/LOGS/exhaustive_eval_full/OMA_newnew/$AOI_ID/-lightglue_superpoint-bestpairs-K5/ba_bruteforce/rpcs_adj"

RAW_EVAL_DIR=$BASE_OUT_DIR/raw_rpcs
AMES_EVAL_DIR=$BASE_OUT_DIR/ames_rpcs
SATBA_EVAL_DIR=$BASE_OUT_DIR/satba_rpcs
MY_EVAL_DIR=$BASE_OUT_DIR/my_rpcs

PAIRWISE_SIM_CSV="$(eval_imagepair_dir "$AOI_ID")/pairwise_image_similarities.csv"
PAIRWISE_SIM_COLUMN="${PAIRWISE_SIM_COLUMN:-dinov2}"
USE_PAIRWISE_SIM_CSV="${USE_PAIRWISE_SIM_CSV:-true}"

HOLDOUT_SEED="${HOLDOUT_SEED:-0}"
MAX_NFEV="${MAX_NFEV:-30}"
LOSS="${LOSS:-linear}"
HELDOUT_EXTRA_ARGS=""

MIN_TRACK_LENGTH="${MIN_TRACK_LENGTH:-3}"
MAX_PAIRS_PER_TRACK="${MAX_PAIRS_PER_TRACK:-100}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file missing: $path"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory missing: $path"
    exit 1
  fi
}

check_static_inputs() {
  require_dir "$SATBA_REPO"
  require_dir "$MATCHES_DIR"
  require_file "$MATCHES_DIR/image_paths.txt"
  require_dir "$CLEANED_TRACKS_DIR"
  require_file "$CLEANED_TRACKS_DIR/cleaned_C.npy"

  require_file "$HELDOUT_SCRIPT"
  require_file "$ROBUSTNESS_SCRIPT"
  require_file "$PAIRWISE_3D_SCRIPT"
  require_file "$HELPER_SCRIPT_DIR/evaluate_heldout_fixed_rpcs.py"

  require_dir "$RAW_RPC_DIR"
  require_dir "$AMES_RPC_DIR"
  require_dir "$SATBA_RPC_DIR"
  require_dir "$MY_RPC_DIR"
  require_dir "$REFERENCE_SELECTED_PAIRS_SOURCE_DIR"
}

heldout_outputs_exist() {
  local eval_dir="$1"
  [[ -f "$eval_dir/heldout_errors.csv" && -f "$eval_dir/C.npy" && -f "$eval_dir/summary.txt" ]] \
    && grep -q "strict_loo_initialization: True" "$eval_dir/summary.txt"
}

run_heldout_if_missing() {
  local method_name="$1"
  local rpc_dir="$2"
  local eval_dir="$3"

  mkdir -p "$eval_dir"

  if heldout_outputs_exist "$eval_dir"; then
    echo "Held-out outputs already exist for $method_name:"
    echo "  $eval_dir/heldout_errors.csv"
    echo "  $eval_dir/C.npy"
    return
  fi

  echo ""
  echo "============================================================"
  echo "Running held-out evaluation on cleaned tracks for $method_name"
  echo "============================================================"
  echo "RPC dir:              $rpc_dir"
  echo "Selected pair source: $REFERENCE_SELECTED_PAIRS_SOURCE_DIR"
  echo "Cleaned tracks dir:   $CLEANED_TRACKS_DIR"
  echo "Output dir:           $eval_dir"
  echo ""

  python "$HELDOUT_SCRIPT" \
    --cleaned_tracks_dir "$CLEANED_TRACKS_DIR" \
    --selected_pairs_source_dir "$REFERENCE_SELECTED_PAIRS_SOURCE_DIR" \
    --matches_dir "$MATCHES_DIR" \
    --rpc_dir "$rpc_dir" \
    --sat_bundleadjust_repo "$SATBA_REPO" \
    --helper_script_dir "$HELPER_SCRIPT_DIR" \
    --output_dir "$eval_dir" \
    --holdout_seed "$HOLDOUT_SEED" \
    --max_nfev "$MAX_NFEV" \
    --loss "$LOSS" \
    --strict_loo_initialization \
    $HELDOUT_EXTRA_ARGS

  if ! heldout_outputs_exist "$eval_dir"; then
    echo "Held-out evaluation for $method_name finished, but expected outputs are missing:"
    echo "  $eval_dir/heldout_errors.csv"
    echo "  $eval_dir/C.npy"
    exit 1
  fi
}

run_robustness_analysis() {
  local out_dir="$BASE_OUT_DIR/dino_robustness"
  mkdir -p "$out_dir"

  echo ""
  echo "============================================================"
  echo "Running robustness analysis by image outlier score"
  echo "============================================================"
  echo "Output dir: $out_dir"
  echo ""

  if [[ "$USE_PAIRWISE_SIM_CSV" == "true" && -f "$PAIRWISE_SIM_CSV" ]]; then
    echo "Using pairwise similarity CSV:"
    echo "  $PAIRWISE_SIM_CSV"
    echo "Similarity column:"
    echo "  $PAIRWISE_SIM_COLUMN"

    python "$ROBUSTNESS_SCRIPT" \
      --method raw:"$RAW_EVAL_DIR" \
      --method ames:"$AMES_EVAL_DIR" \
      --method satba:"$SATBA_EVAL_DIR" \
      --method my:"$MY_EVAL_DIR" \
      --matches_dir "$MATCHES_DIR" \
      --pairwise_similarity_csv "$PAIRWISE_SIM_CSV" \
      --pairwise_similarity_column "$PAIRWISE_SIM_COLUMN" \
      --output_dir "$out_dir" \
      --difficulty_metric heldout_image_outlier_score \
      --image_outlier_top_k "$IMAGE_OUTLIER_TOP_K" \
      --n_quantile_bins 5 \
      --paired_baseline_method ames \
      --paired_comparison_method my
  else
    echo "Pairwise similarity CSV not used or not found."
    echo "Using least-similar DINO pair scores copied from cleaned-track held-out output:"
    echo "  $RAW_EVAL_DIR/dino_pair_scores.csv"

    require_file "$RAW_EVAL_DIR/dino_pair_scores.csv"

    python "$ROBUSTNESS_SCRIPT" \
      --method raw:"$RAW_EVAL_DIR" \
      --method ames:"$AMES_EVAL_DIR" \
      --method satba:"$SATBA_EVAL_DIR" \
      --method my:"$MY_EVAL_DIR" \
      --matches_dir "$MATCHES_DIR" \
      --pairwise_similarity_csv "$RAW_EVAL_DIR/dino_pair_scores.csv" \
      --pairwise_similarity_column dino_similarity \
      --image_1_index_column im_i \
      --image_2_index_column im_j \
      --output_dir "$out_dir" \
      --difficulty_metric heldout_image_outlier_score \
      --image_outlier_top_k "$IMAGE_OUTLIER_TOP_K" \
      --n_quantile_bins 5 \
      --paired_baseline_method ames \
      --paired_comparison_method my
  fi
}

run_pairwise_3d_for_method() {
  local method_name="$1"
  local eval_dir="$2"
  local rpc_dir="$3"
  local out_dir="$BASE_OUT_DIR/pairwise_3d/$method_name"

  mkdir -p "$out_dir"

  echo ""
  echo "============================================================"
  echo "Running pairwise 3D / height consistency for $method_name"
  echo "============================================================"
  echo "Eval dir:   $eval_dir"
  echo "RPC dir:    $rpc_dir"
  echo "Output dir: $out_dir"
  echo ""

  python "$PAIRWISE_3D_SCRIPT" \
    --eval_output_dir "$eval_dir" \
    --matches_dir "$MATCHES_DIR" \
    --rpc_dir "$rpc_dir" \
    --sat_bundleadjust_repo "$SATBA_REPO" \
    --helper_script_dir "$HELPER_SCRIPT_DIR" \
    --output_dir "$out_dir" \
    --min_track_length "$MIN_TRACK_LENGTH" \
    --max_pairs_per_track "$MAX_PAIRS_PER_TRACK" \
    --use_selected_pairs_only
}

print_final_summary() {
  local robustness_out="$BASE_OUT_DIR/dino_robustness"
  local pairwise_out="$BASE_OUT_DIR/pairwise_3d"

  echo ""
  echo "============================================================"
  echo "Final outputs"
  echo "============================================================"

  echo ""
  echo "Primary robustness analysis:"
  echo "  $robustness_out/image_outlier_scores.txt"
  echo "  $robustness_out/summary.txt"
  echo "  $robustness_out/summary_by_method_and_dino_bin.csv"
  echo "  $robustness_out/paired_delta_by_dino_bin.csv"

  echo ""
  echo "Secondary pairwise 3D / height consistency:"
  echo "  $pairwise_out/raw_rpcs/summary.txt"
  echo "  $pairwise_out/ames_rpcs/summary.txt"
  echo "  $pairwise_out/satba_rpcs/summary.txt"
  echo "  $pairwise_out/my_rpcs/summary.txt"
}

check_static_inputs

run_heldout_if_missing "raw_rpcs" "$RAW_RPC_DIR" "$RAW_EVAL_DIR"
run_heldout_if_missing "ames_rpcs" "$AMES_RPC_DIR" "$AMES_EVAL_DIR"
run_heldout_if_missing "satba_rpcs" "$SATBA_RPC_DIR" "$SATBA_EVAL_DIR"
run_heldout_if_missing "my_rpcs" "$MY_RPC_DIR" "$MY_EVAL_DIR"

run_robustness_analysis

run_pairwise_3d_for_method "raw_rpcs" "$RAW_EVAL_DIR" "$RAW_RPC_DIR"
run_pairwise_3d_for_method "ames_rpcs" "$AMES_EVAL_DIR" "$AMES_RPC_DIR"
run_pairwise_3d_for_method "satba_rpcs" "$SATBA_EVAL_DIR" "$SATBA_RPC_DIR"
run_pairwise_3d_for_method "my_rpcs" "$MY_EVAL_DIR" "$MY_RPC_DIR"

print_final_summary

echo ""
echo "Done."
