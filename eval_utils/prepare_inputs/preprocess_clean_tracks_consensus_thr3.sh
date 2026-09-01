#!/usr/bin/env bash
set -euo pipefail

AOI_ID="${AOI_ID:?AOI_ID must be set by the caller}"
FEATURE_TYPE="${FEATURE_TYPE:?FEATURE_TYPE must be set by the caller}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$EVAL_UTILS_DIR/set_paths.sh"

LEAST_PAIR_RANKING_K="${LEAST_PAIR_RANKING_K:-5}"
PAIR_SELECTION_MODE="${PAIR_SELECTION_MODE:-least}"

TRACKS_ROOT="$(eval_tracks_dir "$FEATURE_TYPE" "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")"
MATCHES_DIR="$(eval_feature_matches_dir "$FEATURE_TYPE" "$AOI_ID")"
OUTPUT_DIR="$(eval_cleaned_tracks_dir "$FEATURE_TYPE" "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")"

RAW_RPC_DIR="/home/roger/sat-bundleadjust_github/DATA/data_ready_full/OMA/$AOI_ID/rpcs"
AMES_RPC_DIR="/home/roger/ames_ba/outputs/$AOI_ID/adjusted_rpcs"
SATBA_RPC_DIR="/home/roger/sat-bundleadjust_github/LOGS/exhaustive_eval_full/OMA_newnew/$AOI_ID/-lightglue_superpoint-bestpairs-K5/ba_bruteforce/rpcs_adj"

MAX_OBSERVATION_ERROR_PX="${MAX_OBSERVATION_ERROR_PX:-3.0}"
CLEANING_MODE="${CLEANING_MODE:-fast_proxy}"
REFERENCE_METHOD="${REFERENCE_METHOD:-raw}"
CONSENSUS_RULE="${CONSENSUS_RULE:-at_least_two}"
MIN_METHODS_WITH_VALID_ERROR="${MIN_METHODS_WITH_VALID_ERROR:-2}"
MIN_TRACK_LENGTH_AFTER_CLEANING="${MIN_TRACK_LENGTH_AFTER_CLEANING:-3}"
MAX_REMOVED_FRACTION_PER_TRACK="${MAX_REMOVED_FRACTION_PER_TRACK:-0.5}"
LOSS="${LOSS:-linear}"
F_SCALE="${F_SCALE:-1.0}"
MAX_NFEV="${MAX_NFEV:-30}"

eval_require_dir "$RAW_RPC_DIR"
eval_require_dir "$TRACKS_ROOT"
eval_require_file "$TRACKS_ROOT/C.npy"
eval_require_file "$TRACKS_ROOT/C_v2.npy"
eval_require_file "$(eval_pair_selection_dir "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")/selected_pairs.npy"

mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/preprocess_clean_tracks_consensus.py" \
  --raw_eval_dir "$TRACKS_ROOT" \
  --ames_eval_dir "$TRACKS_ROOT" \
  --satba_eval_dir "$TRACKS_ROOT" \
  --raw_rpc_dir "$RAW_RPC_DIR" \
  --ames_rpc_dir "$AMES_RPC_DIR" \
  --satba_rpc_dir "$SATBA_RPC_DIR" \
  --matches_dir "$MATCHES_DIR" \
  --sat_bundleadjust_repo "$SATBA_REPO" \
  --helper_script_dir "$EVAL_UTILS_DIR/compute_metrics" \
  --output_dir "$OUTPUT_DIR" \
  --reference_method "$REFERENCE_METHOD" \
  --cleaning_mode "$CLEANING_MODE" \
  --max_observation_error_px "$MAX_OBSERVATION_ERROR_PX" \
  --consensus_rule "$CONSENSUS_RULE" \
  --min_methods_with_valid_error "$MIN_METHODS_WITH_VALID_ERROR" \
  --min_track_length_after_cleaning "$MIN_TRACK_LENGTH_AFTER_CLEANING" \
  --max_removed_fraction_per_track "$MAX_REMOVED_FRACTION_PER_TRACK" \
  --loss "$LOSS" \
  --f_scale "$F_SCALE" \
  --max_nfev "$MAX_NFEV"

echo "Prepared cleaned tracks: $OUTPUT_DIR"
