#!/usr/bin/env bash
set -euo pipefail

AOI_ID="${AOI_ID:?AOI_ID must be set by the caller}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UTILS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$EVAL_UTILS_DIR/set_paths.sh"

FEATURE_TYPES="${FEATURE_TYPES:-superpoint aliked}"
LEAST_PAIR_RANKING_K="${LEAST_PAIR_RANKING_K:-5}"
PAIR_SELECTION_MODE="${PAIR_SELECTION_MODE:-least}"
MIN_TRACK_OBSERVATIONS="${MIN_TRACK_OBSERVATIONS:-4}"
FT_MAX_LENGTH="${FT_MAX_LENGTH:-}"
PAIR_OUTPUT_DIR="$(eval_pair_selection_dir "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")"

RAW_RPC_DIR="$EVAL_REPO_ROOT/DATA/data_ready_full/OMA/$AOI_ID/rpcs" # needed because only triangulable pairs are used to build feature tracks

ARGS=(
  python "$SCRIPT_DIR/prepare_least_similar_tracks.py"
  --reference_matches_dir "$(eval_feature_matches_dir superpoint "$AOI_ID")"
  --rpc_dir "$RAW_RPC_DIR"
  --sat_bundleadjust_repo "$SATBA_REPO"
  --pair_output_dir "$PAIR_OUTPUT_DIR"
  --least_pair_ranking_K "$LEAST_PAIR_RANKING_K"
  --pair_selection_mode "$PAIR_SELECTION_MODE"
  --min_track_observations "$MIN_TRACK_OBSERVATIONS"
)

eval_require_dir "$RAW_RPC_DIR"
eval_require_dir "$(eval_feature_matches_dir superpoint "$AOI_ID")"
eval_require_file "$(eval_feature_matches_dir superpoint "$AOI_ID")/image_paths.txt"
eval_require_file "$(eval_feature_matches_dir superpoint "$AOI_ID")/pairwise_matches.npy"

if [[ -n "$FT_MAX_LENGTH" ]]; then
  ARGS+=(--FT_max_length "$FT_MAX_LENGTH")
fi

if [[ "${DISABLE_PAIR_GEOMETRY_FILTER:-false}" == "true" ]]; then
  ARGS+=(--disable_pair_geometry_filter)
fi

if [[ "${KEEP_CONFLICTED_TRACKS:-false}" == "true" ]]; then
  ARGS+=(--keep_conflicted_tracks)
fi

if [[ "${FORCE_PAIR_SELECTION:-false}" == "true" ]]; then
  ARGS+=(--force_pair_selection)
fi

read -r -a FEATURE_TYPE_LIST <<< "$FEATURE_TYPES"
for feature_type in "${FEATURE_TYPE_LIST[@]}"; do
  matches_dir="$(eval_feature_matches_dir "$feature_type" "$AOI_ID")"
  track_dir="$(eval_tracks_dir "$feature_type" "$AOI_ID" "$LEAST_PAIR_RANKING_K" "$PAIR_SELECTION_MODE")"
  eval_require_dir "$matches_dir"
  eval_require_file "$matches_dir/image_paths.txt"
  eval_require_file "$matches_dir/pairwise_matches.npy"
  ARGS+=(--track_output "${feature_type}:${matches_dir}:${track_dir}")
done

"${ARGS[@]}"
