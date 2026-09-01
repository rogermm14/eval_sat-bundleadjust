#!/usr/bin/env bash

# Central directory configuration for eval_utils.
#
# Edit this file when moving data, logs, or outputs. Other shell scripts source
# this file and should not define directory defaults themselves.

if [[ -z "${EVAL_UTILS_DIR:-}" ]]; then
  EVAL_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

SATBA_REPO="/home/roger/sat-bundleadjust_github"

EVAL_ROOT="$SATBA_REPO/EVAL"
EVAL_INPUTS_ROOT="$EVAL_ROOT/inputs"
EVAL_OUTPUTS_ROOT="$EVAL_ROOT/outputs"
EVAL_LOGS_ROOT="$EVAL_ROOT/logs"

IMAGE_ROOT="$SATBA_REPO/DATA/Track3-RGB-OMA"

export EVAL_UTILS_DIR
export SATBA_REPO
export EVAL_ROOT EVAL_INPUTS_ROOT EVAL_OUTPUTS_ROOT EVAL_LOGS_ROOT
export IMAGE_ROOT

eval_die() {
  echo "[set_paths] $*" >&2
  exit 1
}

eval_require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    eval_die "Required path variable is not set in eval_utils/set_paths.sh: $name"
  fi
}

eval_require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    eval_die "Required file missing: $path"
  fi
}

eval_require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    eval_die "Required directory missing: $path"
  fi
}

eval_mkdir() {
  local path="$1"
  mkdir -p "$path"
}

eval_image_dir() {
  local aoi="$1"
  echo "$IMAGE_ROOT/$aoi"
}

eval_feature_matches_dir() {
  local feature_type="$1"
  local aoi="$2"
  case "$feature_type" in
    superpoint_keypt2subpx)
      echo "$EVAL_INPUTS_ROOT/superpoint_lightglue_matching_keypt2subpx/$aoi"
      ;;
    aliked_keypt2subpx)
      echo "$EVAL_INPUTS_ROOT/aliked_lightglue_matching_keypt2subpx/$aoi"
      ;;
    *)
      echo "$EVAL_INPUTS_ROOT/${feature_type}_lightglue_matching/$aoi"
      ;;
  esac
}

eval_keypt2subpx_matches_dir() {
  local feature_type="$1"
  local aoi="$2"
  case "$feature_type" in
    superpoint)
      echo "$EVAL_INPUTS_ROOT/superpoint_lightglue_matching_keypt2subpx/$aoi"
      ;;
    aliked)
      echo "$EVAL_INPUTS_ROOT/aliked_lightglue_matching_keypt2subpx/$aoi"
      ;;
    *)
      echo "$EVAL_INPUTS_ROOT/${feature_type}_lightglue_matching_keypt2subpx/$aoi"
      ;;
  esac
}

eval_imagepair_dir() {
  local aoi="$1"
  echo "$EVAL_INPUTS_ROOT/imagepair_similarities/$aoi"
}

eval_pair_selection_dir() {
  local aoi="$1"
  local k="$2"
  local mode="${3:-least}"
  case "$mode" in
    least)
      echo "$(eval_imagepair_dir "$aoi")/least_similar_K${k}"
      ;;
    most)
      echo "$(eval_imagepair_dir "$aoi")/most_similar_K${k}"
      ;;
    *)
      eval_die "Unknown pair selection mode: $mode"
      ;;
  esac
}

eval_tracks_dir() {
  local feature_type="$1"
  local aoi="$2"
  local k="$3"
  local mode="${4:-least}"
  case "$mode" in
    least)
      echo "$(eval_feature_matches_dir "$feature_type" "$aoi")/tracks_least_similar_K${k}"
      ;;
    most)
      echo "$(eval_feature_matches_dir "$feature_type" "$aoi")/tracks_most_similar_K${k}"
      ;;
    *)
      eval_die "Unknown pair selection mode: $mode"
      ;;
  esac
}

eval_cleaned_tracks_dir() {
  local feature_type="$1"
  local aoi="$2"
  local k="$3"
  local mode="${4:-least}"
  echo "$(eval_tracks_dir "$feature_type" "$aoi" "$k" "$mode")/cleaned_tracks_consensus_thr3"
}

eval_metric_output_root() {
  local top_k="$1"
  echo "$EVAL_OUTPUTS_ROOT/eval_least_similarK5_clean_tracks_image_outlier_topk${top_k}_thr3_SIFT_Cauchy"
}

for _eval_required_path_var in \
  EVAL_UTILS_DIR \
  SATBA_REPO \
  EVAL_ROOT \
  EVAL_INPUTS_ROOT \
  EVAL_OUTPUTS_ROOT \
  EVAL_LOGS_ROOT \
  IMAGE_ROOT
do
  eval_require_var "$_eval_required_path_var"
done
unset _eval_required_path_var
