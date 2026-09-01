# Clean Evaluation Utilities

This folder contains the clean reproduction of the evaluation launched by:

```bash
eval2/run_eval2_least_similarK5_clean_tracks_image_outlier_topk5_thr3_all_aois_SIFT_Cauchy.sh
```

The scripts here do not import code from `eval`, `eval2`, or `old_eval`.  Their
only repository-code dependency is `bundle_adjust`, loaded through
`--sat_bundleadjust_repo`.

## Main Entrypoints

- `prepare_inputs.sh` prepares all inputs for the AOI list in the script, or
  for the whitespace-separated `AOIS` environment variable.
- `eval_least_similarK5_clean_tracks_image_outlier.sh` runs the clean
  SIFT/Cauchy reproduction for the AOI list in the script, or for `AOIS`.
- `prepare_inputs/compute_lightglue_matches.py` creates `image_paths.txt`,
  feature arrays, and `pairwise_matches.npy` for SuperPoint+LightGlue and
  ALIKED+LightGlue.
- `prepare_inputs/compute_pairwise_image_similarities.py` creates
  `pairwise_image_similarities.csv`.
- `prepare_inputs/prepare_least_similar_tracks.py` selects K=5 least-similar
  image pairs and builds `C.npy` / `C_v2.npy` for each feature matcher without
  computing held-out evaluation errors.
- `compute_metrics/evaluate_heldout_fixed_rpcs_from_cleaned_C.py` computes
  fixed-RPC held-out reprojection errors from cleaned tracks.
- `compute_metrics/analyze_heldout_by_dino_similarity.py` computes
  image-outlier/DINO-binned robustness summaries.
- `compute_metrics/evaluate_pairwise_3d_height_consistency.py` computes
  secondary pairwise 3D and height-consistency metrics.
- `prepare_inputs/preprocess_clean_tracks_consensus.py` is included for
  regenerating cleaned tracks when needed.

## Expected Input Layout

By default the shell scripts read and write evaluation files under the
repository-level `EVAL` root:

```text
EVAL/
  inputs/
    superpoint_lightglue_matching/<AOI>/image_paths.txt
    superpoint_lightglue_matching/<AOI>/pairwise_matches.npy
    superpoint_lightglue_matching/<AOI>/tracks_least_similar_K5/C.npy
    superpoint_lightglue_matching/<AOI>/tracks_least_similar_K5/C_v2.npy
    superpoint_lightglue_matching/<AOI>/tracks_least_similar_K5/cleaned_tracks_consensus_thr3/cleaned_C.npy
    aliked_lightglue_matching/<AOI>/image_paths.txt
    aliked_lightglue_matching/<AOI>/pairwise_matches.npy
    aliked_lightglue_matching/<AOI>/tracks_least_similar_K5/C.npy
    aliked_lightglue_matching/<AOI>/tracks_least_similar_K5/C_v2.npy
    aliked_lightglue_matching/<AOI>/tracks_least_similar_K5/cleaned_tracks_consensus_thr3/cleaned_C.npy
    imagepair_similarities/<AOI>/pairwise_image_similarities.csv
    imagepair_similarities/<AOI>/least_similar_K5/selected_least_similar_pairs.npy
  outputs/
  logs/
```

## Preparing Inputs

For one AOI:

```bash
AOIS="OMA_144" bash eval_utils/prepare_inputs.sh
```

For the full historical AOI list:

```bash
bash eval_utils/prepare_inputs.sh
```

The preparation pipeline runs:

1. `prepare_inputs/prepare_lightglue_matches.sh`
2. `prepare_inputs/compute_imagepair_similarities.sh` once per AOI
3. `prepare_inputs/prepare_least_similar_tracks.sh`
4. `prepare_inputs/preprocess_clean_tracks_consensus_thr3.sh`

SuperPoint+LightGlue is the default feature-track input for the reproduced
metrics. Both SuperPoint and ALIKED matching use LightGlue `filter_threshold=0.3`
and fundamental-matrix RANSAC `fundamental_threshold_px=0.3`.
The image-pair-similarity CSV is shared because DINO/SSIM/CLIP image
similarities do not depend on the local feature matcher.

The DINO robustness step only needs the `dinov2` column, so the default
pairwise-similarity runner skips DINOv3 and CLIP. Set `SKIP_DINOV3=false` or
`SKIP_CLIP=false` to produce those historical columns too.

You can also override paths with environment variables:

```bash
AOI_ID=OMA_144 \
MATCHES_DIR=/path/to/superpoint_lightglue_matching/OMA_144 \
BASE_OUT_ROOT=/path/to/output_root \
bash eval_utils/eval_least_similarK5_clean_tracks_image_outlier.sh
```

To run evaluation for one AOI, use:

```bash
AOIS="OMA_144" bash eval_utils/eval_least_similarK5_clean_tracks_image_outlier.sh
```

The RPC roots are configurable too:

```bash
RAW_RPC_DIR=/path/to/raw/rpcs
AMES_RPC_DIR=/path/to/ames/adjusted_rpcs
SATBA_RPC_DIR=/path/to/satba/rpcs_adj
MY_RPC_DIR=/path/to/my/rpcs_adj
```

Historical defaults for AMES/SATBA/MY RPC roots are preserved in the one-AOI
runner, but they can be overridden for cleaner experiments.
