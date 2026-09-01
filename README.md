# eval_sat-bundleadjust

Evaluation utilities and notebooks for comparing RPC bundle adjustment
pipelines on the DFC2019 Track3 multi-date satellite image collections over
the Omaha and Jacksonville areas of interest.

This repository is intended to evaluate:

- AMES Stereo Pipeline bundle adjustment
- SAT-BA v1
- SAT-BA v2

The evaluation protocol is described in the paper
[Robust RPC Bundle Adjustment for Multi-Date Satellite Imagery with Season-Invariant Correspondences](https://arxiv.org/abs/2607.26973).

For the detailed evaluation workflow, see [eval_utils/README.md](eval_utils/README.md).

## Repository Structure

- `eval_utils/` contains the scripts used to prepare evaluation inputs, run the
  fixed-RPC held-out experiments, and compute metrics.
- `notebooks/` contains exploratory notebooks, sanity checks, and code used to
  recreate figures from the paper.
- `tests/` contains regression tests for repository dependency boundaries.

## Dependency Direction

`eval_sat-bundleadjust` depends on `sat-bundleadjust`; `sat-bundleadjust` must
not depend on this repository.

In other words, code under `eval_utils/` and `notebooks/` may import
`bundle_adjust`, but code under `sat-bundleadjust/bundle_adjust/` must not import
or path-reference `eval_utils/` or `notebooks/`.

You can check this boundary with:

```bash
python -m pytest tests/test_dependency_boundaries.py -q
```

By default, the test looks for `sat-bundleadjust` at
`/home/roger/sat-bundleadjust`. To use a different checkout, set:

```bash
SAT_BUNDLEADJUST_REPO=/path/to/sat-bundleadjust \
python -m pytest tests/test_dependency_boundaries.py -q
```

## Requirements

The reference environment runs on Python 3.9.20.

The top-level [requirements.txt](requirements.txt) is the single install recipe
for this repository. It includes the core `sat-bundleadjust` requirements, the
evaluation and notebook dependencies, and `sat-bundleadjust` v2 itself. The
`sat-bundleadjust` v2 requirement is intentionally listed last so it has
priority over older `sat-bundleadjust` installations.

To create a conda environment from scratch:

```bash
conda create -n eval_satba python=3.9.20
conda activate eval_satba

python -m pip install -r requirements.txt
```

The DINOv3 satellite model is gated on Hugging Face. If you recompute DINOv3
similarities, provide a token through `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or
`HUGGING_FACE_HUB_TOKEN`.

## Configuration

Before running the evaluation, edit [eval_utils/set_paths.sh](eval_utils/set_paths.sh)
for your local filesystem. In particular, check:

- `SATBA_REPO`: path to the `sat-bundleadjust` checkout
- `EVAL_ROOT`: root directory for generated inputs, outputs, and logs
- `IMAGE_ROOT`: root directory for the DFC2019 Track3 image crops

Then follow the workflow in [eval_utils/README.md](eval_utils/README.md).
