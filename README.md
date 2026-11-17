# SPE11 cross-scale plume forecasting

Research snapshot for SPE11A-to-SPE11B binary plume forecasting with an
11-channel U-Net. This repository contains corrected experiment code, fixed
experimental metadata, pair-level evaluation tables, and figure-generation
sources. Repository: https://github.com/Youngsoo-Song/spe11-transfer

**Status: private working archive prepared on 2026-11-17, not a DOI-backed public
release.** Large corrected datasets and frozen weights have been inventoried but
are not uploaded here. No new simulation, model training, checkpoint selection,
or inference was performed to prepare this repository. This packaging step does
not add to the study's 38 completed fits (18 primary + 20 low-data).

## What is included

- `research/`: byte-preserved copies retaining the original project-relative
  layout. The `_corrected_reanalysis_20261116` subdirectory is the corrected
  experiment; top-level model/loader files are dependencies, not legacy scores.
- `metadata/source_manifest.csv`: original paths, repository paths, sizes, and
  SHA-256 of all copied research files.
- `metadata/external_assets.csv`: inventory of the full corrected data/caches
  and 38 frozen checkpoints, with hashes and explicit pending availability.
- `environment/`: versions read from the actual existing analysis environment.
- `tools/verify_repository.py`: portable, standard-library-only checksum and
  metric verification. It does not import archived training code or PyTorch.
- `docs/`: protocol, figure source map, availability limitations, and licensing
  status.

## Verify the included results without training or inference

From the cloned repository root, with Python 3.11:

```bash
python tools/verify_repository.py --out _verification
```

This checks copied file hashes, Python syntax without executing the archived
programs, fixed split membership, 410 forecasting pairs per method and seed,
primary means/SD, low-data draw means/SD, and selected consistency counts. It
writes only to a new output directory, refusing to overwrite a previous run.
The results are recalculated from pair-level tables, not from rounded manuscript
values. This is a numerical/archive check; it is not full independent model
re-execution.

## Fixed experiment and interpretation

- Primary label: positive raw gas-saturation change `delta_Sg > 0.001`, with the
  original valid reporting-grid mask. Prediction: sigmoid probability `> 0.5`.
- Corrected pressure: `(PRESSURE - PCGW) * 1e5` Pa where PCGW is present; deltas
  are formed before the training float32 cast. See the original preprocessing
  code for exact restart/grid handling.
- The original dissolved-CO2 channel representation was intentionally retained.
  Pressure and dissolved-CO2 use observed-image max-absolute-value scaling;
  saturation change is unscaled. All 11 channels were not newly re-extracted.
- Source train/validation/test: 141/25/41 scenarios. Target primary weight-update
  train/validation/final-test: 33/25/41 scenarios. There are 10 forecast pairs per
  test scenario, with observations at indices 1--4 and future indices through 5.
- Primary model seeds: 11, 22, 33, 42, 44, 55. Mean/SD are across six seed means.
- Low-data N=5/10: five historical subset draws 101, 202, 303, 404, 505, fixed
  model seed 42, source seed-42 checkpoint, and the same 25 validation and 41
  final-test scenarios. Their SD is **subset SD, not model-seed SD**. N counts
  weight-update scenarios and excludes the 25 validation scenarios.
- All evidence is restricted to fixed SPE11 geometry/facies, the specified
  noise-free simulated inputs, binary extent, scaling, and training protocol.
  Threshold or normalization robustness was not demonstrated by new ablations.

Reference values (full precision is reconstructed from CSV):

| Condition | Scratch IoU | Fine-tuning IoU | Paired gain |
|---|---:|---:|---:|
| N=33, six model seeds | 0.935035 | 0.941166 | +0.006130 |
| N=5, five subset draws | 0.792452 | 0.862492 | +0.070039 |
| N=10, five subset draws | 0.883188 | 0.906037 | +0.022849 |

Corrected B zero-shot IoU is 0.423838 and persistence IoU is 0.563166. Negative
transfer is retained: only 5/6 primary seed gains are positive, with 61/246
negative seed--scenario observations.

## Reproduction levels and pending assets

The included tables support reproduction of reported statistical aggregation.
Selected spatial prediction arrays and figure sources are preserved. Complete
evaluation re-execution additionally requires the full corrected dataset and
frozen weights listed in `metadata/external_assets.csv`. Reconstruction from raw
OPM files additionally needs simulation files, legacy dissolved-CO2 inputs, and
the original pyopmspe11 mapping dependency, which are not included here.

Archived scripts preserve historical paths/hashes. They are **not all portable
entry points** and some contain training or inference actions. Do not import or
run them indiscriminately. See `docs/REPRODUCIBILITY.md` for supported commands
and remaining path requirements. This repository does not claim complete
end-to-end public reproducibility before external assets are deposited.

The manuscript and reviewer correspondence are not part of this repository.
License selection, external archive URLs, citation author list, a frozen release,
and DOI remain to be finalized before publication. See `LICENSING.md` and
`docs/DATA_AVAILABILITY_DRAFT.md`.

