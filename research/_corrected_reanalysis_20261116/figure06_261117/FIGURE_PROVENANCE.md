# Corrected Figure 6

Created 2026-11-17. Post-evaluation explanatory selection only: no training, hyperparameter/checkpoint selection or OPM execution. Exactly 12 batch-size-one CPU inference passes on 4 selected pairs, using 6 unique frozen checkpoints.

## Selection

{
  "selection": [
    {
      "row": 1,
      "reason": "Largest target-learning recovery",
      "scenario_id": "S058",
      "seed": 42,
      "i": 1,
      "j": 5,
      "horizon": 4,
      "zero_shot_original_float32_iou": 0.15531915426254272,
      "scratch_original_float32_iou": 0.9289740920066833,
      "fine_tuning_original_float32_iou": 0.9445359706878662
    },
    {
      "row": 2,
      "reason": "Largest positive fine-tuning gain",
      "scenario_id": "S004",
      "seed": 42,
      "i": 1,
      "j": 2,
      "horizon": 1,
      "zero_shot_original_float32_iou": 0.6095617413520813,
      "scratch_original_float32_iou": 0.8067376017570496,
      "fine_tuning_original_float32_iou": 0.9089183807373047
    },
    {
      "row": 3,
      "reason": "High persistence",
      "scenario_id": "S203",
      "seed": 42,
      "i": 4,
      "j": 5,
      "horizon": 1,
      "zero_shot_original_float32_iou": 0.5832012891769409,
      "scratch_original_float32_iou": 0.950160801410675,
      "fine_tuning_original_float32_iou": 0.9455128312110901
    },
    {
      "row": 4,
      "reason": "Negative transfer (S099)",
      "scenario_id": "S099",
      "seed": 44,
      "i": 1,
      "j": 3,
      "horizon": 2,
      "zero_shot_original_float32_iou": 0.3869132399559021,
      "scratch_original_float32_iou": 0.941678524017334,
      "fine_tuning_original_float32_iou": 0.8877005577087402
    }
  ],
  "posthoc_explanatory_only": true,
  "rules": [
    "seed42: maximum min(scratch,finetune)\u2212zero, with zero<persistence and both target models>zero",
    "seed42: maximum positive pair gain, scenario-level average gain>0, excluding row1 scenario",
    "seed42: maximum persistence, excluding row1/row2 scenarios",
    "seed44/S099: minimum negative pair gain across its 10 pairs"
  ],
  "ties": "scenario_id then i then j ascending",
  "no_checkpoint_or_hyperparameter_selection": true
}

The selection plan was written before inference. These are deliberately selected extreme examples, not a representative estimate or independent test evidence. Rows 1–3 use seed 42; row 4 explicitly uses seed 44/S099. Scenario-level means are used only to check row eligibility and are NOT image labels. Every displayed score is recomputed for the exact row pair.

## Inputs and verification

- Corrected root: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116`
- Input CSVs: paired_forecast_gains.csv, target_scenario_gains.csv, primary_metrics_all.csv, persistence_metrics.csv, corrected dataset index/matrix, and exact per-seed evaluation CSVs. Full paths and hashes in JSON.
- Corrected dataset: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/dataset_v1_raw_sgas_waterpressure/spe11b`. Only selected scenario NPZ and corresponding dynamic/mask caches read. Dataset hashes match preprocessing_checks.csv.
- Verified raw pressure float32 cast matches X[:,0]; rebuilt maxabs pressure/XCO2 normalization matches cached dynamic inputs bitwise. Exactly the original 11-channel temporal/scenario conditioning, raw SGAS channel unchanged. Raw >0.001 labels match saved masks at all six times.
- Frozen checkpoint record: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/MODELS_FROZEN_BEFORE_TEST.json`. All six checkpoint hashes match before inference and again after figure creation. Protocol hash matches freeze record. CPU torch version and 12-thread configuration match the original protocol.
- Architecture class definitions alone extracted from the protocol's hashed source; no training module import or train function executed. Batch size one, eval mode, no_grad, float32, no AMP.
- Learned scores recomputed with original float32 IoU expression and independently with Boolean pixel counts. Agreement with corrected exact pair-level and original per-seed CSVs within 1e−7. Boolean-count IoU is used on figures; original float32 IoU is also stored in verification CSV. Persistence matches its saved CSV within 1e−14.
- Inference probabilities and binary masks, exact inputs and truth saved per row in NPZ. No masks manually modified. Source and dataset hashes rechecked unchanged.
- All 4 selected scenarios belong to the fixed 41-scenario B final test. Only corrected source_best/scratch_best/finetuned_best under primary/seed_42 and primary/seed_44 used. No legacy image, metric, checkpoint or case selection reused. Legacy fig6.jpg was inspected only for the six-column layout, typography and purple/yellow colours.

## Coordinates and layout

Same convention as corrected Figures 2–3: array row zero at the bottom (origin=lower), ascending physical x and elevation z, x∈[0,8.4] km and z∈[0,1.2] km. First column shows these axes for every row; all columns use the same extent. Equal rectangular display boxes intentionally do not preserve physical aspect ratio or a common cross-domain scale. No flip, rotation, smoothing or image resampling applied to model input/target arrays. No injection marker is added, keeping the original forecast layout uncluttered.

Arial, PNG 600 dpi, PDF, four rows and six columns. Each row labels scenario, seed, i, j, observation/target times and lead time. White boxes show exact-pair IoU rounded to three decimals. Full precision values and paths are in `261117_Fig06_corrected_spatial_forecasts_row_verification.csv`.

## Suggested caption

Spatial forecasts for four deliberately selected corrected SPE11B final-test examples: large recovery from zero-shot failure, positive fine-tuning gain, high persistence, and negative transfer. Rows 1–3 use seed 42; row 4 uses seed 44. Models use the frozen N=33 target-training checkpoints. Binary predictions use sigmoid probability >0.5, and targets use raw ΔSg >0.001. Each IoU is recomputed for the displayed forecast pair. These cases were selected after final evaluation for explanation and did not affect model or checkpoint selection. Elevation increases upward; equal display boxes are used for visual comparison.

## Final publication-file check

PNG: 9900×5340 pixels, 600 dpi. PDF: one page with embedded Arial TrueType fonts. Both PNG and independently rasterized PDF visually checked. Saved prediction NPZs independently rechecked against row-verification CSV: persistence threshold, sigmoid >0.5 masks and Boolean exact-pair IoUs all agree. Canonical output: /home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure06_261117. Identical Windows delivery copy: D:/OPM with AI/Submission/5. Machine learning_Science and technology/MLST_Fig06_261117.

