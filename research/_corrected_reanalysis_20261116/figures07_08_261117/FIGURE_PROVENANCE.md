# Corrected Figure 7 and Figure 8

Created 2026-11-17. Only these two figures were produced. No fitting, new checkpoint selection, hyperparameter change or OPM execution. No legacy horizon CSV/IoU/prediction was loaded. Legacy JPEGs/code were inspected only for design. Same gray/purple/blue/orange model colours and purple/yellow binary palette retained. Arial, 600 dpi PNG, PDF with vector text/linework and native reporting-grid raster panels.

## Figure 7 aggregation and reconciliation

Input `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/primary_metrics_all.csv`: SPE11B B_zero, B_scratch, B_finetuned only. Deterministic persistence: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/persistence_metrics.csv`. Six seeds [11, 22, 33, 42, 44, 55]; exactly 41 fixed final-test scenarios and all ten (i,j) pairs for each method/seed. Checked unique keys, exact pair membership and horizon=j−i. Lead time = 5×horizon years. Every horizon contains 164/123/82/41 pairs at 5/10/15/20 yr (4/3/2/1 per scenario).

First compute each seed's mean over the correct forecast pairs at a horizon; then equally average those six means. Shading is ±sample SD(ddof=1) of the six horizon means. Persistence is one deterministic series with seed=-1 in the export and no SD/shading. Pair counts in the horizon CSV are per seed, not six-fold pooled counts.

Global check: sum(n_h×mean_h)/410 for each seed matches its direct 410-pair IoU. The six weighted seed means then match the corrected global mean and seed_results.csv. Unweighted average over the four horizon means is NOT used. All reconciliation errors <1e−14.

```
      model  lead_years  n_pairs_per_seed  mean_iou  sample_sd
persistence           5               164  0.729328        NaN
persistence          10               123  0.547629        NaN
persistence          15                82  0.400524        NaN
persistence          20                41  0.270411        NaN
  zero_shot           5               164  0.540662   0.037296
  zero_shot          10               123  0.406531   0.030127
  zero_shot          15                82  0.313789   0.031191
  zero_shot          20                41  0.228558   0.024165
    scratch           5               164  0.943839   0.001901
    scratch          10               123  0.936503   0.004654
    scratch          15                82  0.927239   0.003755
    scratch          20                41  0.911012   0.004934
fine_tuning           5               164  0.949423   0.002550
fine_tuning          10               123  0.942461   0.003513
fine_tuning          15                82  0.933579   0.003375
fine_tuning          20                41  0.919426   0.008152

      model seed  weighted_horizon_iou  direct_global_iou  difference
persistence mean              0.563166           0.563166         0.0
  zero_shot mean              0.423838           0.423838         0.0
    scratch mean              0.935035           0.935035         0.0
fine_tuning mean              0.941166           0.941166         0.0
```

## Horizon conclusions at the tested points

```
{
  "monotonicity": {
    "persistence": {
      "seed_monotone_decreasing": {
        "-1": true
      },
      "six_seed_mean_monotone_decreasing": true
    },
    "zero_shot": {
      "seed_monotone_decreasing": {
        "11": true,
        "22": true,
        "33": true,
        "42": true,
        "44": true,
        "55": true
      },
      "six_seed_mean_monotone_decreasing": true
    },
    "scratch": {
      "seed_monotone_decreasing": {
        "11": true,
        "22": true,
        "33": true,
        "42": true,
        "44": true,
        "55": true
      },
      "six_seed_mean_monotone_decreasing": true
    },
    "fine_tuning": {
      "seed_monotone_decreasing": {
        "11": true,
        "22": true,
        "33": true,
        "42": true,
        "44": true,
        "55": true
      },
      "six_seed_mean_monotone_decreasing": true
    }
  },
  "mean_finetune_minus_scratch_by_lead": {
    "5": 0.005583987730305329,
    "10": 0.0059579299071294,
    "15": 0.006339869726964209,
    "20": 0.008414567970648212
  },
  "fine_tuning_mean_above_scratch_at_every_horizon": true,
  "positive_finetuning_seeds_by_lead": {
    "5": 6,
    "10": 5,
    "15": 5,
    "20": 4
  }
}
```

Monotonicity refers to discrete 5/10/15/20-year points, not interpolation or extrapolation. Seed-specific checks are separately reported: an aggregate mean trend does not imply every seed or scenario obeys the same ordering. Horizon groups contain different observation/target-time combinations, so their ordering is descriptive and does not isolate lead time causally. No curve smoothing, values reordering, case replacement or model change was used to obtain a trend.

## Figure 8 fixed-target verification

Scenario S089 is in the corrected 41-scenario B final test. Seed 42 was fixed before inference. Target j=5=25 yr for every row; i=1,2,3,4 means 5,10,15,20 yr and lead 20,15,10,5 yr. True mask is bitwise identical in every row. Plan saved before inference.

Corrected dataset: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/dataset_v1_raw_sgas_waterpressure/spe11b/S089.npz`. Cache inputs are checked bitwise against rebuilt original maxabs-normalized corrected NPZ dynamic fields. Pressure X[:,0] matches the raw unrounded water-pressure difference cast once to float32; SGAS is unclipped. Primary raw ΔSg>0.001 mask matches saved mask at all six snapshots. Added the same eight time/scenario constant channels, producing the original eleven channels.

Checkpoint paths:
{
  "zero_shot": "/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/primary/seed_42/source_best.pt",
  "scratch": "/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/primary/seed_42/scratch_best.pt",
  "fine_tuning": "/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/primary/seed_42/finetuned_best.pt"
}

Each checkpoint hash matches MODELS_FROZEN_BEFORE_TEST.json before inference and after plotting. Protocol hash, torch version 2.12.1+cpu, CPU and thread count match the frozen protocol. Only architecture class definitions extracted from the hashed original implementation; no training code imported/executed. Exactly twelve eval-mode, no_grad, batch-size-one float32 CPU forwards, without AMP.

Predictions use sigmoid probability >0.5. Persistence is the observed raw ΔSg>0.001 mask, checked against saved observed labels. Image IoUs are computed from exact pair predictions, agree with the existing corrected pair-level and original seed42 evaluation CSVs within 1e−7 (original float32 versus independent Boolean-count calculation); persistence agrees within 1e−14. All rows' input/probability/mask artifacts saved in NPZ. No scenario-level average is displayed as an image score.

```
scenario  seed  i  j  lead_years  observation_years  target_years  zero_shot_original_float32_iou  scratch_original_float32_iou  fine_tuning_original_float32_iou  persistence_iou  zero_shot_iou  scratch_iou  fine_tuning_iou  paired_gain
    S089    42  1  5          20                  5            25                        0.383583                      0.901961                          0.888258         0.308017       0.383583     0.901961         0.888258    -0.013703
    S089    42  2  5          15                 10            25                        0.440257                      0.913556                          0.900288         0.507384       0.440257     0.913556         0.900288    -0.013268
    S089    42  3  5          10                 15            25                        0.502627                      0.919242                          0.897087         0.714135       0.502627     0.919242         0.897087    -0.022155
    S089    42  4  5           5                 20            25                        0.559237                      0.929303                          0.932255         0.844937       0.559237     0.929303         0.932255     0.002952
```

## Orientation design and limits

Four-by-six layout; same physical-coordinate convention as corrected Figs 2/3/6: x rightward, elevation z upward, origin=lower, no flipping/rotation/smoothing. All panels use x=[0,8.4] km/z=[0,1.2] km and equal rectangular display boxes, intentionally not physical aspect ratio. Large typography and upper-left IoUs match the requested revised Fig6 style. S089 retained regardless of whether fine-tuning wins each individual row. This fixed-target example is descriptive and does not replace the full-test Figure 7 aggregation.

Raw source, dataset/cache and checkpoint SHA-256 hashes unchanged after plotting. Full paths and hashes in JSON. Existing figures/scripts preserved. Re-run this script in the original CPU Python environment with --root and a fresh --out directory. It reproduces aggregation and twelve inference forwards, without fitting.

## Suggested captions

Figure 7. Forecast lead-time dependence on the fixed 41-scenario SPE11B final-test set. The 5-, 10-, 15- and 20-year groups contain 164, 123, 82 and 41 forecast pairs per model seed, respectively. Learned-model curves average six seed-specific horizon means, with shaded ±1 sample standard deviation. Persistence is deterministic. Horizon groups include different observation and target times.

Figure 8. Fixed-target spatial comparison for corrected SPE11B scenario S089, seed 42. Target time is 25 years in all rows; observation times are 5, 10, 15 and 20 years. Exact-pair IoUs are recomputed using the frozen corrected N=33 checkpoints. Targets use raw ΔSg>0.001 and binary predictions use sigmoid probability>0.5. Elevation increases upward and equal display boxes are used.

## Final QA and explicit interpretation

Both PNGs and independently rasterized PDFs were visually reviewed. Figure 7: 6000×3480 px; Figure 8: 9900×5760 px; 600 dpi. Each PDF is one page with embedded Arial TrueType fonts; Figure 7 is vector linework/text. Figure 8 retains native grid rasters with vector labels. Independent standard-library mean/SD and weighted reconciliation checks passed; persisted Figure 8 probabilities, binary masks, fixed targets and exact-pair IoUs were independently rechecked.

At the whole-test seed-mean level all four methods decrease at every tested horizon step, and each learned seed also has this decreasing horizon-mean ordering. Fine-tuning's six-seed mean exceeds scratch at every horizon, but the positive seed counts are 6/6, 5/6, 5/6 and 4/6 at 5/10/15/20 years. This does not establish universal improvement. In the requested S089/seed42 fixed-target example, fine-tuning is below scratch at 20/15/10 years and above it at 5 years. Its four exact-pair IoUs are not monotonically decreasing with lead time. The requested scenario, seed and observations were retained without substitution.

Canonical output: /home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figures07_08_261117. Windows delivery copy: D:/OPM with AI/Submission/5. Machine learning_Science and technology/MLST_Fig07_08_261117.

