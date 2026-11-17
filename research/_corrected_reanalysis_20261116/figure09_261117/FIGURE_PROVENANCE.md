# Corrected low-data Figure 9

Created 2026-11-17. Plot-only revision from the ten corrected N=5/10 subset rows. No training, inference, checkpoint loading or legacy result CSV used. N=33 and N=20/100/141 are not plotted or connected.

## Inputs and replication unit

Authoritative data: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/lowdata_n5_n10_final/corrected_lowdata_subset_results.csv`. All ten row means were independently checked against the corresponding corrected `rep_XX_n_05/metrics_scratch.csv`, `metrics_finetune.csv`, `rep_XX_n_10/...` files: 20 saved evaluation CSVs, each with 410 pairs / 41 scenarios / 10 pairs per scenario. Pair and scenario means agree because each scenario has the same number of pairs. Source paths and SHA-256 hashes are in JSON and were rechecked unchanged after plotting.

Subset seeds: [101, 202, 303, 404, 505]. Model seed is 42 for all rows. The plotted sample SD(ddof=1) describes variation across five overlapping subset draws, not model-seed uncertainty, confidence intervals or predictive uncertainty. N counts scenarios used for target weight updates; B validation 25 labelled scenarios is separate and fixed. The same five draw seeds occur at both N, with N=5 nested in N=10 as documented in the completed experiment. They are not ten independent replications.

## Exact computed statistics

```
 N  n_subset_draws  positive_paired_gains  scratch_iou_mean  scratch_iou_sample_sd  finetuned_iou_mean  finetuned_iou_sample_sd  paired_gain_mean  paired_gain_sample_sd
 5               5                      5          0.792452               0.023210            0.862492                 0.023920          0.070039               0.018144
10               5                      5          0.883188               0.008121            0.906037                 0.009423          0.022849               0.005695
```

## Same-draw gain checks

Both N have 5/5 positive paired gains. Every same draw has N=5 gain greater than N=10 gain:

```
              N5_gain  N10_gain  N5_minus_N10_gain  N5_gain_greater
subset_seed                                                        
101          0.077351  0.020450           0.056901             True
202          0.064229  0.018774           0.045456             True
303          0.054490  0.028328           0.026163             True
404          0.098147  0.029545           0.068603             True
505          0.055979  0.017147           0.038832             True
```

Mean, SD and paired-gain values were calculated from original full-precision subset rows; the user's six-decimal reference values were only compared after calculation. Paired gain SD is calculated from the five paired differences and is not formed from independent method SDs.

## Plot conventions

Blue #1f77b4 is scratch; orange #ff7f0e is fine-tuning. Thin gray lines join matched subset draws. Five marker shapes identify seeds. Large black-edged coloured diamonds show means, shifted +0.18 on the categorical x-axis to keep all raw points visible; black whiskers show ±1 subset sample SD. Small, deterministic horizontal offsets separate raw draw markers; no IoU value is changed. Both panels share the explicit IoU range 0.70–1.00. No fitted learning curve, interpolation, significance test or matched N=33 reference is shown. Arial, 600 dpi PNG and vector PDF.

Suggested caption: Paired corrected low-data comparison on the fixed SPE11B final-test set for N=5 and N=10 target weight-update scenarios. Each gray line connects scratch and source-pretrained fine-tuning models using the same subset draw; marker shapes identify subset seeds. Large diamonds and error bars show the mean and ±1 sample standard deviation across five draws (ddof=1). Model seed 42 and the separate 25-scenario target validation set are fixed. Five of five draws improve at each N, and the gain at N=5 exceeds that at N=10 within every draw. N=33 primary results use a different randomization design and are not included.

Original figures/scripts and source CSVs were preserved. Re-run this Python script in the existing environment with --source pointing at corrected lowdata_n5_n10_final and --out pointing at a fresh folder.

## Final publication-file QA

PNG: 6480×3360 pixels, 600 dpi. PDF: one page, embedded Arial regular/bold TrueType, zero raster image objects (vector points/lines/error bars/text). PNG and independently rasterized PDF visually checked. Independent standard-library calculations of means, sample SDs, paired gains and same-draw N5>N10 checks all passed. Canonical output: /home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure09_261117. Windows delivery copy: D:/OPM with AI/Submission/5. Machine learning_Science and technology/MLST_Fig09_261117.

