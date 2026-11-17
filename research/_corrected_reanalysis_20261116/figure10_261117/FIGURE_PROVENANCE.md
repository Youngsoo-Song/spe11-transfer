# Corrected Figure 10 seed/scenario consistency

Created 2026-11-17. Replaces the role of legacy Figure 12. Only corrected CSVs were used; legacy JPEG/code inspected for design only. No fitting, inference or checkpoint loading. Original figures/scripts preserved.

## Authoritative files

- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/seed_results.csv`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/target_scenario_gains.csv`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv` (test membership validation)

Checked exactly six seeds [11, 22, 33, 42, 44, 55], 41 identical fixed SPE11B final-test scenario IDs for each seed, 246 unique seed–scenario keys, and 10 pairs behind each scenario mean. Scenario gains equal finetuned_iou−scratch_iou and their per-seed average agrees with each seed's global paired gain to <1e−14. No numerical clipping, zero replacement or label substitution.

## Exact seed statistics

```
 seed  B_scratch  B_finetuned  paired_gain  positive_scenarios  negative_scenarios
   11   0.938663     0.941520     0.002857                  29                  12
   22   0.930961     0.939638     0.008677                  35                   6
   33   0.933753     0.943447     0.009695                  35                   6
   42   0.933788     0.943201     0.009413                  36                   5
   44   0.937131     0.937052    -0.000079                  20                  21
   55   0.935916     0.942136     0.006220                  30                  11
```

Mean paired gain = +0.006130404807; sample SD(ddof=1) = 0.003981177972; 5/6 positive seeds. Seed 44 remains -0.000079370853 and is explicitly annotated. SD represents model-seed variation for the N=33 protocol, not low-data subset-draw variation.

## Scenario observations

Negative: 61/246; positive: 185/246; zero: 0. S099/seed44 is -0.026094478369 and its actual point is annotated. Its 10-pair means are scratch=0.938043165207, fine-tuning=0.911948686838; these are scenario means, not single-image pair scores.

Each of the 41 scenarios is assigned one common position, ordered by its six-seed mean gain. All six seed-specific observations remain present at that position, with a small fixed horizontal offset and distinct seed marker. Panel (b) contains exactly 246 core scatter points; negative values are orange, positive values blue, with a black zero line. X tick labels show example scenario IDs, with all rank/ID mappings and raw gains in the observation CSV. No per-scenario aggregation replaces the 246 plotted values. These are repeated observations of the same 41 synthetic scenarios, not 246 independent geological realizations.

Panel (a) preserves scratch blue #1f77b4, fine-tuning orange #ff7f0e and thin gray pairing lines. Small horizontal seed offsets avoid coincident points; IoU values are unchanged. Right panel retains the original blue gain-distribution style with orange negative points added to encode sign. Arial, 600 dpi PNG, vector PDF. Seed shapes are common across both panels. Input/output SHA-256 hashes are recorded in JSON and original CSVs were unchanged after plotting.

Suggested caption: Corrected N=33 seed- and scenario-level consistency. (a) Paired SPE11B test IoUs for six model seeds; gray lines connect scratch and source-pretrained fine-tuning models. The mean paired gain is +0.006130 with sample SD 0.003981, with positive gains in five of six seeds. Seed 44 has a small negative gain. (b) All 246 seed–scenario gains, displayed using a common ordering of the 41 final-test scenarios across seeds. Blue/orange indicate positive/negative gains. There are 61 negative observations, including S099/seed44 at −0.026094. Repeated observations are not independent geological realizations.

Reproduce using the existing Python environment with --root pointing at corrected reanalysis and --out at a fresh directory. No legacy CSV or model dependency is required.

## Final publication-file QA

PNG: 8280×3660 pixels at 600 dpi. PDF: one page with embedded Arial TrueType and zero raster image objects (vector points, pairing lines, text and annotations). PNG and independently rasterized PDF visually checked. Independent standard-library mean/sample-SD calculations and checks of all 246 unique repeated seed–scenario keys, 61 negative gains and S099/seed44 value passed. Canonical output: /home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure10_261117. Windows delivery copy: D:/OPM with AI/Submission/5. Machine learning_Science and technology/MLST_Fig10_261117.

