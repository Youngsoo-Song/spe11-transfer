# Corrected Figure 5

Created 2026-11-17. Only Figure 5 was produced; no model/checkpoint inference, training, or OPM execution.

## Sources and checks

- Primary pair-level CSV: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/primary_metrics_all.csv` (9840 rows).
- Deterministic baseline CSV: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/persistence_metrics.csv` (410 rows).
- Seed means independently checked against `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/method_seed_summary.csv` and `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/seed_results.csv`.
- Every primary metric row, including area error, agrees with its saved `primary/seed_*/<eval_tag>_test_metrics.csv` (24 files).
- Seeds: [11, 22, 33, 42, 44, 55]. Each learned method/seed contains 41 scenarios × 10 forecast pairs. A_test uses SPE11A test; the other methods use SPE11B test. Test membership verified against the corrected scenario matrix.
- Within each seed, mean over 410 forecast pairs; then mean and SD(ddof=1) over 6 seed means. Forecast pairs are not treated as statistical replicates for error bars. Persistence has no seed SD or error bar.
- Corrected area-error panel retained because area errors exist in both pair-level and original per-seed corrected CSVs. Evaluation source `train_primary.py:evaluate_all` points at the corrected dataset and exports these metrics; `train_unet_monitoring_207.py:calc_metrics` defines area error as abs(pred_area−true_area)/(true_area+1e−6). Code was inspected, not imported or run. Existing final evaluation and QA markers were checked. Persistence IoU and area error were rechecked against corrected stored masks for all 410 pairs, with agreement to 1e−14. Learned area scores were verified as existing exports, without rerunning models.
- No legacy result CSV was read. Legacy fig5.jpg and plotting code were used only for design: original five colours, bar width 0.8, positions 0–4, 12° label rotation and two-panel layout retained. Arial used; separator/group labels added to distinguish A and B test domains. IoU axis fixed to 0–1. Mean IoU labels use three decimals. No seed dots added.

## Exact corrected statistics

```
       method test_domain             seeds  replicates  iou_mean  iou_sample_sd  area_rel_error_mean  area_rel_error_sample_sd
       A_test      SPE11A 11,22,33,42,44,55           6  0.945307       0.004917             0.037263                  0.003356
B_persistence      SPE11B     deterministic           1  0.563166            NaN             0.436781                       NaN
       B_zero      SPE11B 11,22,33,42,44,55           6  0.423838       0.031184             0.471138                  0.131979
    B_scratch      SPE11B 11,22,33,42,44,55           6  0.935035       0.002761             0.030216                  0.005575
  B_finetuned      SPE11B 11,22,33,42,44,55           6  0.941166       0.002436             0.024196                  0.002147
```

Fine-tune minus scratch: +0.006130404807; paired sample SD: 0.003981177972. Positive gains in 5/6 seeds.

PNG 600 dpi and vector PDF; source and output SHA-256 hashes in FIGURE_PROVENANCE.json. All source hashes rechecked unchanged after plotting. Existing figures/scripts preserved. Re-run the supplied Python script with `--root` pointing at the corrected folder and `--out` pointing at a fresh directory; uses numpy/pandas/matplotlib/Pillow and Arial fonts, without torch.

Suggested caption: Overall corrected forecasting performance. (a) Intersection over union and (b) relative plume-area error. The green source-domain reference is evaluated on the SPE11A test set; all remaining methods are evaluated on the SPE11B test set. Learned-model bars show means over six seeds, and error bars show ±1 sample standard deviation across seed means. Persistence is deterministic. Metrics within each seed average 410 forecast pairs from 41 held-out scenarios.

## Final publication-file check

PNG: 8100×3120 pixels, 600 dpi. PDF: one page, embedded Arial TrueType, zero raster image objects (vector bars, error bars and text). PNG and independently rendered PDF were visually checked. Canonical output: /home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure05_261117. Windows delivery copy: D:/OPM with AI/Submission/5. Machine learning_Science and technology/MLST_Fig05_261117.

