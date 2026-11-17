# Corrected Figure 10 spatial error decomposition

Created 2026-11-17. This file is separate from the previously produced Figure 10 consistency file. No previous file was overwritten. This request uses Figure 10 for the error decomposition; manuscript numbering itself was not edited.

## Sources and fixed cases

Figure 6 source directory: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure06_261117`. Exactly the requested cases were reused: S058/seed42/i1/j5, S004/seed42/i1/j2, S099/seed44/i1/j3. Saved sigmoid probabilities and prediction masks are the corrected Figure 6 artifacts, not legacy predictions. No model forward pass, fitting or new case/checkpoint selection was performed.

Full source paths and SHA-256 are recorded in JSON. Every saved target matches the corrected NPZ raw ΔSg>0.001 target pixel-for-pixel. Input pressure is verified as corrected raw pressure delta cast to float32; rebuilt image-wise normalized dynamic inputs match the saved Figure 6 inputs. Every binary model mask equals its saved sigmoid probability>0.5. Each checkpoint path/hash matches both Figure 6 provenance and MODELS_FROZEN_BEFORE_TEST.json, for the corrected N=33 protocol. Checkpoint bytes are only hashed, not loaded as models. All source hashes unchanged after plotting.

## Definitions and Section 2.7

TP=prediction∩truth, FP=prediction∩not-truth, FN=not-prediction∩truth, TN=not-prediction∩not-truth. U=TP+FP+FN is checked equal to the binary union. FP/U and FN/U are exact union-normalized counts; exact TP/U + FP/U + FN/U = 1. Each plotted pixel is classified once, and all counts sum to the reporting-grid size.

Section 2.7's existing numerical stabilization ε=10⁻⁶ is NOT changed: its IoU=TP/(U+ε) is retained for figure labels and exact Figure 6 reconciliation. The CSV separately provides exact TP/U, regularized Section-2.7 IoU, their epsilon difference, Figure 6 IoU and their difference. All nine model scores match Figure 6 to <1e−14 and the existing corrected exact-pair evaluation CSV to <1e−7 (historical float32 metric arithmetic). Exact and regularized IoUs round to identical three-decimal labels. No epsilon is added to FP/U or FN/U, as requested.

The CSV contains 12 panel records. The three true-mask panels are marked 'target self-reference': TP=true plume pixel count, FP=FN=0. They are not extra model predictions and have no Figure 6 IoU reference or displayed IoU label. The nine model panels are the scientific error decompositions.

## Pixel counts and normalized errors

```
scenario  seed  i  j         method   TP   FP  FN    TN    U  FP_over_U  FN_over_U  IoU_exact_TP_over_U  IoU_section27_TP_over_U_plus_epsilon  difference_from_Figure6
    S058    42  1  5 true_reference 1750    0   0 23450 1750   0.000000   0.000000             1.000000                              1.000000                      NaN
    S058    42  1  5      zero_shot 1022 4830 728 18620 6580   0.734043   0.110638             0.155319                              0.155319             5.551115e-17
    S058    42  1  5        scratch 1648   24 102 23426 1774   0.013529   0.057497             0.928974                              0.928974             0.000000e+00
    S058    42  1  5    fine_tuning 1720   71  30 23379 1821   0.038990   0.016474             0.944536                              0.944536             1.110223e-16
    S004    42  1  2 true_reference  493    0   0 24707  493   0.000000   0.000000             1.000000                              1.000000                      NaN
    S004    42  1  2      zero_shot  306    9 187 24698  502   0.017928   0.372510             0.609562                              0.609562             0.000000e+00
    S004    42  1  2        scratch  455   71  38 24636  564   0.125887   0.067376             0.806738                              0.806738             0.000000e+00
    S004    42  1  2    fine_tuning  479   34  14 24673  527   0.064516   0.026565             0.908918                              0.908918            -1.110223e-16
    S099    44  1  3 true_reference  702    0   0 24498  702   0.000000   0.000000             1.000000                              1.000000                      NaN
    S099    44  1  3      zero_shot  272    1 430 24497  703   0.001422   0.611664             0.386913                              0.386913             5.551115e-17
    S099    44  1  3        scratch  662    1  40 24497  703   0.001422   0.056899             0.941679                              0.941679             0.000000e+00
    S099    44  1  3    fine_tuning  664   46  38 24452  748   0.061497   0.050802             0.887701                              0.887701             0.000000e+00
```

## Model differences and visual interpretation

Positive ΔFP means more red pixels; negative means fewer. Positive ΔFN means more blue pixels; negative means fewer. Both TP/FP/FN field counts and the total grid-pixel area represented by plotted vector rectangles are checked exactly. Differences among model masks thus agree with their colour-pixel counts, without manual interpretation-based alterations. Ratio denominators U vary between models, so changes in FP/U or FN/U are separately tabulated and not inferred solely from absolute counts. The additional difference CSV includes scratch−zero-shot, fine-tuning−zero-shot and fine-tuning−scratch for every row.

- S058, seed 42, i=1→j=5: fine-tuning minus scratch: ΔTP=+72, ΔFP=+47, ΔFN=-72; ΔIoU=+0.015561899. Red/blue pixel areas change by these exact FP/FN counts, respectively.
- S004, seed 42, i=1→j=2: fine-tuning minus scratch: ΔTP=+24, ΔFP=-37, ΔFN=-24; ΔIoU=+0.102180817. Red/blue pixel areas change by these exact FP/FN counts, respectively.
- S099, seed 44, i=1→j=3: fine-tuning minus scratch: ΔTP=+2, ΔFP=+45, ΔFN=-2; ΔIoU=-0.053977986. Red/blue pixel areas change by these exact FP/FN counts, respectively.

```
 row scenario  seed  i  j                  comparison  delta_TP  delta_FP  delta_FN  delta_U  delta_FP_over_U  delta_FN_over_U  delta_IoU_section27_TP_over_U_plus_epsilon
   1     S058    42  1  5     scratch minus zero_shot     626.0   -4806.0    -626.0  -4806.0        -0.720514        -0.053141                                    0.773655
   1     S058    42  1  5 fine_tuning minus zero_shot     698.0   -4759.0    -698.0  -4759.0        -0.695053        -0.094164                                    0.789217
   1     S058    42  1  5   fine_tuning minus scratch      72.0      47.0     -72.0     47.0         0.025461        -0.041023                                    0.015562
   2     S004    42  1  2     scratch minus zero_shot     149.0      62.0    -149.0     62.0         0.107958        -0.305134                                    0.197176
   2     S004    42  1  2 fine_tuning minus zero_shot     173.0      25.0    -173.0     25.0         0.046588        -0.345944                                    0.299357
   2     S004    42  1  2   fine_tuning minus scratch      24.0     -37.0     -24.0    -37.0        -0.061370        -0.040810                                    0.102181
   3     S099    44  1  3     scratch minus zero_shot     390.0       0.0    -390.0      0.0         0.000000        -0.554765                                    0.554765
   3     S099    44  1  3 fine_tuning minus zero_shot     392.0      45.0    -392.0     45.0         0.060075        -0.560862                                    0.500787
   3     S099    44  1  3   fine_tuning minus scratch       2.0      45.0      -2.0     45.0         0.060075        -0.006097                                   -0.053978
```

These are descriptive pair-level error signatures, not a newly established physical mechanism or universal transfer effect. Section 2.7, model protocol and cases remain unchanged.

## Geometry design and outputs

Three rows/four columns: true future plume, zero-shot, scratch, fine-tuning. Existing legacy Fig11 was inspected only for palette/design: TP=#32CD32, FP=#d62728, FN=#1f77b4, TN=black, true-mask background=#440154 and plume=#FDE725. Arial and larger typography/upper-left IoUs follow the revised corrected Fig6.

Same x=[0,8.4] km, elevation z=[0,1.2] km as corrected Figure 6, 2 and 3. The plotting maps increasing grid row to increasing physical elevation, with no flip, rotation, smoothing or interpolation. Equal rectangular display boxes and Figure 6's display aspect are retained. Coordinate axes shown in reference panels apply identically to model panels.

All class fields are exported in NPZ. The publication PDF is fully vector: exact reporting-pixel rectangles are merged only when identical horizontal runs extend across adjacent rows. The sum of vector rectangle areas in grid-pixel units equals each displayed class count. No raster images or image-generation tool are used. PNG is rendered from the same vector blocks at 600 dpi.

Re-run this script with --root pointing at corrected reanalysis and --out a fresh folder. Saved Figure 6 artifacts, corrected NPZs, checkpoint files for hash verification and corrected evaluation CSV are required. No inference is needed.

## Final publication-file QA

PNG: 8340×4800 pixels at 600 dpi. PDF: one page, embedded Arial regular/bold TrueType, zero raster image objects; plume/error regions are exact vector grid rectangles. PNG and independently rasterized PDF visually checked. Independent rechecking of all nine model class maps and three target-reference panels passed: counts, U, FP/U, FN/U, exact TP/U and existing epsilon-regularized Figure 6 IoUs all agree. No training or inference was performed.

Canonical output: /home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure10_error_decomposition_261117. Windows delivery copy: D:/OPM with AI/Submission/5. Machine learning_Science and technology/MLST_Fig10_error_decomposition_261117.

