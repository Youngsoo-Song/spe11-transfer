# Archived corrected figure source map

All directories below are under `research/_corrected_reanalysis_20261116/`.
Their CSV/JSON provenance and existing PNG/PDF outputs are retained. No figures
were regenerated during packaging.

| Figure/evidence | Directory | Re-execution requirements |
|---|---|---|
| Methods 2/3 initial layout | `methods_figures_261116` | Corrected NPZs and original geometry/mapping dependencies |
| Methods 2/3 equal panel layout | `methods_figures_261117_equal_panels` | Corrected NPZs, prior provenance, historical paths |
| 5 IoU and relative area error | `figure05_261117` | CSVs plus full B masks for original checks |
| 6 spatial examples | `figure06_261117` | Original program performs frozen inference; existing 4 prediction NPZs included |
| 6 larger typography | `figure06_261117_large_type` | Reuses saved corrected Figure 6 predictions |
| 7 horizon and 8 fixed target | `figures07_08_261117` | Original combined program performs inference; 4 saved S089 prediction NPZs included |
| 9 paired low-data draws | `figure09_261117` | Included per-draw evaluation CSVs only, plotting packages and Arial |
| Seed/scenario consistency variant | `figure10_261117` | Included seed/scenario CSVs only |
| 10 TP/FP/FN decomposition | `figure10_error_decomposition_261117` | Saved Figure 6 predictions, corrected NPZs, weights for original hash checks |

Both historically named Figure 10 variants are retained under distinct descriptive
directories. The later user-designated TP/FP/FN decomposition is the Figure 10
candidate; this archive does not renumber or edit the manuscript.

TP/FP/FN cases: S058 seed42 i1/j5, S004 seed42 i1/j2, S099 seed44 i1/j3.
Spatial selections are explanatory after evaluation, not checkpoint selection.
The CSV-derived primary and low-data numerical checks are available through the
portable verifier without any frozen model inference.

