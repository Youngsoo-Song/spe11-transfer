# Figure 6 readability revision

Typography/layout only, 2026-11-17. Source: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure06_261117`.

IoU boxes moved from upper right to upper left and enlarged 8→12 pt. Scenario/seed labels enlarged 8.5→12.5 pt; row details 8.5→11.5 pt; column titles 10→13 pt; legend 9→12 pt; colourbar ticks 8→11 pt and label 9→12 pt; axis ticks 7→10 pt and axis labels 8→11 pt. Long headings/descriptions wrapped to avoid overlap. The canvas height increased from 8.9 to 9.6 inches while width stayed 16.5 inches. Six columns, four rows, purple/yellow binary colours and 600 dpi retained.

Saved prediction NPZs were used directly. No model, torch, checkpoint or original simulation was executed/read for inference. Every exact-pair IoU was independently checked from saved Boolean masks; the verification CSV is byte-identical to the original. All four cases, seeds, i/j values, probabilities/masks, colours, physical coordinates and upward elevation convention are unchanged. Prior source/figure hashes are unchanged. The full corrected dataset/checkpoint paths remain in the copied row-verification CSV and original provenance JSON. Source/output hashes and exact font changes are recorded in the new JSON. Original files are preserved.

Re-run this plot-only script with `--source` pointing to the prior Figure 6 folder and `--out` pointing to a fresh directory. The source folder must contain its saved prediction NPZs, CSV and provenance. No inference is needed.

Final QA: PNG 9900×5760 pixels at 600 dpi; PDF one page with embedded Arial regular/bold TrueType fonts. PNG and independently rasterized PDF visually checked. Text, labels and legend fit without clipping. No new inference passes.

