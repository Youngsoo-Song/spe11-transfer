# Reproduction scope and historical code

## Supported now: statistics, checksums, and archived evidence

Run `python tools/verify_repository.py --out _verification` from the repository
root. This requires only the Python standard library. The command reads CSVs,
checks split/pair membership, computes seed/subset statistics with sample SD, and
checks all copied source hashes. It does not evaluate models, select checkpoints,
install packages, or run training/simulation.

The top-level legacy-named model/loader files in `research/` are included because
the corrected protocol AST-extracts definitions from them. They are not a license
to reuse legacy data or results. Do not execute the legacy training entry point.
Only historical N=5/10 `training_scenarios.csv` membership was copied from the
legacy low-data directories; no legacy low-data metrics or weights are included.

## Original analysis environment

The observed environment is Python 3.11.15, NumPy 1.26.4, pandas 3.0.3,
Matplotlib 3.11.0, PyTorch 2.12.1+cpu, 12 Torch threads, and resdata 6.3.2.
Exact installed versions are in `environment/recorded_environment.json`.
`requirements-analysis.txt` lists plotting/table packages only, not a complete
OPM/PyTorch installation recipe. No environment was installed for packaging.
Historical legacy Torch version was not recorded; do not claim it was identical.

## Figure generation

All source scripts and original provenance are retained without changes. Some
support explicit `--root` / `--source` and `--out` arguments; others retain
absolute paths. Arial must be available locally (not redistributed). Use a fresh
output directory, because original scripts guard against overwriting.

For example, the archived corrected low-data Figure 9 script can use the included
per-draw evaluation tables, with no dataset/weights or model execution:

```bash
python research/_corrected_reanalysis_20261116/figure09_261117/261117_Fig09_corrected_lowdata.py --source research/_corrected_reanalysis_20261116/lowdata_n5_n10_final --out _reproduced_fig09
```

This command was not run during repository packaging. It needs the recorded
plotting packages and Arial at the path expected by the script.

Figure 5's original plotting script additionally checks full corrected B masks
even though its bars use CSVs. Until those external assets are restored, use the
portable verification command to reproduce the numeric statistics, not a bypass
of the original mask checks. Figures 6 and the combined 7/8 original programs
perform frozen inference; their saved predictions are included for inspection,
but the original programs were not rerun here. See `FIGURE_INDEX.md`.

## Full frozen model evaluation and raw preprocessing: external assets required

The complete corrected dataset/caches and 18 primary + 20 low-data weights are
listed in `metadata/external_assets.csv` with SHA-256, but not shipped in Git.
Raw simulation UNRST/EGRID/INIT files, deck mapping files, and the historical
dissolved-CO2 input arrays are also not shipped. The original preprocessing
imports the local pyopmspe11 mapping source fingerprinted separately.

Original JSON manifests and CSV path columns intentionally retain historical
absolute paths. They are evidence, not current download locations. Several
programs validate the original protocol/file hashes; rewriting JSON paths would
break that evidence. A future portable evaluator should resolve archive paths
without altering the original manifests or changing model/data conditions.

No full end-to-end re-execution claim is made for this preparation snapshot.
The task here only packages existing evidence; AI training remains closed at 38
fits. No threshold/scaling ablation or additional seeds/N values are introduced.

