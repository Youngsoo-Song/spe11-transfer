"""Reproduce corrected Figure 5 from frozen CSVs only; no torch/model execution.
Run: existing corrected venv python -B this_file.py [--root CORRECTED_ROOT] [--out NEW_DIR]
Existing outputs are never overwritten. Arial is loaded from Windows Fonts.
"""
from pathlib import Path
import argparse, os, sys, json, hashlib
sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, default=Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116'))
ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
args = ap.parse_args()
R, O = args.root.resolve(), args.out.resolve()
O.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(O / 'qa/mplconfig')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

STEM = '261117_Fig05_corrected_performance'
assert not any((O / (STEM + ext)).exists() for ext in ['.png', '.pdf', '_summary.csv'])
SEEDS = [11, 22, 33, 42, 44, 55]
TAGS = ['A_test', 'B_zero', 'B_scratch', 'B_finetuned']
KEYS = ['seed', 'eval_tag', 'scenario_id', 'i', 'j', 'horizon']
METRICS = ['iou', 'dice', 'area_rel_error']
hashes = {}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    hashes[str(p)] = sha(p)
    return pd.read_csv(p)
m = read(R / 'primary_metrics_all.csv')
p = read(R / 'persistence_metrics.csv')
summary_saved = read(R / 'method_seed_summary.csv')
seed_saved = read(R / 'seed_results.csv')
matrix = read(R / 'dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv')
assert sorted(m.seed.unique()) == SEEDS
assert len(m) == 9840 and len(p) == 410 and set(m.eval_tag) == set(TAGS)
assert not m.duplicated(KEYS).any()
assert set(p.seed) == {-1} and set(p.eval_tag) == {'B_persistence'}
assert np.isfinite(m[METRICS]).all().all() and np.isfinite(p[METRICS]).all().all()
assert m.iou.between(0, 1).all()
assert (m.area_rel_error >= 0).all()
test_a = set(matrix.loc[matrix.source_A_split == 'test', 'scenario_id'])
test_b = set(matrix.loc[matrix.target_B_role == 'zero_shot_and_final_test', 'scenario_id'])
assert len(test_a) == len(test_b) == 41
for (seed, tag), g in m.groupby(['seed', 'eval_tag']):
    assert len(g) == 410 and g.groupby('scenario_id').size().eq(10).all()
    assert set(g.scenario_id) == (test_a if tag == 'A_test' else test_b)
    assert set(g['case']) == ({'spe11a'} if tag == 'A_test' else {'spe11b'})
    original = read(R / 'primary' / f'seed_{seed}' / f'{tag}_test_metrics.csv')
    a = g.sort_values(KEYS).reset_index(drop=True)
    b = original.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns], b, check_exact=False, rtol=1e-14, atol=1e-14)
# Derive seed means from pair-level corrected CSV, not rounded reference values.
s = m.groupby(['seed', 'eval_tag'])[METRICS].mean().reset_index()
pd.testing.assert_frame_equal(s.sort_values(['seed','eval_tag']).reset_index(drop=True),
    summary_saved.sort_values(['seed','eval_tag']).reset_index(drop=True),
    check_exact=False, rtol=1e-13, atol=1e-13)
pivot = s.pivot(index='seed', columns='eval_tag', values='iou').loc[SEEDS]
saved = seed_saved.set_index('seed').loc[SEEDS]
assert np.allclose(pivot[TAGS], saved[TAGS], rtol=0, atol=1e-14)
gain = pivot.B_finetuned - pivot.B_scratch
assert np.allclose(gain, saved.paired_gain, rtol=0, atol=1e-14)
# Confirm deterministic persistence area errors directly against the corrected
# stored binary masks. No predictions, checkpoints, or simulations are run.
assert set(p.scenario_id) == test_b
for sid, g in p.groupby('scenario_id'):
    path = R / 'dataset_v1_raw_sgas_waterpressure/spe11b' / f'{sid}.npz'
    hashes[str(path)] = sha(path)
    with np.load(path, allow_pickle=False) as z:
        masks = z['plume_mask'].astype(bool)
    for row in g.itertuples():
        observed, target = masks[row.i], masks[row.j]
        area = abs(int(observed.sum()) - int(target.sum())) / (int(target.sum()) + 1e-6)
        iou = (observed & target).sum() / ((observed | target).sum() + 1e-6)
        assert abs(area - row.area_rel_error) < 1e-14 and abs(iou - row.iou) < 1e-14
for f in ['PROTOCOL.json', 'DATASET_COMPLETE.json', 'FINAL_EVALUATION_COMPLETE.json',
          'FINAL_QA.json', 'train_primary.py', 'summarize_results.py']:
    hashes[str(R / f)] = sha(R / f)
hashes[str(R.parent / 'train_unet_monitoring_207.py')] = sha(R.parent / 'train_unet_monitoring_207.py')
assert json.loads((R / 'FINAL_EVALUATION_COMPLETE.json').read_text())['learned_rows'] == 9840
protocol = json.loads((R / 'PROTOCOL.json').read_text())
assert protocol['primary_threshold'] == .001
dataset = json.loads((R / 'DATASET_COMPLETE.json').read_text())
assert Path(dataset['dataset_version']).name == 'dataset_v1_raw_sgas_waterpressure'

ORDER = ['A_test', 'B_persistence', 'B_zero', 'B_scratch', 'B_finetuned']
LABELS = ['A within-domain', 'Persistence', 'A→B zero-shot', 'B scratch', 'A pretrain\n+ B fine-tune']
COLORS = ['#2ca02c', '#7f7f7f', '#9467bd', '#1f77b4', '#ff7f0e']
rows = []
for tag in ORDER:
    g = s[s.eval_tag == tag]
    row = {'method':tag, 'test_domain':'SPE11A' if tag == 'A_test' else 'SPE11B',
        'seeds':','.join(map(str,SEEDS)) if tag != 'B_persistence' else 'deterministic',
        'replicates':6 if tag != 'B_persistence' else 1}
    for metric in ['iou', 'area_rel_error']:
        row[metric + '_mean'] = float(p[metric].mean() if tag == 'B_persistence' else g[metric].mean())
        row[metric + '_sample_sd'] = np.nan if tag == 'B_persistence' else float(g[metric].std(ddof=1))
    rows.append(row)
result = pd.DataFrame(rows)
result.to_csv(O / (STEM + '_summary.csv'), index=False)
s.to_csv(O / (STEM + '_seed_metrics.csv'), index=False)
pd.DataFrame({'seed':SEEDS, 'fine_tune_minus_scratch_iou':gain.to_numpy()}).to_csv(O / (STEM + '_paired_gain.csv'),index=False)
fontdir = Path('/mnt/c/Windows/Fonts') if os.name != 'nt' else Path('C:/Windows/Fonts')
for f in ['arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf']:
    font_manager.fontManager.addfont(str(fontdir/f))
assert 'arial' in font_manager.findfont('Arial', fallback_to_default=False).lower()
plt.rcParams.update({'font.family':'Arial', 'font.size':10, 'axes.labelsize':11,
    'xtick.labelsize':9, 'ytick.labelsize':10, 'axes.linewidth':.8,
    'pdf.fonttype':42, 'ps.fonttype':42, 'figure.facecolor':'white'})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
x = np.arange(5)
for ax, metric, ylabel in zip(axes, ['iou','area_rel_error'], ['IoU','Relative plume-area error']):
    means = result[metric+'_mean'].to_numpy()
    sd = result[metric+'_sample_sd'].to_numpy()
    ax.bar(x, means, width=.8, color=COLORS, edgecolor='black', linewidth=.8)
    learned = np.array([0, 2, 3, 4])
    ax.errorbar(x[learned], means[learned], yerr=sd[learned], fmt='none',
                color='black', capsize=4, elinewidth=1.1, capthick=1.1)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, LABELS, rotation=12)
    ax.set_xlim(-.65,4.65)
    ax.set_ylim(0, 1 if metric == 'iou' else .7)
    ax.grid(axis='y', alpha=.2)
    ax.axvline(.5, color='.6', lw=.8, ls='--')
    ax.text(0, 1.035, 'SPE11A test', transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
    ax.text(2.5, 1.035, 'SPE11B test', transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
    if metric == 'iou':
        for i, value in enumerate(means):
            err = 0 if np.isnan(sd[i]) else sd[i]
            ax.text(i, value+err+.014, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
for ax, letter in zip(axes,['(a)','(b)']):
    ax.text(.5,-.245,letter,transform=ax.transAxes,ha='center',fontsize=11)
fig.text(.5,.018,'Learned models: mean ± sample SD across 6 seeds (ddof = 1). Persistence: deterministic, no error bar.',
         ha='center',fontsize=8.5)
fig.subplots_adjust(left=.065,right=.985,bottom=.255,top=.89,wspace=.24)
for ext in ['png','pdf']:
    fig.savefig(O/(STEM+'.'+ext),dpi=600)
fig.savefig(O/'qa/preview.png',dpi=160)
plt.close(fig)
assert all(sha(path) == digest for path,digest in hashes.items())
hashes[str(Path(__file__).resolve())] = sha(__file__)
manifest = {'source_hashes':hashes,'seeds':SEEDS,'summary':rows,
    'paired_gain_mean':float(gain.mean()),'paired_gain_sample_sd':float(gain.std(ddof=1)),
    'positive_gain_seeds':int((gain>0).sum()), 'area_error_panel_retained':True,
    'area_metric':'abs(predicted_pixel_area - target_pixel_area)/(target_pixel_area + 1e-6)',
    'aggregation':'410 forecast-pair values averaged per seed; then unweighted mean and sample SD across six seeds',
    'no_training_or_checkpoint_execution':True,
    'output_hashes':{STEM+'.'+e:sha(O/(STEM+'.'+e)) for e in ['png','pdf']}}
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
report = f'''# Corrected Figure 5

Created 2026-11-17. Only Figure 5 was produced; no model/checkpoint inference, training, or OPM execution.

## Sources and checks

- Primary pair-level CSV: `{R / 'primary_metrics_all.csv'}` (9840 rows).
- Deterministic baseline CSV: `{R / 'persistence_metrics.csv'}` (410 rows).
- Seed means independently checked against `{R / 'method_seed_summary.csv'}` and `{R / 'seed_results.csv'}`.
- Every primary metric row, including area error, agrees with its saved `primary/seed_*/<eval_tag>_test_metrics.csv` (24 files).
- Seeds: {SEEDS}. Each learned method/seed contains 41 scenarios × 10 forecast pairs. A_test uses SPE11A test; the other methods use SPE11B test. Test membership verified against the corrected scenario matrix.
- Within each seed, mean over 410 forecast pairs; then mean and SD(ddof=1) over 6 seed means. Forecast pairs are not treated as statistical replicates for error bars. Persistence has no seed SD or error bar.
- Corrected area-error panel retained because area errors exist in both pair-level and original per-seed corrected CSVs. Evaluation source `train_primary.py:evaluate_all` points at the corrected dataset and exports these metrics; `train_unet_monitoring_207.py:calc_metrics` defines area error as abs(pred_area−true_area)/(true_area+1e−6). Code was inspected, not imported or run. Existing final evaluation and QA markers were checked. Persistence IoU and area error were rechecked against corrected stored masks for all 410 pairs, with agreement to 1e−14. Learned area scores were verified as existing exports, without rerunning models.
- No legacy result CSV was read. Legacy fig5.jpg and plotting code were used only for design: original five colours, bar width 0.8, positions 0–4, 12° label rotation and two-panel layout retained. Arial used; separator/group labels added to distinguish A and B test domains. IoU axis fixed to 0–1. Mean IoU labels use three decimals. No seed dots added.

## Exact corrected statistics

```
{result.to_string(index=False)}
```

Fine-tune minus scratch: {gain.mean():+.12f}; paired sample SD: {gain.std(ddof=1):.12f}. Positive gains in {int((gain>0).sum())}/6 seeds.

PNG 600 dpi and vector PDF; source and output SHA-256 hashes in FIGURE_PROVENANCE.json. All source hashes rechecked unchanged after plotting. Existing figures/scripts preserved. Re-run the supplied Python script with `--root` pointing at the corrected folder and `--out` pointing at a fresh directory; uses numpy/pandas/matplotlib/Pillow and Arial fonts, without torch.

Suggested caption: Overall corrected forecasting performance. (a) Intersection over union and (b) relative plume-area error. The green source-domain reference is evaluated on the SPE11A test set; all remaining methods are evaluated on the SPE11B test set. Learned-model bars show means over six seeds, and error bars show ±1 sample standard deviation across seed means. Persistence is deterministic. Metrics within each seed average 410 forecast pairs from 41 held-out scenarios.
'''
(O/'FIGURE_PROVENANCE.md').write_text(report,encoding='utf-8')
print(result.to_string(index=False))
print('PAIRED GAIN',gain.mean(),'SD',gain.std(ddof=1))
print('SAVED',O)
