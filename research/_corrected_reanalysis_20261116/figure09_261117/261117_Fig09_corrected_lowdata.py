"""Corrected Fig9: plot-only low-data paired draws; no model/torch/inference.
Run with existing opm-ai-env Python -B; use --out NEW_DIR for reproduction.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,default=Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/lowdata_n5_n10_final'))
ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args();R=args.source.resolve();O=args.out.resolve()
O.mkdir(parents=True,exist_ok=True);(O/'qa').mkdir(exist_ok=True)
STEM='261117_Fig09_corrected_lowdata'
assert not (O/(STEM+'.png')).exists() and not (O/(STEM+'.pdf')).exists()
os.environ['MPLCONFIGDIR']=str(O/'qa/mplconfig')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

hashes={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):hashes[str(p)]=sha(p);return pd.read_csv(p)
source=R/'corrected_lowdata_subset_results.csv'
data=read(source)
SEEDS=[101,202,303,404,505]
assert len(data)==10 and set(data.N)=={5,10}
assert not data.duplicated(['N','subset_seed']).any()
assert set(data.model_seed)=={42} and set(data.validation_scenarios)=={25} and set(data.test_scenarios)=={41}
assert np.allclose(data.finetuned_iou-data.scratch_iou,data.paired_gain,rtol=0,atol=1e-14)
counts={}
for N,g in data.groupby('N'):
    assert sorted(g.subset_seed)==SEEDS
    assert (g.paired_gain>0).all()
    counts[int(N)]=int((g.paired_gain>0).sum())
for row in data.itertuples():
    assert row.subset_seed==SEEDS[row.replicate-1]
    folder=R/f'rep_{row.replicate:02d}_n_{row.N:02d}'
    for method,col in [('scratch','scratch_iou'),('finetune','finetuned_iou')]:
        raw=read(folder/f'metrics_{method}.csv')
        assert len(raw)==410 and raw.scenario_id.nunique()==41 and raw.groupby('scenario_id').size().eq(10).all()
        assert raw.iou.between(0,1).all()
        assert abs(raw.iou.mean()-getattr(row,col))<1e-14
        assert abs(raw.groupby('scenario_id').iou.mean().mean()-getattr(row,col))<1e-14
stats=[]
for N,g in data.groupby('N'):
    r={'N':int(N),'n_subset_draws':5,'positive_paired_gains':counts[int(N)]}
    for metric in ['scratch_iou','finetuned_iou','paired_gain']:
        r[metric+'_mean']=float(g[metric].mean());r[metric+'_sample_sd']=float(g[metric].std(ddof=1))
    stats.append(r)
stats=pd.DataFrame(stats)
comparison=data.pivot(index='subset_seed',columns='N',values='paired_gain').loc[SEEDS]
comparison.columns=['N5_gain','N10_gain'];comparison['N5_minus_N10_gain']=comparison.N5_gain-comparison.N10_gain
comparison['N5_gain_greater']=comparison.N5_minus_N10_gain>0
assert comparison.N5_gain_greater.all()
# Values supplied by the user serve only as a rounded cross-check after calculation.
reference={5:[.792452,.023210,.862492,.023920,.070039,.018144],10:[.883188,.008121,.906037,.009423,.022849,.005695]}
for r in stats.itertuples():
    actual=[r.scratch_iou_mean,r.scratch_iou_sample_sd,r.finetuned_iou_mean,r.finetuned_iou_sample_sd,r.paired_gain_mean,r.paired_gain_sample_sd]
    assert np.all(np.abs(np.array(actual)-reference[r.N])<=.00000051)
used=data[['replicate','subset_seed','N','scratch_iou','finetuned_iou','paired_gain','model_seed','validation_scenarios','test_scenarios']].sort_values(['N','subset_seed'])
used.to_csv(O/(STEM+'_used_subset_data.csv'),index=False)
stats.to_csv(O/(STEM+'_summary.csv'),index=False)
comparison.to_csv(O/(STEM+'_same_draw_gain_check.csv'))
fontdir=Path('/mnt/c/Windows/Fonts') if os.name!='nt' else Path('C:/Windows/Fonts')
for f in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:font_manager.fontManager.addfont(str(fontdir/f))
assert 'arial' in font_manager.findfont('Arial',fallback_to_default=False).lower()
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.labelsize':13,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#1f77b4';ORANGE='#ff7f0e';MARKERS=['o','s','^','v','P'];JITTER=np.linspace(-.06,.06,5)
fig,axes=plt.subplots(1,2,figsize=(10.8,5.6),sharey=True)
for ax,N in zip(axes,[5,10]):
    g=used[used.N==N].set_index('subset_seed').loc[SEEDS]
    stat=stats[stats.N==N].iloc[0]
    for k,seed in enumerate(SEEDS):
        vals=g.loc[seed,['scratch_iou','finetuned_iou']].to_numpy(dtype=float)
        xs=np.array([0.,1.])+JITTER[k]
        ax.plot(xs,vals,color='.65',lw=.85,zorder=1)
        for x,y,color in zip(xs,vals,[BLUE,ORANGE]):
            ax.scatter(x,y,s=51,marker=MARKERS[k],facecolor=color,edgecolor='white',linewidth=.45,zorder=3)
    # Mean glyphs shifted right so none of the five raw draw markers is hidden.
    for x,metric,color in [(0,'scratch_iou',BLUE),(1,'finetuned_iou',ORANGE)]:
        ax.errorbar(x+.18,stat[metric+'_mean'],yerr=stat[metric+'_sample_sd'],fmt='D',
            color=color,markeredgecolor='black',markeredgewidth=.8,markersize=10,
            ecolor='black',elinewidth=1.2,capsize=4,zorder=5)
    ax.set_title(f'N = {N}',fontsize=15,weight='bold',pad=10)
    ax.set_xlim(-.25,1.4);ax.set_ylim(.70,1.00)
    ax.set_xticks([0,1],['B scratch','A pretrain +\nB fine-tune'])
    ax.set_yticks(np.arange(.70,1.001,.05));ax.tick_params(labelsize=12)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.text(.5,.955,f'Paired gain: {stat.paired_gain_mean:+.4f} ± {stat.paired_gain_sample_sd:.4f}\n5/5 draws improve',
        transform=ax.transAxes,ha='center',va='top',fontsize=11.5,linespacing=1.3)
axes[0].set_ylabel('Mean IoU on SPE11B final-test set',fontsize=13)
handles=[Line2D([],[],marker=marker,color='.65',lw=.85,markerfacecolor='.5',markersize=7,label=f'Draw {seed}') for seed,marker in zip(SEEDS,MARKERS)]
handles.append(Line2D([],[],marker='D',color='black',linestyle='none',markersize=9,label='Mean ± 1 subset SD'))
fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.09),frameon=False,fontsize=10.5,columnspacing=2.0)
fig.text(.5,.025,'5 subset draws per N; model seed 42. B validation: 25 additional labelled scenarios.',ha='center',fontsize=10.5)
fig.subplots_adjust(left=.095,right=.985,bottom=.31,top=.89,wspace=.15)
for ext in ['png','pdf']:fig.savefig(O/(STEM+'.'+ext),dpi=600)
fig.savefig(O/'qa/preview.png',dpi=160);plt.close(fig)
assert all(sha(p)==h for p,h in hashes.items())
hashes[str(Path(__file__).resolve())]=sha(__file__)
manifest={'source_sha256':hashes,'subset_seeds':SEEDS,'model_seed':42,'summary':stats.to_dict('records'),
    'same_draw_gain_check':comparison.reset_index().to_dict('records'),'positive_draws_by_N':counts,
    'model_fits':0,'inference_passes':0,'mean_marker_horizontal_offset':.18,
    'axis_range':[.7,1.0],'output_sha256':{STEM+'.'+e:sha(O/(STEM+'.'+e)) for e in ['png','pdf']}}
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
report=f'''# Corrected low-data Figure 9

Created 2026-11-17. Plot-only revision from the ten corrected N=5/10 subset rows. No training, inference, checkpoint loading or legacy result CSV used. N=33 and N=20/100/141 are not plotted or connected.

## Inputs and replication unit

Authoritative data: `{source}`. All ten row means were independently checked against the corresponding corrected `rep_XX_n_05/metrics_scratch.csv`, `metrics_finetune.csv`, `rep_XX_n_10/...` files: 20 saved evaluation CSVs, each with 410 pairs / 41 scenarios / 10 pairs per scenario. Pair and scenario means agree because each scenario has the same number of pairs. Source paths and SHA-256 hashes are in JSON and were rechecked unchanged after plotting.

Subset seeds: {SEEDS}. Model seed is 42 for all rows. The plotted sample SD(ddof=1) describes variation across five overlapping subset draws, not model-seed uncertainty, confidence intervals or predictive uncertainty. N counts scenarios used for target weight updates; B validation 25 labelled scenarios is separate and fixed. The same five draw seeds occur at both N, with N=5 nested in N=10 as documented in the completed experiment. They are not ten independent replications.

## Exact computed statistics

```
{stats.to_string(index=False)}
```

## Same-draw gain checks

Both N have 5/5 positive paired gains. Every same draw has N=5 gain greater than N=10 gain:

```
{comparison.to_string()}
```

Mean, SD and paired-gain values were calculated from original full-precision subset rows; the user's six-decimal reference values were only compared after calculation. Paired gain SD is calculated from the five paired differences and is not formed from independent method SDs.

## Plot conventions

Blue #1f77b4 is scratch; orange #ff7f0e is fine-tuning. Thin gray lines join matched subset draws. Five marker shapes identify seeds. Large black-edged coloured diamonds show means, shifted +0.18 on the categorical x-axis to keep all raw points visible; black whiskers show ±1 subset sample SD. Small, deterministic horizontal offsets separate raw draw markers; no IoU value is changed. Both panels share the explicit IoU range 0.70–1.00. No fitted learning curve, interpolation, significance test or matched N=33 reference is shown. Arial, 600 dpi PNG and vector PDF.

Suggested caption: Paired corrected low-data comparison on the fixed SPE11B final-test set for N=5 and N=10 target weight-update scenarios. Each gray line connects scratch and source-pretrained fine-tuning models using the same subset draw; marker shapes identify subset seeds. Large diamonds and error bars show the mean and ±1 sample standard deviation across five draws (ddof=1). Model seed 42 and the separate 25-scenario target validation set are fixed. Five of five draws improve at each N, and the gain at N=5 exceeds that at N=10 within every draw. N=33 primary results use a different randomization design and are not included.

Original figures/scripts and source CSVs were preserved. Re-run this Python script in the existing environment with --source pointing at corrected lowdata_n5_n10_final and --out pointing at a fresh folder.
'''
(O/'FIGURE_PROVENANCE.md').write_text(report,encoding='utf-8')
print('SUMMARY\n',stats.to_string(index=False),flush=True)
print('DRAW COMPARISON\n',comparison.to_string(),flush=True)
print('COMPLETE',O,flush=True)
