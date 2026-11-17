"""Corrected N=33 consistency: CSV-only plot. No torch, inference or fitting.
Reproduce using existing opm-ai-env Python -B, with --out a fresh directory.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,default=Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116'))
ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args();R=args.root.resolve();O=args.out.resolve()
O.mkdir(parents=True,exist_ok=True);(O/'qa').mkdir(exist_ok=True)
STEM='261117_Fig10_corrected_consistency'
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
def read(name):
    p=R/name;hashes[str(p)]=sha(p);return pd.read_csv(p)
SEEDS=[11,22,33,42,44,55];MARKERS=['o','s','^','v','P','X'];OFFSETS=np.linspace(-.24,.24,6)
BLUE='#1f77b4';ORANGE='#ff7f0e'
seed=read('seed_results.csv').sort_values('seed').reset_index(drop=True)
scenario=read('target_scenario_gains.csv')
matrix=read('dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv')
testids=set(matrix[matrix.target_B_role=='zero_shot_and_final_test'].scenario_id)
assert len(testids)==41 and seed.seed.tolist()==SEEDS
assert len(scenario)==246 and not scenario.duplicated(['seed','scenario_id']).any()
assert set(scenario.seed)==set(SEEDS) and set(scenario.scenario_id)==testids
assert scenario.pairs.eq(10).all()
assert np.allclose(scenario.finetuned_iou-scenario.scratch_iou,scenario.iou_gain,rtol=0,atol=1e-14)
assert np.allclose(seed.B_finetuned-seed.B_scratch,seed.paired_gain,rtol=0,atol=1e-14)
for s,g in scenario.groupby('seed'):
    assert len(g)==41 and set(g.scenario_id)==testids
    ref=seed[seed.seed==s].iloc[0]
    for col,seedcol in [('finetuned_iou','B_finetuned'),('scratch_iou','B_scratch'),('iou_gain','paired_gain')]:
        assert abs(g[col].mean()-ref[seedcol])<1e-14
    assert int((g.iou_gain<0).sum())==ref.negative_scenarios
    assert int((g.iou_gain>0).sum())==ref.positive_scenarios
gain=seed.B_finetuned-seed.B_scratch
mean=float(gain.mean());sd=float(gain.std(ddof=1))
assert abs(mean-.006130)<.00000051 and abs(sd-.003981)<.00000051
assert int((gain>0).sum())==5 and int((gain<0).sum())==1
g44=float(gain[seed.seed==44].iloc[0]);assert g44<0 and abs(g44-(-.000079))<.00000051
negative=int((scenario.iou_gain<0).sum());positive=int((scenario.iou_gain>0).sum());zeros=int((scenario.iou_gain==0).sum())
assert negative==61 and positive==185 and zeros==0
target=scenario[(scenario.seed==44)&(scenario.scenario_id=='S099')]
assert len(target)==1
target=target.iloc[0];assert abs(target.iou_gain-(-.026094))<.00000051
# A common scenario ordering keeps every scenario's six repeated seed values
# aligned. No pooling into 246 independent scenario ranks.
ordering=scenario.groupby('scenario_id').iou_gain.mean().reset_index().sort_values(['iou_gain','scenario_id']).scenario_id.tolist()
rank={sid:k+1 for k,sid in enumerate(ordering)}
used=scenario.copy();used['scenario_rank']=used.scenario_id.map(rank)
used['gain_sign']=np.where(used.iou_gain<0,'negative','positive')
used['display_x']=used.scenario_rank+used.seed.map(dict(zip(SEEDS,OFFSETS)))
used=used.sort_values(['scenario_rank','seed'])
assert len(used)==246 and used.groupby('scenario_id').size().eq(6).all()
seed_used=seed[['seed','B_scratch','B_finetuned','paired_gain','positive_scenarios','negative_scenarios']]
seed_used.to_csv(O/(STEM+'_seed_gains.csv'),index=False)
used.to_csv(O/(STEM+'_scenario_observations.csv'),index=False)
fontdir=Path('/mnt/c/Windows/Fonts') if os.name!='nt' else Path('C:/Windows/Fonts')
for f in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:font_manager.fontManager.addfont(str(fontdir/f))
assert 'arial' in font_manager.findfont('Arial',fallback_to_default=False).lower()
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(13.8,6.1),gridspec_kw={'width_ratios':[1,1.18]})
ax=axes[0]
for k,r in seed.iterrows():
    xs=np.array([0.,1.])+np.linspace(-.045,.045,6)[k]
    ax.plot(xs,[r.B_scratch,r.B_finetuned],color='.63',lw=1.2,zorder=1)
    for x,y,color in zip(xs,[r.B_scratch,r.B_finetuned],[BLUE,ORANGE]):
        ax.scatter(x,y,marker=MARKERS[k],s=65,facecolor=color,edgecolor='white',linewidth=.4,zorder=3)
ax.set_xlim(-.18,1.18);ax.set_ylim(.929,.9455)
ax.set_xticks([0,1],['B scratch','A pretrain\n+ B fine-tune'])
ax.set_ylabel('Mean SPE11B test IoU',fontsize=13)
ax.grid(alpha=.2);ax.set_axisbelow(True)
ax.text(.97,.045,f'Mean paired ΔIoU = {mean:+.6f}\nSample SD = {sd:.6f}\nFine-tune > scratch: 5/6 seeds',
    transform=ax.transAxes,ha='right',va='bottom',fontsize=10.5,
    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
r44=seed[seed.seed==44].iloc[0]
ax.annotate(f'Seed 44: {g44:+.6f}',xy=(.52,(r44.B_scratch+r44.B_finetuned)/2),
    xytext=(.47,.93605),ha='center',fontsize=10.5,
    arrowprops=dict(arrowstyle='-',color='.35',lw=.8),
    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
ax.set_title('(a) Six paired model seeds',fontsize=13.5,pad=11)

ax=axes[1]
for s,marker in zip(SEEDS,MARKERS):
    g=used[used.seed==s]
    ax.scatter(g.display_x,g.iou_gain,s=25,marker=marker,
        c=np.where(g.iou_gain<0,ORANGE,BLUE),edgecolors='white',linewidth=.25,zorder=3)
assert sum(len(collection.get_offsets()) for collection in ax.collections)==246
ax.axhline(0,color='black',lw=1.,zorder=2)
ax.set_xlim(0,42);ax.set_ylim(-.033,max(.039,float(scenario.iou_gain.max())+.006))
ticks=[1,11,21,31,41]
ax.set_xticks(ticks,[ordering[t-1] for t in ticks],fontsize=10)
ax.set_xlabel('41 final-test scenarios\n(common ordering by six-seed mean gain)',fontsize=11.5)
ax.set_ylabel('Fine-tune − scratch IoU',fontsize=13)
ax.grid(alpha=.2);ax.set_axisbelow(True)
ax.text(.025,.965,'246 observations = 41 scenarios × 6 seeds',transform=ax.transAxes,ha='left',va='top',fontsize=10.5)
ax.text(.975,.035,f'Negative: {negative}/246\nPositive: {positive}/246',transform=ax.transAxes,
    ha='right',va='bottom',fontsize=10.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
tx=rank['S099']+dict(zip(SEEDS,OFFSETS))[44]
# Arrow targets the actual seed44 point, not the scenario's six-seed mean.
ax.annotate(f'S099, seed 44\n{target.iou_gain:+.6f}',xy=(tx,float(target.iou_gain)),
    xytext=(.38,.12),textcoords='axes fraction',ha='left',va='center',fontsize=10.5,
    arrowprops=dict(arrowstyle='->',lw=.9,color='.25'),
    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
ax.set_title('(b) Repeated seed–scenario observations',fontsize=13.5,pad=11)
handles=[Line2D([],[],marker=marker,ls='none',color='.45',markersize=7,label=f'Seed {s}') for s,marker in zip(SEEDS,MARKERS)]
handles += [Line2D([],[],marker='o',ls='none',color=BLUE,markersize=7,label='Positive gain'),
            Line2D([],[],marker='o',ls='none',color=ORANGE,markersize=7,label='Negative gain')]
fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.055),frameon=False,fontsize=10.5,columnspacing=1.7)
fig.text(.5,.014,'The same 41 scenarios repeat across seeds; 246 observations are not independent geological realizations.',ha='center',fontsize=10)
fig.subplots_adjust(left=.075,right=.985,bottom=.285,top=.89,wspace=.28)
for ext in ['png','pdf']:fig.savefig(O/(STEM+'.'+ext),dpi=600)
fig.savefig(O/'qa/preview.png',dpi=150);plt.close(fig)
assert all(sha(p)==h for p,h in hashes.items())
hashes[str(Path(__file__).resolve())]=sha(__file__)
checks={'seeds':SEEDS,'seed_gains':seed_used[['seed','paired_gain']].to_dict('records'),
    'mean_paired_gain':mean,'sample_SD_ddof1':sd,'positive_seeds':5,'negative_seeds':1,
    'seed44_gain':g44,'negative_seed_scenario_count':negative,'positive_seed_scenario_count':positive,'zero_count':zeros,
    'total_observations':246,'unique_scenarios':41,'S099_seed44_gain':float(target.iou_gain),
    'source_sha256':hashes,'panel_b_core_observation_markers':246,
    'source_data_modified':False,'training_steps':0,'inference_passes':0,
    'output_sha256':{STEM+'.'+e:sha(O/(STEM+'.'+e)) for e in ['png','pdf']}}
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
report=f'''# Corrected Figure 10 seed/scenario consistency

Created 2026-11-17. Replaces the role of legacy Figure 12. Only corrected CSVs were used; legacy JPEG/code inspected for design only. No fitting, inference or checkpoint loading. Original figures/scripts preserved.

## Authoritative files

- `{R/'seed_results.csv'}`
- `{R/'target_scenario_gains.csv'}`
- `{R/'dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv'}` (test membership validation)

Checked exactly six seeds {SEEDS}, 41 identical fixed SPE11B final-test scenario IDs for each seed, 246 unique seed–scenario keys, and 10 pairs behind each scenario mean. Scenario gains equal finetuned_iou−scratch_iou and their per-seed average agrees with each seed's global paired gain to <1e−14. No numerical clipping, zero replacement or label substitution.

## Exact seed statistics

```
{seed_used.to_string(index=False)}
```

Mean paired gain = {mean:+.12f}; sample SD(ddof=1) = {sd:.12f}; 5/6 positive seeds. Seed 44 remains {g44:+.12f} and is explicitly annotated. SD represents model-seed variation for the N=33 protocol, not low-data subset-draw variation.

## Scenario observations

Negative: {negative}/246; positive: {positive}/246; zero: {zeros}. S099/seed44 is {target.iou_gain:+.12f} and its actual point is annotated. Its 10-pair means are scratch={target.scratch_iou:.12f}, fine-tuning={target.finetuned_iou:.12f}; these are scenario means, not single-image pair scores.

Each of the 41 scenarios is assigned one common position, ordered by its six-seed mean gain. All six seed-specific observations remain present at that position, with a small fixed horizontal offset and distinct seed marker. Panel (b) contains exactly 246 core scatter points; negative values are orange, positive values blue, with a black zero line. X tick labels show example scenario IDs, with all rank/ID mappings and raw gains in the observation CSV. No per-scenario aggregation replaces the 246 plotted values. These are repeated observations of the same 41 synthetic scenarios, not 246 independent geological realizations.

Panel (a) preserves scratch blue #1f77b4, fine-tuning orange #ff7f0e and thin gray pairing lines. Small horizontal seed offsets avoid coincident points; IoU values are unchanged. Right panel retains the original blue gain-distribution style with orange negative points added to encode sign. Arial, 600 dpi PNG, vector PDF. Seed shapes are common across both panels. Input/output SHA-256 hashes are recorded in JSON and original CSVs were unchanged after plotting.

Suggested caption: Corrected N=33 seed- and scenario-level consistency. (a) Paired SPE11B test IoUs for six model seeds; gray lines connect scratch and source-pretrained fine-tuning models. The mean paired gain is +0.006130 with sample SD 0.003981, with positive gains in five of six seeds. Seed 44 has a small negative gain. (b) All 246 seed–scenario gains, displayed using a common ordering of the 41 final-test scenarios across seeds. Blue/orange indicate positive/negative gains. There are 61 negative observations, including S099/seed44 at −0.026094. Repeated observations are not independent geological realizations.

Reproduce using the existing Python environment with --root pointing at corrected reanalysis and --out at a fresh directory. No legacy CSV or model dependency is required.
'''
(O/'FIGURE_PROVENANCE.md').write_text(report,encoding='utf-8')
print(seed_used.to_string(index=False),flush=True)
print('CHECKS',json.dumps({k:v for k,v in checks.items() if k not in ['source_sha256','output_sha256','seed_gains']}),flush=True)
print('COMPLETE',O,flush=True)
