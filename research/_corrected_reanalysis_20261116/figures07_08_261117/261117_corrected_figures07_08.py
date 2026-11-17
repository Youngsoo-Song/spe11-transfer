"""Corrected Figs 7/8: CSV aggregation + 12 frozen CPU forwards, no fitting.
Run with /home/yssongvd/opm-ai-env/bin/python -B; --out must be a fresh folder.
"""
from pathlib import Path
import os,sys,json,hashlib,ast,math,argparse
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,default=Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116'))
ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args();R=args.root.resolve();O=args.out.resolve();P=R.parent
O.mkdir(parents=True,exist_ok=True);(O/'qa').mkdir(exist_ok=True)
assert not (O/'FIXED_TARGET_PLAN.json').exists(),'Use a fresh output directory'
os.environ['MPLCONFIGDIR']=str(O/'qa/mplconfig')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

S7='261117_Fig07_corrected_horizon';S8='261117_Fig08_corrected_fixed_target'
SEEDS=[11,22,33,42,44,55];METHODS=['persistence','zero_shot','scratch','fine_tuning']
TAGS={'B_zero':'zero_shot','B_scratch':'scratch','B_finetuned':'fine_tuning'}
COUNTS={5:164,10:123,15:82,20:41}
hashes={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def remember(p):hashes[str(p)]=sha(p);return p
def read(name):return pd.read_csv(remember(R/name))
protocol=json.loads(remember(R/'PROTOCOL.json').read_text())
freeze=json.loads(remember(R/'MODELS_FROZEN_BEFORE_TEST.json').read_text())
assert sha(R/'PROTOCOL.json')==freeze['protocol_sha256'] and freeze['test_scores_seen'] is False
assert protocol['primary_threshold']==.001 and protocol['prediction_threshold']==.5
assert protocol['counts']['spe11b_finetune_train']==33
assert protocol['device']=='cpu' and torch.__version__==protocol['torch']
torch.set_num_threads(protocol['threads'])
m=read('primary_metrics_all.csv');p=read('persistence_metrics.csv')
seed_results=read('seed_results.csv')
matrix=read('dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv')
index=read('dataset_v1_raw_sgas_waterpressure/dataset_index.csv')
testids=set(matrix[matrix.target_B_role=='zero_shot_and_final_test'].scenario_id)
assert len(testids)==41 and 'S089' in testids
b=m[m.eval_tag.isin(TAGS)].copy();b['model']=b.eval_tag.map(TAGS)
assert len(b)==7380 and sorted(b.seed.unique())==SEEDS
assert set(b['case'])=={'spe11b'} and set(p['case'])=={'spe11b'}
assert not b.duplicated(['model','seed','scenario_id','i','j']).any()
expected_pairs={(i,j) for i in range(1,5) for j in range(i+1,6)}
for group in [g for _,g in b.groupby(['model','seed'])]+[p]:
    assert len(group)==410 and set(group.scenario_id)==testids
    assert np.array_equal(group.horizon.to_numpy(),(group.j-group.i).to_numpy())
    for sid,g in group.groupby('scenario_id'):assert set(zip(g.i,g.j))==expected_pairs
    assert group.iou.between(0,1).all()
b['lead_years']=5*b.horizon
p=p.copy();p['model']='persistence';p['lead_years']=5*p.horizon
h=b.groupby(['model','seed','lead_years']).agg(n_pairs=('iou','size'),mean_iou=('iou','mean')).reset_index()
hp=p.groupby(['model','seed','lead_years']).agg(n_pairs=('iou','size'),mean_iou=('iou','mean')).reset_index()
h=pd.concat([h,hp],ignore_index=True).sort_values(['model','seed','lead_years']).reset_index(drop=True)
assert len(h)==76
assert h.apply(lambda r:r.n_pairs==COUNTS[r.lead_years],axis=1).all()
h.to_csv(O/'261117_corrected_horizon_metrics.csv',index=False)
summary=[];reconciliation=[];monotonic={}
for method in METHODS:
    mh=h[h.model==method]
    for lead,g in mh.groupby('lead_years'):
        assert len(g)==(1 if method=='persistence' else 6)
        summary.append({'model':method,'lead_years':int(lead),'n_pairs_per_seed':COUNTS[lead],
            'mean_iou':float(g.mean_iou.mean()),'sample_sd':float(g.mean_iou.std(ddof=1)) if method!='persistence' else np.nan})
    direct=p if method=='persistence' else b[b.model==method]
    seed_means=[];weighted=[]
    for seed,g in mh.groupby('seed'):
        wm=float(np.average(g.mean_iou,weights=g.n_pairs));dm=float(direct[direct.seed==seed].iou.mean())
        assert abs(wm-dm)<1e-14
        if method!='persistence':
            tag=next(k for k,v in TAGS.items() if v==method)
            assert abs(dm-seed_results.set_index('seed').loc[seed,tag])<1e-14
        else:assert abs(dm-seed_results.B_persistence.iloc[0])<1e-14
        seed_means.append(dm);weighted.append(wm)
        reconciliation.append({'model':method,'seed':int(seed),'weighted_horizon_iou':wm,'direct_global_iou':dm,'difference':wm-dm})
    reconciliation.append({'model':method,'seed':'mean','weighted_horizon_iou':float(np.mean(weighted)),
        'direct_global_iou':float(np.mean(seed_means)),'difference':float(np.mean(weighted)-np.mean(seed_means))})
    monotonic[method]={'seed_monotone_decreasing':{str(int(seed)):bool(np.all(np.diff(g.sort_values('lead_years').mean_iou)<0)) for seed,g in mh.groupby('seed')}}
summary=pd.DataFrame(summary);reconciliation=pd.DataFrame(reconciliation)
summary.to_csv(O/'261117_corrected_horizon_summary.csv',index=False)
reconciliation.to_csv(O/'261117_corrected_horizon_global_reconciliation.csv',index=False)
for method in METHODS:
    g=summary[summary.model==method].sort_values('lead_years')
    monotonic[method]['six_seed_mean_monotone_decreasing']=bool(np.all(np.diff(g.mean_iou)<0))
means=summary.pivot(index='lead_years',columns='model',values='mean_iou')
gains=means.fine_tuning-means.scratch
learned_pivot=h[h.model!='persistence'].pivot(index=['seed','lead_years'],columns='model',values='mean_iou')
seed_gain=learned_pivot.fine_tuning-learned_pivot.scratch
diagnostics={'monotonicity':monotonic,'mean_finetune_minus_scratch_by_lead':{str(k):float(v) for k,v in gains.items()},
    'fine_tuning_mean_above_scratch_at_every_horizon':bool((gains>0).all()),
    'positive_finetuning_seeds_by_lead':{str(k):int((g>0).sum()) for k,g in seed_gain.groupby(level='lead_years')}}
(O/'HORIZON_DIAGNOSTICS.json').write_text(json.dumps(diagnostics,indent=2))
print('HORIZON SUMMARY\n',summary.to_string(index=False),flush=True)
print('GLOBAL RECONCILIATION\n',reconciliation[reconciliation.seed=='mean'].to_string(index=False),flush=True)
print('DIAGNOSTICS',json.dumps(diagnostics),flush=True)

SID='S089';SEED=42
plan={'scenario':SID,'seed':SEED,'target_index':5,'target_time_yr':25,
    'observation_indices':[1,2,3,4],'observation_times_yr':[5,10,15,20],
    'lead_years':[20,15,10,5],'selection':'user-requested fixed scenario/target; seed42 fixed before inference',
    'checkpoint_selection_or_training':False}
(O/'FIXED_TARGET_PLAN.json').write_text(json.dumps(plan,indent=2))
source=remember(P/'train_unet_monitoring_207.py')
assert sha(source)==protocol['source_functions_sha256']
tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['DoubleConv','SmallUNet']]
assert len(nodes)==2
ns={'torch':torch,'nn':nn};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),ns)
Model=ns['SmallUNet'];models={};ckpaths={}
frozen={c['path']:c['sha256'] for c in freeze['checkpoints']}
for method,file in [('zero_shot','source_best.pt'),('scratch','scratch_best.pt'),('fine_tuning','finetuned_best.pt')]:
    path=R/'primary/seed_42'/file
    assert str(path) in frozen and sha(path)==frozen[str(path)]
    remember(path);ckpaths[method]=str(path)
    model=Model(in_ch=11,base=16);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval();models[method]=model
npz=remember(R/f'dataset_v1_raw_sgas_waterpressure/spe11b/{SID}.npz')
pre=read('preprocessing_checks.csv')
assert sha(npz)==pre[(pre.case=='spe11b')&(pre.scenario_id==SID)].iloc[0].npz_sha256
with np.load(npz,allow_pickle=False) as z:
    a={k:z[k].copy() for k in ['X','delta_sgas_raw','delta_pressure_raw_Pa','plume_mask','valid_mask','times','x','z']}
assert np.array_equal(a['times'],[0,5,10,15,20,25])
assert a['X'].shape==(6,3,60,420) and np.all(np.diff(a['x'])>0) and np.all(np.diff(a['z'])>0)
assert np.array_equal(a['plume_mask'],(a['delta_sgas_raw']>.001)&a['valid_mask'][None].astype(bool))
assert np.array_equal(a['X'][:,0],np.nan_to_num(a['delta_pressure_raw_Pa'],nan=0).astype(np.float32))
dyn=np.load(remember(npz.with_name(SID+'_dynamic.npy')));masks=np.load(remember(npz.with_name(SID+'_mask.npy')))
rebuilt=a['X'][1:5].copy()
for t in range(4):
    for c in [0,2]:
        scale=np.nanmax(np.abs(rebuilt[t,c]))
        if scale>0:rebuilt[t,c]/=scale
assert np.array_equal(dyn,rebuilt) and np.array_equal(masks,a['plume_mask'])
row=index[(index.case=='spe11b')&(index.scenario_id==SID)].iloc[0];mr=matrix[matrix.scenario_id==SID].iloc[0]
evaluation_files={method:read(f'primary/seed_42/{tag}_test_metrics.csv') for tag,method in TAGS.items()}
images=[];pair_records=[]
for i in [1,2,3,4]:
    j=5;lead=5*(j-i);truth=masks[j].astype(bool);observed=a['delta_sgas_raw'][i]
    vals=[i/5.,j/5.,j/5.-i/5.,(row.q_mult-1.)/.35,math.log(float(row.kz_rel_mult),2.),
        (row.global_perm_mult-1.)/.25,(mr.well1_rate_mult-1.)/.7,(mr.well2_rate_mult-1.)/.7]
    const=np.stack([np.full((60,420),v,dtype=np.float32) for v in vals])
    X=np.nan_to_num(np.concatenate([dyn[i-1],const]),nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    assert X.shape==(11,60,420)
    predictions={'persistence':(observed>.001)&a['valid_mask'].astype(bool)};probabilities={}
    assert np.array_equal(predictions['persistence'],masks[i].astype(bool))
    record={'scenario':SID,'seed':SEED,'i':i,'j':j,'lead_years':lead,'observation_years':i*5,'target_years':25}
    for method,model in models.items():
        with torch.no_grad():
            probability=torch.sigmoid(model(torch.from_numpy(X[None])))
            pred=probability>.5;pf=pred.float();y=torch.from_numpy(truth[None,None].astype(np.float32))
            float_iou=float(((pf*y).sum()/(((pf+y)>0).float().sum()+1e-6)).item())
        predictions[method]=pred[0,0].numpy();probabilities[method+'_probability']=probability[0,0].numpy()
        ref=b[(b.model==method)&(b.seed==42)&(b.scenario_id==SID)&(b.i==i)&(b.j==5)]
        ev=evaluation_files[method];er=ev[(ev.scenario_id==SID)&(ev.i==i)&(ev.j==5)]
        assert len(ref)==len(er)==1 and abs(float_iou-ref.iloc[0].iou)<1e-7 and abs(float_iou-er.iloc[0].iou)<1e-7
        record[method+'_original_float32_iou']=float_iou
    for method,pred in predictions.items():
        score=float((pred&truth).sum()/((pred|truth).sum()+1e-6));record[method+'_iou']=score
        if method=='persistence':
            ref=p[(p.scenario_id==SID)&(p.i==i)&(p.j==j)];assert len(ref)==1 and abs(score-ref.iloc[0].iou)<1e-14
        else:assert abs(score-record[method+'_original_float32_iou'])<1e-7
    record['paired_gain']=record['fine_tuning_iou']-record['scratch_iou']
    images.append((observed,truth,predictions));pair_records.append(record)
    np.savez_compressed(O/f'S089_seed42_i{i}_j5_predictions.npz',input_11ch=X,observed_delta_sgas_raw=observed,
        true_mask=truth,**predictions,**probabilities)
assert all(np.array_equal(images[0][1],im[1]) for im in images)
pd.DataFrame(pair_records).to_csv(O/'261117_corrected_fig08_pair_metrics.csv',index=False)
print('FIXED TARGET PAIRS\n',pd.DataFrame(pair_records).to_string(index=False),flush=True)

fontdir=Path('/mnt/c/Windows/Fonts') if os.name!='nt' else Path('C:/Windows/Fonts')
for f in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:font_manager.fontManager.addfont(str(fontdir/f))
assert 'arial' in font_manager.findfont('Arial',fallback_to_default=False).lower()
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42})
colors=['#7f7f7f','#9467bd','#1f77b4','#ff7f0e']
labels=['Persistence','A→B zero-shot','B scratch','A pretrain + B fine-tune']
fig,ax=plt.subplots(figsize=(10,5.8))
for method,color,label in zip(METHODS,colors,labels):
    g=summary[summary.model==method].sort_values('lead_years');xx=g.lead_years.to_numpy();yy=g.mean_iou.to_numpy()
    ax.plot(xx,yy,color=color,marker='o',markersize=7,linewidth=2.5,label=label)
    if method!='persistence':
        sd=g.sample_sd.to_numpy();ax.fill_between(xx,yy-sd,yy+sd,color=color,alpha=.12)
ax.set_xlim(4.3,20.7);ax.set_ylim(0,1);ax.set_xticks([5,10,15,20]);ax.set_yticks(np.arange(0,1.01,.2))
ax.set_xlabel('Prediction lead time, Δt (yr)',fontsize=14)
ax.set_ylabel('Mean IoU on SPE11B final-test set',fontsize=14)
ax.tick_params(labelsize=12);ax.grid(alpha=.2);ax.legend(loc='lower left',frameon=False,fontsize=12)
fig.subplots_adjust(left=.105,right=.985,top=.965,bottom=.18)
fig.text(.55,.025,'Shading: ±1 sample SD across 6 seeds (ddof=1). Persistence is deterministic.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(O/(S7+'.'+ext),dpi=600)
fig.savefig(O/'qa/Fig07_preview.png',dpi=140);plt.close(fig)

cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#D9D9D9');binary=ListedColormap(['#440154','#FDE725'])
fig,axs=plt.subplots(4,6,figsize=(16.5,9.6))
titles=['Observed\nΔSg','True future\nplume','Persistence','A→B\nzero-shot','B scratch','A pretrain +\nB fine-tune']
for r,(rec,ims) in enumerate(zip(pair_records,images)):
    observed,truth,preds=ims
    fields=[observed,truth,*[preds[k] for k in METHODS]]
    for c,field in enumerate(fields):
        ax=axs[r,c];im=ax.imshow(field,origin='lower',extent=[0,8.4,0,1.2],aspect='auto',interpolation='nearest',
            cmap=cmap if c==0 else binary,vmin=0,vmax=.9 if c==0 else 1)
        assert np.array_equal(np.asarray(im.get_array()),field,equal_nan=True)
        ax.set_xticks([]);ax.set_yticks([])
        if r==0:ax.set_title(titles[c],pad=10,fontsize=13)
        if c>=2:ax.text(.025,.975,f'IoU={rec[METHODS[c-2]+"_iou"]:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=12,
            bbox=dict(facecolor='white',edgecolor='.5',alpha=.94,pad=2))
    ax=axs[r,0];ax.set_xticks([0,4.2,8.4]);ax.set_yticks([0,.6,1.2]);ax.tick_params(labelsize=10,length=2,pad=2)
    ax.set_xlabel('x (km)',fontsize=11,labelpad=2);ax.set_ylabel('z (km)',fontsize=11,labelpad=2)
fig.subplots_adjust(left=.18,right=.99,bottom=.15,top=.90,wspace=.075,hspace=.34)
for r,rec in enumerate(pair_records):
    pos=axs[r,0].get_position();center=(pos.y0+pos.y1)/2
    fig.text(.012,center+.033,'S089 | seed 42',fontsize=12.5,weight='bold',ha='left',va='bottom')
    text=f'i={rec["i"]} → j=5\n{rec["observation_years"]} → 25 yr\nLead time: {rec["lead_years"]} yr'
    fig.text(.012,center+.020,text,fontsize=11.5,ha='left',va='top',linespacing=1.18)
cax=fig.add_axes([.24,.065,.36,.018]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,.9),cmap=cmap),cax=cax,orientation='horizontal')
cb.ax.tick_params(labelsize=11);cb.set_label('Observed raw ΔSg',fontsize=12,labelpad=3)
fig.legend(handles=[Patch(facecolor='#440154',label='Background'),Patch(facecolor='#FDE725',label='Plume (raw ΔSg > 0.001)')],
    loc='lower right',bbox_to_anchor=(.985,.052),frameon=False,ncol=2,fontsize=12)
fig.text(.5,.009,'Fixed target: 25 yr. Frozen corrected N=33 checkpoints; equal display boxes; elevation increases upward.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(O/(S8+'.'+ext),dpi=600)
fig.savefig(O/'qa/Fig08_preview.png',dpi=130);plt.close(fig)
assert all(sha(path)==hsh for path,hsh in hashes.items())
remember(Path(__file__).resolve())
manifest={'source_sha256':hashes,'checkpoints':ckpaths,'corrected_dataset':str(npz),'fixed_target_plan':plan,
    'horizon_diagnostics':diagnostics,'new_training_steps':0,'CPU_forward_passes':12,'torch':torch.__version__,
    'output_sha256':{stem+'.'+e:sha(O/(stem+'.'+e)) for stem in [S7,S8] for e in ['png','pdf']}}
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
report=f'''# Corrected Figure 7 and Figure 8

Created 2026-11-17. Only these two figures were produced. No fitting, new checkpoint selection, hyperparameter change or OPM execution. No legacy horizon CSV/IoU/prediction was loaded. Legacy JPEGs/code were inspected only for design. Same gray/purple/blue/orange model colours and purple/yellow binary palette retained. Arial, 600 dpi PNG, PDF with vector text/linework and native reporting-grid raster panels.

## Figure 7 aggregation and reconciliation

Input `{R/'primary_metrics_all.csv'}`: SPE11B B_zero, B_scratch, B_finetuned only. Deterministic persistence: `{R/'persistence_metrics.csv'}`. Six seeds {SEEDS}; exactly 41 fixed final-test scenarios and all ten (i,j) pairs for each method/seed. Checked unique keys, exact pair membership and horizon=j−i. Lead time = 5×horizon years. Every horizon contains 164/123/82/41 pairs at 5/10/15/20 yr (4/3/2/1 per scenario).

First compute each seed's mean over the correct forecast pairs at a horizon; then equally average those six means. Shading is ±sample SD(ddof=1) of the six horizon means. Persistence is one deterministic series with seed=-1 in the export and no SD/shading. Pair counts in the horizon CSV are per seed, not six-fold pooled counts.

Global check: sum(n_h×mean_h)/410 for each seed matches its direct 410-pair IoU. The six weighted seed means then match the corrected global mean and seed_results.csv. Unweighted average over the four horizon means is NOT used. All reconciliation errors <1e−14.

```
{summary.to_string(index=False)}

{reconciliation[reconciliation.seed=='mean'].to_string(index=False)}
```

## Horizon conclusions at the tested points

```
{json.dumps(diagnostics,indent=2)}
```

Monotonicity refers to discrete 5/10/15/20-year points, not interpolation or extrapolation. Seed-specific checks are separately reported: an aggregate mean trend does not imply every seed or scenario obeys the same ordering. Horizon groups contain different observation/target-time combinations, so their ordering is descriptive and does not isolate lead time causally. No curve smoothing, values reordering, case replacement or model change was used to obtain a trend.

## Figure 8 fixed-target verification

Scenario S089 is in the corrected 41-scenario B final test. Seed 42 was fixed before inference. Target j=5=25 yr for every row; i=1,2,3,4 means 5,10,15,20 yr and lead 20,15,10,5 yr. True mask is bitwise identical in every row. Plan saved before inference.

Corrected dataset: `{npz}`. Cache inputs are checked bitwise against rebuilt original maxabs-normalized corrected NPZ dynamic fields. Pressure X[:,0] matches the raw unrounded water-pressure difference cast once to float32; SGAS is unclipped. Primary raw ΔSg>0.001 mask matches saved mask at all six snapshots. Added the same eight time/scenario constant channels, producing the original eleven channels.

Checkpoint paths:
{json.dumps(ckpaths,indent=2)}

Each checkpoint hash matches MODELS_FROZEN_BEFORE_TEST.json before inference and after plotting. Protocol hash, torch version {torch.__version__}, CPU and thread count match the frozen protocol. Only architecture class definitions extracted from the hashed original implementation; no training code imported/executed. Exactly twelve eval-mode, no_grad, batch-size-one float32 CPU forwards, without AMP.

Predictions use sigmoid probability >0.5. Persistence is the observed raw ΔSg>0.001 mask, checked against saved observed labels. Image IoUs are computed from exact pair predictions, agree with the existing corrected pair-level and original seed42 evaluation CSVs within 1e−7 (original float32 versus independent Boolean-count calculation); persistence agrees within 1e−14. All rows' input/probability/mask artifacts saved in NPZ. No scenario-level average is displayed as an image score.

```
{pd.DataFrame(pair_records).to_string(index=False)}
```

## Orientation design and limits

Four-by-six layout; same physical-coordinate convention as corrected Figs 2/3/6: x rightward, elevation z upward, origin=lower, no flipping/rotation/smoothing. All panels use x=[0,8.4] km/z=[0,1.2] km and equal rectangular display boxes, intentionally not physical aspect ratio. Large typography and upper-left IoUs match the requested revised Fig6 style. S089 retained regardless of whether fine-tuning wins each individual row. This fixed-target example is descriptive and does not replace the full-test Figure 7 aggregation.

Raw source, dataset/cache and checkpoint SHA-256 hashes unchanged after plotting. Full paths and hashes in JSON. Existing figures/scripts preserved. Re-run this script in the original CPU Python environment with --root and a fresh --out directory. It reproduces aggregation and twelve inference forwards, without fitting.

## Suggested captions

Figure 7. Forecast lead-time dependence on the fixed 41-scenario SPE11B final-test set. The 5-, 10-, 15- and 20-year groups contain 164, 123, 82 and 41 forecast pairs per model seed, respectively. Learned-model curves average six seed-specific horizon means, with shaded ±1 sample standard deviation. Persistence is deterministic. Horizon groups include different observation and target times.

Figure 8. Fixed-target spatial comparison for corrected SPE11B scenario S089, seed 42. Target time is 25 years in all rows; observation times are 5, 10, 15 and 20 years. Exact-pair IoUs are recomputed using the frozen corrected N=33 checkpoints. Targets use raw ΔSg>0.001 and binary predictions use sigmoid probability>0.5. Elevation increases upward and equal display boxes are used.
'''
(O/'FIGURE_PROVENANCE.md').write_text(report,encoding='utf-8')
print('COMPLETE',O,flush=True)
