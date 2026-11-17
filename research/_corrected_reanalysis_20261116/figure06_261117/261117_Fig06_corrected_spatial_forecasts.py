"""Corrected Fig 6: deterministic post-evaluation examples, 12 frozen CPU forwards.
No training code is imported/executed. Run with original opm-ai-env Python -B.
Use --out NEW_DIR to rerun; original outputs/checkpoints/data are never overwritten.
"""
from pathlib import Path
import os, sys, ast, json, hashlib, math, argparse
sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116'))
parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
args = parser.parse_args()
R,O = args.root.resolve(),args.out.resolve(); P=R.parent
STEM='261117_Fig06_corrected_spatial_forecasts'
O.mkdir(parents=True,exist_ok=True); (O/'qa').mkdir(exist_ok=True)
assert not (O/'SELECTION_FROZEN_BEFORE_INFERENCE.json').exists(), 'Use a fresh output directory'
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

hashes={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def remember(p):hashes[str(p)]=sha(p);return p
def read(name):return pd.read_csv(remember(R/name))
protocol=json.loads(remember(R/'PROTOCOL.json').read_text())
freeze=json.loads(remember(R/'MODELS_FROZEN_BEFORE_TEST.json').read_text())
assert sha(R/'PROTOCOL.json')==freeze['protocol_sha256'] and freeze['test_scores_seen'] is False
assert torch.__version__==protocol['torch'] and protocol['device']=='cpu'
torch.set_num_threads(protocol['threads'])
assert protocol['primary_threshold']==.001 and protocol['prediction_threshold']==.5
matrix=read('dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv')
index=read('dataset_v1_raw_sgas_waterpressure/dataset_index.csv')
testids=set(matrix.loc[matrix.target_B_role=='zero_shot_and_final_test','scenario_id'])
assert len(testids)==41 and protocol['counts']['spe11b_finetune_train']==33
pair=read('paired_forecast_gains.csv'); scenario=read('target_scenario_gains.csv')
metrics=read('primary_metrics_all.csv'); pers=read('persistence_metrics.csv')
keys=['seed','scenario_id','i','j','horizon']
assert not pair.duplicated(keys).any() and len(pair)==2460
assert set(pair.scenario_id)==testids
aggregate=pair.groupby(['seed','scenario_id']).iou_gain.mean().reset_index()
joined=aggregate.merge(scenario,on=['seed','scenario_id'],suffixes=('_pair','_scenario'),validate='one_to_one')
assert np.allclose(joined.iou_gain_pair,joined.iou_gain_scenario,rtol=0,atol=1e-14)
seed42=pair[pair.seed==42].copy()
zero=metrics[(metrics.seed==42)&(metrics.eval_tag=='B_zero')][keys+['iou']].rename(columns={'iou':'zero_iou'})
seed42=seed42.merge(zero,on=keys,validate='one_to_one').merge(
    pers[['scenario_id','i','j','horizon','iou']].rename(columns={'iou':'persistence_iou'}),
    on=['scenario_id','i','j','horizon'],validate='one_to_one')
seed42['recovery']=seed42[['iou_scratch','iou_finetuned']].min(axis=1)-seed42.zero_iou
def pick(table,column,ascending=False):
    assert len(table)>0
    return table.sort_values([column,'scenario_id','i','j'],ascending=[ascending,True,True,True]).iloc[0]
r1=pick(seed42[(seed42.zero_iou<seed42.persistence_iou)&(seed42.iou_scratch>seed42.zero_iou)&(seed42.iou_finetuned>seed42.zero_iou)],'recovery')
positive=set(scenario[(scenario.seed==42)&(scenario.iou_gain>0)].scenario_id)
r2=pick(seed42[(seed42.scenario_id!=r1.scenario_id)&seed42.scenario_id.isin(positive)&(seed42.iou_gain>0)],'iou_gain')
r3=pick(seed42[~seed42.scenario_id.isin([r1.scenario_id,r2.scenario_id])],'persistence_iou')
r4=pick(pair[(pair.seed==44)&(pair.scenario_id=='S099')&(pair.iou_gain<0)],'iou_gain',True)
selected=[r1,r2,r3,r4]
reasons=['Largest target-learning recovery','Largest positive fine-tuning gain','High persistence','Negative transfer (S099)']
selection=[]
for k,r in enumerate(selected):
    selection.append(dict(row=k+1,reason=reasons[k],scenario_id=r.scenario_id,seed=int(r.seed),i=int(r.i),j=int(r.j),horizon=int(r.horizon)))
plan={'selection':selection,'posthoc_explanatory_only':True,
 'rules':['seed42: maximum min(scratch,finetune)−zero, with zero<persistence and both target models>zero',
 'seed42: maximum positive pair gain, scenario-level average gain>0, excluding row1 scenario',
 'seed42: maximum persistence, excluding row1/row2 scenarios',
 'seed44/S099: minimum negative pair gain across its 10 pairs'],
 'ties':'scenario_id then i then j ascending','no_checkpoint_or_hyperparameter_selection':True}
(O/'SELECTION_FROZEN_BEFORE_INFERENCE.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print('SELECTION FIXED',json.dumps(selection),flush=True)

# Extract only the architecture definitions from the original frozen protocol's
# implementation. Do not import a training/legacy plotting module.
source=remember(P/'train_unet_monitoring_207.py')
assert sha(source)==protocol['source_functions_sha256']
tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ['DoubleConv','SmallUNet']]
assert len(nodes)==2
ns={'torch':torch,'nn':nn}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),ns)
Model=ns['SmallUNet']
frozen={rec['path']:rec['sha256'] for rec in freeze['checkpoints']}
model_cache={}; checkpoint_paths={}; records=[]; images=[]
precheck=read('preprocessing_checks.csv')
for sel in selection:
    sid,seed,i,j=sel['scenario_id'],sel['seed'],sel['i'],sel['j']
    assert sid in testids
    npzpath=R/'dataset_v1_raw_sgas_waterpressure/spe11b'/f'{sid}.npz'
    remember(npzpath)
    assert hashes[str(npzpath)]==precheck[(precheck.case=='spe11b')&(precheck.scenario_id==sid)].iloc[0].npz_sha256
    with np.load(npzpath,allow_pickle=False) as z:
        a={k:z[k].copy() for k in ['X','delta_sgas_raw','delta_pressure_raw_Pa','plume_mask','valid_mask','times','x','z']}
    assert a['X'].shape==(6,3,60,420)
    assert np.all(np.diff(a['x'])>0) and np.all(np.diff(a['z'])>0)
    assert np.array_equal(a['plume_mask'],(a['delta_sgas_raw']>.001)&a['valid_mask'][None].astype(bool))
    assert np.array_equal(a['X'][:,0],np.nan_to_num(a['delta_pressure_raw_Pa'],nan=0).astype(np.float32))
    dynpath=remember(npzpath.with_name(f'{sid}_dynamic.npy'))
    maskpath=remember(npzpath.with_name(f'{sid}_mask.npy'))
    dyn=np.load(dynpath); savedm=np.load(maskpath)
    rebuilt=a['X'][1:5].copy()
    for t in range(4):
        for c in [0,2]:
            vmax=np.nanmax(np.abs(rebuilt[t,c]))
            if vmax>0:rebuilt[t,c]/=vmax
    assert np.array_equal(dyn,rebuilt) and np.array_equal(savedm,a['plume_mask'])
    row=index[(index.case=='spe11b')&(index.scenario_id==sid)].iloc[0]
    mr=matrix[matrix.scenario_id==sid].iloc[0]
    vals=[i/5.,j/5.,j/5.-i/5.,(row.q_mult-1.)/.35,math.log(float(row.kz_rel_mult),2.),
          (row.global_perm_mult-1.)/.25,(mr.well1_rate_mult-1.)/.7,(mr.well2_rate_mult-1.)/.7]
    const=np.stack([np.full((60,420),v,dtype=np.float32) for v in vals])
    X=np.nan_to_num(np.concatenate([dyn[i-1],const]),nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    assert X.shape==(11,60,420)
    observed=a['delta_sgas_raw'][i]; truth=savedm[j].astype(bool)
    persistence=(observed>.001)&a['valid_mask'].astype(bool)
    assert np.array_equal(persistence,savedm[i].astype(bool))
    preds={'persistence':persistence}; probs={}; paths={}
    for method,filename,tag in [('zero_shot','source_best.pt','B_zero'),('scratch','scratch_best.pt','B_scratch'),('fine_tuning','finetuned_best.pt','B_finetuned')]:
        ck=R/'primary'/f'seed_{seed}'/filename
        assert str(ck) in frozen and sha(ck)==frozen[str(ck)]
        remember(ck); paths[method]=str(ck)
        key=(seed,filename)
        if key not in model_cache:
            model=Model(in_ch=11,base=16)
            model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True));model.eval()
            model_cache[key]=model
        with torch.no_grad():
            logits=model_cache[key](torch.from_numpy(X[None]))
            probability=torch.sigmoid(logits)
            pred=probability>.5
            y=torch.from_numpy(truth[None,None].astype(np.float32))
            pred_f=pred.float()
            score=float(((pred_f*y).sum()/(((pred_f+y)>0).float().sum()+1e-6)).item())
        probs[method]=probability[0,0].numpy(); preds[method]=pred[0,0].numpy()
        match=metrics[(metrics.seed==seed)&(metrics.scenario_id==sid)&(metrics.i==i)&(metrics.j==j)&(metrics.eval_tag==tag)]
        assert len(match)==1 and abs(score-float(match.iloc[0].iou))<1e-7,(sel,method,score,match.iou.tolist())
        direct=read(f'primary/seed_{seed}/{tag}_test_metrics.csv')
        dm=direct[(direct.scenario_id==sid)&(direct.i==i)&(direct.j==j)]
        assert len(dm)==1 and abs(score-dm.iloc[0].iou)<1e-7
        sel[method+'_original_float32_iou']=score
    def iou(mask):return float((mask&truth).sum()/((mask|truth).sum()+1e-6))
    scores={name:iou(pred) for name,pred in preds.items()}
    pm=pers[(pers.scenario_id==sid)&(pers.i==i)&(pers.j==j)]
    assert len(pm)==1 and abs(scores['persistence']-pm.iloc[0].iou)<1e-14
    oldpair=pair[(pair.seed==seed)&(pair.scenario_id==sid)&(pair.i==i)&(pair.j==j)].iloc[0]
    assert abs(scores['fine_tuning']-oldpair.iou_finetuned)<1e-7
    assert abs(scores['scratch']-oldpair.iou_scratch)<1e-7
    assert sel['row']!=4 or scores['fine_tuning']<scores['scratch']
    sc=scenario[(scenario.seed==seed)&(scenario.scenario_id==sid)].iloc[0]
    record={**sel,'observation_time_yr':float(a['times'][i]),'target_time_yr':float(a['times'][j]),
        'lead_time_yr':float(a['times'][j]-a['times'][i]),
        **{k+'_iou':v for k,v in scores.items()},'paired_gain':scores['fine_tuning']-scores['scratch'],
        'scenario_10_pair_mean_gain_reference_only':float(sc.iou_gain),'corrected_npz':str(npzpath),
        **{k+'_checkpoint':v for k,v in paths.items()}}
    records.append(record); images.append((observed,truth,preds))
    checkpoint_paths[str(seed)]=paths
    np.savez_compressed(O/f'row{sel["row"]}_{sid}_seed{seed}_i{i}_j{j}_predictions.npz',
        input_11ch=X,observed_delta_sgas_raw=observed,true_mask=truth,**preds,
        **{k+'_probability':v for k,v in probs.items()})
    print('VERIFIED',record,flush=True)

pd.DataFrame(records).to_csv(O/(STEM+'_row_verification.csv'),index=False)
fontdir=Path('/mnt/c/Windows/Fonts')
for f in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:font_manager.fontManager.addfont(str(fontdir/f))
assert 'arial' in font_manager.findfont('Arial',fallback_to_default=False).lower()
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':10,'axes.linewidth':.7,'pdf.fonttype':42})
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#D9D9D9')
binary=ListedColormap(['#440154','#FDE725'])
fig,axs=plt.subplots(4,6,figsize=(16.5,8.9))
titles=['Observed ΔSg','True future plume','Persistence','A→B zero-shot','B scratch','A pretrain + B fine-tune']
for r,(rec,ims) in enumerate(zip(records,images)):
    observed,truth,preds=ims
    for c,field in enumerate([observed,truth,preds['persistence'],preds['zero_shot'],preds['scratch'],preds['fine_tuning']]):
        ax=axs[r,c]
        im=ax.imshow(field,origin='lower',extent=[0,8.4,0,1.2],aspect='auto',interpolation='nearest',
            cmap=cmap if c==0 else binary,vmin=0,vmax=.9 if c==0 else 1)
        assert np.array_equal(np.asarray(im.get_array()),field,equal_nan=True)
        ax.set_xticks([]);ax.set_yticks([])
        if r==0:ax.set_title(titles[c],pad=8)
        if c>=2:
            method=['persistence','zero_shot','scratch','fine_tuning'][c-2]
            ax.text(.97,.97,f'IoU={rec[method+"_iou"]:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=8,
                bbox=dict(facecolor='white',edgecolor='.5',alpha=.9,pad=1.5))
    ax=axs[r,0]
    ax.set_xticks([0,4.2,8.4]);ax.set_yticks([0,.6,1.2]);ax.tick_params(labelsize=7,length=2,pad=1)
    ax.set_xlabel('x (km)',fontsize=8,labelpad=1);ax.set_ylabel('z (km)',fontsize=8,labelpad=1)
    label=f'{rec["scenario_id"]} | seed {rec["seed"]}\ni={rec["i"]} → j={rec["j"]}\n{rec["observation_time_yr"]:g} → {rec["target_time_yr"]:g} yr; lead {rec["lead_time_yr"]:g} yr\n{rec["reason"]}'
    fig.text(.011,.848-r*.207,label,fontsize=8.5,ha='left',va='center')
fig.subplots_adjust(left=.18,right=.99,bottom=.14,top=.93,wspace=.075,hspace=.30)
cax=fig.add_axes([.24,.066,.36,.013])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,.9),cmap=cmap),cax=cax,orientation='horizontal')
cb.ax.tick_params(labelsize=8);cb.set_label('Observed raw ΔSg',fontsize=9,labelpad=2)
fig.legend(handles=[Patch(facecolor='#440154',label='Background'),Patch(facecolor='#FDE725',label='Plume (raw ΔSg > 0.001)')],
    loc='lower right',bbox_to_anchor=(.985,.052),frameon=False,ncol=2,fontsize=9)
fig.text(.5,.012,'Post-evaluation explanatory examples; frozen N=33 checkpoints. Equal display boxes; elevation increases upward.',ha='center',fontsize=8)
for ext in ['png','pdf']:
    assert not (O/(STEM+'.'+ext)).exists()
    fig.savefig(O/(STEM+'.'+ext),dpi=600)
fig.savefig(O/'qa/preview.png',dpi=140);plt.close(fig)
assert all(sha(p)==h for p,h in hashes.items())
manifest={'selection':plan,'records':records,'source_sha256':hashes,'checkpoint_paths':checkpoint_paths,
    'torch':torch.__version__,'threads':torch.get_num_threads(),'device':'cpu','forward_passes':12,
    'training_steps':0,'prediction_threshold':'> 0.5 sigmoid','label_threshold':'> 0.001 raw delta SGAS',
    'orientation':'ascending x/right, z elevation/up, origin lower; no flips, rotations or resampling; equal display boxes',
    'output_sha256':{STEM+'.'+e:sha(O/(STEM+'.'+e)) for e in ['png','pdf']}}
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
report=f'''# Corrected Figure 6

Created 2026-11-17. Post-evaluation explanatory selection only: no training, hyperparameter/checkpoint selection or OPM execution. Exactly 12 batch-size-one CPU inference passes on 4 selected pairs, using 6 unique frozen checkpoints.

## Selection

{json.dumps(plan,indent=2)}

The selection plan was written before inference. These are deliberately selected extreme examples, not a representative estimate or independent test evidence. Rows 1–3 use seed 42; row 4 explicitly uses seed 44/S099. Scenario-level means are used only to check row eligibility and are NOT image labels. Every displayed score is recomputed for the exact row pair.

## Inputs and verification

- Corrected root: `{R}`
- Input CSVs: paired_forecast_gains.csv, target_scenario_gains.csv, primary_metrics_all.csv, persistence_metrics.csv, corrected dataset index/matrix, and exact per-seed evaluation CSVs. Full paths and hashes in JSON.
- Corrected dataset: `{R/'dataset_v1_raw_sgas_waterpressure/spe11b'}`. Only selected scenario NPZ and corresponding dynamic/mask caches read. Dataset hashes match preprocessing_checks.csv.
- Verified raw pressure float32 cast matches X[:,0]; rebuilt maxabs pressure/XCO2 normalization matches cached dynamic inputs bitwise. Exactly the original 11-channel temporal/scenario conditioning, raw SGAS channel unchanged. Raw >0.001 labels match saved masks at all six times.
- Frozen checkpoint record: `{R/'MODELS_FROZEN_BEFORE_TEST.json'}`. All six checkpoint hashes match before inference and again after figure creation. Protocol hash matches freeze record. CPU torch version and 12-thread configuration match the original protocol.
- Architecture class definitions alone extracted from the protocol's hashed source; no training module import or train function executed. Batch size one, eval mode, no_grad, float32, no AMP.
- Learned scores recomputed with original float32 IoU expression and independently with Boolean pixel counts. Agreement with corrected exact pair-level and original per-seed CSVs within 1e−7. Boolean-count IoU is used on figures; original float32 IoU is also stored in verification CSV. Persistence matches its saved CSV within 1e−14.
- Inference probabilities and binary masks, exact inputs and truth saved per row in NPZ. No masks manually modified. Source and dataset hashes rechecked unchanged.
- All 4 selected scenarios belong to the fixed 41-scenario B final test. Only corrected source_best/scratch_best/finetuned_best under primary/seed_42 and primary/seed_44 used. No legacy image, metric, checkpoint or case selection reused. Legacy fig6.jpg was inspected only for the six-column layout, typography and purple/yellow colours.

## Coordinates and layout

Same convention as corrected Figures 2–3: array row zero at the bottom (origin=lower), ascending physical x and elevation z, x∈[0,8.4] km and z∈[0,1.2] km. First column shows these axes for every row; all columns use the same extent. Equal rectangular display boxes intentionally do not preserve physical aspect ratio or a common cross-domain scale. No flip, rotation, smoothing or image resampling applied to model input/target arrays. No injection marker is added, keeping the original forecast layout uncluttered.

Arial, PNG 600 dpi, PDF, four rows and six columns. Each row labels scenario, seed, i, j, observation/target times and lead time. White boxes show exact-pair IoU rounded to three decimals. Full precision values and paths are in `{STEM}_row_verification.csv`.

## Suggested caption

Spatial forecasts for four deliberately selected corrected SPE11B final-test examples: large recovery from zero-shot failure, positive fine-tuning gain, high persistence, and negative transfer. Rows 1–3 use seed 42; row 4 uses seed 44. Models use the frozen N=33 target-training checkpoints. Binary predictions use sigmoid probability >0.5, and targets use raw ΔSg >0.001. Each IoU is recomputed for the displayed forecast pair. These cases were selected after final evaluation for explanation and did not affect model or checkpoint selection. Elevation increases upward; equal display boxes are used for visual comparison.
'''
(O/'FIGURE_PROVENANCE.md').write_text(report,encoding='utf-8')
print('FIGURE COMPLETE',O,flush=True)
