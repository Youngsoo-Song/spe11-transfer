"""Vector TP/FP/FN decomposition of saved corrected Fig6 predictions only.
No torch/model inference or fitting. Existing results never overwritten.
Run existing Python -B; --out must be a fresh directory.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,default=Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116'))
ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args();R=args.root.resolve();O=args.out.resolve();S=R/'figure06_261117'
O.mkdir(parents=True,exist_ok=True);(O/'qa').mkdir(exist_ok=True)
STEM='261117_Fig10_corrected_error_decomposition'
assert not (O/(STEM+'.png')).exists() and not (O/(STEM+'.pdf')).exists()
os.environ['MPLCONFIGDIR']=str(O/'qa/mplconfig')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch,Rectangle
from matplotlib.collections import PatchCollection
hashes={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def remember(p):hashes[str(p)]=sha(p);return p
def read(p):return pd.read_csv(remember(p))
fig6=read(S/'261117_Fig06_corrected_spatial_forecasts_row_verification.csv')
prior=json.loads(remember(S/'FIGURE_PROVENANCE.json').read_text())
freeze=json.loads(remember(R/'MODELS_FROZEN_BEFORE_TEST.json').read_text())
protocol=json.loads(remember(R/'PROTOCOL.json').read_text())
assert sha(R/'PROTOCOL.json')==freeze['protocol_sha256'] and freeze['test_scores_seen'] is False
assert protocol['counts']['spe11b_finetune_train']==33 and protocol['primary_threshold']==.001
frozen={c['path']:c['sha256'] for c in freeze['checkpoints']}
evaluation=read(R/'primary_metrics_all.csv')
cases=[('S058',42,1,5),('S004',42,1,2),('S099',44,1,3)]
methods=['zero_shot','scratch','fine_tuning']
tags=dict(zip(methods,['B_zero','B_scratch','B_finetuned']))
TP_COLOR='#32CD32';FP_COLOR='#d62728';FN_COLOR='#1f77b4'
ERROR_COLORS={0:'black',1:TP_COLOR,2:FP_COLOR,3:FN_COLOR}
records=[];fields=[];rowmeta=[];checkpoint_paths={}
for row_index,(sid,seed,i,j) in enumerate(cases,1):
    ref=fig6[(fig6.scenario_id==sid)&(fig6.seed==seed)&(fig6.i==i)&(fig6.j==j)]
    assert len(ref)==1
    ref=ref.iloc[0]
    path=remember(S/f'row{int(ref.row)}_{sid}_seed{seed}_i{i}_j{j}_predictions.npz')
    with np.load(path,allow_pickle=False) as z:
        saved={k:z[k].copy() for k in ['input_11ch','observed_delta_sgas_raw','true_mask',*methods,*[m+'_probability' for m in methods]]}
    rawpath=remember(Path(ref.corrected_npz))
    assert 'dataset_v1_raw_sgas_waterpressure' in str(rawpath)
    assert sha(rawpath)==prior['source_sha256'][str(rawpath)]
    with np.load(rawpath,allow_pickle=False) as z:
        valid=z['valid_mask'].astype(bool);truth=(z['delta_sgas_raw'][j]>.001)&valid
        assert np.array_equal(truth,z['plume_mask'][j].astype(bool))
        assert np.array_equal(saved['true_mask'].astype(bool),truth)
        assert np.array_equal(saved['observed_delta_sgas_raw'],z['delta_sgas_raw'][i],equal_nan=True)
        assert np.all(np.diff(z['x'])>0) and np.all(np.diff(z['z'])>0)
        assert np.array_equal(z['X'][:,0],np.nan_to_num(z['delta_pressure_raw_Pa'],nan=0).astype(np.float32))
        dyn=z['X'][i].copy()
        for c in [0,2]:
            scale=np.nanmax(np.abs(dyn[c]))
            if scale>0:dyn[c]/=scale
        assert np.array_equal(dyn,saved['input_11ch'][:3])
    assert truth.shape==(60,420) and truth.any()
    maps=[]
    for method in ['true_reference',*methods]:
        pred=truth if method=='true_reference' else saved[method].astype(bool)
        ck=''
        if method!='true_reference':
            assert np.array_equal(pred,saved[method+'_probability']>.5)
            ck=str(ref[method+'_checkpoint'])
            assert ck in frozen and sha(ck)==frozen[ck]==prior['source_sha256'][ck]
            remember(Path(ck));checkpoint_paths[f'{seed}_{method}']=ck
        tp=pred&truth;fp=pred&~truth;fn=~pred&truth;tn=~pred&~truth
        counts={k:int(v.sum()) for k,v in [('TP',tp),('FP',fp),('FN',fn),('TN',tn)]}
        U=counts['TP']+counts['FP']+counts['FN']
        assert U==int((pred|truth).sum()) and sum(counts.values())==truth.size
        assert not (tp&fp).any() and not (tp&fn).any() and not (fp&fn).any()
        exact=counts['TP']/U;section27=counts['TP']/(U+1e-6)
        fp_u=counts['FP']/U;fn_u=counts['FN']/U
        assert abs(exact+fp_u+fn_u-1)<1e-14
        ref_iou=np.nan if method=='true_reference' else float(ref[method+'_iou'])
        if method!='true_reference':
            assert abs(section27-ref_iou)<1e-14
            assert f'{exact:.3f}'==f'{ref_iou:.3f}'
            ev=evaluation[(evaluation.seed==seed)&(evaluation.scenario_id==sid)&(evaluation.i==i)&(evaluation.j==j)&(evaluation.eval_tag==tags[method])]
            assert len(ev)==1 and abs(section27-ev.iloc[0].iou)<1e-7
        # 0 TN, 1 TP, 2 FP, 3 FN; these exact masks are the plotted content.
        error=np.zeros(truth.shape,dtype=np.uint8);error[tp]=1;error[fp]=2;error[fn]=3
        assert all(int((error==code).sum())==counts[label] for code,label in [(0,'TN'),(1,'TP'),(2,'FP'),(3,'FN')])
        record=dict(row=row_index,scenario=sid,seed=seed,i=i,j=j,
            observation_years=float(ref.observation_time_yr),target_years=float(ref.target_time_yr),lead_years=float(ref.lead_time_yr),
            method=method,panel_role='target self-reference' if method=='true_reference' else 'model prediction',
            **counts,U=U,FP_over_U=fp_u,FN_over_U=fn_u,IoU_exact_TP_over_U=exact,
            IoU_section27_TP_over_U_plus_epsilon=section27,IoU_Figure6=ref_iou,
            difference_from_Figure6=np.nan if method=='true_reference' else section27-ref_iou,
            epsilon_effect=exact-section27,corrected_npz=str(rawpath),saved_Figure6_prediction=str(path),checkpoint=ck)
        records.append(record)
        if method!='true_reference':maps.append(error)
    rowmeta.append(ref);fields.append((truth,maps))
    np.savez_compressed(O/f'{sid}_seed{seed}_i{i}_j{j}_error_maps.npz',true_mask=truth,
        **{m:em for m,em in zip(methods,maps)})
df=pd.DataFrame(records)
df.to_csv(O/(STEM+'_verification.csv'),index=False)
changes=[]
for row,g in df[df.method!='true_reference'].groupby('row'):
    d=g.set_index('method')
    for before,after in [('zero_shot','scratch'),('zero_shot','fine_tuning'),('scratch','fine_tuning')]:
        a,b=d.loc[before],d.loc[after]
        rec=dict(row=int(row),scenario=a.scenario,seed=int(a.seed),i=int(a.i),j=int(a.j),comparison=f'{after} minus {before}')
        for k in ['TP','FP','FN','U','FP_over_U','FN_over_U','IoU_section27_TP_over_U_plus_epsilon']:
            rec['delta_'+k]=float(b[k]-a[k])
        changes.append(rec)
changes=pd.DataFrame(changes);changes.to_csv(O/(STEM+'_model_differences.csv'),index=False)

def vector_cells(ax,array,palette):
    """Exact grid-pixel rectangles, merged only when adjacent runs are identical.
    This preserves class pixel areas and produces a genuinely vector PDF.
    """
    H,W=array.shape;active={};finished=[]
    for j in range(H):
        a=array[j];bounds=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,W]
        runs={(int(start),int(end),int(a[start])) for start,end in zip(bounds[:-1],bounds[1:]) if a[start]!=0}
        for key in list(active):
            if key not in runs:
                startrow=active.pop(key);finished.append((*key,startrow,j))
        for key in runs:
            if key not in active:active[key]=j
    for key,startrow in active.items():finished.append((*key,startrow,H))
    patches=[];colors=[];areas={k:0 for k in palette}
    for start,end,code,bottom,top in finished:
        patches.append(Rectangle((start*8.4/W,bottom*1.2/H),(end-start)*8.4/W,(top-bottom)*1.2/H))
        colors.append(palette[code]);areas[code]+=(end-start)*(top-bottom)
    for code in palette:
        if code!=0:assert areas[code]==int((array==code).sum())
    ax.set_facecolor(palette[0])
    collection=PatchCollection(patches,facecolors=colors,edgecolors='none',antialiaseds=False,rasterized=False)
    ax.add_collection(collection);ax.set_xlim(0,8.4);ax.set_ylim(0,1.2);ax.set_aspect('auto')
    ax.set_xticks([]);ax.set_yticks([])
    return len(patches)

fontdir=Path('/mnt/c/Windows/Fonts') if os.name!='nt' else Path('C:/Windows/Fonts')
for f in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:font_manager.fontManager.addfont(str(fontdir/f))
assert 'arial' in font_manager.findfont('Arial',fallback_to_default=False).lower()
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(13.9,8.0));titles=['True future plume','A→B zero-shot','B scratch','A pretrain +\nB fine-tune']
panel_w=2.58
# Match the display aspect of the requested enlarged corrected Figure 6.
fig6_w=(16.5*(.99-.18))/(6+5*.075);fig6_h=(9.6*(.90-.15))/(4+3*.34)
panel_h=panel_w/(fig6_w/fig6_h);vector_rectangle_count=0
for row,(ref,ims,bottom) in enumerate(zip(rowmeta,fields,[5.45,3.18,.91]),1):
    truth,maps=ims
    for col,array in enumerate([truth.astype(np.uint8),*maps]):
        ax=fig.add_axes([(2.77+col*(panel_w+.22))/13.9,bottom/8.0,panel_w/13.9,panel_h/8.0])
        vector_rectangle_count+=vector_cells(ax,array,{0:'#440154',1:'#FDE725'} if col==0 else ERROR_COLORS)
        if row==1:ax.set_title(titles[col],fontsize=13,pad=10)
        if col==0:
            ax.set_xticks([0,4.2,8.4]);ax.set_yticks([0,.6,1.2]);ax.tick_params(labelsize=10,length=2,pad=2)
            ax.set_xlabel('x (km)',fontsize=11,labelpad=2);ax.set_ylabel('z (km)',fontsize=11,labelpad=2)
        else:
            metric=df[(df.row==row)&(df.method==methods[col-1])].iloc[0]
            ax.text(.025,.975,f'IoU={metric.IoU_section27_TP_over_U_plus_epsilon:.3f}',transform=ax.transAxes,
                ha='left',va='top',fontsize=12,bbox=dict(facecolor='white',edgecolor='.5',alpha=.94,pad=2))
    center=(bottom+panel_h/2)/8.0
    fig.text(.012,center+.035,f'{ref.scenario_id} | seed {int(ref.seed)}',fontsize=12.5,weight='bold',ha='left',va='bottom')
    label=f'i={int(ref.i)} → j={int(ref.j)}\n{ref.observation_time_yr:g} → {ref.target_time_yr:g} yr\nLead time: {ref.lead_time_yr:g} yr'
    fig.text(.012,center+.019,label,fontsize=11.5,ha='left',va='top',linespacing=1.18)
fig.legend(handles=[Patch(facecolor=TP_COLOR,edgecolor='black',label='True positive'),
    Patch(facecolor=FP_COLOR,edgecolor='black',label='False positive'),Patch(facecolor=FN_COLOR,edgecolor='black',label='False negative')],
    loc='lower center',bbox_to_anchor=(.61,.027),frameon=False,ncol=3,fontsize=12,columnspacing=2)
fig.text(.61,.012,'Exact Figure 6 cases and predictions; black = true negative; elevation increases upward.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(O/(STEM+'.'+ext),dpi=600)
fig.savefig(O/'qa/preview.png',dpi=140);plt.close(fig)
assert all(sha(p)==h for p,h in hashes.items())
remember(Path(__file__).resolve())
manifest={'source_sha256':hashes,'cases':cases,'checkpoint_paths':checkpoint_paths,'new_inference_passes':0,'training_steps':0,
    'plotted_error_classes':ERROR_COLORS,'background':'TN black; target reference purple/yellow',
    'vector_rectangle_count':vector_rectangle_count,'orientation':'physical x right/elevation z up; no flips; exact extent [0,8.4]×[0,1.2] km',
    'same_display_aspect_as_Fig6_large_type':True,'model_panel_count':9,'reference_panel_count':3,
    'output_sha256':{STEM+'.'+e:sha(O/(STEM+'.'+e)) for e in ['png','pdf']}}
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
interpretations=[]
for row,g in changes.groupby('row'):
    c=g[g.comparison=='fine_tuning minus scratch'].iloc[0]
    interpretations.append(f'- {c.scenario}, seed {int(c.seed)}, i={int(c.i)}→j={int(c.j)}: fine-tuning minus scratch: ΔTP={c.delta_TP:+g}, ΔFP={c.delta_FP:+g}, ΔFN={c.delta_FN:+g}; ΔIoU={c.delta_IoU_section27_TP_over_U_plus_epsilon:+.9f}. Red/blue pixel areas change by these exact FP/FN counts, respectively.')
report=f'''# Corrected Figure 10 spatial error decomposition

Created 2026-11-17. This file is separate from the previously produced Figure 10 consistency file. No previous file was overwritten. This request uses Figure 10 for the error decomposition; manuscript numbering itself was not edited.

## Sources and fixed cases

Figure 6 source directory: `{S}`. Exactly the requested cases were reused: S058/seed42/i1/j5, S004/seed42/i1/j2, S099/seed44/i1/j3. Saved sigmoid probabilities and prediction masks are the corrected Figure 6 artifacts, not legacy predictions. No model forward pass, fitting or new case/checkpoint selection was performed.

Full source paths and SHA-256 are recorded in JSON. Every saved target matches the corrected NPZ raw ΔSg>0.001 target pixel-for-pixel. Input pressure is verified as corrected raw pressure delta cast to float32; rebuilt image-wise normalized dynamic inputs match the saved Figure 6 inputs. Every binary model mask equals its saved sigmoid probability>0.5. Each checkpoint path/hash matches both Figure 6 provenance and MODELS_FROZEN_BEFORE_TEST.json, for the corrected N=33 protocol. Checkpoint bytes are only hashed, not loaded as models. All source hashes unchanged after plotting.

## Definitions and Section 2.7

TP=prediction∩truth, FP=prediction∩not-truth, FN=not-prediction∩truth, TN=not-prediction∩not-truth. U=TP+FP+FN is checked equal to the binary union. FP/U and FN/U are exact union-normalized counts; exact TP/U + FP/U + FN/U = 1. Each plotted pixel is classified once, and all counts sum to the reporting-grid size.

Section 2.7's existing numerical stabilization ε=10⁻⁶ is NOT changed: its IoU=TP/(U+ε) is retained for figure labels and exact Figure 6 reconciliation. The CSV separately provides exact TP/U, regularized Section-2.7 IoU, their epsilon difference, Figure 6 IoU and their difference. All nine model scores match Figure 6 to <1e−14 and the existing corrected exact-pair evaluation CSV to <1e−7 (historical float32 metric arithmetic). Exact and regularized IoUs round to identical three-decimal labels. No epsilon is added to FP/U or FN/U, as requested.

The CSV contains 12 panel records. The three true-mask panels are marked 'target self-reference': TP=true plume pixel count, FP=FN=0. They are not extra model predictions and have no Figure 6 IoU reference or displayed IoU label. The nine model panels are the scientific error decompositions.

## Pixel counts and normalized errors

```
{df[['scenario','seed','i','j','method','TP','FP','FN','TN','U','FP_over_U','FN_over_U','IoU_exact_TP_over_U','IoU_section27_TP_over_U_plus_epsilon','difference_from_Figure6']].to_string(index=False)}
```

## Model differences and visual interpretation

Positive ΔFP means more red pixels; negative means fewer. Positive ΔFN means more blue pixels; negative means fewer. Both TP/FP/FN field counts and the total grid-pixel area represented by plotted vector rectangles are checked exactly. Differences among model masks thus agree with their colour-pixel counts, without manual interpretation-based alterations. Ratio denominators U vary between models, so changes in FP/U or FN/U are separately tabulated and not inferred solely from absolute counts. The additional difference CSV includes scratch−zero-shot, fine-tuning−zero-shot and fine-tuning−scratch for every row.

{chr(10).join(interpretations)}

```
{changes.to_string(index=False)}
```

These are descriptive pair-level error signatures, not a newly established physical mechanism or universal transfer effect. Section 2.7, model protocol and cases remain unchanged.

## Geometry design and outputs

Three rows/four columns: true future plume, zero-shot, scratch, fine-tuning. Existing legacy Fig11 was inspected only for palette/design: TP=#32CD32, FP=#d62728, FN=#1f77b4, TN=black, true-mask background=#440154 and plume=#FDE725. Arial and larger typography/upper-left IoUs follow the revised corrected Fig6.

Same x=[0,8.4] km, elevation z=[0,1.2] km as corrected Figure 6, 2 and 3. The plotting maps increasing grid row to increasing physical elevation, with no flip, rotation, smoothing or interpolation. Equal rectangular display boxes and Figure 6's display aspect are retained. Coordinate axes shown in reference panels apply identically to model panels.

All class fields are exported in NPZ. The publication PDF is fully vector: exact reporting-pixel rectangles are merged only when identical horizontal runs extend across adjacent rows. The sum of vector rectangle areas in grid-pixel units equals each displayed class count. No raster images or image-generation tool are used. PNG is rendered from the same vector blocks at 600 dpi.

Re-run this script with --root pointing at corrected reanalysis and --out a fresh folder. Saved Figure 6 artifacts, corrected NPZs, checkpoint files for hash verification and corrected evaluation CSV are required. No inference is needed.
'''
(O/'FIGURE_PROVENANCE.md').write_text(report,encoding='utf-8')
print(df[['scenario','seed','i','j','method','TP','FP','FN','U','FP_over_U','FN_over_U','IoU_section27_TP_over_U_plus_epsilon']].to_string(index=False),flush=True)
print('MODEL DIFFERENCES\n',changes.to_string(index=False),flush=True)
print('COMPLETE',O,flush=True)
