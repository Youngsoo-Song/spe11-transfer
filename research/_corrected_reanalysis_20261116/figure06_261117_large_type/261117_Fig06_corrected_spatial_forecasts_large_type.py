"""Typography-only replot from saved corrected predictions; no torch/inference."""
from pathlib import Path
import os,sys,json,hashlib,csv,argparse
sys.dont_write_bytecode=True
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,default=Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/figure06_261117'))
ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args();S=args.source.resolve();O=args.out.resolve()
O.mkdir(parents=True,exist_ok=True);(O/'qa').mkdir(exist_ok=True)
BASE='261117_Fig06_corrected_spatial_forecasts'
STEM=BASE+'_large_type'
assert not (O/(STEM+'.png')).exists() and not (O/(STEM+'.pdf')).exists()
os.environ['MPLCONFIGDIR']=str(O/'qa/mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sourcefiles=[S/(BASE+'_row_verification.csv'),S/'FIGURE_PROVENANCE.json',S/(BASE+'.png'),S/(BASE+'.pdf')]
with open(sourcefiles[0],newline='') as f:rows=list(csv.DictReader(f))
images=[]
for r in rows:
    path=S/f'row{r["row"]}_{r["scenario_id"]}_seed{r["seed"]}_i{r["i"]}_j{r["j"]}_predictions.npz'
    sourcefiles.append(path)
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in ['observed_delta_sgas_raw','true_mask','persistence','zero_shot','scratch','fine_tuning']}
    truth=a['true_mask'].astype(bool)
    for method in ['persistence','zero_shot','scratch','fine_tuning']:
        pred=a[method].astype(bool)
        iou=(pred&truth).sum()/((pred|truth).sum()+1e-6)
        assert abs(iou-float(r[method+'_iou']))<1e-14
    images.append(a)
hashes={str(p):sha(p) for p in sourcefiles}
fontdir=Path('/mnt/c/Windows/Fonts') if os.name!='nt' else Path('C:/Windows/Fonts')
for f in ['arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf']:font_manager.fontManager.addfont(str(fontdir/f))
assert 'arial' in font_manager.findfont('Arial',fallback_to_default=False).lower()
plt.rcParams.update({'font.family':'Arial','font.size':12,'axes.titlesize':13,'axes.linewidth':.7,'pdf.fonttype':42})
cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#D9D9D9')
binary=ListedColormap(['#440154','#FDE725'])
fig,axs=plt.subplots(4,6,figsize=(16.5,9.6))
titles=['Observed\nΔSg','True future\nplume','Persistence','A→B\nzero-shot','B scratch','A pretrain +\nB fine-tune']
methods=['persistence','zero_shot','scratch','fine_tuning']
reasons=['Largest target-learning\nrecovery','Largest positive\nfine-tuning gain','High persistence','Negative transfer\n(S099)']
for ri,(r,a) in enumerate(zip(rows,images)):
    for ci,key in enumerate(['observed_delta_sgas_raw','true_mask',*methods]):
        ax=axs[ri,ci];field=a[key]
        im=ax.imshow(field,origin='lower',extent=[0,8.4,0,1.2],aspect='auto',interpolation='nearest',
            cmap=cmap if ci==0 else binary,vmin=0,vmax=.9 if ci==0 else 1)
        assert np.array_equal(np.asarray(im.get_array()),field,equal_nan=True)
        ax.set_xticks([]);ax.set_yticks([])
        if ri==0:ax.set_title(titles[ci],pad=10)
        if ci>=2:
            label=f'IoU={float(r[key+"_iou"]):.3f}'
            ax.text(.025,.975,label,transform=ax.transAxes,ha='left',va='top',fontsize=12,
                bbox=dict(facecolor='white',edgecolor='.5',alpha=.94,pad=2.0))
    ax=axs[ri,0]
    ax.set_xticks([0,4.2,8.4]);ax.set_yticks([0,.6,1.2]);ax.tick_params(labelsize=10,length=2,pad=2)
    ax.set_xlabel('x (km)',fontsize=11,labelpad=2);ax.set_ylabel('z (km)',fontsize=11,labelpad=2)
fig.subplots_adjust(left=.18,right=.99,bottom=.15,top=.90,wspace=.075,hspace=.34)
for ri,r in enumerate(rows):
    bounds=axs[ri,0].get_position();center=(bounds.y0+bounds.y1)/2
    header=f'{r["scenario_id"]} | seed {r["seed"]}'
    times=f'i={r["i"]} → j={r["j"]}; lead {float(r["lead_time_yr"]):g} yr\n{float(r["observation_time_yr"]):g} → {float(r["target_time_yr"]):g} yr'
    fig.text(.012,center+.033,header,fontsize=12.5,weight='bold',ha='left',va='bottom')
    fig.text(.012,center+.020,times+'\n'+reasons[ri],fontsize=11.5,ha='left',va='top',linespacing=1.18)
cax=fig.add_axes([.24,.065,.36,.018])
cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,.9),cmap=cmap),cax=cax,orientation='horizontal')
cb.ax.tick_params(labelsize=11);cb.set_label('Observed raw ΔSg',fontsize=12,labelpad=3)
fig.legend(handles=[Patch(facecolor='#440154',label='Background'),Patch(facecolor='#FDE725',label='Plume (raw ΔSg > 0.001)')],
    loc='lower right',bbox_to_anchor=(.985,.052),frameon=False,ncol=2,fontsize=12)
fig.text(.5,.009,'Post-evaluation examples; frozen N=33 checkpoints. Equal display boxes; elevation increases upward.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(O/(STEM+'.'+ext),dpi=600)
fig.savefig(O/'qa/preview.png',dpi=130);plt.close(fig)
(O/(STEM+'_row_verification.csv')).write_bytes(sourcefiles[0].read_bytes())
assert all(sha(p)==h for p,h in hashes.items())
changes={'IoU':{'from':8,'to':12,'position':'upper left'},'row_scenario_seed':{'from':8.5,'to':12.5},
    'row_details':{'from':8.5,'to':11.5},'column_titles':{'from':10,'to':13},
    'legend':{'from':9,'to':12},'colourbar_ticks':{'from':8,'to':11},'colourbar_label':{'from':9,'to':12},
    'axis_ticks':{'from':7,'to':10},'axis_labels':{'from':8,'to':11}}
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps({'source_sha256':hashes,'changes':changes,
    'new_inference_passes':0,'training_steps':0,'cases_masks_predictions_scores_unchanged':True,
    'prior_provenance':str(S/'FIGURE_PROVENANCE.json'),
    'output_sha256':{STEM+'.'+e:sha(O/(STEM+'.'+e)) for e in ['png','pdf']}},indent=2),encoding='utf-8')
(O/'FIGURE_PROVENANCE.md').write_text(f'''# Figure 6 readability revision

Typography/layout only, 2026-11-17. Source: `{S}`.

IoU boxes moved from upper right to upper left and enlarged 8→12 pt. Scenario/seed labels enlarged 8.5→12.5 pt; row details 8.5→11.5 pt; column titles 10→13 pt; legend 9→12 pt; colourbar ticks 8→11 pt and label 9→12 pt; axis ticks 7→10 pt and axis labels 8→11 pt. Long headings/descriptions wrapped to avoid overlap. The canvas height increased from 8.9 to 9.6 inches while width stayed 16.5 inches. Six columns, four rows, purple/yellow binary colours and 600 dpi retained.

Saved prediction NPZs were used directly. No model, torch, checkpoint or original simulation was executed/read for inference. Every exact-pair IoU was independently checked from saved Boolean masks; the verification CSV is byte-identical to the original. All four cases, seeds, i/j values, probabilities/masks, colours, physical coordinates and upward elevation convention are unchanged. Prior source/figure hashes are unchanged. The full corrected dataset/checkpoint paths remain in the copied row-verification CSV and original provenance JSON. Source/output hashes and exact font changes are recorded in the new JSON. Original files are preserved.

Re-run this plot-only script with `--source` pointing to the prior Figure 6 folder and `--out` pointing to a fresh directory. The source folder must contain its saved prediction NPZs, CSV and provenance. No inference is needed.
''',encoding='utf-8')
print('SAVED',O,flush=True)
