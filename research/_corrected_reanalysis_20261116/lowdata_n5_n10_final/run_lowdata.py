#!/usr/bin/env python3
"""Exactly 20 final low-data fits; fixed historical subsets/model seed; no experiment expansion."""
import sys,ast,json,time,hashlib,math,random,re,argparse,contextlib
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;PRIMARY=ROOT.parent;PROJECT=PRIMARY.parent
DATA=PRIMARY/'dataset_v1_raw_sgas_waterpressure';LEGACY=PROJECT/'v2_analysis/lowdata_subset_robustness'
SOURCE=PRIMARY/'primary/seed_42/source_best.pt';DEVICE='cpu';DRAWS=[101,202,303,404,505]
assert sys.executable=='/home/yssongvd/opm-ai-env/bin/python'
assert torch.__version__=='2.12.1+cpu' and torch.get_num_threads()==12

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def jsave(path,obj):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,default=str),encoding='utf8');t.replace(path)
def tprint(*args,**kwargs):
 print(*args,**kwargs,flush=True)
 hit=re.search(r'\[(.*?)\] epoch (\d+).*loss=([\d.e+-]+).*val_iou=([\d.e+-]+)',' '.join(str(a) for a in args))
 if hit:jsave(ROOT/'progress.json',{'stage':hit[1],'epoch':int(hit[2]),'train_loss':float(hit[3]),'val_iou':float(hit[4]),'unix':time.time()})

basefile=PROJECT/'train_unet_monitoring_207.py';tree=ast.parse(basefile.read_text())
names=['DoubleConv','SmallUNet','loss_fn','calc_metrics','train_model','evaluate']
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
ns={'torch':torch,'nn':nn,'np':np,'pd':pd,'DataLoader':DataLoader,'DEVICE':DEVICE,'print':tprint}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(basefile),'exec'),ns)
ptree=ast.parse((PRIMARY/'train_primary.py').read_text())
klass=next(n for n in ptree.body if isinstance(n,ast.ClassDef) and n.name=='CachedPairs')
ns.update(Path=Path,DATA=DATA,math=math,Dataset=Dataset)
exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'unchanged_CachedPairs','exec'),ns)
Pairs=ns['CachedPairs'];Model=ns['SmallUNet']
index=pd.read_csv(DATA/'dataset_index.csv');matrix=pd.read_csv(PROJECT/'scenario_matrix_valid_207.csv')
for c in ['well1_rate_mult','well2_rate_mult']:index[c]=index.scenario_id.map(matrix.set_index('scenario_id')[c])
def ids(role):return sorted(index[(index.case=='spe11b')&(index.target_B_role==role)].scenario_id.unique().tolist())
def seed42():random.seed(42);np.random.seed(42);torch.manual_seed(42)
def verify_protocol():
 p=json.loads((ROOT/'PROTOCOL.json').read_text())
 assert p['script_sha256']==sha(Path(__file__)) and p['training_functions_sha256']==sha(basefile)
 assert p['source_checkpoint_sha256']==sha(SOURCE)
 assert p['primary_protocol_sha256']==sha(PRIMARY/'PROTOCOL.json')
 assert p['dataset_complete_sha256']==sha(PRIMARY/'DATASET_COMPLETE.json')
 assert p['matrix_sha256']==sha(PROJECT/'scenario_matrix_valid_207.csv')
 assert len(p['jobs'])==10 and sum(len(x['methods']) for x in p['jobs'])==20
 return p

def setup():
 assert not (ROOT/'PROTOCOL.json').exists()
 cfg=json.loads((PRIMARY/'PROTOCOL.json').read_text())
 assert sha(basefile)==cfg['source_functions_sha256']
 assert sha(PRIMARY/'train_primary.py')==cfg['script_sha256']
 assert json.loads((PRIMARY/'primary/seed_42/COMPLETE.json').read_text())['checkpoints']['source_best.pt']==sha(SOURCE)
 pool=ids('finetune_train');val=ids('finetune_val');test=ids('zero_shot_and_final_test')
 assert len(pool)==33 and len(val)==25 and len(test)==41
 assert not(set(pool)&set(val) or set(pool)&set(test) or set(val)&set(test))
 jobs=[];rows=[];provenance=[]
 for rep,draw in enumerate(DRAWS,1):
  perm=[str(x) for x in np.random.default_rng(draw).permutation(pool)]
  for n in [5,10]:
   file=LEGACY/f'rep_{rep:02d}_n_{n:02d}/training_scenarios.csv'
   train=pd.read_csv(file).scenario_id.tolist()
   assert train==perm[:n],('saved subset vs historical RNG recipe differ',rep,n)
   assert len(set(train))==n and set(train)<=set(pool)
   out=ROOT/f'rep_{rep:02d}_n_{n:02d}';out.mkdir(exist_ok=False)
   # Exact byte-for-byte copy of existing scenario order, no new draw selection.
   (out/'training_scenarios.csv').write_bytes(file.read_bytes())
   jobs.append({'replicate':rep,'subset_seed':draw,'n_target_train':n,'train_ids':train,'methods':['finetune','scratch']})
   for order,sid in enumerate(train):rows.append({'replicate':rep,'subset_seed':draw,'N':n,'order':order,'scenario_id':sid})
   provenance.append({'path':str(file),'sha256':sha(file),'copied_sha256':sha(out/'training_scenarios.csv')})
 pd.DataFrame(rows).to_csv(ROOT/'subset_manifest.csv',index=False)
 pd.DataFrame(provenance).to_csv(ROOT/'subset_provenance.csv',index=False)
 config={'primary_threshold':.001,'jobs':jobs,'total_training_jobs':20,'N_values':[5,10],
  'model_seed':42,'subset_seeds':DRAWS,'source_checkpoint':str(SOURCE),'source_checkpoint_sha256':sha(SOURCE),
  'source_selection':'corrected primary seed42, A validation best epoch24; no B test-based source choice',
  'train_pool':pool,'validation_ids':val,'test_ids':test,'validation_count':25,'test_count':41,
  'epochs_per_job':30,'batch_size':4,'lr':.001,'optimizer':'original Adam defaults','loss':'original BCE+soft-Dice',
  'architecture':'SmallUNet(in_ch=11,base=16)','prediction_threshold':.5,'num_workers':0,
  'normalization':'same primary observed-image maxabs pressure/XCO2; deltaSGAS unscaled',
  'device':'cpu','torch':torch.__version__,'threads':12,'amp':False,'compile':False,
  'seed_reset':'Python/NumPy/Torch reset to42 separately before each fine-tune AND scratch model construction, as legacy08',
  'subset_restore':'saved CSVs copied byte-for-byte, order and historical default_rng recipe match; N5 nested within N10',
  'test_isolation':'all20 validation-selected checkpoints frozen before any new low-data test evaluation',
  'N33_reference':'existing corrected6-seed gain0.006130404806718577 only; no new N33 training',
  'stop_rule':'This is the final AI training. No threshold/scaling/otherN/seed/architecture/geology/noise/UQ expansion.',
  'script_sha256':sha(Path(__file__)),'training_functions_sha256':sha(basefile),
  'primary_protocol_sha256':sha(PRIMARY/'PROTOCOL.json'),'dataset_complete_sha256':sha(PRIMARY/'DATASET_COMPLETE.json'),
  'matrix_sha256':sha(PROJECT/'scenario_matrix_valid_207.csv'),'registered_unix':time.time(),
  'known_restore_limits':'Historical legacy PyTorch version was not recorded. Current existing CPU environment retained. Full-data primary used continuous per-seed RNG; legacy08 resets seed42 per method, which is reproduced here.'}
 jsave(ROOT/'PROTOCOL.json',config)
 jsave(ROOT/'ANALYSIS_PLAN.json',{'metric':'average10 forecast-pair IoUs per each of41 scenarios, then average41 scenarios',
  'paired_gain':'fine_tuned minus scratch for same subset/scenario/i/j','SD':'sample SD across5 subset draws, not model-seed uncertainty',
  'negative_transfer':'scenario mean paired gain <0; export all5x2x41 cases and negative-only table',
  'repetition':'report exact positive count/5 for each N; no subset exclusions or p-value selection',
  'comparison':'matched legacy08 draws; legacy07 seed2026 headline is separately identified, not a matched counterpart',
  'budget':'N is weight-update scenario count;25 target validation scenarios also labelled. N5 needs5+25=30,N10 needs10+25=35 target development scenarios, test41 additional benchmark only.',
  'claim_limits':'one source/model seed42,five overlapping/nested subset draws,fixed geology,noise-free inputs,binary extent,sampled horizons',
  'stop':'20 fits only; final evidence synthesis without more AI learning','registered_unix':time.time()})
 print('RESTORED10 EXACT SUBSETS;20 JOBS REGISTERED',flush=True)
 print(json.dumps({'pool':len(pool),'val':len(val),'test':len(test),'draws':DRAWS,'model_seed':42,'source_sha':sha(SOURCE)},indent=2),flush=True)

def train():
 p=verify_protocol();assert not (ROOT/'TRAINING_STARTED.json').exists()
 jsave(ROOT/'TRAINING_STARTED.json',{'unix':time.time(),'planned_jobs':20})
 val=Pairs('spe11b',p['validation_ids'],index);completed=[];start=time.time()
 for job in p['jobs']:
  rep=job['replicate'];n=job['n_target_train'];out=ROOT/f'rep_{rep:02d}_n_{n:02d}'
  train_ds=Pairs('spe11b',job['train_ids'],index)
  initial=[]
  for method in ['finetune','scratch']:
   seed42();model=Model(in_ch=11,base=16)
   if method=='finetune':model.load_state_dict(torch.load(SOURCE,map_location=DEVICE,weights_only=True))
   rng_hash=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest();initial.append(rng_hash)
   checkpoint=out/f'{method}_best.pt';assert not checkpoint.exists()
   before=time.perf_counter();cpu=time.process_time()
   with open(out/f'train_{method}.log','x') as log,contextlib.redirect_stdout(log):
    hist=ns['train_model'](model,DataLoader(train_ds,batch_size=4,shuffle=True,num_workers=0),
       DataLoader(val,batch_size=4,shuffle=False,num_workers=0),epochs=30,lr=.001,save_path=checkpoint,tag=f'rep{rep}_N{n}_{method}')
   wall=time.perf_counter()-before;cpu=time.process_time()-cpu
   assert len(hist)==30 and np.isfinite(hist[['train_loss','val_iou','val_dice','val_area_rel_error']].to_numpy()).all()
   hist.to_csv(out/f'history_{method}.csv',index=False)
   rec={'replicate':rep,'subset_seed':job['subset_seed'],'N':n,'method':method,'model_seed':42,
    'epochs':30,'best_epoch':int(hist.loc[hist.val_iou.idxmax(),'epoch']),'best_val_iou':float(hist.val_iou.max()),
    'wall_seconds':wall,'process_CPU_seconds':cpu,'initial_training_RNG_sha256':rng_hash,
    'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint)}
   jsave(out/f'{method}_COMPLETE.json',rec);completed.append(rec)
   pd.DataFrame(completed).to_csv(ROOT/'training_costs.csv',index=False)
   print('FIT_COMPLETE',json.dumps(rec),flush=True)
  assert initial[0]==initial[1],'Paired scratch/FT training RNG starts differ'
 assert len(completed)==20
 frozen={'checkpoints':completed,'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'source_sha256':sha(SOURCE),
         'new_lowdata_test_scores_seen':False,'frozen_unix':time.time(),'total_train_wall_seconds':time.time()-start}
 jsave(ROOT/'MODELS_FROZEN_BEFORE_TEST.json',frozen)
 print('ALL20 LOWDATA CHECKPOINTS FROZEN;NO FURTHER TRAINING',flush=True)

def evaluate():
 p=verify_protocol();f=json.loads((ROOT/'MODELS_FROZEN_BEFORE_TEST.json').read_text())
 assert len(f['checkpoints'])==20 and f['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
 assert not (ROOT/'metrics_all.csv').exists()
 assert all(sha(Path(r['checkpoint']))==r['checkpoint_sha256'] for r in f['checkpoints'])
 test=Pairs('spe11b',p['test_ids'],index);results=[]
 for rec in f['checkpoints']:
  model=Model(in_ch=11,base=16);model.load_state_dict(torch.load(rec['checkpoint'],map_location=DEVICE,weights_only=True))
  res=ns['evaluate'](model,test,rec['method']);assert len(res)==410
  res['replicate']=rec['replicate'];res['subset_seed']=rec['subset_seed'];res['N']=rec['N'];res['model_seed']=42
  res.to_csv(Path(rec['checkpoint']).parent/f'metrics_{rec["method"]}.csv',index=False);results.append(res)
  print('TEST_COMPLETE',rec['replicate'],rec['N'],rec['method'],flush=True)
 allmetrics=pd.concat(results,ignore_index=True);assert len(allmetrics)==8200
 allmetrics.to_csv(ROOT/'metrics_all.csv',index=False)
 jsave(ROOT/'TRAIN_AND_TEST_COMPLETE.json',{'jobs':20,'epochs':600,'test_rows':8200,'test_scenarios':41,'validation_scenarios':25,'unix':time.time(),'additional_training_prohibited':True})
 print('LOWDATA TRAINING AND EVALUATION COMPLETE;STOPPING TRAINING',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['setup','train','evaluate']);args=a.parse_args()
 {'setup':setup,'train':train,'evaluate':evaluate}[args.action]()
