#!/usr/bin/env python3
"""Frozen primary experiment, original CPU protocol, no test-driven changes.
prepare -> preflight -> train42 -> gate -> train remaining seeds -> freeze -> test -> report.
"""
import os,sys,ast,json,hashlib,math,random,time,re,argparse,contextlib,copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; DATA=ROOT/'dataset_v1_raw_sgas_waterpressure'
DEVICE='cpu'; SEEDS=[42,11,22,33,44,55]
# Exactly the existing CPU environment's default, recorded before starting any training.
assert torch.__version__=='2.12.1+cpu' and torch.get_num_threads()==12,(torch.__version__,torch.get_num_threads())

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def savejson(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf8');tmp.replace(path)

def training_print(*args,**kwargs):
 msg=' '.join(str(x) for x in args)
 print(*args,**kwargs,flush=True)
 hit=re.search(r'\[(.*?)\] epoch (\d+).*loss=([\d.e+-]+).*val_iou=([\d.e+-]+)',msg)
 if hit:savejson(ROOT/'progress.json',{'stage':hit[1],'epoch':int(hit[2]),'train_loss':float(hit[3]),'val_iou':float(hit[4]),'updated_unix':time.time()})

source=PROJECT/'train_unet_monitoring_207.py'
tree=ast.parse(source.read_text())
names=['DoubleConv','SmallUNet','loss_fn','calc_metrics','train_model','evaluate','binary_metrics_np','evaluate_persistence']
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
assert len(nodes)==len(names)
ns={'torch':torch,'nn':nn,'np':np,'pd':pd,'DataLoader':DataLoader,'DEVICE':DEVICE,'print':training_print}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),ns)
SmallUNet=ns['SmallUNet']
index=pd.read_csv(DATA/'dataset_index.csv'); matrix=pd.read_csv(PROJECT/'scenario_matrix_valid_207.csv')
for c in ['well1_rate_mult','well2_rate_mult']:
 index[c]=index.scenario_id.map(matrix.set_index('scenario_id')[c])
assert len(index)==414 and len(matrix)==207

class CachedPairs(Dataset):
 def __init__(self,case,sids,idx):
  self.case=case;self.samples=[];self.cache={}
  for sid in sids:
   row=idx[(idx.case==case)&(idx.scenario_id==sid)].iloc[0]
   self.cache[sid]=(np.load(DATA/case/f'{sid}_dynamic.npy',mmap_mode='r'),np.load(DATA/case/f'{sid}_mask.npy',mmap_mode='r'))
   for i in range(1,5):
    for j in range(i+1,6):
     self.samples.append({'case':case,'scenario_id':sid,'npz_path':row.npz_path,'i':i,'j':j,
       **{k:float(row[k]) for k in ['q_mult','kz_rel_mult','global_perm_mult','well1_rate_mult','well2_rate_mult']}})
 def __len__(self):return len(self.samples)
 def __getitem__(self,k):
  s=self.samples[k];i=s['i'];j=s['j'];dyn,masks=self.cache[s['scenario_id']]
  a=dyn[i-1];H,W=a.shape[-2:]
  tau_i=i/5.;tau_j=j/5.;dtau=tau_j-tau_i
  vals=[tau_i,tau_j,dtau,(s['q_mult']-1.)/.35,math.log(float(s['kz_rel_mult']),2.),
        (s['global_perm_mult']-1.)/.25,(s['well1_rate_mult']-1.)/.7,(s['well2_rate_mult']-1.)/.7]
  const=np.stack([np.full((H,W),v,dtype=np.float32) for v in vals],axis=0)
  X=np.nan_to_num(np.concatenate([a,const],axis=0),nan=0.,posinf=0.,neginf=0.).astype(np.float32)
  y=masks[j][None].astype(np.float32)
  return torch.from_numpy(X),torch.from_numpy(y),s['scenario_id'],i,j

def ids(case,kind):
 col='source_A_split' if case=='spe11a' else 'target_B_role'
 return index[(index.case==case)&(index[col]==kind)].scenario_id.tolist()

def dataset(case,kind):return CachedPairs(case,ids(case,kind),index)

def frozen_protocol():
 config={'primary_threshold':.001,'sensitivity_thresholds':[.01,.05],'legacy_threshold':.097,
  'seeds_in_execution_order':SEEDS,'source_epochs':25,'target_scratch_epochs':30,'target_finetune_epochs':30,
  'stage_order':['A_source','B_finetune','B_scratch'],'optimizer':'Adam','lr':1e-3,'batch_size':4,'num_workers':0,
  'architecture':'original SmallUNet(in_ch=11,base=16)','loss':'original BCE+soft-Dice','selection':'max validation IoU, strict >, original',
  'prediction_threshold':.5,'normalization':'unchanged original observed image maxabs pressure/XCO2; SGAS unscaled',
  'device':'cpu','torch':torch.__version__,'threads':12,'amp':False,'compile':False,'memory_format':'original contiguous',
  'dataset_complete_sha256':sha(ROOT/'DATASET_COMPLETE.json'),'script_sha256':sha(Path(__file__)),
  'source_functions_sha256':sha(source),'dataset_class_source_sha256':sha(PROJECT/'v2_analysis/03_channel_ablation.py'),
  'counts':{f'{case}_{role}':len(ids(case,role)) for case,roles in [('spe11a',['train','val','test']),('spe11b',['finetune_train','finetune_val','zero_shot_and_final_test'])] for role in roles},
  'deferred_test':'test forward passes deferred until all 18 selected checkpoints fixed; original evaluation DataLoader RNG draws preserved',
  'no_new_experiments':'no sensitivity training, scaling ablation, or low-data training in this primary run',
  'legacy_runtime_limit':'legacy logs confirm CPU; historical PyTorch version was not recorded in those logs'}
 path=ROOT/'PROTOCOL.json'
 if path.exists():assert json.loads(path.read_text())==config,'Protocol changed after registration'
 else:savejson(path,config)
 return config

def preflight():
 assert (ROOT/'DATASET_COMPLETE.json').exists()
 conf=frozen_protocol()
 # The completed audit is not repeated; check the new cached loader against the original Dataset implementation.
 tree=ast.parse((PROJECT/'v2_analysis/03_channel_ablation.py').read_text())
 klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AblationDataset')
 env={'Path':Path,'math':math,'np':np,'pd':pd,'torch':torch,'Dataset':Dataset}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'original_AblationDataset','exec'),env)
 checks=[]
 for case,roles in [('spe11a',['train','val']),('spe11b',['finetune_train','finetune_val'])]:
  for role in roles:
   sid=ids(case,role)[0]
   original=env['AblationDataset'](case,[sid],index,[0,1,2]);cached=CachedPairs(case,[sid],index)
   assert len(original)==len(cached)==10
   for k in range(10):
    a=original[k];b=cached[k]
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2:]==b[2:]
    assert a[0].shape[0]==11 and torch.isfinite(a[0]).all()
   checks.append({'case':case,'role':role,'scenario_id':sid,'forecast_pairs':10,'input_and_label_bitwise_equal':True})
 a_train=set(ids('spe11a','train'));a_val=set(ids('spe11a','val'));a_test=set(ids('spe11a','test'))
 b_train=set(ids('spe11b','finetune_train'));b_val=set(ids('spe11b','finetune_val'));b_test=set(ids('spe11b','zero_shot_and_final_test'))
 assert not a_train&a_val and not a_train&a_test and not a_val&a_test
 assert not b_train&b_val and not b_train&b_test and not b_val&b_test
 assert not (a_train|a_val)&b_test and len(b_test)==41
 # Check deferred-evaluation RNG accounting using train data only.
 for case,role in [('spe11a','train'),('spe11b','finetune_train')]:
  small=CachedPairs(case,[ids(case,role)[0]],index);model=SmallUNet(in_ch=11,base=16)
  state=torch.get_rng_state().clone()
  ns['evaluate'](model,small,'TRAIN_ONLY_RNG_CHECK')
  real=torch.get_rng_state().clone();torch.set_rng_state(state)
  consume_original_evaluation_rng(model,'PREFLIGHT_'+case)
  assert torch.equal(real,torch.get_rng_state()),'Deferred evaluation RNG differs from original'
  torch.set_rng_state(state)
 pd.DataFrame(checks).to_csv(ROOT/'loader_compatibility_checks.csv',index=False)
 savejson(ROOT/'PREFLIGHT_PASS.json',{'train_validation_only':True,'bitwise_loader_comparison_pairs':40,'split_checks':'pass','protocol_sha256':sha(ROOT/'PROTOCOL.json')})
 print('PREFLIGHT PASS',conf['counts'],flush=True)

def consume_original_evaluation_rng(model,tag):
 # Original evaluate: one unshuffled, num_workers=0 DataLoader iterator; no dropout or stochastic layers.
 # Its only RNG draw is _BaseDataLoaderIter's base seed. No test fields or scores are read here.
 model.to(DEVICE);model.eval()
 before=torch.get_rng_state().clone()
 iterator=iter(DataLoader([],batch_size=1,shuffle=False));del iterator
 after=torch.get_rng_state().clone()
 assert not torch.equal(before,after)
 with open(ROOT/'deferred_evaluation_rng.jsonl','a') as f:
  f.write(json.dumps({'tag':tag,'before':hashlib.sha256(before.numpy().tobytes()).hexdigest(),'after':hashlib.sha256(after.numpy().tobytes()).hexdigest()})+'\n')

def loadmodel(path):
 model=SmallUNet(in_ch=11,base=16)
 model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
 return model

def train_one(seed):
 assert (ROOT/'PREFLIGHT_PASS.json').exists()
 frozen_protocol()
 out=ROOT/'primary'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'COMPLETE.json').exists(),'No repeat training of completed seed'
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 atr=dataset('spe11a','train');av=dataset('spe11a','val')
 btr=dataset('spe11b','finetune_train');bv=dataset('spe11b','finetune_val')
 durations={};started=time.time()
 def fit(model,tr,va,epochs,path,tag):
  t=time.time()
  hist=ns['train_model'](model,DataLoader(tr,batch_size=4,shuffle=True,num_workers=0),
       DataLoader(va,batch_size=4,shuffle=False,num_workers=0),epochs=epochs,lr=1e-3,save_path=path,tag=tag)
  assert len(hist)==epochs and np.isfinite(hist[['train_loss','val_iou','val_dice','val_area_rel_error']].to_numpy()).all()
  hist.to_csv(out/f'history_{tag}.csv',index=False)
  durations[tag]=time.time()-t
  return hist
 sourcepath=out/'source_best.pt';ftpath=out/'finetuned_best.pt';scratchpath=out/'scratch_best.pt'
 hist=fit(SmallUNet(in_ch=11,base=16),atr,av,25,sourcepath,f'seed{seed}_A_source')
 src=loadmodel(sourcepath)
 consume_original_evaluation_rng(src,f'seed{seed}_A_test_deferred');consume_original_evaluation_rng(src,f'seed{seed}_B_zero_deferred')
 ft=SmallUNet(in_ch=11,base=16);ft.load_state_dict(torch.load(sourcepath,map_location=DEVICE,weights_only=True))
 histft=fit(ft,btr,bv,30,ftpath,f'seed{seed}_B_finetune')
 ft=loadmodel(ftpath);consume_original_evaluation_rng(ft,f'seed{seed}_B_finetuned_deferred')
 histsc=fit(SmallUNet(in_ch=11,base=16),btr,bv,30,scratchpath,f'seed{seed}_B_scratch')
 scratch=loadmodel(scratchpath);consume_original_evaluation_rng(scratch,f'seed{seed}_B_scratch_deferred')
 result={'seed':seed,'train_validation_only':True,'epochs_completed':[25,30,30],
   'selection':{name:{'best_epoch':int(h.loc[h.val_iou.idxmax(),'epoch']),'best_val_iou':float(h.val_iou.max())}
      for name,h in [('source',hist),('finetune',histft),('scratch',histsc)]},
   'checkpoints':{p.name:sha(p) for p in [sourcepath,ftpath,scratchpath]},'stage_seconds':durations,'elapsed_seconds':time.time()-started}
 savejson(out/'COMPLETE.json',result)
 if seed==42:savejson(ROOT/'SEED42_PIPELINE_PASS.json',{'seed':42,'gate':'finite full 25/30/30 train/validation, exact loader compatibility, no test inspected','checkpoint_selection':'validation only'})
 print('SEED_COMPLETE',json.dumps(result),flush=True)

def freeze_models():
 frozen_protocol();assert (ROOT/'SEED42_PIPELINE_PASS.json').exists()
 files=[]
 for seed in SEEDS:
  folder=ROOT/'primary'/f'seed_{seed}';d=json.loads((folder/'COMPLETE.json').read_text())
  for name,digest in d['checkpoints'].items():
   path=folder/name;assert sha(path)==digest;files.append({'seed':seed,'path':str(path),'sha256':digest})
 frozen={'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'checkpoints':files,'test_scores_seen':False,'freeze_unix':time.time()}
 assert len(files)==18
 savejson(ROOT/'MODELS_FROZEN_BEFORE_TEST.json',frozen)
 print('ALL 18 CHECKPOINTS FROZEN BEFORE TEST',flush=True)

def evaluate_all():
 frozen=json.loads((ROOT/'MODELS_FROZEN_BEFORE_TEST.json').read_text())
 assert sha(ROOT/'PROTOCOL.json')==frozen['protocol_sha256']
 for rec in frozen['checkpoints']:assert sha(Path(rec['path']))==rec['sha256']
 assert not (ROOT/'primary_metrics_all.csv').exists(),'Do not re-run final test'
 atest=dataset('spe11a','test');btest=dataset('spe11b','zero_shot_and_final_test')
 rows=[]
 persistence=ns['evaluate_persistence'](btest,'B_persistence');persistence['seed']=-1
 for seed in SEEDS:
  folder=ROOT/'primary'/f'seed_{seed}'
  for checkpoint,ds,tag in [('source_best.pt',atest,'A_test'),('source_best.pt',btest,'B_zero'),('scratch_best.pt',btest,'B_scratch'),('finetuned_best.pt',btest,'B_finetuned')]:
   model=loadmodel(folder/checkpoint);df=ns['evaluate'](model,ds,tag);df['seed']=seed
   assert len(df)==410 and np.isfinite(df[['iou','dice','area_rel_error']]).all().all()
   df.to_csv(folder/f'{tag}_test_metrics.csv',index=False);rows.append(df)
  print('FINAL_TEST_COMPLETED seed',seed,flush=True)
 persistence.to_csv(ROOT/'persistence_metrics.csv',index=False)
 allmetrics=pd.concat(rows,ignore_index=True);allmetrics.to_csv(ROOT/'primary_metrics_all.csv',index=False)
 savejson(ROOT/'FINAL_EVALUATION_COMPLETE.json',{'learned_rows':len(allmetrics),'persistence_rows':len(persistence),'fixed_B_test_scenarios':41,'forecast_pairs_per_scenario':10,'no_test_tuning':True})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','train','freeze','evaluate']);parser.add_argument('--seed',type=int)
 args=parser.parse_args()
 if args.action=='preflight':preflight()
 elif args.action=='train':assert args.seed in SEEDS;train_one(args.seed)
 elif args.action=='freeze':freeze_models()
 elif args.action=='evaluate':evaluate_all()
