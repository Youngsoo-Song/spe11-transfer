#!/usr/bin/env python3
import os,sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from resdata.grid import Grid
from resdata.resfile import ResdataFile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; DATA=ROOT/'dataset_v1_raw_sgas_waterpressure'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 assert not (ROOT/'DATASET_COMPLETE.json').exists(),'Dataset is immutable once complete'
 DATA.mkdir(exist_ok=True)
 spec=importlib.util.spec_from_file_location('completed_audit_mapping',PROJECT/'_review_audit_20261116/audit.py')
 au=importlib.util.module_from_spec(spec); spec.loader.exec_module(au)
 matrix=pd.read_csv(PROJECT/'scenario_matrix_valid_207.csv'); legacy=pd.read_csv(PROJECT/'ai_dataset_npz_207/dataset_index.csv')
 assert len(matrix)==207 and len(legacy)==414
 expected=pd.read_csv(PROJECT/'_review_audit_20261116/threshold_comparison.csv')
 geom=[]; checks=[]; provenance=[]; index=[]; started=time.time()
 for case in ['spe11a','spe11b']:
  dest=DATA/case; dest.mkdir(exist_ok=True)
  for k,row in enumerate(legacy[legacy.case==case].to_dict('records')):
   sid=row['scenario_id']; run=PROJECT/'runs'/case/sid
   with np.load(row['npz_path']) as z:
    times=z['times'].copy(); x=z['x'].copy(); zz=z['z'].copy(); legacy_xco2=z['X'][:,2].copy()
   nx,nz=len(x),len(zz)
   un=run/'flow'/f'{sid}.UNRST'; eg=run/'flow'/f'{sid}.EGRID'; ini=run/'flow'/f'{sid}.INIT'
   rf=ResdataFile(str(un)); it=ResdataFile(str(ini)); grid=Grid(str(eg))
   mp=au.get_mapping(case,sid,grid,it,nx,nz,geom); ids=mp['ids']; active=np.flatnonzero(mp['act'])
   valid=mp['act'][ids].astype(bool).reshape(nz,nx)
   lines=(run/'deck/dt.txt').read_text().splitlines(); ti=float(lines[0]); skip=int(lines[1]); ts=np.fromstring(lines[2],sep=' ')
   factor=3600. if case=='spe11a' else 31536000.
   sg=[]; pressure=[]; indices=[]
   def reporting(a):
    field=np.full(len(mp['act']),np.nan,dtype=np.float64); field[active]=a
    return field[ids].reshape(nz,nx)
   for t in times:
    h=np.flatnonzero(np.isclose(ts,float(t)*factor,rtol=0,atol=1e-4)); assert len(h)==1
    n=int(h[0])+skip; indices.append(n)
    assert abs(float(np.array(rf['DOUBHEAD'][n])[0])*86400.-ti-float(t)*factor)<1e-4
    s=np.array(rf['SGAS'][n]); p=np.array(rf['PRESSURE'][n])
    assert (s>=0).all() and np.isfinite(s).all()
    if rf.has_kw('PCGW'):p-=np.array(rf['PCGW'][n])
    sg.append(reporting(s)); pressure.append(reporting(p*1e5))
   sg=np.stack(sg); pressure=np.stack(pressure)
   ds=np.maximum(sg-sg[0:1],0); dp=pressure-pressure[0:1]
   assert np.nanmax(sg[0])==0 and np.isfinite(pressure[:,valid]).all()
   masks={f'mask_{tag}':((ds>thr)&valid[None]).astype(np.uint8) for tag,thr in [('0001',.001),('001',.01),('005',.05),('legacy0097',.097)]}
   for j,t in enumerate(times):
    for tag,thr in [('0001',.001),('001',.01),('005',.05),('legacy0097',.097)]:
     e=expected[(expected.case==case)&(expected.scenario_id==sid)&(expected.time==float(t))&(expected.threshold==thr)].iloc[0]
     assert int(masks[f'mask_{tag}'][j].sum())==int(e.raw_plume_pixels),(case,sid,t,thr)
   X=np.stack([np.nan_to_num(dp,nan=0.).astype(np.float32),np.nan_to_num(ds,nan=0.).astype(np.float32),legacy_xco2.astype(np.float32)],axis=1)
   assert np.isfinite(X).all() and X.shape==(6,3,nz,nx)
   path=dest/f'{sid}.npz'; assert not path.exists()
   np.savez_compressed(path,times=times,x=x,z=zz,sgas_raw=sg,water_pressure_raw_Pa=pressure,
      delta_sgas_raw=ds,delta_pressure_raw_Pa=dp,X=X,plume_mask=masks['mask_0001'],
      valid_mask=valid.astype(np.uint8),restart_indices=np.array(indices),**masks)
   # Training cache has exactly the original per-observation float32 maxabs scaling.
   dyn=X[1:5].copy()
   for j in range(4):
    for c in [0,2]:
     vmax=np.nanmax(np.abs(dyn[j,c]))
     if vmax>0:dyn[j,c]/=vmax
   np.save(dest/f'{sid}_dynamic.npy',dyn)
   np.save(dest/f'{sid}_mask.npy',masks['mask_0001'])
   # Reopen output, check lossless raw doubles and unrounded delta before accepting it.
   with np.load(path) as test:
    assert np.array_equal(test['sgas_raw'],sg,equal_nan=True)
    assert np.array_equal(test['water_pressure_raw_Pa'],pressure,equal_nan=True)
    assert np.array_equal(test['plume_mask'],masks['mask_0001'])
    assert np.array_equal(test['X'][:,2],legacy_xco2)
   checks.append({'case':case,'scenario_id':sid,'snapshots':6,'valid_pixels':int(valid.sum()),
    'raw_float64_roundtrip_exact':True,'mask_matches_completed_audit':True,'xco2_legacy_identical':True,
    'delta_pressure_float32_max_error_Pa':float(np.max(np.abs(X[:,0][:,valid].astype(float)-dp[:,valid]))),
    'low_sgas_gt0001_le0097_pixels':int(((ds>.001)&(ds<=.097)&valid[None]).sum()),'npz_sha256':sha(path)})
   for p in [un,eg,ini,run/'deck/dt.txt',Path(row['npz_path'])]:provenance.append({'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size})
   row.update(npz_path=str(path),n_channels=3,representation='corrected_primary_raw_dSg_gt_0.001')
   index.append(row); rf.close(); it.close()
   if k==0 or (k+1)%25==0:print('DATASET',case,k+1,'/207','elapsed_s',round(time.time()-started),flush=True)
 idx=pd.DataFrame(index)
 idx.to_csv(DATA/'dataset_index.csv',index=False); matrix.to_csv(DATA/'scenario_matrix_valid_207.csv',index=False)
 for name,rows in [('preprocessing_checks',checks),('input_provenance',provenance),('geometry_manifest',geom)]:pd.DataFrame(rows).to_csv(ROOT/f'{name}.csv',index=False)
 complete={'dataset_version':str(DATA),'domain_scenario_pairs':414,'snapshots':2484,'primary_threshold':.001,
  'sensitivity_thresholds':[.01,.05],'legacy_threshold':.097,'training_dynamic_channels':3,'final_model_channels':11,
  'raw_dtype':'float64, NaN outside native active mapping','training_dtype':'float32, zero fill as legacy',
  'pressure_formula':'(PRESSURE-PCGW)*1e5 Pa; delta formed in float64 before one float32 cast',
  'xco2':'legacy X[:,2] retained exactly to isolate authorized SGAS/pressure corrections',
  'normalization':'original observed-image maxabs for pressure and XCO2; no fitted statistics',
  'index_sha256':sha(DATA/'dataset_index.csv'),'matrix_sha256':sha(PROJECT/'scenario_matrix_valid_207.csv'),
  'prepare_script_sha256':sha(Path(__file__)),'elapsed_seconds':time.time()-started}
 (ROOT/'DATASET_COMPLETE.json').write_text(json.dumps(complete,indent=2),encoding='utf8')
 print(json.dumps(complete,indent=2),flush=True)
if __name__=='__main__':main()
