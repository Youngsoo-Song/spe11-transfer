#!/usr/bin/env python3
"""Read-only raw OPM audit; all writes constrained to this script's directory.
Run with existing /home/yssongvd/opm-ai-env/bin/python -B audit.py [--pilot].
No simulation, training, installation, or imports of training scripts.
"""
import argparse, ast, contextlib, hashlib, importlib.util, io, json, platform, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from resdata.grid import Grid
from resdata.resfile import ResdataFile
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
SOURCE = Path('/home/yssongvd/pyopmspe11/src/pyopmspe11/visualization/data.py')
THRESHOLDS = [.001, .01, .05, .097, .1, .15]
CACHE = {}
spec = importlib.util.spec_from_file_location('audit_original_data', SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def rounded(a):
    return np.array([float(f'{v:.3e}') for v in np.ravel(a)]).reshape(a.shape)

def fraction(a,b): return float(a/b) if b else np.nan

def stats(a, prefix):
    a=np.asarray(a, dtype=float)
    return {prefix+k:float(v) for k,v in zip(['min','q05','median','q95','max','mean','std'],
          [a.min(), *np.quantile(a,[.05,.5,.95]), a.max(), a.mean(), a.std()])}

def norm(a):
    v=float(np.max(np.abs(a)))
    return a/v if v>0 else a.copy()

def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else np.nan

def get_mapping(case, sid, g, init, nx, nz, geometry_rows):
    run=PROJECT/'runs'/case/sid
    act=np.array(g.export_actnum(),dtype=np.int8)
    sat=np.array(init['SATNUM'][0])
    key=(case,nx,nz,sha(run/'deck/centers.txt'),sha(run/'deck/corners.txt'),
         hashlib.sha256(act.tobytes()).hexdigest(),hashlib.sha256(sat.tobytes()).hexdigest())
    if key not in CACHE:
        dig={'path':str(run), 'case':case,'use':'resdata', 'init':init,
             'dims':[2.8,1.,1.2] if case=='spe11a' else [8400.,1.,1200.],
             'nxyz':[nx,1,nz], 'nocellsr':nx*nz,'noxzr':nx*nz,
             'noxz':g.nx*g.nz,'gxyz':[g.nx,g.ny,g.nz],
             'actind':np.flatnonzero(act).tolist(),'dense_t':[],'times':[],'mode':'audit_mapping'}
        dil_holder={}
        old=original.get_corners
        def capture(d,dil):
            old(d,dil); dil_holder['dil']=dil
        original.get_corners=capture
        try:
            original.dense_data(dig)
        finally: original.get_corners=old
        ids=np.asarray(dil_holder['dil']['cell_cent'],dtype=int)
        # writer idc = -nx*(nz-idz)+idx is identical to this flat index modulo N.
        assert np.array_equal(np.arange(nx*nz),np.array([-nx*(nz-k)+i for k in range(nz) for i in range(nx)])%(nx*nz))
        centers=np.loadtxt(run/'deck/centers.txt',delimiter=',')
        ai=np.flatnonzero(act)
        xyz=np.array([g.get_xyz(global_index=int(i)) for i in ai])
        center_error=float(np.max(np.abs(xyz-centers[ai])))
        assert center_error < (1e-6 if case=='spe11a' else .01), center_error
        CACHE[key]={'ids':ids, 'act':act, 'center_error':center_error,
                    'key':list(key), 'inactive_reporting':len(dig['actindr'])}
        print('MAPPING',case,nx,nz,'EGRID-center max error',center_error,flush=True)
    m=CACHE[key]
    geometry_rows.append({'case':case,'scenario_id':sid,'mapping_key':hashlib.sha256(str(key).encode()).hexdigest(),
                         'native_nx':g.nx,'native_ny':g.ny,'native_nz':g.nz,'native_active_cells':int(act.sum()),
                         'report_nx':nx,'report_nz':nz,'report_active_native_centers':int(np.unique(m['ids'][act[m['ids']]>0]).size),
                         'EGRID_deck_center_max_abs_error_m':m['center_error']})
    return m

def extract(case,sid,threshold_rows,pressure_rows,checks,geometry_rows,provenance):
    run=PROJECT/'runs'/case/sid
    npzpath=PROJECT/'ai_dataset_npz_207'/case/f'{sid}.npz'
    with np.load(npzpath) as z:
        saved={k:z[k] for k in z.files}
    times=saved['times'].astype(float)
    xs=saved['x'].astype(float); zs=saved['z'].astype(float)
    nx,nz=len(xs),len(zs)
    rpath=run/'flow'/f'{sid}.UNRST'
    rst=ResdataFile(str(rpath)); init=ResdataFile(str(run/'flow'/f'{sid}.INIT'))
    g=Grid(str(run/'flow'/f'{sid}.EGRID'))
    mapping=get_mapping(case,sid,g,init,nx,nz,geometry_rows)
    ids=mapping['ids']; act=mapping['act']; ai=np.flatnonzero(act)
    a_lines=(run/'deck/dt.txt').read_text().splitlines()
    time_initial=float(a_lines[0]); skip=int(a_lines[1]); deck_times=np.fromstring(a_lines[2],sep=' ')
    factor=3600. if case=='spe11a' else 31536000.
    rst_indices=[]
    raw_sg=[]; raw_p=[]; gas_p=[]; csv_sg=[]; csv_p=[]; csv_x=[]
    signed_negative=0
    max_time_error=0.
    for t in times:
        hit=np.flatnonzero(np.isclose(deck_times,t*factor,rtol=0,atol=1e-4))
        assert len(hit)==1,(case,sid,t,hit)
        n=int(hit[0])+skip
        rt=float(np.array(rst['DOUBHEAD'][n])[0])*86400.-time_initial
        max_time_error=max(max_time_error,abs(rt-t*factor))
        assert abs(rt-t*factor)<max(1e-4,factor*1e-8),(case,sid,t,rt)
        rst_indices.append(n)
        sg=np.array(rst['SGAS'][n]); p=np.array(rst['PRESSURE'][n]); signed_negative+=int((sg<0).sum())
        pg=p.copy()*1e5
        if rst.has_kw('PCGW'): p-=np.array(rst['PCGW'][n])
        p*=1e5
        def report(a):
            full=np.full(len(act),np.nan if case=='spe11a' else 0.)
            full[ai]=a
            return full[ids].reshape(nz,nx)
        raw_sg.append(report(sg)); raw_p.append(report(p)); gas_p.append(report(pg))
        suffix='h' if case=='spe11a' else 'y'
        f=run/'data'/f'{case}_spatial_map_{t:g}{suffix}.csv'
        df=pd.read_csv(f); df.columns=[c.strip().lstrip('#').strip() for c in df.columns]
        assert len(df)==nx*nz and not df.duplicated(['x [m]','z [m]']).any()
        df=df.sort_values(['z [m]','x [m]'])
        # NPZ coordinates are float32 copies of CSV coordinates.
        assert np.array_equal(np.sort(df['x [m]'].unique()).astype(np.float32),saved['x'])
        assert np.array_equal(np.sort(df['z [m]'].unique()).astype(np.float32),saved['z'])
        csv_sg.append(pd.to_numeric(df['gas saturation [-]'],errors='coerce').to_numpy().reshape(nz,nx))
        csv_p.append(pd.to_numeric(df['pressure [Pa]'],errors='coerce').to_numpy().reshape(nz,nx))
        csv_x.append(pd.to_numeric(df['mass fraction of CO2 in liquid [-]'],errors='coerce').fillna(0).to_numpy().reshape(nz,nx))
    raw_sg=np.stack(raw_sg); raw_p=np.stack(raw_p); gas_p=np.stack(gas_p)
    csv_sg=np.stack(csv_sg); csv_p=np.stack(csv_p); csv_x=np.stack(csv_x)
    finite=np.isfinite(raw_p)
    assert np.array_equal(finite,np.isfinite(csv_p)),(case,sid,'finite mismatch')
    assert np.array_equal(np.isfinite(raw_sg),np.isfinite(csv_sg))
    assert signed_negative==0,(case,sid,'negative SGAS: signed/abs definitions differ')
    predicted_sg=rounded(np.abs(raw_sg)*(np.abs(raw_sg)>.097))
    predicted_p=rounded(raw_p)
    sg_mismatch=int(np.sum(predicted_sg[finite]!=csv_sg[finite]))
    p_mismatch=int(np.sum(predicted_p[finite]!=csv_p[finite]))
    assert sg_mismatch==0 and p_mismatch==0,(case,sid,'CSV reproduction',sg_mismatch,p_mismatch)
    sg0=np.nan_to_num(csv_sg,nan=0.).astype(np.float32)
    p0=np.nan_to_num(csv_p,nan=0.).astype(np.float32)
    csv_ds=np.clip(sg0-sg0[0:1],0,None)
    csv_dp=p0-p0[0:1]
    oldmask=saved['plume_mask'].astype(bool)
    reconstructed=csv_ds>np.float32(.001); reconstructed[0]=False
    mask_mismatch=int((oldmask!=reconstructed).sum())
    assert mask_mismatch==0 and np.array_equal(csv_ds,saved['delta_sgas'])
    assert np.array_equal(csv_dp,saved['delta_pressure'])
    assert np.array_equal(saved['X'][:,0],csv_dp) and np.array_equal(saved['X'][:,1],csv_ds)
    assert np.array_equal(saved['X'][:,2],csv_x.astype(np.float32))
    ds=np.maximum(raw_sg-raw_sg[0:1],0)
    dp=raw_p-raw_p[0:1]
    cellarea=(2.8/nx*1.2/nz) if case=='spe11a' else (8400./nx*1200./nz)
    for j,t in enumerate(times):
        valid=finite[j]
        old=oldmask[j]&valid
        for th in THRESHOLDS:
            new=(ds[j]>th)&valid
            nr=int(new.sum()); no=int(old.sum()); inter=int((new&old).sum()); union=int((new|old).sum())
            missing=int((new&~old).sum()); extra=int((old&~new).sum())
            threshold_rows.append({'case':case,'scenario_id':sid,'time':t,'time_unit':'h' if case=='spe11a' else 'y',
              'restart_index':rst_indices[j],'threshold':th,'valid_reporting_pixels':int(valid.sum()),'cell_area_m2':cellarea,
              'raw_plume_pixels':nr,'stored_plume_pixels':no,'raw_plume_area_m2':nr*cellarea,'stored_plume_area_m2':no*cellarea,
              'intersection_pixels':inter,'union_pixels':union,'mask_iou':fraction(inter,union),
              'empty_both':int(union==0),'missing_raw_pixels':missing,'missing_raw_pixel_fraction':fraction(missing,nr),
              'stored_extra_pixels':extra,'stored_extra_pixel_fraction':fraction(extra,no),
              'stored_area_relative_bias':fraction(no-nr,nr)})
        rp=raw_p[j][valid]; cp=csv_p[j][valid]; rd=dp[j][valid]
        cd=csv_p[j][valid]-csv_p[0][valid]; nd=saved['delta_pressure'][j][valid].astype(float)
        pe=cp-rp; de=cd-rd; ne=nd-rd
        rn=norm(rd); cn=norm(cd); nn=norm(saved['delta_pressure'][j])[valid].astype(float)
        # Reproduce actual float32 normalization used by final 11-channel dataset.
        physical_nonzero=np.abs(rd)>1.
        lost=physical_nonzero&(cd==0)
        row={'case':case,'scenario_id':sid,'time':t,'time_unit':'h' if case=='spe11a' else 'y','restart_index':rst_indices[j],
          'valid_reporting_pixels':int(valid.sum()),'native_unique_pressure':int(np.unique(np.array(rst['PRESSURE'][rst_indices[j]])).size),
          'unique_raw_pressure':int(np.unique(rp).size),'unique_csv_pressure':int(np.unique(cp).size),
          'unique_npz_pressure':int(np.unique(p0[j][valid]).size),'unique_raw_delta_pressure':int(np.unique(rd).size),
          'unique_csv_delta_pressure':int(np.unique(cd).size),'unique_npz_delta_pressure':int(np.unique(nd).size),
          'pressure_mae_Pa':float(np.abs(pe).mean()),'pressure_rmse_Pa':float(np.sqrt(np.mean(pe**2))),
          'pressure_max_abs_error_Pa':float(np.abs(pe).max()),
          'delta_pressure_mae_Pa':float(np.abs(de).mean()),'delta_pressure_rmse_Pa':float(np.sqrt(np.mean(de**2))),
          'delta_pressure_max_abs_error_Pa':float(np.abs(de).max()),
          'npz_delta_pressure_rmse_Pa':float(np.sqrt(np.mean(ne**2))),
          'csv_to_npz_delta_max_abs_error_Pa':float(np.max(np.abs(nd-cd))),
          'delta_pressure_rmse_over_raw_maxabs':fraction(np.sqrt(np.mean(de**2)),np.abs(rd).max()),
          'delta_spatial_pearson':corr(rd,cd),'delta_std_csv_over_raw':fraction(np.std(cd),np.std(rd)),
          'csv_zero_delta_fraction':float(np.mean(cd==0)),'raw_zero_delta_fraction_1e8Pa':float(np.mean(np.abs(rd)<=1e-8)),
          'raw_gt1Pa_pixels':int(physical_nonzero.sum()),'lost_gt1Pa_pixels':int(lost.sum()),
          'lost_gt1Pa_fraction':fraction(lost.sum(),physical_nonzero.sum()),
          'raw_maxabs_delta_pressure_Pa':float(np.abs(rd).max()),'csv_maxabs_delta_pressure_Pa':float(np.abs(cd).max()),
          'normalized_delta_rmse':float(np.sqrt(np.mean((cn-rn)**2))),
          'normalized_npz_vs_raw_rmse':float(np.sqrt(np.mean((nn-rn)**2))),
          'normalized_delta_max_abs_error':float(np.max(np.abs(cn-rn))),
          'unique_normalized_raw_delta':int(np.unique(rn).size),'unique_normalized_csv_delta':int(np.unique(cn).size),
          'PCGW_subtraction_max_Pa':float(np.max(np.abs(gas_p[j][valid]-rp))),
          'csv_format_quantum_min_Pa':float(np.min(10.**(np.floor(np.log10(np.abs(rp)))-3))),
          'csv_format_quantum_max_Pa':float(np.max(10.**(np.floor(np.log10(np.abs(rp)))-3))),
          'xco2_image_max_removed':float(np.max(np.abs(saved['X'][j,2])))}
        for arr,pre in [(rp,'raw_pressure_'),(cp,'csv_pressure_'),(rd,'raw_delta_'),(cd,'csv_delta_'),(rn,'raw_norm_'),(cn,'csv_norm_')]: row.update(stats(arr,pre))
        pressure_rows.append(row)
    checks.append({'case':case,'scenario_id':sid,'csv_sgas_mismatch_pixels':sg_mismatch,'csv_pressure_mismatch_pixels':p_mismatch,
                   'npz_mask_mismatch_pixels':mask_mismatch,'restart_max_time_error_s':max_time_error,
                   'initial_raw_sgas_max':float(np.nanmax(raw_sg[0])), 'signed_negative_sgas_values':signed_negative,
                   'raw_pressure_nan_pixels':int((~finite[0]).sum()),'npz_active_mask_pixels':int(saved['active_mask'].sum()),
                   'raw_finite_pixels':int(finite[0].sum()),'restart_indices':json.dumps(rst_indices)})
    for f in [rpath,run/'flow'/f'{sid}.EGRID',run/'flow'/f'{sid}.INIT',npzpath,run/'deck/dt.txt']:
        provenance.append({'path':str(f),'size_bytes':f.stat().st_size,'sha256':sha(f)})
    rst.close(); init.close()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pilot',action='store_true'); args=parser.parse_args()
    start=time.time()
    mat=pd.read_csv(PROJECT/'scenario_matrix_valid_207.csv')
    sids=mat.scenario_id.tolist()
    assert len(sids)==207 and len(set(sids))==207
    sids=[s for s in ['S000','S188'] if s in sids]+[s for s in sids if s not in ['S000','S188']]
    if args.pilot: sids=sids[:2]
    tr=[]; pr=[]; checks=[]; geometries=[]; prov=[]; failures=[]
    for case in ['spe11a','spe11b']:
        for k,sid in enumerate(sids):
            try: extract(case,sid,tr,pr,checks,geometries,prov)
            except Exception as e:
                failures.append({'case':case,'scenario_id':sid,'error':repr(e),'traceback':traceback.format_exc()})
                print('FAIL',case,sid,repr(e),flush=True)
                if args.pilot: traceback.print_exc()
            if k<2 or (k+1)%25==0: print('PROGRESS',case,k+1,'/',len(sids),'elapsed_s',round(time.time()-start),flush=True)
    prefix='pilot_' if args.pilot else ''
    for name,rows in [('threshold_comparison',tr),('pressure_precision',pr),('reproduction_checks',checks),('geometry_checks',geometries),('input_provenance',prov)]:
        pd.DataFrame(rows).to_csv(OUT/f'{prefix}{name}.csv',index=False)
    (OUT/f'{prefix}failures.json').write_text(json.dumps(failures,indent=2),encoding='utf8')
    sources=[SOURCE,PROJECT/'build_crossscale_npz_dataset_207.py',PROJECT/'train_unet_monitoring_207.py',
             PROJECT/'v2_analysis/03_channel_ablation.py',PROJECT/'v2_analysis/05_final_spx_multiseed.py',
             PROJECT/'v2_analysis/06_finalize_six_seed_summary.py',PROJECT/'scenario_matrix_valid_207.csv']
    evidence={}
    for p in sources:
        text=p.read_text(); lines=text.splitlines()
        evidence[str(p)]={'sha256':sha(p),'line_count':len(lines),'relevant_lines':
          [{'line':i+1,'text':s} for i,s in enumerate(lines) if any(q in s for q in ['SGAS_THR','sgas_array','PCGW','PRESSURE','cell_cent','argmin','delta_sgas','delta_pressure','plume_mask','Xall[i','yall[j','vmax','a / vmax','tau_i','tau_j','dtau','in_ch','dyn_channels=[','for i in range','for j in range','active_mask ='])]}
    (OUT/f'{prefix}code_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    env={'python':sys.executable,'version':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
         'project':str(PROJECT),'source':str(SOURCE),'requested_scenarios_per_domain':len(sids),'successful_domain_scenario_pairs':len(checks),
         'failures':len(failures),'elapsed_seconds':time.time()-start,'mapping_cache_entries':len(CACHE),'pilot':args.pilot}
    (OUT/f'{prefix}run_summary.json').write_text(json.dumps(env,indent=2),encoding='utf8')
    print(json.dumps(env,indent=2),flush=True)
    if failures: sys.exit(2)
if __name__=='__main__': main()
