#!/usr/bin/env python3
import json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;P=R.parent

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 assert (R/'TRAIN_AND_TEST_COMPLETE.json').exists()
 assert not (R/'RESULTS_COMPLETE.json').exists()
 m=pd.read_csv(R/'primary_metrics_all.csv');pers=pd.read_csv(R/'persistence_metrics.csv')
 assert len(m)==9840 and len(pers)==410
 seeds=sorted(int(x) for x in m.seed.unique());assert seeds==[11,22,33,42,44,55]
 summary=m.groupby(['seed','eval_tag'])[['iou','dice','area_rel_error']].mean().reset_index()
 pivot=summary.pivot(index='seed',columns='eval_tag',values='iou').reset_index()
 pivot['B_persistence']=pers.iou.mean();pivot['paired_gain']=pivot.B_finetuned-pivot.B_scratch
 pivot['zero_minus_persistence']=pivot.B_zero-pivot.B_persistence
 summary.to_csv(R/'method_seed_summary.csv',index=False)
 keys=['seed','scenario_id','i','j','horizon']
 ft=m[m.eval_tag=='B_finetuned'][keys+['iou','dice','area_rel_error']]
 sc=m[m.eval_tag=='B_scratch'][keys+['iou','dice','area_rel_error']]
 paired=ft.merge(sc,on=keys,suffixes=('_finetuned','_scratch'),validate='one_to_one');assert len(paired)==2460
 paired['iou_gain']=paired.iou_finetuned-paired.iou_scratch
 paired['dice_gain']=paired.dice_finetuned-paired.dice_scratch
 paired['area_error_reduction']=paired.area_rel_error_scratch-paired.area_rel_error_finetuned
 paired.to_csv(R/'paired_forecast_gains.csv',index=False)
 scenario=paired.groupby(['seed','scenario_id']).agg(finetuned_iou=('iou_finetuned','mean'),scratch_iou=('iou_scratch','mean'),
   iou_gain=('iou_gain','mean'),dice_gain=('dice_gain','mean'),area_error_reduction=('area_error_reduction','mean'),pairs=('iou_gain','size')).reset_index()
 assert len(scenario)==246 and scenario.pairs.eq(10).all()
 scenario['negative_transfer']=scenario.iou_gain<0
 scenario.to_csv(R/'target_scenario_gains.csv',index=False)
 scenario[scenario.negative_transfer].sort_values('iou_gain').to_csv(R/'negative_transfer_cases.csv',index=False)
 combined=scenario.groupby('scenario_id').agg(mean_gain=('iou_gain','mean'),sd_gain=('iou_gain','std'),min_gain=('iou_gain','min'),
  max_gain=('iou_gain','max'),negative_seed_count=('negative_transfer','sum'),mean_finetuned_iou=('finetuned_iou','mean'),mean_scratch_iou=('scratch_iou','mean')).reset_index()
 assert len(combined)==41;combined.to_csv(R/'target_scenario_gain_across_seeds.csv',index=False)
 counts=scenario.groupby('seed').agg(positive_scenarios=('iou_gain',lambda a:int((a>0).sum())),negative_scenarios=('negative_transfer','sum')).reset_index()
 pivot=pivot.merge(counts,on='seed');pivot.to_csv(R/'seed_results.csv',index=False)
 legacy=pd.read_csv(P/'v2_analysis/tables/final_six_seed_summary.csv')
 mat=pd.read_csv(R/'dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv')
 testids=mat[mat.target_B_role=='zero_shot_and_final_test'].scenario_id.tolist();assert len(testids)==41
 oldp=[]
 for sid in testids:
  with np.load(P/'ai_dataset_npz_207/spe11b'/f'{sid}.npz') as z:mask=z['plume_mask'].astype(bool)
  for i in range(1,5):
   for j in range(i+1,6):
    a,b=mask[i],mask[j];oldp.append(np.logical_and(a,b).sum()/(np.logical_or(a,b).sum()+1e-6))
 oldpers=float(np.mean(oldp));rows=[]
 mapping={'A_test':'A_test_iou','B_zero':'B_zero_iou','B_scratch':'B_scratch_iou','B_finetuned':'B_finetuned_iou','paired_gain':'delta_iou'}
 for seed in seeds:
  old=legacy[legacy.seed==seed].iloc[0];new=pivot[pivot.seed==seed].iloc[0]
  for method,col in mapping.items():
   rows.append({'seed':seed,'method':method,'legacy_value':float(old[col]),'corrected_value':float(new[method]),
    'corrected_minus_legacy':float(new[method]-old[col]),'legacy_label_threshold_effective':.097,'corrected_label_threshold':.001,
    'legacy_input':'CSV rounded pressure, SGAS clipped .097','corrected_input':'UNRST raw SGAS/water pressure, unchanged XCO2 and scaling',
    'interpretation':'Different truth masks across versions; paired gain is calculated within each version'})
  rows.append({'seed':seed,'method':'B_persistence','legacy_value':oldpers,'corrected_value':float(new.B_persistence),
   'corrected_minus_legacy':float(new.B_persistence-oldpers),'legacy_label_threshold_effective':.097,'corrected_label_threshold':.001,
   'legacy_input':'legacy masks','corrected_input':'corrected masks','interpretation':'Nonlearned baseline, different truth masks'})
 comp=pd.DataFrame(rows);comp.to_csv(R/'legacy_vs_corrected.csv',index=False)
 lm=pd.read_csv(P/'v2_analysis/final_spx_multiseed/final_spx_multiseed_metrics.csv')
 l42=pd.read_csv(P/'v2_analysis/channel_ablation/channel_ablation_metrics_all.csv')
 l42=l42[l42.configuration=='SPX_pressure_dsgas_xco2'].copy();l42['seed']=42
 lm=pd.concat([lm,l42],ignore_index=True)
 lm['method']=lm.eval_tag.map(lambda s:'ft' if s.endswith('_B_finetuned') else 'sc' if s.endswith('_B_scratch') else 'other')
 li=lm[lm.method.isin(['ft','sc'])].groupby(['seed','scenario_id','method']).iou.mean().unstack('method').reset_index()
 li['legacy_iou_gain']=li.ft-li.sc
 scenario.merge(li[['seed','scenario_id','legacy_iou_gain']],on=['seed','scenario_id'],validate='one_to_one').to_csv(R/'legacy_vs_corrected_scenario_gain.csv',index=False)
 zero_all=bool((pivot.zero_minus_persistence<0).all());zero_mean=float(pivot.B_zero.mean()-pers.iou.mean())
 gain=float(pivot.paired_gain.mean());gain_sd=float(pivot.paired_gain.std(ddof=1));npos=int((pivot.paired_gain>0).sum())
 dz=('유지 가능: 6개 seed 모두 corrected persistence보다 낮다.' if zero_all else '평균 실패로만 제한: 일부 seed는 persistence보다 낮지 않다.' if zero_mean<0 else '기존 failure claim 유지 불가: corrected 평균 zero-shot이 persistence보다 낮지 않다.')
 df=('평균 개선과 6/6 seed 방향 일치가 확인됐다. 효과 크기를 보고하며 일반적 우월성으로 확대하지 않는다.' if gain>0 and npos==6 else '평균 개선은 있으나 seed별 방향이 달라 일관된 fine-tuning 우위 주장을 축소한다.' if gain>0 else '기존 fine-tuning improvement claim 유지 불가. 평균 이득이 양수가 아니다.')
 stats={c:{'mean':float(pivot[c].mean()),'sd':float(pivot[c].std(ddof=1)),'min':float(pivot[c].min()),'max':float(pivot[c].max())} for c in ['A_test','B_zero','B_scratch','B_finetuned','paired_gain']}
 stats.update(persistence_iou=float(pers.iou.mean()),zero_minus_persistence=zero_mean,positive_gain_seeds=npos,
  negative_seed_scenario_pairs=int(scenario.negative_transfer.sum()),negative_mean_scenarios=int((combined.mean_gain<0).sum()))
 (R/'RESULT_STATISTICS.json').write_text(json.dumps(stats,indent=2))
 tab='\n'.join(f'| {int(r.seed)} | {r.A_test:.6f} | {r.B_zero:.6f} | {r.B_scratch:.6f} | {r.B_finetuned:.6f} | {r.paired_gain:+.6f} | {int(r.positive_scenarios)}/41 | {int(r.negative_scenarios)}/41 |' for r in pivot.itertuples())
 ct='\n'.join(f'| {method} | {d.legacy_value.mean():.6f} | {d.corrected_value.mean():.6f} |' for method,d in comp.groupby('method',sort=False))
 worst=scenario.sort_values('iou_gain').head(8)
 wt='\n'.join(f'| {int(r.seed)} | {r.scenario_id} | {r.scratch_iou:.6f} | {r.finetuned_iou:.6f} | {r.iou_gain:+.6f} |' for r in worst.itertuples())
 durations=[json.loads((R/'primary'/f'seed_{s}'/'COMPLETE.json').read_text()) for s in seeds]
 sc42=scenario[scenario.seed==42]
 report=f'''# Corrected SPE11 primary experiment results

2026-11-16 Asia/Seoul. Corrected primary raw ΔSg>0.001 결과다. Legacy0.097은 별도 열로 비교했다. Figure와 원고는 수정하지 않았다.

## 핵심 결과

Corrected persistence IoU **{pers.iou.mean():.6f}**,B zero-shot **{stats['B_zero']['mean']:.6f} ± {stats['B_zero']['sd']:.6f}**,scratch **{stats['B_scratch']['mean']:.6f} ± {stats['B_scratch']['sd']:.6f}**,fine-tuning **{stats['B_finetuned']['mean']:.6f} ± {stats['B_finetuned']['sd']:.6f}**다. ±는6개 seed 평균 IoU의 sample SD이며 예측 신뢰구간이 아니다.

Fine-tune−scratch paired gain **{gain:+.6f} ± {gain_sd:.6f}**,양수 seed **{npos}/6**다. Seed42 positive scenario {int((sc42.iou_gain>0).sum())}/41,negative {int((sc42.iou_gain<0).sum())}/41이다. 전체 negative seed–scenario는 {int(scenario.negative_transfer.sum())}/246,6-seed 평균 gain이 음수인 scenario는 {int((combined.mean_gain<0).sum())}/41이다.

{dz}

{df}

## 데이터와 고정 조건

기존 `/home/yssongvd/opm-ai-env/bin/python`,torch2.12.1+cpu,기본12 threads를 사용했다. CUDA 별도 설치는 certificate chain 오류로 실패했고 사용자가 CPU 진행을 허용했다. 기존 환경은 변경하지 않았다. 공식 설치 명령 확인 출처는 [PyTorch versions](https://pytorch.org/get-started/previous-versions/)다.

원래 SmallUNet(in_ch=11,base=16),Adam lr1e-3,batch4,num_workers0,BCE+soft-Dice,source25/fine-tune30/scratch30 epochs,strict maximum validation IoU checkpoint,sigmoid0.5를 유지했다. AMP,torch.compile,memory-format 변경,thread 최적화는 사용하지 않았다. 임시 속도 측정용 모델은 본 실험 checkpoint가 아니다.

원래 모델/loss/train/evaluate 함수는 AST로 그대로 실행했다. 새 cached loader는 원래 AblationDataset과 train/validation40개 pair에서 input·label이 bitwise 동일했다. Cache는 입력 계산 결과만 보존하며 sample 순서·optimizer를 바꾸지 않는다. Test forward를 보류할 때 원래 unshuffled DataLoader의 RNG 소비를 보존했고 train data에서 난수 상태의 일치를 확인했다.

207쌍,reporting grid,시점,split,11채널과 관측영상별 pressure/XCO2 maxabs scaling을 유지했다. A train/val/test141/25/41,B train/val/final33/25/41이다. 모든10개 i<j monitoring pair를 scenario 단위로 같은 split에 유지했다.

UNRST의 clipping 없는 SGAS와 수상 pressure=(PRESSURE−PCGW)×1e5 Pa를 추출해 raw float64로 보존했다. Delta도 float64에서 먼저 계산한 뒤 학습용 float32로 한 번 변환했다. 비유효 영역은 raw NaN,학습0이다. Primary raw ΔSg>0.001,사전 sensitivity0.01/0.05,legacy0.097 mask를 분리했다.

**수정 범위를 SGAS·압력으로 제한하기 위해 XCO2는 기존 X[:,2]를 정확히 유지했다.** XCO2의 기존 CSV 유한 정밀도와 amplitude 제거는 그대로 남는다. XCO2까지 바꾼 새 실험으로 해석하지 않는다.

414개 NPZ/2,484개 snapshot의 raw lossless roundtrip,completed audit의 threshold별 pixel 수,기존 좌표·시점·split을 전처리 검산했다. 완료된 감사를 반복하거나 OPM을 새로 실행하지 않았다.

## Seed별 결과

| seed | A test | B zero-shot | B scratch | B fine-tune | paired gain | positive scenario | negative scenario |
|---:|---:|---:|---:|---:|---:|---:|---:|
{tab}

고정 B test41개×10 pair=410개/seed다. A test도410개/seed다. 불리한 seed를 제외하거나 epoch를 늘리지 않았다. Seed42는 train/validation-only full25/30/30 pipeline gate를 통과했고6-seed 집계에 한 번만 포함했다.

## Legacy 대 corrected

| method | legacy mean | corrected mean |
|---|---:|---:|
{ct}

Legacy는 effective0.097 label·clipped/rounded inputs,corrected는0.001 label·raw SGAS/pressure다. **Truth mask도 다르므로 cross-version IoU 차이를 동일 표적에서의 교정 효과라고 해석하지 않는다.** 각 버전 내부 paired gain을 따로 보고한다. Legacy persistence도 old masks로 별도 계산했다. Legacy IoU를 corrected 결과로 재사용하지 않았다.

## Scenario와 negative transfer

Target_scenario_gains.csv에246개 seed–scenario의10 pair 평균,target_scenario_gain_across_seeds.csv에41개 scenario의 seed 평균·SD·음수 seed 수를 저장했다. 모든 음수 사례는 negative_transfer_cases.csv다. 아래는 gain이 가장 낮은8개를 기계적으로 정렬했다.

| seed | scenario | scratch | fine-tuned | gain |
|---:|---|---:|---:|---:|
{wt}

Paired_forecast_gains.csv는2,460개 직접 paired gain이다. 반복 pair를 독립 geology나 random seed로 간주하지 않았다. Corrected N5/10 학습은 이번 N33 primary에 포함되지 않아 legacy low-data gain을 그대로 유지할 수 없다.

## Test 격리와 한계

PROTOCOL.json·ANALYSIS_PLAN.json을 test 전에 기록했다. 모든18개 checkpoint hash를 MODELS_FROZEN_BEFORE_TEST.json에 고정한 후 한 번 test 평가했다. Model selection·pipeline gate·오류 수정에 test score를 쓰지 않았다. 6개 seed 학습 실행시간 합계는 {sum(x['elapsed_seconds'] for x in durations)/3600:.2f}시간이다.

Legacy 당시 PyTorch 버전은 로그에 없어 역사적 software stack의 완전 동일성을 증명할 수 없다. 이번에는 현재 기존 CPU 환경과 기본 thread·학습 코드·hyperparameter·seed 난수 순서를 유지했다. 데이터 교정으로 loss와 selected epoch가 바뀌는 것은 결과다.

이번에는 primary6-seed만 실행했다. Scaling ablation,sensitivity retraining,low-data,새 seed·architecture를 추가하지 않았다. 필요한 다음 선택은 NEXT_DECISION.md에 기록하고 실행을 멈춘다.
'''
 (R/'CORRECTED_RESULTS.md').write_text(report,encoding='utf8')
 if zero_mean>=0:
  route='기존 zero-shot failure를 중심 명제로 유지할 수 없어 제목과 논문 질문을 변경해야 한다. Corrected 표현에서의 전이 성능과 초기화 효과로 질문을 좁히고, fine-tuning gain은 위의 실제 수치와 일관성 범위로만 보고한다. Low-data 중심 주장은 corrected 근거가 없으므로 축소한다.'
 elif gain<=0:
  route='사전학습의 성능 향상을 중심 기여로 유지하기 어렵다. 논문 질문을 현재 조건에서의 zero-shot 한계와 target learning의 필요성으로 축소/변경한다. 긍정적인 gain을 찾기 위한 추가 학습은 하지 않는다.'
 else:
  route='Primary의 조건부 평균 이득을 보고하면서 R1-5의 제한된 사전 threshold sensitivity와 범위에 맞는 R1-2 설명/대조를 마친 뒤 재투고하는 방향이 합리적이다. Low-data 중심 제목·초록은 corrected 근거가 없으므로 축소해야 한다.'
 decision=f'''# Corrected primary 이후 판단

2026-11-16. Primary6-seed 학습·평가가 정상 종료됐다. 여기서 실행을 멈추고 사전 정의하지 않은 실험을 추가하지 않는다.

## Zero-shot failure claim

**{dz}**

Zero-shot 평균 {stats['B_zero']['mean']:.6f},corrected persistence {pers.iou.mean():.6f},차이 {zero_mean:+.6f}다. 고정 SPE11 구조·현재11채널·image-max scaling의 조건부 결과이며 scale 단독 인과효과는 아니다.

## Fine-tuning improvement와 low-data claim

**{df}**

Paired gain 평균 {gain:+.6f},seed SD {gain_sd:.6f},양수 {npos}/6이다. 모든 negative scenario를 공개했다. 원래6/6 일관성 문장은 새6개 seed가 지지하는 범위에서만 유지한다.

**Low-data claim은 이번 N33 primary로 유지 확정할 수 없다.** Legacy N5/10 결과는0.097 문제다. 제목·초록 중심에 남기려면 사전 고정 N5/10·subset draw에서 corrected scratch/fine-tune을 별도 실행해야 한다. 그 작업을 하지 않으면 기존 수치·중심 주장을 제거/축소한다.

## R1-2 normalization 대조

이번에는 scaling을 유지해 raw 교정과 normalization 변경을 섞지 않았다. Δp/XCO2 진폭 제거의 한계는 남는다. 질문을 image-based extent transfer의 조건부 비교로 좁히면 Methods·limitation으로 범위를 설명할 수 있다. 동등한 physical input이나 본질적인 physical transfer failure를 주장하려면 고정 물리 reference scaling 대조가 필요하다.

대조의 scale·채널 수·protocol은 독립 물리 설계 또는 train/validation에서 사전 고정하고 seed42 source/scratch/fine-tune 한 세트로 제한할 수 있다. Primary test를 보고 유리한 scale을 탐색하지 않는다. 이번에는 이 대조를 실행하지 않았다.

## R1-5 threshold 재학습

Raw0.01/0.05 label은 생성했지만 모델은0.001로만 학습했다. 후속 sensitivity에서도 입력 SGAS는 clipping 없이 유지하고 label 및 persistence mask에만 해당 threshold를 적용해야 한다. 면적 민감도만으로 AI ranking robustness를 입증하지 않는다. 직접 대응할 후속 범위는 사전0.01/0.05,seed42의 source/scratch/fine-tune 각각 한 세트,총6 jobs(170 model-epochs)까지다. 최소 범위로 한 threshold만 선택한다면 실행 전에 선택을 고정한다. 새로운 threshold·seed·epoch를 결과에 맞춰 늘리지 않는다. 이번에는 threshold 재학습을 실행하지 않았다.

## 재투고 또는 질문 변경

**{route}**

Binary extent,fixed facies,synthetic clean input 범위를 초록부터 결론까지 일관되게 쓴다. New geology,새architecture,DA,noise,UQ를 모두 추가하지 않는다. 기존 결론에 반하는 결과는 그대로 반영한다. Figure·원고는 이번에 변경하지 않았다. 핵심 결과 보고와 함께 추가 실행을 멈춘다.
'''
 (R/'NEXT_DECISION.md').write_text(decision,encoding='utf8')
 prov=pd.read_csv(R/'input_provenance.csv');assert all(sha(Path(x.path))==x.sha256 for x in prov.itertuples())
 checks=pd.read_csv(R/'preprocessing_checks.csv')
 assert all(sha(R/'dataset_v1_raw_sgas_waterpressure'/x.case/f'{x.scenario_id}.npz')==x.npz_sha256 for x in checks.itertuples())
 cfg=json.loads((R/'PROTOCOL.json').read_text()); dc=json.loads((R/'DATASET_COMPLETE.json').read_text())
 assert sha(P/'scenario_matrix_valid_207.csv')==dc['matrix_sha256']
 assert sha(P/'train_unet_monitoring_207.py')==cfg['source_functions_sha256']
 assert sha(R/'train_primary.py')==cfg['script_sha256']
 frozen=json.loads((R/'MODELS_FROZEN_BEFORE_TEST.json').read_text())
 assert all(sha(Path(x['path']))==x['sha256'] for x in frozen['checkpoints'])
 assert sha(R/'PROTOCOL.json')==frozen['protocol_sha256']
 done={'seeds':seeds,'training_jobs':18,'epochs':510,'learned_test_rows':9840,'persistence_rows':410,'paired_forecast_rows':2460,
  'seed_scenario_rows':246,'target_scenarios':41,'input_files_unchanged':2070,'selected_checkpoints_unchanged_after_freeze':18,
  'low_data_trained':False,'sensitivity_trained':False,'complete_unix':time.time()}
 (R/'RESULTS_COMPLETE.json').write_text(json.dumps(done,indent=2))
 print(pivot.to_string(index=False));print(json.dumps(stats,indent=2));print('FINAL REPORTS COMPLETE')
if __name__=='__main__':main()
