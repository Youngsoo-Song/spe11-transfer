# Corrected SPE11 primary experiment results

2026-11-16 Asia/Seoul. Corrected primary raw ΔSg>0.001 결과다. Legacy0.097은 별도 열로 비교했다. Figure와 원고는 수정하지 않았다.

## 핵심 결과

Corrected persistence IoU **0.563166**,B zero-shot **0.423838 ± 0.031184**,scratch **0.935035 ± 0.002761**,fine-tuning **0.941166 ± 0.002436**다. ±는6개 seed 평균 IoU의 sample SD이며 예측 신뢰구간이 아니다.

Fine-tune−scratch paired gain **+0.006130 ± 0.003981**,양수 seed **5/6**다. Seed44의 평균 차이 −0.000079는 수치상 사실상 동률에 가깝지만, 엄밀히는 기존 6/6 양수 주장에 해당하지 않는다. Seed42 positive scenario 36/41,negative 5/41이다. 전체 negative seed–scenario는 61/246,6-seed 평균 gain이 음수인 scenario는 0/41이다.

유지 가능: 6개 seed 모두 corrected persistence보다 낮다.

평균 개선은 있으나 seed별 방향이 달라 일관된 fine-tuning 우위 주장을 축소한다.

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
| 11 | 0.944751 | 0.418610 | 0.938663 | 0.941520 | +0.002857 | 29/41 | 12/41 |
| 22 | 0.940116 | 0.384267 | 0.930961 | 0.939638 | +0.008677 | 35/41 | 6/41 |
| 33 | 0.938898 | 0.455706 | 0.933753 | 0.943447 | +0.009695 | 35/41 | 6/41 |
| 42 | 0.949213 | 0.426444 | 0.933788 | 0.943201 | +0.009413 | 36/41 | 5/41 |
| 44 | 0.948149 | 0.395884 | 0.937131 | 0.937052 | -0.000079 | 20/41 | 21/41 |
| 55 | 0.950714 | 0.462117 | 0.935916 | 0.942136 | +0.006220 | 30/41 | 11/41 |

고정 B test41개×10 pair=410개/seed다. A test도410개/seed다. 불리한 seed를 제외하거나 epoch를 늘리지 않았다. Seed42는 train/validation-only full25/30/30 pipeline gate를 통과했고6-seed 집계에 한 번만 포함했다.

## Legacy 대 corrected

| method | legacy mean | corrected mean |
|---|---:|---:|
| A_test | 0.946857 | 0.945307 |
| B_zero | 0.452625 | 0.423838 |
| B_scratch | 0.929181 | 0.935035 |
| B_finetuned | 0.938024 | 0.941166 |
| paired_gain | 0.008842 | 0.006130 |
| B_persistence | 0.547309 | 0.563166 |

Legacy는 effective0.097 label·clipped/rounded inputs,corrected는0.001 label·raw SGAS/pressure다. **Truth mask도 다르므로 cross-version IoU 차이를 동일 표적에서의 교정 효과라고 해석하지 않는다.** 각 버전 내부 paired gain을 따로 보고한다. Legacy persistence도 old masks로 별도 계산했다. Legacy IoU를 corrected 결과로 재사용하지 않았다.

## Scenario와 negative transfer

target_scenario_gains.csv에246개 seed–scenario의10 pair 평균,target_scenario_gain_across_seeds.csv에41개 scenario의 seed 평균·SD·음수 seed 수를 저장했다. 모든 음수 사례는 negative_transfer_cases.csv다. 아래는 gain이 가장 낮은8개를 기계적으로 정렬했다.

| seed | scenario | scratch | fine-tuned | gain |
|---:|---|---:|---:|---:|
| 44 | S099 | 0.938043 | 0.911949 | -0.026094 |
| 33 | S085 | 0.936729 | 0.917281 | -0.019448 |
| 44 | S155 | 0.943816 | 0.927982 | -0.015834 |
| 44 | S028 | 0.943222 | 0.929633 | -0.013589 |
| 44 | S024 | 0.904952 | 0.891960 | -0.012991 |
| 44 | S051 | 0.913487 | 0.901016 | -0.012472 |
| 44 | S175 | 0.941860 | 0.930701 | -0.011160 |
| 55 | S037 | 0.957766 | 0.947212 | -0.010554 |

paired_forecast_gains.csv는2,460개 직접 paired gain이다. 반복 pair를 독립 geology나 random seed로 간주하지 않았다. Corrected N5/10 학습은 이번 N33 primary에 포함되지 않아 legacy low-data gain을 그대로 유지할 수 없다.

## Test 격리와 한계

PROTOCOL.json·ANALYSIS_PLAN.json을 test 전에 기록했다. 모든18개 checkpoint hash를 MODELS_FROZEN_BEFORE_TEST.json에 고정한 후 한 번 test 평가했다. Model selection·pipeline gate·오류 수정에 test score를 쓰지 않았다. 6개 seed 학습 실행시간 합계는 1.95시간이다.

Legacy 당시 PyTorch 버전은 로그에 없어 역사적 software stack의 완전 동일성을 증명할 수 없다. 이번에는 현재 기존 CPU 환경과 기본 thread·학습 코드·hyperparameter·seed 난수 순서를 유지했다. 데이터 교정으로 loss와 selected epoch가 바뀌는 것은 결과다.

이번에는 primary6-seed만 실행했다. Scaling ablation,sensitivity retraining,low-data,새 seed·architecture를 추가하지 않았다. 필요한 다음 선택은 NEXT_DECISION.md에 기록하고 실행을 멈춘다.
