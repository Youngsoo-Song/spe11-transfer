# Corrected N=5/10 low-data 최종 결과

사전에 고정한 N=5/10 × 기존 subset draw 5개 × scratch/fine-tuning의 **20개 학습, 600 epoch**와 고정 B test 평가를 정상 완료했다. 이 작업을 마지막으로 추가 AI 학습을 종료한다. 날짜/작업 ID: 20261116.

## 1. 복원한 조건과 변경하지 않은 사항

- 기존 `v2_analysis/08_lowdata_subset_robustness.py`와 10개 `training_scenarios.csv`를 확인했다. CSV를 byte-for-byte 복사했으며 시나리오 순서와 기존 NumPy RNG 생성 방식도 일치했다. 복원하지 못해 새로 추첨한 subset은 없다.
- Train pool 33개, subset seed 101/202/303/404/505를 사용했다. 각 draw에서 N=5는 N=10의 prefix다. 모델 seed는 모든 방법에서 42다. 이 5개 draw는 기존 `07` 실험의 단일 subset seed 2026과 별개다.
- 각 방법의 모델 생성 직전에 Python/NumPy/Torch seed를 각각 42로 초기화했다. Fine-tuning만 corrected primary의 A-only source seed 42 checkpoint를 불러왔다. B=33 fine-tuned 가중치는 사용하지 않았다. 대응하는 두 방법의 학습 시작 Torch RNG hash가 모든 subset에서 같았다.
- Source는 A validation으로 선택된 epoch 24 checkpoint다. SHA256: `44216fd5fa7c8903e4c0c7b39082b6271c0f9d59f10800d6337c1b6ed999b26a`. 이번 결과를 보고 6개 source 중 선택하지 않았다. 추가 source 학습은 없다.
- 11채널 SmallUNet(base 16), Adam lr=0.001, batch=4, workers=0, BCE+soft-Dice, 30 epochs, validation 최고 IoU(strict >) checkpoint, sigmoid 출력 threshold 0.5를 유지했다. ΔSg는 unscaled, pressure/XCO2는 관측 이미지별 max-absolute normalization을 그대로 사용했다.
- 원본 UNRST에서 만든 corrected raw ΔSg > 0.001 데이터셋과 기존 reporting grid/관측 시점/split을 재사용했다. Pressure 반올림 손실을 교정했으며 XCO2는 primary와 동일하게 기존 표현을 유지했다. 0.01/0.05/0.097 표적에 대한 추가 모델 학습은 하지 않았다.
- CPU torch 2.12.1+cpu, 12 threads로 실행했다. 설치, AMP, compile 또는 epoch 축소는 없었다. 역사적 legacy PyTorch 버전은 기록이 없어 그 부분까지 완전히 동일하다고 인증할 수는 없다.

원래 `08`의 N=20 loop는 실행하지 않았다. 방법별 seed reset을 복원했으므로 N=20 생략과 test 평가를 뒤로 미룬 것이 다음 N=5/10 학습의 RNG 시작 상태를 바꾸지 않는다. 반면 corrected primary의 per-seed 연속 RNG 흐름과 이번 방법별 reset은 다르며 이를 숨기지 않는다.

## 2. Test 격리와 집계 단위

B validation 25개(250 forecast pairs), final test 41개(410 pairs)를 유지했다. 20개 validation-selected checkpoint를 모두 freeze한 뒤 test를 평가했다. Test 결과로 subset, seed, hyperparameter, checkpoint를 바꾸지 않았다. 이 test는 legacy 및 corrected primary와 동일한 benchmark holdout이며 새로 확보한 외부 독립 검증 세트는 아니다.

각 scenario의 10개 forecast-pair IoU를 평균하고 41개 scenario를 평균했다. Pair 수가 같으므로 410개 평균과 같다. Paired gain은 동일 subset/시나리오/관측-예측 시점의 fine-tuning − scratch다.

## 3. 전체 subset 결과

| Replicate | Subset seed | N | Scratch IoU | Fine-tuning IoU | Paired gain | Negative scenarios |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 101 | 5 | 0.779789 | 0.857140 | +0.077351 | 3/41 |
| 1 | 101 | 10 | 0.881102 | 0.901552 | +0.020450 | 6/41 |
| 2 | 202 | 5 | 0.824668 | 0.888897 | +0.064229 | 0/41 |
| 2 | 202 | 10 | 0.896149 | 0.914923 | +0.018774 | 5/41 |
| 3 | 303 | 5 | 0.808832 | 0.863322 | +0.054490 | 6/41 |
| 3 | 303 | 10 | 0.877868 | 0.906195 | +0.028328 | 1/41 |
| 4 | 404 | 5 | 0.779119 | 0.877266 | +0.098147 | 1/41 |
| 4 | 404 | 10 | 0.885296 | 0.914840 | +0.029545 | 1/41 |
| 5 | 505 | 5 | 0.769855 | 0.825834 | +0.055979 | 8/41 |
| 5 | 505 | 10 | 0.875526 | 0.892673 | +0.017147 | 11/41 |

## 4. 다섯 subset의 평균과 표준편차

| N | Scratch IoU | Fine-tuning IoU | Paired gain | Positive subsets | N=33 gain 대비 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.792452 ± 0.023210 | 0.862492 ± 0.023920 | +0.070039 ± 0.018144 | 5/5 | 11.42 |
| 10 | 0.883188 ± 0.008121 | 0.906037 ± 0.009423 | +0.022849 ± 0.005695 | 5/5 | 3.73 |

±는 서로 겹치는 5개 subset 사이의 sample SD(ddof=1)다. 모델 seed 간 불확실성, calibrated predictive uncertainty 또는 신뢰구간이 아니다. Source/model seed는 42 하나이며, N=5/10은 draw 안에서 nested이므로 10개 독립 반복으로 합치지 않는다.

두 N 모두 5/5 draw에서 평균 이득이 양수다. N=5의 이득 범위는 +0.054490~+0.098147, N=10은 +0.017147~+0.029545다. N=33 corrected primary의 6-seed 평균 +0.006130보다 각각 약 11.42배와 3.73배 크다. N=33 seed 42만의 gain은 +0.009413이다. N=33과 이번 실험은 randomization 설계가 달라 배율은 설명용 reference이며 완전한 matched learning curve가 아니다.

N마다 30 epochs라도 gradient update 수는 다르다. 이 설계는 고정 epoch protocol에서의 초기화 이득을 보여주며 순수 sample complexity와 최적화 진전도를 분리하지 않는다. 더 오래 학습한 scratch와도 같은 격차가 유지되는지, 동일 정확도에 필요한 정확한 데이터 감소율은 입증하지 않았다.

## 5. Legacy와 corrected의 구분

| N | Legacy scratch | Legacy fine-tuning | Legacy 5-draw gain | Corrected 5-draw gain |
|---:|---:|---:|---:|---:|
| 5 | 0.785567 | 0.844846 | +0.059279 ± 0.046168 | +0.070039 ± 0.018144 |
| 10 | 0.874345 | 0.898371 | +0.024026 ± 0.010768 | +0.022849 ± 0.005695 |

위 legacy는 동일한 saved subset을 사용한 `08` 결과다. 기존 논문의 단일 seed-2026 learning-curve headline(+0.0562/+0.0624)과 혼동하지 않는다. 단일 draw는 `legacy_single2026_reference_only.csv`로 분리했다.

Legacy는 clipping으로 사실상 0.097인 표적과 기존 입력/source를, corrected는 raw 0.001 표적과 교정 입력/source를 사용한다. 따라서 버전 간 IoU 차이는 같은 정답에 대한 순수 전처리 효과가 아니다. Legacy IoU를 corrected 결과로 재사용하지 않았다.

## 6. Negative transfer

Scenario 평균 gain이 0 미만인 경우를 negative transfer로 정의했다. N=5는 18/205개, N=10은 24/205개이며 총 42/410개다. 같은 41개 scenario가 draw/N에 걸쳐 반복되므로 410개 독립 지질 사례라는 뜻은 아니다.

가장 큰 감소는 replicate 5, N=10, S188: scratch 0.835566 → fine-tuning 0.709796, gain −0.125770이다. 전체 410개 scenario-level 결과는 `scenario_gains.csv`, 음수 42개 전부는 `negative_transfer_cases.csv`, 4,100개 forecast-pair gain은 `paired_forecast_gains.csv`에 보존했다. 평균 개선이 개별 시나리오의 개선을 보장하지 않는다.

## 7. 학습 비용과 label budget

20개 fit의 wall-time 합계는 1619.7초(26.99분), process CPU-time 합계는 19426.6초(5.396 CPU-hours)다. CPU-time은 여러 thread의 사용시간 합으로 wall-time과 다르다. Evaluation 시간과 기존 source 사전학습 비용은 이 20-fit 합계에 포함하지 않았다. 개별 비용·선택 epoch는 `training_costs.csv`에 기록했다.

Source seed 42는 A train 141개와 A validation 25개를 사용했다. 기존 primary source fit 비용 737.8초는 이번에는 재지출하지 않았지만 무료로 간주할 수 없다.

**N은 target weight update에 사용한 시나리오 수다.** B validation 25개도 labelled target data다. 개발 label budget은 N=5에서 30개, N=10에서 35개이며 test 41개는 별도 benchmark 평가 자료다. 이미 생성된 207쌍 ensemble 안에서 subset을 비교했으므로 'target simulation 5개만 필요하다' 또는 총 simulation/label 비용 절감률을 증명한 결과가 아니다.

## 8. 종료와 판정

**조건부 Go:** 고정 validation/학습 protocol에서 제한된 target 학습 시나리오를 사용할 때 source initialization의 반복 이득을 핵심 결과로 유지할 수 있다. 보편적 data efficiency, 총비용 절감, 물리적으로 보존적인 현장 surrogate 주장은 유지하지 않는다.

Primary와 통합한 유지/철회 claim, 미검증 사항, 필수 원고 수정 및 JESTECH rejection risk는 `FINAL_SUBMISSION_DECISION.md`에 정리했다. 추가 threshold/normalization/다른 N/seed/model/geometry/noise/UQ 학습은 하지 않는다. 기존 dataset/checkpoint/원고/Figure는 덮어쓰지 않았다.
