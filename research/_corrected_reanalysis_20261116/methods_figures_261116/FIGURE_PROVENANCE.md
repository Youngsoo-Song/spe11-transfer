# Figure 2 and Figure 3 corrected 데이터 출처

작성일: 2026-11-16. MLST Research Paper의 Methods Figure 2·3만 재생성했다. 원고와 Results Figure는 수정하지 않았다.

## 입력 선택 및 기존 코드 조사

- 기존 `v2_analysis/11_generate_manuscript_figures_14.py:88–124`는 INDEX 경유 NPZ를 읽고, 공통 시나리오 S000을 BASE_SID로 우선 선택한다.
- 기존 `v2_analysis/30_make_FINAL_11ch_manuscript_figures_physorient.py:794–1046`의 make_fig02/make_fig03은 `X[:,1]` 및 마지막 저장 plume_mask를 사용한다. Figure 2는 전 시점, Figure 3은 snapshot 5이다.
- 별도 Figure 2 코드 `v2_analysis/33_make_Fig02_typography_matched_physorient.py`도 S000의 X[:,1]을 사용한다. 기존 렌더링은 origin=lower, aspect=auto, 물리 좌표 눈금 제거였다. 더 오래된 colored 코드에는 origin=upper도 남아 있으므로 파일명의 orientation을 근거로 뒤집지 않았다.
- 기존 생성 모듈은 학습/모델 관련 모듈을 최상위에서 import하므로 실행하거나 import하지 않았다. 코드 텍스트만 읽었다.
- 같은 Methods 예시 S000을 유지했다. S000은 성능이나 외관을 보고 새로 선택하지 않았다. 시나리오 조건: q_mult=0.7, kz_rel_mult=0.5, global_perm_mult=0.8, lower_dominant; source_A_split=test, target_B_role=zero_shot_and_final_test이다. 방법 설명용 시각화이며 모델·부분집합 선택에 사용하지 않았다.
- `ai_dataset_npz_207`의 NPZ, clipped saturation, X[:,1], 모델 prediction은 이번 생성에서 읽지 않았다.

## Corrected 파일 및 시간

- 데이터셋: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/dataset_v1_raw_sgas_waterpressure`
- 연속장: NPZ의 float64 `delta_sgas_raw`. `sgas_raw`의 초기값 차이와 정확히 일치하는지 확인했다. 원본 생성식은 max(Sg(t)−Sg(0), 0)이며 이 예시의 Sg(0)는 0이다. 0.097 clipping이나 CSV 반올림을 적용하지 않았다.
- Binary target: NPZ `plume_mask`, 보조 키 `mask_0001`과 동일. Figure 3에는 재구성한 별도 mask 대신 이 저장 mask를 직접 전달했다.
- Figure 2: 두 도메인 모두 snapshot index 0,1,2,3,4,5. SPE11A 시간 0,1,2,3,4,5 h; SPE11B 시간 0,5,10,15,20,25 yr.
- Figure 3: SPE11A snapshot 5 = 5 h; SPE11B snapshot 5 = 25 yr. 연속장·mask·overlay가 같은 시나리오 및 같은 시점이다.
- 두 그림에서 동일 선형 색상 범위 0–0.9를 사용했다. 해당 raw 최댓값은 약 0.899907(A), 0.858079(B)로 범위 안에 있다. 색상 범위는 saturation 데이터나 mask를 잘라내지 않는다.

## 좌표 격자 방향 및 주입 위치

- A: reporting grid H×W=120×280, x∈[0,2.8] m, z∈[0,1.2] m. B: 60×420, x∈[0,8400] m, z∈[0,1200] m; 그림에는 양 축을 모두 km로 환산했다.
- NPZ x/z는 reporting-cell centre이며 오름차순이다. 전체 cell edge는 pyopmspe11 dense_data의 linspace 경계와 같다. float32 저장 중심 좌표와 이상 중심의 허용 오차 내 일치를 확인했다.
- pyopmspe11 `visualization/data.py:840–938`는 `z_elevation=Lz−native_depth`로 native centres를 바꾸고 reporting z를 작은 값부터 순회한다. 따라서 row 0은 바닥이다. 모든 연속장·mask에 origin=lower를 동일하게 적용하며 flip, 회전, smoothing, 재표본화를 하지 않았다.
- 기존 감사의 get_mapping을 S000 두 도메인에만 호출하여 기존 UNRST SGAS와 corrected NPZ의 6개 시점을 재대조했다. raw 불일치 0 pixel, mask 임계값 불일치 0 pixel이다. UNRST를 읽었으며 OPM은 실행하지 않았다.
- `imshow(..., aspect=equal)`을 사용하고 실제 그려진 축의 가로/세로 비를 검사했다. A=2.8/1.2=2.333333, B=8400/1200=7이다. B 패널이 더 납작한 것은 실제 geometry 때문이다. 두 도메인을 공통 물리 축척으로 그리지는 않았다.
- 기존 Fig01 geometry의 nominal 주입 위치 A (0.9,0.3)/(1.7,0.7) m, B (2700,300)/(5100,700) m와 입력 파일의 #Well 1/2를 대조했다. 그림의 삼각형은 실제 시뮬레이션에 사용된 셀 중심을 표시하므로 nominal 좌표와 작은 격자 차이가 있다.
- A는 COMPDAT의 INJ0/INJ1 completion cell, B는 SOURCE cell이다. 아래 표의 I/J/K는 1-based, z는 위로 증가하는 elevation이다.

| Domain | Injection | Keyword | I,J,K | Actual x (m) | Actual z elevation (m) | Nominal x,z (m) |
|---|---|---|---|---:|---:|---|
| SPE11A | 1 | COMPDAT | (18, 1, 63) | 0.886666656 | 0.29523809 | (0.9, 0.3) |
| SPE11A | 2 | COMPDAT | (36, 1, 26) | 1.70333332 | 0.70549449 | (1.7, 0.7) |
| SPE11B | 1 | SOURCE | (23, 1, 49) | 2680 | 305.068893 | (2700, 300) |
| SPE11B | 2 | SOURCE | (53, 1, 26) | 5080 | 693.849228 | (5100, 700) |

- Figure 3 overlay는 저장 mask의 true pixel 중 이웃 false pixel과 접하는 정확한 cell edge를 흰 선으로 표시한다. contour 보간·팽창·침식은 적용하지 않았다.
- 연속장의 inactive reporting cell(NaN)은 회색, binary mask에서는 저장된 0으로 표시한다. mask와 맞추기 위해 NaN을 plume 값으로 바꾸지 않았다.

## 검산 결과

- 각 domain/snapshot의 저장 mask와 `(delta_sgas_raw > 0.001) & valid_mask` 비교: 전체 12개 시점에서 0 pixel mismatch.
- Figure 3의 두 binary panel: Matplotlib image array와 저장 mask를 직접 array_equal로 검증했다. PNG는 출판용 렌더링이므로 스크린 픽셀 수와 reporting-grid 픽셀 수가 같다는 의미는 아니다.
- `figure_pixel_checks.csv`: 시점별 mask pixel 수, 0.001<ΔSg≤0.097 plume pixel 수, restart index, raw 검증 결과.
- `figure_provenance.json`: NPZ, UNRST, EGRID, INIT, input/deck/centres/corners/dt, 관련 원본 코드와 생성 코드의 SHA-256, geometry 및 버전.
- 생성 전후 입력 파일 SHA-256을 재확인하여 원본 변경이 없음을 확인했다.

## 산출물 및 범위

- PNG: Arial, 600 dpi. PDF: Arial TrueType subset을 포함한 문서; vector text와 원본 reporting-grid raster를 사용하며 savefig dpi=600.
- 생성 중 새 패키지를 설치하지 않았다. 기존 corrected venv_cuda의 Python/Numpy/Matplotlib와 Windows의 실제 Arial 글꼴을 사용했다. 환경 이름에 CUDA가 있지만 모델이나 GPU 계산을 실행하지 않았다.
- 예시의 target 정의·전처리와 좌표 일치를 보여주는 Methods 도해다. threshold robustness, normalization robustness, 모델 성능 또는 다른 geometry의 일반화를 새로 입증하지 않는다.
- Figure 2/3 외 그림, 원고, checkpoint, NPZ, 기존 결과는 수정하지 않았다.

## 제안 영문 캡션

**Figure 2.** Evolution of raw gas-saturation change in scenario S000 on the original reporting grids. Snapshot indices 0–5 correspond to 0–5 h in SPE11A and 0–25 yr in SPE11B. Colours share a linear 0–0.9 scale. The vertical coordinate is elevation, and triangles mark the injection-cell centres used in the simulation. Each panel preserves its domain’s physical aspect ratio; the domains are not drawn to a common physical scale.

**Figure 3.** Construction of the binary plume target from unclipped, UNRST-derived gas-saturation change for S000 at snapshot 5 (5 h in SPE11A; 25 yr in SPE11B). The middle panels directly display the corrected NPZ masks, which match raw ΔSg > 0.001 on valid reporting cells with zero pixel differences. White edges in the right panels trace the exact boundaries of the stored plume pixels. Triangles mark injection-cell centres. Physical aspect ratios and upward-increasing elevation are preserved.

## 입력 파일 해시

- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/dataset_v1_raw_sgas_waterpressure/spe11a/S000.npz`
  - SHA-256: `393950a4d157b894a56ae3253999e502336baa7538bb60d730bca7ab447dc77d`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11a/S000/deck/S000.DATA`
  - SHA-256: `d49f4da162e2507f86bbefa267315a907a13732493829becf3e261f65023721d`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/inputs/S000/spe11a_S000.txt`
  - SHA-256: `d2069227c78a53bf85e9425ab817d5cf5df136c2a1b3737b047acdf2e635e07e`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11a/S000/flow/S000.EGRID`
  - SHA-256: `121da5ead1cefd6ca105c1803303ed724521874bdcc56d3069f64608b5a8846c`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11a/S000/flow/S000.INIT`
  - SHA-256: `6aa00028f892d06b68b3d555e4d14fc0c3e864f896a96125d9134beb9c306af4`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11a/S000/flow/S000.UNRST`
  - SHA-256: `6115047a10f5aca3f72dee59826264644b6aeaff60a968dd1d3818aad398dfe3`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11a/S000/deck/centers.txt`
  - SHA-256: `0d318ecc1aee991cbf37bc574386d48cfbc344b9a8d2ef90efb3351a025b67d9`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11a/S000/deck/corners.txt`
  - SHA-256: `74eb934388ca9a75ede48661a218e08103bbf9c3f04369bc7e27f7ae7dc20726`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11a/S000/deck/dt.txt`
  - SHA-256: `993740ead960dabebe621b2094f81b7ead88de4117e62e04b4c39c3da48f4971`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/dataset_v1_raw_sgas_waterpressure/spe11b/S000.npz`
  - SHA-256: `21d943386730e851e76bf0b973d60b1a095256a41d2a5e4c93206b5986c49230`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11b/S000/deck/S000.DATA`
  - SHA-256: `cbd3bb85c5467ee74e554db12ee400f558bc63f9726c02955fedb75981946997`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/inputs/S000/spe11b_S000.txt`
  - SHA-256: `e750fd07961a1b6dbfcb9bc116db149741e977a5383ec5b6f8cc7d4306c5d9f3`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11b/S000/flow/S000.EGRID`
  - SHA-256: `9163eb0a7b0d88346d55ae528c29b7712dacb3b7ced7ff644323efd46bff3205`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11b/S000/flow/S000.INIT`
  - SHA-256: `c6949bf1a5fee072dcffe572a7ebb121d1d336324d0a2e990591dc5e13d24e3b`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11b/S000/flow/S000.UNRST`
  - SHA-256: `66d5420859f852c70d724d83f548610a11ec65aa23e43abb964436128525a193`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11b/S000/deck/centers.txt`
  - SHA-256: `a08352b79be744a74ec6aef27f295cf72222210970782fe5635c756df9633c3c`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11b/S000/deck/corners.txt`
  - SHA-256: `9e3f020635c539339a7ac24b5b3d6715bb9cc1c529c74156ce5f395e39c8db34`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/runs/spe11b/S000/deck/dt.txt`
  - SHA-256: `b3a1943ccfd4c46ab37273d74ba5f3dee7256fc84b38eb6af549e06b84367856`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/methods_figures_261116/generate_methods_figures.py`
  - SHA-256: `74f40ce24184a06b3cda087b767b100e162c0a4303d511800523f2ed18d988c5`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/prepare_dataset.py`
  - SHA-256: `e1c2f291cdc0c5af693e6f16e5141f6c7ae7cd93240e0fd9a488bc45bb2e9d8d`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/DATASET_COMPLETE.json`
  - SHA-256: `4a5ce5d505f830551e3125e60d076afe3a3122361656bee8516521e687e27c7b`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/_review_audit_20261116/audit.py`
  - SHA-256: `31455caea008100de71b4cd0cb3ca32fd712d4011afd57fa294c670dda940f2e`
- `/home/yssongvd/pyopmspe11/src/pyopmspe11/visualization/data.py`
  - SHA-256: `580e758c0bd193774f0bef00140af4cb91078f6c8e47d902f801b76f343bc3de`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/v2_analysis/11_generate_manuscript_figures_14.py`
  - SHA-256: `f00aab94fb498001315288d7fb0446cc42ab0cb5709af9611241fff6779c1440`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/v2_analysis/30_make_FINAL_11ch_manuscript_figures_physorient.py`
  - SHA-256: `fdeb153e58bd1590fd386c62761bfbbe09ced1e90277012de1e63155b3ac47aa`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/v2_analysis/33_make_Fig02_typography_matched_physorient.py`
  - SHA-256: `d06f715bf60ef7da96c1e5a08aa5e7f5d97883538c3ce9104d7e4c539c8d1ae4`
- `/home/yssongvd/opm_runs/spe11_crossscale_project/v2_analysis/43_make_Fig01_orientation_fixed.py`
  - SHA-256: `0fa0f75a29d1588f642b8b0adf43ff83360bc55676a8feb12699048250ef08cf`

## 최종 파일 확인

- PNG 두 개와 PDF를 독립적으로 rasterize한 두 preview를 모두 시각적으로 확인했다. 축·시간 표기·주입 위치·경계선의 잘림이나 중첩이 없다.
- 두 PNG의 실제 DPI 메타데이터는 599.9988이며 600 dpi 저장 결과이다. Figure 2는 8880×2610 px, Figure 3은 6840×3090 px이다.
- PDF는 각각 1페이지이다. ArialMT와 Arial-BoldMT의 TrueType subset 글꼴이 파일 안에 포함되어 있음을 확인했다.
- 영문 캡션에는 다음 설명도 포함할 수 있다: “Grey regions in the continuous-field panels denote inactive reporting cells.” 저장 mask에서는 해당 셀이 0인 사실과 구분한다.
- WSL 원본 산출물: `/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116/methods_figures_261116`
- Windows 전달용 동일 사본: `D:/OPM with AI/Submission/5. Machine learning_Science and technology/MLST_methods_figures_261116`

이번 작업은 Figure 2·3에서 종료했다.
