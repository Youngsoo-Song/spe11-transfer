# Figure 2·3 동일 패널 크기 수정

수정일: 2026-11-17. 사용자 제공 `Figures/260914/fig2.jpg`, `fig3.jpg`를 표시 크기의 참고로 사용했다. 날짜 prefix `261116_`는 앞선 요청을 유지했다. 이전 그림과 원고·Results Figure를 덮어쓰지 않았다.

## 수정한 사항

- Figure 2는 A/B 각각 6개 패널을 모두 같은 2.01×1.461818 inch 직사각형으로 표시한다(표시 종횡비 1.375).
- Figure 3은 A/B 각각 3개 패널을 모두 같은 3.08×1.822485 inch 직사각형으로 표시한다(표시 종횡비 1.69).
- 이는 비교를 위한 동일한 표시 상자이다. 물리적 x/z 종횡비 또는 도메인 간 공통 물리 축척을 보존한 도해가 아니다. 각 그림 안에서 실제 좌표값·단위는 유지했다.
- 패널의 최종 렌더링 너비·높이가 모두 같은지 검사했다. B의 ΔSg나 mask 배열에 resize, 보간, 반전, 회전, smoothing을 적용하지 않았다. Matplotlib의 `aspect=auto`가 패널 안의 표시 축척만 바꾼다.

## 유지한 자료와 검산

- Corrected `dataset_v1_raw_sgas_waterpressure/spe11a/S000.npz`, `spe11b/S000.npz`의 `delta_sgas_raw`와 `plume_mask`를 직접 사용했다.
- 이전 2026-11-16 출처 기록의 NPZ SHA-256과 동일함을 확인했고 생성 후에도 해시가 같다. `ai_dataset_npz_207`은 읽지 않았다.
- Figure 2: snapshot 0–5, SPE11A=0,1,2,3,4,5 h / SPE11B=0,5,10,15,20,25 yr.
- Figure 3: snapshot 5, SPE11A=5 h / SPE11B=25 yr. Binary panel은 저장 mask 그 자체이다.
- 두 도메인의 모든 12개 시점에서 raw ΔSg > 0.001 및 valid_mask와 저장 plume_mask의 불일치가 0 pixel이다. `mask_0001`도 동일하다.
- 연속장·binary panel에 전달한 Matplotlib 배열을 원본 배열과 직접 비교했다. NaN 위치를 포함하여 일치한다.
- 좌표는 x가 오른쪽, z elevation이 위쪽으로 증가하고 origin=lower이다. 첨부된 예전 JPEG의 상하 방향은 복제하지 않았다. 원본 reporting-grid 좌표로 확인된 방향을 유지했다.
- 실제 injection-cell centre와 physical coordinate는 이전에 EGRID·COMPDAT(A)·SOURCE(B)로 검증한 metadata를 재사용했다. 상세 geometry 근거는 `../methods_figures_261116/FIGURE_PROVENANCE.md` 및 이전 JSON에 있다. 이번에는 UNRST 전체 감사를 반복하지 않았다.
- 같은 선형 색상 범위 0–0.9, Arial, PNG 600 dpi 및 PDF를 유지했다. 회색은 연속장의 inactive reporting cell이며, binary mask에서는 저장된 0으로 남아 있다.
- OPM, 모델, checkpoint, 학습 코드를 실행하지 않았다. 이번 변경은 Figure 2·3의 표시 배치 수정으로 한정했다.

## 사용 가능한 영문 캡션 문장

“SPE11A and SPE11B are displayed in equal-sized rectangular panels for visual comparison; the physical aspect ratios and a common physical scale are not preserved. The coordinate values remain in physical units, and elevation increases upward.”

Figure 3의 흰 선은 저장 mask에서 인접 background pixel과 접하는 정확한 cell edge이다. contour 보간 또는 mask 재정의는 하지 않았다.

입력/출력 SHA-256과 실제 패널 크기는 `figure_provenance.json`, 시점별 mask 검산은 `figure_pixel_checks.csv`, 재생성 코드는 `generate_equal_panel_figures.py`에 기록했다.

## 최종 파일 확인

PNG와 독립적으로 rasterize한 PDF 모두 시각적으로 확인했다. 두 PDF는 각각 1페이지이며 Arial TrueType subset 글꼴이 포함되어 있다. 실제 PNG DPI는 599.9988이며 Figure 2는 8880×3360 px, Figure 3은 6840×4110 px이다. 색상표와 범례의 겹침을 정리했다. 이 패널 배치 수정본만 최종 산출물로 사용하며 첫 배치 초안은 qa/layout_v1에 보존했다.

