"""Re-layout only corrected Methods Figs 2/3 into equal A/B panel boxes."""
from pathlib import Path
import os
import sys
import hashlib
import json
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / 'qa/mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path('/home/yssongvd/opm_runs/spe11_crossscale_project/_corrected_reanalysis_20261116')
PREVIOUS = ROOT / 'methods_figures_261116'
PRIOR = json.loads((PREVIOUS / 'figure_provenance.json').read_text())
STEMS = ['261116_Fig02_corrected_equal_panels', '261116_Fig03_corrected_equal_panels']
FIGURE_INCHES = [(14.8, 5.6), (11.4, 6.85)]
PANEL_INCHES = [(2.01, 2.01 / 1.375), (3.08, 3.08 / 1.69)]
QA = OUT / 'qa'
QA.mkdir(parents=True, exist_ok=True)
assert not any((OUT / f'{stem}.{ext}').exists() for stem in STEMS for ext in ['png', 'pdf'])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

for name in ['arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf']:
    font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/' + name)
assert Path(font_manager.findfont('Arial', fallback_to_default=False)).name.lower() == 'arial.ttf'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 9, 'axes.linewidth': .65, 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
CONT = plt.get_cmap('viridis').copy()
CONT.set_bad('#D9D9D9')
BINARY = ListedColormap([CONT(0.), CONT(1.)])
VMAX = .9
INJECTION = Line2D([], [], marker='^', mfc='none', mec='#17B7D7', mew=1.,
    linestyle='none', label='Injection-cell centres (1, 2)')
checks, arrays, sources, measured = [], {}, {}, {}
domains = {d['case']: d for d in PRIOR['domains']}
for case, d in domains.items():
    path = Path(d['npz'])
    assert 'dataset_v1_raw_sgas_waterpressure' in str(path)
    sources[str(path)] = sha(path)
    assert sources[str(path)] == PRIOR['sources_sha256'][str(path)]
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in ['delta_sgas_raw', 'plume_mask', 'valid_mask', 'mask_0001', 'times', 'x', 'z']}
    assert np.array_equal(a['times'], d['times'])
    assert np.all(np.diff(a['z']) > 0) and np.all(np.diff(a['x']) > 0)
    assert np.array_equal((a['delta_sgas_raw'] > .001) & a['valid_mask'][None].astype(bool), a['plume_mask'])
    assert np.array_equal(a['plume_mask'], a['mask_0001'])
    arrays[case] = a
    for k in range(6):
        checks.append({'domain': case, 'scenario': 'S000', 'snapshot': k,
            'time': float(a['times'][k]), 'time_unit': d['timeunit'],
            'mask_mismatch_pixels': 0, 'plume_pixels': int(a['plume_mask'][k].sum())})

def add_axes(fig, left, bottom, width, height):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])

def panel(ax, case, field, binary=False, numbers=False):
    d = domains[case]
    lx, lz = d['Lx'] / d['scale'], d['Lz'] / d['scale']
    # Same physical coordinates and array orientation; only the display box changes.
    im = ax.imshow(field, extent=[0, lx, 0, lz], origin='lower', aspect='auto',
        interpolation='nearest', cmap=BINARY if binary else CONT,
        vmin=0, vmax=1 if binary else VMAX)
    assert np.array_equal(np.asarray(im.get_array()), field, equal_nan=True)
    ax.set_xlim(0, lx); ax.set_ylim(0, lz)
    ax.set_xticks([0, lx / 2, lx]); ax.set_yticks([0, lz / 2, lz])
    ax.set_xlabel(f'x ({d["unit"]})', labelpad=1.5)
    ax.set_ylabel(f'z ({d["unit"]})', labelpad=1.5)
    ax.tick_params(length=2.5, pad=2)
    for p in d['injectors']:
        x, z = p['x_m'] / d['scale'], p['z_elevation_m'] / d['scale']
        ax.plot(x, z, marker='^', ms=4.7, mfc='none', mec='#28D9F8', mew=.9, linestyle='none', zorder=8)
        if numbers:
            ax.annotate(p['label'], (x, z), xytext=(4, -1), textcoords='offset points', color='white', fontsize=7.5)
    return im

def boundary_segments(mask, d):
    nz, nx = mask.shape
    xe = np.linspace(0, d['Lx'] / d['scale'], nx + 1)
    ze = np.linspace(0, d['Lz'] / d['scale'], nz + 1)
    pad = np.pad(mask.astype(bool), 1)
    lines = []
    for j, i in np.argwhere(mask):
        x0, x1, z0, z1 = xe[i], xe[i+1], ze[j], ze[j+1]
        if not pad[j, i+1]: lines.append([(x0,z0), (x1,z0)])
        if not pad[j+2, i+1]: lines.append([(x0,z1), (x1,z1)])
        if not pad[j+1, i]: lines.append([(x0,z0), (x0,z1)])
        if not pad[j+1, i+2]: lines.append([(x1,z0), (x1,z1)])
    return lines

def save(fig, index):
    fig.canvas.draw()
    sizes = [tuple(ax.get_window_extent().size / fig.dpi) for ax in fig.axes if ax.images]
    assert len(sizes) == [12, 6][index]
    assert np.allclose(sizes, PANEL_INCHES[index], rtol=0, atol=1e-10)
    measured[STEMS[index]] = sizes
    for ax in fig.axes:
        if ax.images:
            assert ax.images[0].origin == 'lower'
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{STEMS[index]}.{ext}', dpi=600)
    fig.savefig(QA / f'{STEMS[index]}_preview.png', dpi=160)
    plt.close(fig)

fig = plt.figure(figsize=FIGURE_INCHES[0])
w, h = PANEL_INCHES[0]
for case, bottom, heading in [('spe11a', 3.40, 5.30), ('spe11b', 1.03, 2.93)]:
    d, a = domains[case], arrays[case]
    fig.text(.038, heading / 5.6, f'{d["name"]}  |  S000', fontsize=11, weight='bold')
    for k in range(6):
        ax = add_axes(fig, .54 + k * (w + .35), bottom, w, h)
        im = panel(ax, case, a['delta_sgas_raw'][k], numbers=(k == 0))
        ax.set_title(f'{a["times"][k]:g} {d["timeunit"]}', pad=5)
        if k: ax.set_ylabel('')
cax = add_axes(fig, 4.0, .40, 6.8, .12)
cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, .2, .4, .6, .8, .9])
cb.set_label('Raw gas-saturation change, ΔSg (–)', labelpad=2)
fig.legend(handles=[INJECTION, Patch(facecolor='#D9D9D9', label='Inactive cells')],
    loc='lower left', bbox_to_anchor=(.033, .008), frameon=False)
save(fig, 0)

fig = plt.figure(figsize=FIGURE_INCHES[1])
w, h = PANEL_INCHES[1]
for case, bottom, heading in [('spe11a', 4.26, 6.54), ('spe11b', 1.44, 3.72)]:
    d, a = domains[case], arrays[case]
    fig.text(.049, heading / 6.85, f'{d["name"]}  |  S000  |  {a["times"][5]:g} {d["timeunit"]}', fontsize=11, weight='bold')
    titles = ['Continuous raw ΔSg', 'Binary target: raw ΔSg > 0.001', 'Target boundary overlay']
    for col in range(3):
        ax = add_axes(fig, .56 + col * (w + .62), bottom, w, h)
        mask = a['plume_mask'][5].astype(bool)
        panel(ax, case, mask if col == 1 else a['delta_sgas_raw'][5], binary=(col == 1), numbers=(col == 0))
        ax.set_title(titles[col], pad=5)
        if col: ax.set_ylabel('')
        if col == 2:
            ax.add_collection(LineCollection(boundary_segments(mask, d), colors='white', linewidths=.75, zorder=6))
cax = add_axes(fig, 3.65, .87, 4.2, .12)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, VMAX), cmap=CONT), cax=cax,
    orientation='horizontal', ticks=[0, .2, .4, .6, .8, .9])
cb.set_label('Raw gas-saturation change, ΔSg (–)', labelpad=2)
fig.legend(handles=[Patch(facecolor=BINARY(0), label='Background'), Patch(facecolor=BINARY(1), label='Plume'),
    Patch(facecolor='#D9D9D9', label='Inactive cells (raw)'), INJECTION],
    ncol=4, loc='lower center', bbox_to_anchor=(.5, .005), frameon=False)
save(fig, 1)

assert all(sha(p) == digest for p, digest in sources.items())
import csv
with open(OUT / 'figure_pixel_checks.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=checks[0].keys()); writer.writeheader(); writer.writerows(checks)
manifest = {'revision_date': '2026-11-17', 'scenario': 'S000', 'figures': STEMS,
    'figure_inches': FIGURE_INCHES, 'panel_inches': PANEL_INCHES, 'measured_panel_inches': measured,
    'display': 'equal rectangular A/B panel boxes; physical aspect ratios not preserved',
    'orientation': 'origin=lower, ascending elevation; unchanged arrays; no flips/rotation',
    'threshold': .001, 'all_mask_mismatch_pixels': 0, 'source_sha256': sources,
    'prior_provenance_path': str(PREVIOUS / 'figure_provenance.json'),
    'prior_provenance_sha256': sha(PREVIOUS / 'figure_provenance.json'),
    'code_sha256': sha(__file__), 'no_model_or_OPM_execution': True,
    'generated_sha256': {f'{s}.{e}': sha(OUT / f'{s}.{e}') for s in STEMS for e in ['png', 'pdf']}}
(OUT / 'figure_provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(OUT / 'FIGURE_PROVENANCE.md').write_text('''# Figure 2·3 동일 패널 크기 수정

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
''', encoding='utf-8')
for stem in STEMS:
    with Image.open(OUT / f'{stem}.png') as image:
        assert all(abs(x - 600) < .1 for x in image.info['dpi'])
        print(stem, image.size, image.info['dpi'])
print('PASS: equal A/B panel dimensions; corrected arrays unchanged; 0 mask mismatch pixels.')
