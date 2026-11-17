#!/usr/bin/env python3
"""Only corrected Methods Figs 2 and 3. No training/model imports or simulation."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / 'qa' / 'mplconfig')
import hashlib
import importlib.util
import json
import platform
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from resdata.grid import Grid
from resdata.resfile import ResdataFile
from PIL import Image

PROJECT = Path('/home/yssongvd/opm_runs/spe11_crossscale_project')
CORRECTED = PROJECT / '_corrected_reanalysis_20261116'
DATASET = CORRECTED / 'dataset_v1_raw_sgas_waterpressure'
SID = 'S000'  # same BASE_SID as original Methods figures, not selected using results
THRESHOLD = .001
STEMS = ['261116_Fig02_corrected_raw_dSg_evolution', '261116_Fig03_corrected_target_construction']
FONTDIR = Path('/mnt/c/Windows/Fonts')
DOMAINS = {
    'spe11a': dict(name='SPE11A', Lx=2.8, Lz=1.2, scale=1., unit='m', timeunit='h',
                   times=[0, 1, 2, 3, 4, 5], native=(55, 1, 74), report=(120, 280)),
    'spe11b': dict(name='SPE11B', Lx=8400., Lz=1200., scale=1000., unit='km', timeunit='yr',
                   times=[0, 5, 10, 15, 20, 25], native=(83, 1, 58), report=(60, 420)),
}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def configure_fonts():
    for name in ['arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf']:
        font_manager.fontManager.addfont(str(FONTDIR / name))
    actual = font_manager.findfont('Arial', fallback_to_default=False)
    assert Path(actual).name.lower() == 'arial.ttf', actual
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'legend.fontsize': 9, 'axes.linewidth': .65,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'image.interpolation': 'nearest',
    })
    return actual

def load_and_check():
    # Import only the completed geometric mapping helper; never import legacy
    # plotting modules, which also initialize model-related code at module level.
    spec = importlib.util.spec_from_file_location('completed_geometry', PROJECT / '_review_audit_20261116/audit.py')
    au = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(au)
    checks, hashes, geometry = [], {}, []
    prechecks = pd.read_csv(CORRECTED / 'preprocessing_checks.csv')
    for case, d in DOMAINS.items():
        npz = DATASET / case / f'{SID}.npz'
        assert 'dataset_v1_raw_sgas_waterpressure' in str(npz)
        hashes[str(npz)] = sha(npz)
        expected = prechecks[(prechecks.case == case) & (prechecks.scenario_id == SID)].iloc[0]
        assert hashes[str(npz)] == expected.npz_sha256
        with np.load(npz, allow_pickle=False) as z:
            d['arrays'] = {k: z[k].copy() for k in ['x', 'z', 'times', 'delta_sgas_raw', 'sgas_raw', 'valid_mask', 'plume_mask', 'mask_0001', 'restart_indices']}
        a = d['arrays']
        assert np.array_equal(a['times'], d['times'])
        assert a['delta_sgas_raw'].shape == (6, *d['report'])
        assert np.all(np.diff(a['x']) > 0) and np.all(np.diff(a['z']) > 0)
        nz, nx = d['report']
        d['xe'] = np.linspace(0., d['Lx'], nx + 1) / d['scale']
        d['ze'] = np.linspace(0., d['Lz'], nz + 1) / d['scale']
        idealx = (d['xe'][:-1] + d['xe'][1:]) * .5 * d['scale']
        idealz = (d['ze'][:-1] + d['ze'][1:]) * .5 * d['scale']
        assert np.allclose(a['x'], idealx, rtol=2e-7, atol=1e-7)
        assert np.allclose(a['z'], idealz, rtol=2e-7, atol=1e-7)
        valid = a['valid_mask'].astype(bool)
        assert np.array_equal(np.isfinite(a['delta_sgas_raw']), np.broadcast_to(valid, a['delta_sgas_raw'].shape))
        assert np.array_equal(a['delta_sgas_raw'], np.maximum(a['sgas_raw'] - a['sgas_raw'][0:1], 0), equal_nan=True)
        calc = (a['delta_sgas_raw'] > THRESHOLD) & valid[None]
        assert np.array_equal(calc, a['plume_mask'].astype(bool))
        assert np.array_equal(a['mask_0001'], a['plume_mask'])
        d['npz_path'] = npz
        run = PROJECT / 'runs' / case / SID
        deck = run / 'deck' / f'{SID}.DATA'
        inputfile = PROJECT / 'inputs' / SID / f'{case}_{SID}.txt'
        gridpath = run / 'flow' / f'{SID}.EGRID'
        initpath = run / 'flow' / f'{SID}.INIT'
        unpath = run / 'flow' / f'{SID}.UNRST'
        for path in [deck, inputfile, gridpath, initpath, unpath, run / 'deck/centers.txt', run / 'deck/corners.txt', run / 'deck/dt.txt']:
            hashes[str(path)] = sha(path)
        grid = Grid(str(gridpath))
        assert (grid.nx, grid.ny, grid.nz) == d['native']
        rf, init = ResdataFile(str(unpath)), ResdataFile(str(initpath))
        mp = au.get_mapping(case, SID, grid, init, nx, nz, geometry)
        ids, act = mp['ids'], mp['act'].astype(bool)
        assert np.array_equal(act[ids].reshape(nz, nx), valid)
        # Verify ONLY the displayed S000 snapshots against existing UNRST. This
        # does not generate a dataset or change the completed study's results.
        raw_mismatch = []
        time_error = []
        for k, n in enumerate(a['restart_indices']):
            native = np.asarray(rf['SGAS'][int(n)])
            full = np.full(act.size, np.nan)
            full[act] = native
            mapped = full[ids].reshape(nz, nx)
            raw_mismatch.append(int(np.count_nonzero(~((mapped == a['sgas_raw'][k]) | (np.isnan(mapped) & np.isnan(a['sgas_raw'][k]))))))
            factor = 3600. if case == 'spe11a' else 31536000.
            t = float(np.asarray(rf['DOUBHEAD'][int(n)])[0]) * 86400.
            time_error.append(abs(t - float(a['times'][k]) * factor))
        assert max(raw_mismatch) == 0 and max(time_error) < 1e-4
        text = re.sub(r'--[^\n]*', '', deck.read_text())
        if case == 'spe11a':
            block = text.split('COMPDAT', 1)[1].split('\n/')[0]
            records = re.findall(r"'INJ([01])'\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", block)
            assert len(records) == 2
            ijks = [(int(i), int(j), int(k)) for _, i, j, k, end in records if k == end]
            d['injection_keyword'] = 'COMPDAT'
        else:
            block = text.split('\nSOURCE\n', 1)[1].split('\n/')[0]
            ijks = [tuple(map(int, r)) for r in re.findall(r'(\d+)\s+(\d+)\s+(\d+)\s+GAS', block)]
            assert len(ijks) == 2
            d['injection_keyword'] = 'SOURCE'
        nominal = []
        for line in inputfile.read_text().splitlines():
            if re.search(r'#Well [12]', line):
                p = list(map(float, line.split('#')[0].split()))
                nominal.append((p[1], p[3]))
        assert len(nominal) == 2
        d['injectors'] = []
        for idx, (ijk, nom) in enumerate(zip(ijks, nominal), 1):
            c = grid.get_xyz(ijk=tuple(v - 1 for v in ijk))
            elev = d['Lz'] - c[2]
            pts = np.loadtxt(run / 'deck/centers.txt', delimiter=',')
            gi = (ijk[2] - 1) * grid.nx * grid.ny + (ijk[1] - 1) * grid.nx + ijk[0] - 1
            assert np.allclose(c, pts[gi], rtol=0., atol=1e-3)
            d['injectors'].append(dict(label=str(idx), ijk=ijk, x_m=c[0], z_elevation_m=elev,
                                        nominal_x_m=nom[0], nominal_z_m=nom[1]))
        rf.close(); init.close()
        for k, t in enumerate(a['times']):
            m = a['plume_mask'][k].astype(bool)
            ds = a['delta_sgas_raw'][k]
            checks.append(dict(case=case, scenario=SID, snapshot=k, time=float(t), unit=d['timeunit'],
                restart_index=int(a['restart_indices'][k]), raw_UNRST_mismatch_pixels=raw_mismatch[k],
                stored_mask_vs_raw_threshold_mismatch_pixels=int((calc[k] != m).sum()),
                plume_pixels=int(m.sum()), low_saturation_plume_pixels_0001_to_0097=int(((ds > .001) & (ds <= .097) & valid).sum()),
                max_raw_delta_sgas=float(np.nanmax(ds)), valid_reporting_pixels=int(valid.sum()),
                orientation='ascending z elevation; row zero bottom; no flip/rotation'))
    return checks, hashes, geometry

def image_panel(ax, case, field, binary=False, numbers=False):
    d = DOMAINS[case]
    xe, ze = d['xe'], d['ze']
    im = ax.imshow(field, extent=[xe[0], xe[-1], ze[0], ze[-1]], origin='lower',
        interpolation='nearest', aspect='equal', cmap=BINARY if binary else CONT,
        vmin=0., vmax=1. if binary else VMAX)
    if binary:
        assert np.array_equal(np.asarray(im.get_array()), field)
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ze[0], ze[-1])
    ax.set_xticks([0, .5 * xe[-1], xe[-1]])
    ax.set_yticks([0, .5 * ze[-1], ze[-1]])
    ax.tick_params(length=2.5, pad=2)
    ax.set_xlabel(f'x ({d["unit"]})', labelpad=1.5)
    ax.set_ylabel(f'z ({d["unit"]})', labelpad=1.5)
    for p in d['injectors']:
        x, z = p['x_m'] / d['scale'], p['z_elevation_m'] / d['scale']
        ax.plot(x, z, marker='^', ms=4.7, mfc='none', mec='#28D9F8', mew=.9, linestyle='none', zorder=8)
        if numbers:
            ax.annotate(p['label'], (x, z), xytext=(4, -1), textcoords='offset points',
                fontsize=7.5, color='white', zorder=9)
    return im

def exact_pixel_edges(mask, xe, ze):
    """Draw only exposed edges of true pixels; no contour interpolation/smoothing."""
    segments = []
    padded = np.pad(mask.astype(bool), 1)
    for j, i in np.argwhere(mask):
        x0, x1, z0, z1 = xe[i], xe[i + 1], ze[j], ze[j + 1]
        if not padded[j, i + 1]: segments.append([(x0, z0), (x1, z0)])
        if not padded[j + 2, i + 1]: segments.append([(x0, z1), (x1, z1)])
        if not padded[j + 1, i]: segments.append([(x0, z0), (x0, z1)])
        if not padded[j + 1, i + 2]: segments.append([(x1, z0), (x1, z1)])
    return segments

def axes_inches(fig, x, y, w, h):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])

def save(fig, stem):
    # Assert the axes geometry after layout, not only a plotting parameter.
    fig.canvas.draw()
    for ax in fig.axes:
        if ax.get_images():
            bounds = ax.get_window_extent()
            ratio = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
            assert abs(bounds.width / bounds.height - ratio) < 1e-8
            assert ax.images[0].origin == 'lower'
    for ext in ['png', 'pdf']:
        p = OUT / f'{stem}.{ext}'
        assert not p.exists(), f'Refusing overwrite: {p}'
        fig.savefig(p, dpi=600, metadata={'Creator': 'Corrected Methods figures; no model execution'} if ext == 'pdf' else None)
    qa = OUT / 'qa'; qa.mkdir(exist_ok=True)
    fig.savefig(qa / f'{stem}_preview.png', dpi=160)
    plt.close(fig)

def make_fig2():
    fig = plt.figure(figsize=(14.8, 4.35))
    w = 2.01
    left, gap = .54, .35
    for case, bottom, head in [('spe11a', 2.48, 3.95), ('spe11b', 1.05, 1.95)]:
        d = DOMAINS[case]
        fig.text(.038, head / 4.35, f'{d["name"]}  |  {SID}', fontsize=11, weight='bold')
        for k in range(6):
            h = w * d['Lz'] / d['Lx']
            ax = axes_inches(fig, left + k * (w + gap), bottom, w, h)
            im = image_panel(ax, case, d['arrays']['delta_sgas_raw'][k], numbers=(k == 0))
            ax.set_title(f'Snapshot {k}  |  {d["arrays"]["times"][k]:g} {d["timeunit"]}', pad=5)
            if k > 0:
                ax.set_ylabel('')
    cax = axes_inches(fig, 4.0, .41, 6.8, .12)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, .2, .4, .6, .8, VMAX])
    cb.ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '0.9'])
    cb.set_label('Raw gas-saturation change, ΔSg (–)', labelpad=2)
    fig.legend(handles=[INJECTION], loc='lower left', bbox_to_anchor=(.035, .015), frameon=False)
    save(fig, STEMS[0])

def make_fig3():
    fig = plt.figure(figsize=(11.4, 5.15))
    w, left, gap = 3.08, .56, .62
    for case, bottom, head in [('spe11a', 2.73, 4.75), ('spe11b', 1.20, 2.23)]:
        d = DOMAINS[case]
        a = d['arrays']; raw = a['delta_sgas_raw'][5]; mask = a['plume_mask'][5].astype(bool)
        fig.text(.049, head / 5.15, f'{d["name"]}  |  {SID}  |  snapshot 5, {a["times"][5]:g} {d["timeunit"]}',
                 fontsize=11, weight='bold')
        titles = ['Continuous raw ΔSg', 'Binary target: raw ΔSg > 0.001', 'Exact mask boundary on raw ΔSg']
        for col in range(3):
            h = w * d['Lz'] / d['Lx']
            ax = axes_inches(fig, left + col * (w + gap), bottom, w, h)
            im = image_panel(ax, case, mask if col == 1 else raw, binary=(col == 1), numbers=(col == 0))
            ax.set_title(titles[col], pad=5)
            if col > 0: ax.set_ylabel('')
            if col == 2:
                ax.add_collection(LineCollection(exact_pixel_edges(mask, d['xe'], d['ze']),
                                                colors='white', linewidths=.75, zorder=6))
    cax = axes_inches(fig, 3.65, .64, 4.2, .12)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, VMAX), cmap=CONT), cax=cax,
                      orientation='horizontal', ticks=[0, .2, .4, .6, .8, VMAX])
    cb.ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '0.9'])
    cb.set_label('Raw gas-saturation change, ΔSg (–)', labelpad=2)
    fig.legend(handles=[Patch(facecolor=BINARY(0), label='Background'),
        Patch(facecolor=BINARY(1), label='Plume'),
        Line2D([], [], color='gray', lw=1., label='White line: stored-mask pixel edges'), INJECTION],
        ncol=4, loc='lower center', bbox_to_anchor=(.5, .005), frameon=False, columnspacing=1.4)
    save(fig, STEMS[1])

def provenance(checks, hashes, geometry, font):
    pd.DataFrame(checks).to_csv(OUT / 'figure_pixel_checks.csv', index=False)
    records = []
    for case, d in DOMAINS.items():
        records.append({k: v for k, v in d.items() if k not in ['arrays', 'xe', 'ze', 'npz_path']} |
                       {'case': case, 'npz': str(d['npz_path'])})
    manifest = dict(scenario=SID, figures=STEMS, threshold=THRESHOLD,
        domains=records, geometry_mapping=geometry, sources_sha256=hashes,
        generated_sha256={f'{s}.{e}': sha(OUT / f'{s}.{e}') for s in STEMS for e in ['png', 'pdf']},
        Arial_font=font, matplotlib=matplotlib.__version__, numpy=np.__version__, python=platform.python_version(),
        no_models_or_checkpoints_executed=True, no_simulation=True)
    (OUT / 'figure_provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    lines = ['# Figure 2 and Figure 3 corrected 데이터 출처', '',
        '작성일: 2026-11-16. MLST Research Paper의 Methods Figure 2·3만 재생성했다. 원고와 Results Figure는 수정하지 않았다.', '',
        '## 입력 선택 및 기존 코드 조사', '',
        '- 기존 `v2_analysis/11_generate_manuscript_figures_14.py:88–124`는 INDEX 경유 NPZ를 읽고, 공통 시나리오 S000을 BASE_SID로 우선 선택한다.',
        '- 기존 `v2_analysis/30_make_FINAL_11ch_manuscript_figures_physorient.py:794–1046`의 make_fig02/make_fig03은 `X[:,1]` 및 마지막 저장 plume_mask를 사용한다. Figure 2는 전 시점, Figure 3은 snapshot 5이다.',
        '- 별도 Figure 2 코드 `v2_analysis/33_make_Fig02_typography_matched_physorient.py`도 S000의 X[:,1]을 사용한다. 기존 렌더링은 origin=lower, aspect=auto, 물리 좌표 눈금 제거였다. 더 오래된 colored 코드에는 origin=upper도 남아 있으므로 파일명의 orientation을 근거로 뒤집지 않았다.',
        '- 기존 생성 모듈은 학습/모델 관련 모듈을 최상위에서 import하므로 실행하거나 import하지 않았다. 코드 텍스트만 읽었다.',
        '- 같은 Methods 예시 S000을 유지했다. S000은 성능이나 외관을 보고 새로 선택하지 않았다. 시나리오 조건: q_mult=0.7, kz_rel_mult=0.5, global_perm_mult=0.8, lower_dominant; source_A_split=test, target_B_role=zero_shot_and_final_test이다. 방법 설명용 시각화이며 모델·부분집합 선택에 사용하지 않았다.',
        '- `ai_dataset_npz_207`의 NPZ, clipped saturation, X[:,1], 모델 prediction은 이번 생성에서 읽지 않았다.', '',
        '## Corrected 파일 및 시간', '',
        f'- 데이터셋: `{DATASET}`',
        '- 연속장: NPZ의 float64 `delta_sgas_raw`. `sgas_raw`의 초기값 차이와 정확히 일치하는지 확인했다. 원본 생성식은 max(Sg(t)−Sg(0), 0)이며 이 예시의 Sg(0)는 0이다. 0.097 clipping이나 CSV 반올림을 적용하지 않았다.',
        '- Binary target: NPZ `plume_mask`, 보조 키 `mask_0001`과 동일. Figure 3에는 재구성한 별도 mask 대신 이 저장 mask를 직접 전달했다.',
        '- Figure 2: 두 도메인 모두 snapshot index 0,1,2,3,4,5. SPE11A 시간 0,1,2,3,4,5 h; SPE11B 시간 0,5,10,15,20,25 yr.',
        '- Figure 3: SPE11A snapshot 5 = 5 h; SPE11B snapshot 5 = 25 yr. 연속장·mask·overlay가 같은 시나리오 및 같은 시점이다.',
        '- 두 그림에서 동일 선형 색상 범위 0–0.9를 사용했다. 해당 raw 최댓값은 약 0.899907(A), 0.858079(B)로 범위 안에 있다. 색상 범위는 saturation 데이터나 mask를 잘라내지 않는다.', '',
        '## 좌표 격자 방향 및 주입 위치', '',
        '- A: reporting grid H×W=120×280, x∈[0,2.8] m, z∈[0,1.2] m. B: 60×420, x∈[0,8400] m, z∈[0,1200] m; 그림에는 양 축을 모두 km로 환산했다.',
        '- NPZ x/z는 reporting-cell centre이며 오름차순이다. 전체 cell edge는 pyopmspe11 dense_data의 linspace 경계와 같다. float32 저장 중심 좌표와 이상 중심의 허용 오차 내 일치를 확인했다.',
        '- pyopmspe11 `visualization/data.py:840–938`는 `z_elevation=Lz−native_depth`로 native centres를 바꾸고 reporting z를 작은 값부터 순회한다. 따라서 row 0은 바닥이다. 모든 연속장·mask에 origin=lower를 동일하게 적용하며 flip, 회전, smoothing, 재표본화를 하지 않았다.',
        '- 기존 감사의 get_mapping을 S000 두 도메인에만 호출하여 기존 UNRST SGAS와 corrected NPZ의 6개 시점을 재대조했다. raw 불일치 0 pixel, mask 임계값 불일치 0 pixel이다. UNRST를 읽었으며 OPM은 실행하지 않았다.',
        '- `imshow(..., aspect=equal)`을 사용하고 실제 그려진 축의 가로/세로 비를 검사했다. A=2.8/1.2=2.333333, B=8400/1200=7이다. B 패널이 더 납작한 것은 실제 geometry 때문이다. 두 도메인을 공통 물리 축척으로 그리지는 않았다.',
        '- 기존 Fig01 geometry의 nominal 주입 위치 A (0.9,0.3)/(1.7,0.7) m, B (2700,300)/(5100,700) m와 입력 파일의 #Well 1/2를 대조했다. 그림의 삼각형은 실제 시뮬레이션에 사용된 셀 중심을 표시하므로 nominal 좌표와 작은 격자 차이가 있다.',
        '- A는 COMPDAT의 INJ0/INJ1 completion cell, B는 SOURCE cell이다. 아래 표의 I/J/K는 1-based, z는 위로 증가하는 elevation이다.', '',
        '| Domain | Injection | Keyword | I,J,K | Actual x (m) | Actual z elevation (m) | Nominal x,z (m) |',
        '|---|---|---|---|---:|---:|---|']
    for case, d in DOMAINS.items():
        for p in d['injectors']:
            lines.append(f'| {d["name"]} | {p["label"]} | {d["injection_keyword"]} | {p["ijk"]} | {p["x_m"]:.9g} | {p["z_elevation_m"]:.9g} | ({p["nominal_x_m"]:g}, {p["nominal_z_m"]:g}) |')
    lines += ['', '- Figure 3 overlay는 저장 mask의 true pixel 중 이웃 false pixel과 접하는 정확한 cell edge를 흰 선으로 표시한다. contour 보간·팽창·침식은 적용하지 않았다.',
        '- 연속장의 inactive reporting cell(NaN)은 회색, binary mask에서는 저장된 0으로 표시한다. mask와 맞추기 위해 NaN을 plume 값으로 바꾸지 않았다.', '',
        '## 검산 결과', '',
        '- 각 domain/snapshot의 저장 mask와 `(delta_sgas_raw > 0.001) & valid_mask` 비교: 전체 12개 시점에서 0 pixel mismatch.',
        '- Figure 3의 두 binary panel: Matplotlib image array와 저장 mask를 직접 array_equal로 검증했다. PNG는 출판용 렌더링이므로 스크린 픽셀 수와 reporting-grid 픽셀 수가 같다는 의미는 아니다.',
        '- `figure_pixel_checks.csv`: 시점별 mask pixel 수, 0.001<ΔSg≤0.097 plume pixel 수, restart index, raw 검증 결과.',
        '- `figure_provenance.json`: NPZ, UNRST, EGRID, INIT, input/deck/centres/corners/dt, 관련 원본 코드와 생성 코드의 SHA-256, geometry 및 버전.',
        '- 생성 전후 입력 파일 SHA-256을 재확인하여 원본 변경이 없음을 확인했다.', '',
        '## 산출물 및 범위', '',
        '- PNG: Arial, 600 dpi. PDF: Arial TrueType subset을 포함한 문서; vector text와 원본 reporting-grid raster를 사용하며 savefig dpi=600.',
        '- 생성 중 새 패키지를 설치하지 않았다. 기존 corrected venv_cuda의 Python/Numpy/Matplotlib와 Windows의 실제 Arial 글꼴을 사용했다. 환경 이름에 CUDA가 있지만 모델이나 GPU 계산을 실행하지 않았다.',
        '- 예시의 target 정의·전처리와 좌표 일치를 보여주는 Methods 도해다. threshold robustness, normalization robustness, 모델 성능 또는 다른 geometry의 일반화를 새로 입증하지 않는다.',
        '- Figure 2/3 외 그림, 원고, checkpoint, NPZ, 기존 결과는 수정하지 않았다.', '',
        '## 제안 영문 캡션', '',
        '**Figure 2.** Evolution of raw gas-saturation change in scenario S000 on the original reporting grids. Snapshot indices 0–5 correspond to 0–5 h in SPE11A and 0–25 yr in SPE11B. Colours share a linear 0–0.9 scale. The vertical coordinate is elevation, and triangles mark the injection-cell centres used in the simulation. Each panel preserves its domain’s physical aspect ratio; the domains are not drawn to a common physical scale.', '',
        '**Figure 3.** Construction of the binary plume target from unclipped, UNRST-derived gas-saturation change for S000 at snapshot 5 (5 h in SPE11A; 25 yr in SPE11B). The middle panels directly display the corrected NPZ masks, which match raw ΔSg > 0.001 on valid reporting cells with zero pixel differences. White edges in the right panels trace the exact boundaries of the stored plume pixels. Triangles mark injection-cell centres. Physical aspect ratios and upward-increasing elevation are preserved.', '',
        '## 입력 파일 해시', '']
    for path, digest in hashes.items():
        lines.append(f'- `{path}`\n  - SHA-256: `{digest}`')
    (OUT / 'FIGURE_PROVENANCE.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'FIGURE_PROVENANCE.md').exists(), 'Refuse to overwrite a completed figure package'
    font = configure_fonts()
    checks, hashes, geometry = load_and_check()
    for p in [Path(__file__), CORRECTED / 'prepare_dataset.py', CORRECTED / 'DATASET_COMPLETE.json',
              PROJECT / '_review_audit_20261116/audit.py',
              Path('/home/yssongvd/pyopmspe11/src/pyopmspe11/visualization/data.py'),
              PROJECT / 'v2_analysis/11_generate_manuscript_figures_14.py',
              PROJECT / 'v2_analysis/30_make_FINAL_11ch_manuscript_figures_physorient.py',
              PROJECT / 'v2_analysis/33_make_Fig02_typography_matched_physorient.py',
              PROJECT / 'v2_analysis/43_make_Fig01_orientation_fixed.py']:
        hashes[str(p)] = sha(p)
    CONT = plt.get_cmap('viridis').copy(); CONT.set_bad('#D9D9D9')
    BINARY = ListedColormap([CONT(0.), CONT(1.)])
    VMAX = .9
    INJECTION = Line2D([], [], marker='^', mfc='none', mec='#17B7D7', mew=1., linestyle='none',
                       label='Injection-cell centres (1, 2)')
    make_fig2(); make_fig3()
    assert all(sha(Path(p)) == digest for p, digest in hashes.items())
    provenance(checks, hashes, geometry, font)
    for s in STEMS:
        with Image.open(OUT / f'{s}.png') as im:
            assert all(abs(v - 600) < .1 for v in im.info['dpi'])
            print(s, im.size, im.info['dpi'])
    print('PASS: displayed raw SGAS and stored masks verified; no model/checkpoint/simulation execution.')
