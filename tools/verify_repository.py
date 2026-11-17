"""Recompute archived statistics with stdlib only; no model code is executed."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'research/_corrected_reanalysis_20261116'
SEEDS = [11, 22, 33, 42, 44, 55]
TAGS = ['A_test', 'B_zero', 'B_scratch', 'B_finetuned']
PAIRS = {(i, j) for i in range(1, 5) for j in range(i + 1, 6)}

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)

def check_pairs(rows, sids, case):
    assert len(rows) == 410 and {r['scenario_id'] for r in rows} == sids
    assert {r['case'] for r in rows} == {case}
    for sid in sids:
        g = [r for r in rows if r['scenario_id'] == sid]
        assert len(g) == 10 and {(int(r['i']), int(r['j'])) for r in g} == PAIRS
    for r in rows:
        assert int(r['horizon']) == int(r['j']) - int(r['i'])
        assert 0 <= float(r['iou']) <= 1 and float(r['area_rel_error']) >= 0

def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    out = ap.parse_args().out.resolve()
    assert not out.exists(), 'Use a new output directory'
    manifest = read(REPO / 'metadata/source_manifest.csv')
    for rec in manifest:
        p = REPO / rec['repository_path']
        assert p.stat().st_size == int(rec['size_bytes']) and sha(p) == rec['sha256'], str(p)
    pyfiles = list((REPO / 'research').rglob('*.py'))
    for p in pyfiles:
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    matrix = read(R / 'dataset_v1_raw_sgas_waterpressure/scenario_matrix_valid_207.csv')
    assert len(matrix) == 207
    atest = {r['scenario_id'] for r in matrix if r['source_A_split'] == 'test'}
    btest = {r['scenario_id'] for r in matrix if r['target_B_role'] == 'zero_shot_and_final_test'}
    assert len(atest) == len(btest) == 41
    m = read(R / 'primary_metrics_all.csv'); p = read(R / 'persistence_metrics.csv')
    assert len(m) == 9840 and sorted({int(r['seed']) for r in m}) == SEEDS
    assert {r['eval_tag'] for r in m} == set(TAGS)
    check_pairs(p, btest, 'spe11b')
    assert {r['eval_tag'] for r in p} == {'B_persistence'}
    seedrows = []; groups = {}
    for seed in SEEDS:
        for tag in TAGS:
            g = [r for r in m if int(r['seed']) == seed and r['eval_tag'] == tag]
            check_pairs(g, atest if tag == 'A_test' else btest, 'spe11a' if tag == 'A_test' else 'spe11b')
            rec = {'seed': seed, 'method': tag, 'n_pairs': len(g)}
            for metric in ['iou', 'area_rel_error']:
                rec[metric] = st.mean(float(r[metric]) for r in g)
            seedrows.append(rec); groups[seed, tag] = rec
    saved = read(R / 'seed_results.csv')
    for r in saved:
        for tag in TAGS:
            close(groups[int(r['seed']), tag]['iou'], float(r[tag]))
    summary = []
    for tag in TAGS + ['B_persistence']:
        rec = {'method': tag, 'n_model_seeds': 0 if tag == 'B_persistence' else 6, 'n_pairs_per_seed_or_baseline': 410}
        for metric in ['iou', 'area_rel_error']:
            values = [st.mean(float(r[metric]) for r in p)] if tag == 'B_persistence' else [groups[s, tag][metric] for s in SEEDS]
            rec[metric + '_mean'] = st.mean(values)
            rec[metric + '_sample_sd'] = '' if len(values) == 1 else st.stdev(values)
        summary.append(rec)
    gains = [groups[s, 'B_finetuned']['iou'] - groups[s, 'B_scratch']['iou'] for s in SEEDS]
    close(st.mean(gains), 0.006130404806718577)
    assert sum(g > 0 for g in gains) == 5
    scenario = read(R / 'target_scenario_gains.csv')
    gaincol = next(k for k in scenario[0] if k in {'paired_gain', 'gain', 'iou_gain', 'finetune_minus_scratch'})
    assert len(scenario) == 246
    negative = sum(float(r[gaincol]) < 0 for r in scenario)
    assert negative == 61
    low = R / 'lowdata_n5_n10_final'; draws = read(low / 'corrected_lowdata_subset_results.csv')
    assert len(draws) == 10
    lowstats = []
    for n in [5, 10]:
        g = [r for r in draws if int(r['N']) == n]
        assert sorted(int(r['subset_seed']) for r in g) == [101, 202, 303, 404, 505]
        for r in g:
            assert int(r['model_seed']) == 42 and int(r['validation_scenarios']) == 25 and int(r['test_scenarios']) == 41
            folder = low / ('rep_%02d_n_%02d' % (int(r['replicate']), n))
            for method, col in [('scratch', 'scratch_iou'), ('finetune', 'finetuned_iou')]:
                raw = read(folder / ('metrics_' + method + '.csv'))
                check_pairs(raw, btest, 'spe11b')
                close(st.mean(float(x['iou']) for x in raw), float(r[col]))
            close(float(r['finetuned_iou']) - float(r['scratch_iou']), float(r['paired_gain']))
        rec = {'N': n, 'n_subset_draws': len(g), 'model_seed': 42, 'positive_draws': sum(float(r['paired_gain']) > 0 for r in g)}
        assert rec['positive_draws'] == 5
        for col in ['scratch_iou', 'finetuned_iou', 'paired_gain']:
            v = [float(r[col]) for r in g]; rec[col + '_mean'] = st.mean(v); rec[col + '_subset_sample_sd'] = st.stdev(v)
        lowstats.append(rec)
    for seed in [101, 202, 303, 404, 505]:
        values = {int(r['N']): float(r['paired_gain']) for r in draws if int(r['subset_seed']) == seed}
        assert values[5] > values[10]
    out.mkdir(parents=True)
    write_csv(out / 'primary_summary.csv', summary)
    write_csv(out / 'primary_seed_means.csv', seedrows)
    write_csv(out / 'lowdata_summary.csv', lowstats)
    result = {'status': 'PASS', 'copied_file_checksums_verified': len(manifest),
              'archived_python_syntax_checked_without_execution': len(pyfiles),
              'primary_pair_rows': len(m), 'persistence_pair_rows': len(p),
              'primary_paired_gain_mean': st.mean(gains), 'primary_paired_gain_sample_sd': st.stdev(gains),
              'primary_positive_seed_gains': sum(g > 0 for g in gains),
              'negative_seed_scenario_observations': negative, 'total_seed_scenario_observations': len(scenario),
              'lowdata_draws': len(draws), 'new_training_fits': 0, 'new_inference': 0,
              'scope': 'CSV aggregation and archived file checks; no full pixel/model reconstruction'}
    (out / 'VERIFICATION.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
