"""Complete-cohort reports and paired provenance checks for application studies."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import zipfile
from functools import lru_cache
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / 'artifacts/pfad_registration'
ANATOMIES = ('foot', 'tibia', 'fibula')
SPECIMENS = tuple(f'specimen{i:02}' for i in range(1, 15))
KEYS = ('virtual_target_rms_displacement_mm', 'registration_recall',
        'rotation_error_deg', 'translation_error_mm', 'registration_only_seconds')


@lru_cache(maxsize=1)
def retired_cache():
    return zipfile.ZipFile(APP / 'nested_retention_v1/retired_inner_cache.zip')


def read_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    rel = path.relative_to(APP / 'nested_retention_v1').as_posix()
    return retired_cache().read(rel)


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def analyze(root, methods, proposed, n_starts=16):
    rows, hashes = [], {}
    for specimen in SPECIMENS:
        for anatomy in ANATOMIES:
            path = root / 'cases' / f'{specimen}_{anatomy}.json'
            d = json.loads(path.read_text())
            assert d['status'] == 'complete' and len(d['records']) == len(methods) * n_starts
            if (root / 'protocol.json').exists():
                assert d['protocol_sha256'] == sha(root / 'protocol.json')
            pred = ROOT / 'results/first_arrival_refinement/rebuild_v2/selector_attentive_loocv/nested_policy' / f'{specimen}_{anatomy}.npz'
            assert d['input_prediction_sha256'] == sha(pred)
            assert d['target_mesh_sha256'] == sha(Path(__import__('os').environ.get('PFAD_DATA_ROOT', str(ROOT / 'data/UltraBones100k'))) / specimen / 'CT_bone_segmentations' / (anatomy + '.stl'))
            hashes[str(path.relative_to(ROOT))] = sha(path)
            for r in d['records']:
                assert r['specimen'] == specimen and r['anatomy'] == anatomy
                assert r['method'] in methods and np.isfinite([r[k] for k in KEYS]).all()
                T = np.array(r['estimated_transform'])
                assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
                assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-6
            identities = {(r['condition'], r['draw']) for r in d['records']}
            assert len(identities) == n_starts
            for condition, draw in identities:
                group = [r for r in d['records'] if (r['condition'], r['draw']) == (condition, draw)]
                assert {r['method'] for r in group} == set(methods) and len(group) == len(methods)
                assert all(np.array_equal(r['imposed_transform'], group[0]['imposed_transform']) for r in group)
            rows.extend(d['records'])
    values = {}
    for method in methods:
        values[method] = {}
        for key in KEYS:
            v = []
            for specimen in SPECIMENS:
                case_means = []
                for anatomy in ANATOMIES:
                    group = [r for r in rows if r['specimen'] == specimen and r['anatomy'] == anatomy
                             and r['method'] == method and r['level'] > 0]
                    assert len(group) == n_starts - 1
                    case_means.append(np.mean([r[key] for r in group]))
                if key == 'registration_recall':
                    # Equal case sizes make success counts an exact equivalent
                    # of anatomy averaging; avoid floating pseudo-differences.
                    successful = sum(r[key] for r in rows if r['specimen'] == specimen
                                     and r['method'] == method and r['level'] > 0)
                    v.append(float(successful / (3 * (n_starts - 1))))
                else:
                    v.append(float(np.mean(case_means)))
            values[method][key] = v
    summary = {m: {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
                   for k, v in d.items()} for m, d in values.items()}
    comparisons = {}
    rng = np.random.default_rng(20260906)
    idx = rng.integers(0, 14, (100000, 14))
    for method in methods:
        if method == proposed:
            continue
        comparisons[method] = {}
        for key in KEYS[:2]:
            ref, new = np.array(values[method][key]), np.array(values[proposed][key])
            gain = ref - new if key.endswith('_mm') else new - ref
            test_gain = gain
            if key == 'registration_recall':
                denominator = 3 * (n_starts - 1)
                test_gain = np.rint(new * denominator) - np.rint(ref * denominator)
                gain = test_gain / denominator
            comparisons[method][key] = dict(
                mean_gain_favoring_pfad=float(gain.mean()),
                gain_95ci=np.quantile(gain[idx].mean(axis=1), [.025, .975]).tolist(),
                wilcoxon_two_sided_p=float(wilcoxon(test_gain, method='auto').pvalue) if np.any(test_gain != 0) else 1.,
                wins=int((gain > 0).sum()), losses=int((gain < 0).sum()), ties=int((gain == 0).sum()))
    by_anatomy = {a: {m: {k: float(np.mean([r[k] for r in rows if r['anatomy'] == a
                          and r['method'] == m and r['level'] > 0])) for k in KEYS} for m in methods} for a in ANATOMIES}
    by_start = {c: {m: {k: float(np.mean([r[k] for r in rows if r['condition'] == c
                       and r['method'] == m])) for k in KEYS} for m in methods}
                for c in ('aligned', '5deg_5mm', '10deg_10mm', '20deg_20mm')}
    result = dict(status='complete', specimens=14, cases=42, rows=len(rows), primary_rows=42 * (n_starts - 1) * len(methods),
                  methods=methods, proposed=proposed, summary=summary, comparisons=comparisons,
                  specimen_values=values, by_anatomy_descriptive=by_anatomy, by_start_descriptive=by_start,
                  case_sha256=hashes,
                  QA=dict(all_cases_complete=True, unique_paired_starts=True, transforms_finite_and_rigid=True,
                          prediction_and_CT_hashes_unchanged=True,
                          numerical_exceptions=sum(r['error'] is not None for r in rows),
                          zero_final_fitness=sum(r['fitness'] == 0 for r in rows)))
    return result, rows


def holm(tests):
    running = 0.
    for rank, obj in enumerate(sorted(tests, key=lambda d: d['wilcoxon_two_sided_p'])):
        running = max(running, min(1., (len(tests) - rank) * obj['wilcoxon_two_sided_p']))
        obj['holm_p'] = running
        obj['holm_family_size'] = len(tests)


def write_csv(path, rows):
    fields = ['backend', 'specimen', 'anatomy', 'method', 'condition', 'level', 'draw', 'retention',
              *KEYS, 'fitness', 'error']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=['solvers', 'convergence', 'nested'], required=True)
    args = parser.parse_args()
    if args.study in ('solvers', 'convergence'):
        solver_root = APP / 'solver_sensitivity_v1'
        out = solver_root if args.study == 'solvers' else APP / 'convergence_v1'
        folders = {'point_to_point_l2': solver_root / 'point_to_point_l2',
                   'point_to_plane_l2': solver_root / 'point_to_plane_l2',
                   'point_to_plane_huber': ROOT / 'artifacts/pfad_registration_application_20260906'}
        if args.study == 'convergence':
            folders['point_to_point_l2_extended'] = out
        methods = ('raw', 'random_matched_refined', 'geometry_matched_refined', 'pfad_refined')
        results, rows = {}, []
        for backend, folder in folders.items():
            results[backend], part = analyze(folder, methods, 'pfad_refined')
            rows.extend([dict(r, backend=backend) for r in part])
        holm([t for s in results.values() for cmp in s['comparisons'].values() for t in cmp.values()])
        payload = dict(status='complete', design='retrospective solver sensitivity, not backend selection',
                       total_rows=len(rows), backends=results)
    else:
        out = APP / 'nested_retention_v1'
        methods = ('raw', 'all_refined', 'random_policy', 'geometry_policy', 'pfad_policy')
        payload, rows = analyze(out, methods, 'pfad_policy')
        holm([t for cmp in payload['comparisons'].values() for t in cmp.values()])
        policies = {s: json.loads((out / 'policies' / (s + '.json')).read_text()) for s in SPECIMENS}
        for s, p in policies.items():
            assert s not in p['training_specimens'] and len(p['training_specimens']) == 13
            ledger = json.loads(read_bytes(out / 'inner_scores' / (s + '.json')))
            assert ledger['scores_sha256'] == sha(out / 'inner_scores' / (s + '.npz'))
            for fold in ledger['folds']:
                assert s not in fold['training_specimens'] and s not in fold['validation_specimens']
                assert not set(fold['training_specimens']).intersection(fold['validation_specimens'])
            for rel, expected_hash in p['input_sha256'].items():
                assert sha(out / rel) == expected_hash
            for anatomy in ANATOMIES:
                case = json.loads((out / 'cases' / f'{s}_{anatomy}.json').read_text())
                assert case['policy_sha256'] == sha(out / 'policies' / (s + '.json'))
        payload['selected_retentions'] = {s: {m: d['retention'] for m, d in p['chosen'].items()} for s, p in policies.items()}
        payload['QA']['strict_inner_and_outer_membership'] = True
        payload['QA']['policy_input_and_output_hashes_verified'] = True
        payload['design'] = 'retrospective nested application-policy development; previously inspected cohort'
        payload['inner_model_fits'] = 42
    (out / 'summary_complete.json').write_text(json.dumps(payload, indent=2))
    write_csv(out / 'all_trials.csv', rows)
    print(json.dumps(payload['backends'] if args.study != 'nested' else payload['summary'], indent=2))


if __name__ == '__main__':
    main()
