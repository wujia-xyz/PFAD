"""Application-policy cross-validation with specimen-separated score fitting.

The controls need no fitted score and may cache their inner registration trials.
PFAD inner scores exclude both the outer specimen and each inner validation
fold. A policy file is committed before the selected outer policies are run.
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import concurrent.futures
from functools import lru_cache
import importlib.util
import json
import math
import time
import numpy as np
import open3d as o3d
import trimesh

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('base', ROOT / 'scripts/audits/pfad_registration_application/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT = ROOT / 'artifacts/pfad_registration/nested_retention_v1'
CONFIG = ROOT / 'configs/pfad/registration/nested_retention_v1.json'
FRACTIONS = (.5, .7, .9, 1.)
SPECIMENS = tuple(f'specimen{i:02}' for i in range(1, 15))
CASES = tuple((s, a) for s in SPECIMENS for a in base.ANATOMIES)
METHODS = ('raw', 'all_refined', 'random_policy', 'geometry_policy', 'pfad_policy')


@lru_cache(maxsize=128)
def sha(path):
    return base.sha(path)


@lru_cache(maxsize=42)
def source_data(specimen, anatomy):
    path = base.PRED / f'{specimen}_{anatomy}.npz'
    previous_path = base.OUT / 'cases' / f'{specimen}_{anatomy}.json'
    previous = json.loads(previous_path.read_text())
    assert sha(path) == previous['input_prediction_sha256']
    with np.load(path) as z:
        raw = z['ray_origins_mm'].astype(float) + z['ray_depths_mm'][:, None].astype(float) * z['ray_directions'].astype(float)
        refined = z['ray_origins_mm'].astype(float) + (z['ray_depths_mm'].astype(float) + z['correction'].astype(float))[:, None] * z['ray_directions'].astype(float)
        scores = dict(geometry=-z['cross_sweep_nearest_mm'].astype(float),
                      random=np.random.default_rng(previous['case_seed'] + 40).random(len(raw)),
                      pfad=z['selector_anatomy_probability'] * z['selector_surface_probability'])
        counts = z['counts'].astype(int)
    center = np.median(raw, axis=0)
    assert counts.sum() == len(raw)
    assert np.isfinite(raw).all() and np.isfinite(refined).all()
    return raw - center, refined - center, center, counts, scores, previous


@lru_cache(maxsize=42)
def target_data(specimen, anatomy):
    # Called only after the source masks have been constructed.
    _, _, center, _, _, previous = source_data(specimen, anatomy)
    mesh_path = base.DATA / specimen / 'CT_bone_segmentations' / (anatomy + '.stl')
    assert sha(mesh_path) == previous['target_mesh_sha256']
    mesh = trimesh.load_mesh(mesh_path, process=False)
    seed = previous['case_seed']
    points, normals = base.sample_mesh(mesh, 80000, seed + 1)
    targets, _ = base.sample_mesh(mesh, 4096, seed + 2)
    clouds = [base.voxel(points - center, v, normals) for v in (2., 1., .5)]
    return clouds, targets - center


def selection(score, counts, fraction):
    assert np.isfinite(score).all() and 0 < fraction <= 1
    keep = np.zeros(len(score), dtype=bool)
    cursor = 0
    for n in counts:
        idx = np.arange(cursor, cursor + n)
        k = min(n, max(3, math.ceil(fraction * int(n))))
        order = np.lexsort((idx, -score[idx]))
        keep[idx[order[:k]]] = True
        assert keep[idx].sum() == k
        cursor += n
    return keep


def register_case(job):
    stage, specimen, anatomy, outer = job
    name = specimen + '_' + anatomy
    if stage == 'controls':
        dest = OUT / 'inner_registration/controls' / (name + '.json')
    elif stage == 'inner':
        assert specimen != outer
        dest = OUT / 'inner_registration' / outer / (name + '.json')
    else:
        assert stage == 'outer' and specimen == outer
        dest = OUT / 'cases' / (name + '.json')
    stamp = dict(protocol_sha256=sha(CONFIG), runner_sha256=sha(Path(__file__)),
                 input_prediction_sha256=sha(base.PRED / (name + '.npz')))
    if stage == 'inner':
        scores_path = OUT / 'inner_scores' / (outer + '.npz')
        score_ledger = json.loads(scores_path.with_suffix('.json').read_text())
        assert score_ledger['scores_sha256'] == sha(scores_path)
        assert score_ledger['protocol_sha256'] == sha(CONFIG)
        assert outer not in score_ledger['training_specimens']
        stamp['inner_scores_sha256'] = sha(scores_path)
    if stage == 'outer':
        policy_path = OUT / 'policies' / (outer + '.json')
        policy = json.loads(policy_path.read_text())
        assert outer not in policy['training_specimens']
        assert policy['protocol_sha256'] == sha(CONFIG)
        stamp['policy_sha256'] = sha(policy_path)
    if dest.exists():
        payload = json.loads(dest.read_text())
        assert payload['status'] == 'complete'
        assert all(payload[k] == v for k, v in stamp.items())
        return dict(case=name, stage=stage, outer=outer, reused=True)
    tick = time.perf_counter()
    raw, refined, _, counts, scores, previous = source_data(specimen, anatomy)
    if stage == 'inner':
        with np.load(scores_path) as z:
            scores = dict(pfad=z[name])
        configurations = [('pfad', f) for f in FRACTIONS]
    elif stage == 'controls':
        configurations = [(m, f) for m in ('random', 'geometry') for f in FRACTIONS]
    else:
        configurations = [('all_refined', 1.)] + [
            (m, policy['chosen'][m]['retention']) for m in ('random', 'geometry', 'pfad')]
    # Source selection has no target CT dependency.
    masks = {(m, f): selection(scores[m], counts, f) if m != 'all_refined'
             else np.ones(len(raw), bool) for m, f in configurations}
    target_clouds, targets = target_data(specimen, anatomy)
    perturbations = [r for r in previous['records'] if r['method'] == 'raw' and
                     (stage == 'outer' or (r['level'] > 0 and r['draw'] == 0))]
    assert len(perturbations) == (16 if stage == 'outer' else 3)
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.HuberLoss(k=2.))
    records = []
    if stage == 'outer':
        records.extend([dict(r, reused_from_original=True) for r in perturbations])
    full_rows = None
    for method, fraction in configurations:
        output_method = method + '_policy' if stage == 'outer' and method != 'all_refined' else method
        if fraction == 1. and full_rows is not None:
            records.extend([dict(r, method=output_method, shared_full_retention_result=True) for r in full_rows])
            continue
        points = refined[masks[(method, fraction)]]
        clouds = [base.voxel(points, v, cap=6000, seed=previous['case_seed'] + 3 + k)
                  for k, v in enumerate((2., 1., .5))]
        current = []
        for perturb in perturbations:
            D = np.array(perturb['imposed_transform'])
            T = np.eye(4)
            error, fitness = None, 0.
            start = time.perf_counter()
            try:
                for k, (gate, iterations) in enumerate(zip((30., 15., 5.), (50, 30, 20))):
                    cloud = o3d.geometry.PointCloud(clouds[k])
                    cloud.transform(D)
                    result = o3d.pipelines.registration.registration_icp(cloud, target_clouds[k], gate, T,
                        estimator, o3d.pipelines.registration.ICPConvergenceCriteria(
                            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iterations))
                    T = np.asarray(result.transformation).copy()
                    fitness = float(result.fitness)
                assert np.isfinite(T).all()
                assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
                assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-6
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                T = np.eye(4)
            elapsed = time.perf_counter() - start
            row = base.pose_metrics(T, D, targets)
            if error or fitness == 0:
                row['registration_recall'] = 0.
            row.update({k: perturb[k] for k in ('specimen', 'anatomy', 'condition', 'level', 'draw')})
            row.update(method=output_method, retention=fraction, selected_count=len(points),
                       registration_only_seconds=elapsed, error=error,
                       fitness=fitness, imposed_transform=D.tolist(), estimated_transform=T.tolist())
            current.append(row)
        if fraction == 1.:
            full_rows = current
        records.extend(current)
    payload = dict(status='complete', stage=stage, outer=outer, specimen=specimen,
        anatomy=anatomy, records=records, **stamp, target_mesh_sha256=previous['target_mesh_sha256'],
        case_seconds=time.perf_counter() - tick)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, separators=(',', ':')))
    temporary.replace(dest)
    return dict(case=name, stage=stage, outer=outer, records=len(records),
                seconds=round(payload['case_seconds'], 2), errors=sum(r['error'] is not None for r in records))


def select_policies():
    (OUT / 'policies').mkdir(exist_ok=True)
    for outer in SPECIMENS:
        training = [s for s in SPECIMENS if s != outer]
        rows, inputs = [], {}
        for specimen in training:
            for anatomy in base.ANATOMIES:
                for subdir in ('controls', outer):
                    path = OUT / 'inner_registration' / subdir / f'{specimen}_{anatomy}.json'
                    d = json.loads(path.read_text())
                    assert d['status'] == 'complete' and d['protocol_sha256'] == sha(CONFIG)
                    assert d['specimen'] != outer
                    rows.extend(d['records'])
                    inputs[str(path.relative_to(OUT))] = sha(path)
        candidates, chosen = {}, {}
        for method in ('random', 'geometry', 'pfad'):
            candidates[method] = []
            for fraction in FRACTIONS:
                values = []
                for specimen in training:
                    anatomy_values = []
                    for anatomy in base.ANATOMIES:
                        group = [r for r in rows if r['method'] == method and r['retention'] == fraction
                                 and r['specimen'] == specimen and r['anatomy'] == anatomy]
                        assert len(group) == 3
                        anatomy_values.append(np.mean([r['virtual_target_rms_displacement_mm'] for r in group]))
                    values.append(float(np.mean(anatomy_values)))
                candidates[method].append(dict(retention=fraction, mean_virtual_rms_mm=float(np.mean(values)),
                                               specimen_values=values))
            chosen[method] = min(candidates[method], key=lambda v: (v['mean_virtual_rms_mm'], -v['retention']))
        payload = dict(status='frozen_before_outer_registration', outer=outer, training_specimens=training,
                       chosen=chosen, candidates=candidates, input_sha256=inputs, protocol_sha256=sha(CONFIG))
        path = OUT / 'policies' / (outer + '.json')
        if path.exists():
            assert json.loads(path.read_text()) == payload
        else:
            path.write_text(json.dumps(payload, indent=2))
        print('POLICY', outer, {m: c['retention'] for m, c in chosen.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['controls', 'inner', 'outer'], required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    assert json.loads(CONFIG.read_text())['retention_candidates'] == list(FRACTIONS)
    assert base.sanity()['metric_convention_verified']
    if args.stage == 'controls':
        jobs = [('controls', s, a, None) for s, a in CASES]
    elif args.stage == 'inner':
        for s in SPECIMENS:
            assert (OUT / 'inner_scores' / (s + '.json')).is_file()
        jobs = [('inner', s, a, outer) for outer in SPECIMENS for s, a in CASES if s != outer]
    else:
        select_policies()
        jobs = [('outer', s, a, s) for s, a in CASES]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(register_case, jobs)):
            if args.stage != 'inner' or i % 13 == 0 or i == len(jobs) - 1:
                print(json.dumps(result), flush=True)
    print('COMPLETE', args.stage, len(jobs), flush=True)


if __name__ == '__main__':
    main()
