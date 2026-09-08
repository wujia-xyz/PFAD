"""Tenfold ICP iteration budget, fixed before outcomes; every case is retained."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import concurrent.futures
import importlib.util
import json
import time
import numpy as np
import open3d as o3d
import trimesh

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('base', ROOT / 'scripts/audits/pfad_registration_application/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CONFIG = ROOT / 'configs/pfad/registration/convergence_v1.json'
OUT = ROOT / 'artifacts/pfad_registration/convergence_v1'


def run_case(case):
    s, a = case
    dest = OUT / 'cases' / f'{s}_{a}.json'
    pred = base.PRED / f'{s}_{a}.npz'
    stamp = dict(protocol_sha256=base.sha(CONFIG), executed_source_sha256=base.sha(__file__),
                 input_prediction_sha256=base.sha(pred))
    if dest.exists():
        d = json.loads(dest.read_text())
        assert d['status'] == 'complete' and all(d[k] == v for k, v in stamp.items())
        return dict(case=s + '_' + a, reused=True)
    previous = json.loads((base.OUT / 'cases' / f'{s}_{a}.json').read_text())
    seed = previous['case_seed']
    sources, center, metadata = base.build_sources(pred, seed)
    mesh_path = base.DATA / s / 'CT_bone_segmentations' / (a + '.stl')
    mesh = trimesh.load_mesh(mesh_path, process=False)
    points, normals = base.sample_mesh(mesh, 80000, seed + 1)
    targets, _ = base.sample_mesh(mesh, 4096, seed + 2)
    targets -= center
    target_clouds = [base.voxel(points - center, v, normals) for v in (2., 1., .5)]
    source_clouds = {m: [base.voxel(p, v, cap=6000, seed=seed + 3 + k)
                         for k, v in enumerate((2., 1., .5))] for m, p in sources.items()}
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    rows = []
    start_case = time.perf_counter()
    for index, perturb in enumerate(r for r in previous['records'] if r['method'] == 'raw'):
        D = np.array(perturb['imposed_transform'])
        for method in base.METHODS[index % 4:] + base.METHODS[:index % 4]:
            start = time.perf_counter()
            T, error, fitness = np.eye(4), None, 0.
            try:
                for k, (gate, iterations) in enumerate(zip((30., 15., 5.), (500, 300, 200))):
                    cloud = o3d.geometry.PointCloud(source_clouds[method][k])
                    cloud.transform(D)
                    result = o3d.pipelines.registration.registration_icp(cloud, target_clouds[k], gate, T,
                        estimator, o3d.pipelines.registration.ICPConvergenceCriteria(
                            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iterations))
                    T = np.asarray(result.transformation).copy()
                    fitness = float(result.fitness)
                assert np.isfinite(T).all()
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                T = np.eye(4)
            elapsed = time.perf_counter() - start
            row = base.pose_metrics(T, D, targets)
            if error or fitness == 0:
                row['registration_recall'] = 0.
            row.update({k: perturb[k] for k in ('specimen', 'anatomy', 'condition', 'level', 'draw')})
            row.update(method=method, registration_only_seconds=elapsed, fitness=fitness, error=error,
                       imposed_transform=D.tolist(), estimated_transform=T.tolist())
            rows.append(row)
    payload = dict(status='complete', specimen=s, anatomy=a, records=rows,
                   target_mesh_sha256=base.sha(mesh_path), **stamp,
                   metadata=metadata, case_seconds=time.perf_counter() - start_case)
    dest.write_text(json.dumps(payload, separators=(',', ':')))
    return dict(case=s + '_' + a, seconds=round(payload['case_seconds'], 2))


if __name__ == '__main__':
    assert json.loads(CONFIG.read_text())['iterations'] == [500, 300, 200]
    (OUT / 'protocol.json').write_bytes(CONFIG.read_bytes())
    (OUT / 'executed_runner.py').write_bytes(Path(__file__).read_bytes())
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(run_case, [(f'specimen{i:02}', a) for i in range(1, 15) for a in base.ANATOMIES]):
            print(json.dumps(result), flush=True)
    print('COMPLETE 42 cases, 2688 registrations', flush=True)
