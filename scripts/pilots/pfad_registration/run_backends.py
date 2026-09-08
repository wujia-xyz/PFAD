"""Retrospective solver sensitivity of the frozen PFAD application experiment.

Adds the two standard least-squares ICP estimators to the existing Huber
point-to-plane experiment. It does not select a backend from test outcomes.
Every backend and every case is retained. See the frozen protocol for scope.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import platform
import sys
import time

import numpy as np
import open3d as o3d
import trimesh

ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = ROOT / 'scripts/audits/pfad_registration_application/run.py'
spec = importlib.util.spec_from_file_location('original_registration', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OUT = ROOT / 'artifacts/pfad_registration/solver_sensitivity_v1'
CONFIG = ROOT / 'configs/pfad/registration/solver_sensitivity_v1.json'
BACKENDS = ('point_to_point_l2', 'point_to_plane_l2')


def case_run(case):
    specimen, anatomy = case
    case_name = specimen + '_' + anatomy
    previous_path = base.OUT / 'cases' / (case_name + '.json')
    previous = json.loads(previous_path.read_text())
    archive = base.PRED / (case_name + '.npz')
    assert previous['input_prediction_sha256'] == base.sha(archive)
    seed = previous['case_seed']
    sources, center, metadata = base.build_sources(archive, seed)
    mesh_path = base.DATA / specimen / 'CT_bone_segmentations' / (anatomy + '.stl')
    mesh = trimesh.load_mesh(mesh_path, process=False)
    assert previous['target_mesh_sha256'] == base.sha(mesh_path)
    points, normals = base.sample_mesh(mesh, 80000, seed + 1)
    targets, _ = base.sample_mesh(mesh, 4096, seed + 2)
    targets -= center
    target_clouds = [base.voxel(points - center, v, normals) for v in (2., 1., .5)]
    source_clouds = {m: [base.voxel(p, v, cap=6000, seed=seed + 3 + k)
                        for k, v in enumerate((2., 1., .5))] for m, p in sources.items()}
    perturbations = [r for r in previous['records'] if r['method'] == 'raw']
    assert len(perturbations) == 16
    status = []
    for backend in BACKENDS:
        destination = OUT / backend / 'cases' / (case_name + '.json')
        if destination.exists():
            existing = json.loads(destination.read_text())
            assert existing['protocol_sha256'] == base.sha(CONFIG)
            assert existing['input_prediction_sha256'] == base.sha(archive)
            assert existing['executed_source_sha256'] == base.sha(__file__)
            assert existing['status'] == 'complete'
            status.append({'backend': backend, 'reused': True})
            continue
        if backend == 'point_to_point_l2':
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        else:
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        rows = []
        tick = time.perf_counter()
        for index, perturb in enumerate(perturbations):
            D = np.array(perturb['imposed_transform'])
            order = base.METHODS[index % 4:] + base.METHODS[:index % 4]
            for method in order:
                T = np.eye(4)
                error = None
                fitness = 0.
                start = time.perf_counter()
                try:
                    for k, (gate, iterations) in enumerate(zip((30., 15., 5.), (50, 30, 20))):
                        cloud = o3d.geometry.PointCloud(source_clouds[method][k])
                        cloud.transform(D)
                        result = o3d.pipelines.registration.registration_icp(
                            cloud, target_clouds[k], gate, T, estimator,
                            o3d.pipelines.registration.ICPConvergenceCriteria(
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
                row.update(method=method, backend=backend, registration_only_seconds=elapsed,
                           fitness=fitness, error=error, imposed_transform=D.tolist(),
                           estimated_transform=T.tolist())
                rows.append(row)
        payload = dict(status='complete', specimen=specimen, anatomy=anatomy, backend=backend,
                       records=rows, metadata=metadata, input_prediction_sha256=base.sha(archive),
                       target_mesh_sha256=base.sha(mesh_path), protocol_sha256=base.sha(CONFIG),
                       original_case_sha256=base.sha(previous_path), executed_source_sha256=base.sha(__file__),
                       case_seconds=time.perf_counter() - tick)
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, separators=(',', ':')))
        temporary.replace(destination)
        status.append(dict(backend=backend, seconds=round(payload['case_seconds'], 2),
                           errors=sum(r['error'] is not None for r in rows)))
    return dict(case=case_name, results=status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    assert config['added_backends'] == list(BACKENDS)
    OUT.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        (OUT / backend / 'cases').mkdir(parents=True, exist_ok=True)
        frozen = OUT / backend / 'protocol.json'
        if frozen.exists():
            assert base.sha(frozen) == base.sha(CONFIG)
        else:
            frozen.write_bytes(CONFIG.read_bytes())
    provenance = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                      open3d=o3d.__version__, numpy=np.__version__, trimesh=trimesh.__version__,
                      source_sha256=base.sha(__file__), base_source_sha256=base.sha(BASE_PATH),
                      protocol_sha256=base.sha(CONFIG), numerical_sanity=base.sanity())
    (OUT / 'execution.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)
    cases = [(f'specimen{sid:02}', a) for sid in range(1, 15) for a in base.ANATOMIES]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(case_run, cases):
            print(json.dumps(result), flush=True)
    for backend in BACKENDS:
        base.OUT = OUT / backend
        base.summarize()


if __name__ == '__main__':
    main()
