"""Independent moment-form check of all reported virtual-target displacements."""
from pathlib import Path
import importlib.util
import json
from functools import lru_cache
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('sampling', Path(__file__).with_name('run.py'))
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)
APP = ROOT / 'artifacts/pfad_registration'


@lru_cache(maxsize=42)
def moments(specimen, anatomy):
    case = json.loads((sampling.OUT / 'cases' / f'{specimen}_{anatomy}.json').read_text())
    center = np.array(case['metadata']['reference_center_from_raw_ultrasound_mm'])
    mesh = trimesh.load_mesh(sampling.DATA / specimen / 'CT_bone_segmentations' / (anatomy + '.stl'), process=False)
    points, _ = sampling.sample_mesh(mesh, 4096, case['case_seed'] + 2)
    points -= center
    mean = points.mean(axis=0)
    covariance = (points - mean).T @ (points - mean) / len(points)
    return mean, covariance


if __name__ == '__main__':
    roots = [sampling.OUT, APP / 'solver_sensitivity_v1/point_to_point_l2',
             APP / 'solver_sensitivity_v1/point_to_plane_l2', APP / 'convergence_v1',
             APP / 'nested_retention_v1']
    maximum = np.zeros(3)
    total = 0
    for root in roots:
        paths = sorted((root / 'cases').glob('*.json'))
        assert len(paths) == 42
        for path in paths:
            case = json.loads(path.read_text())
            mean, covariance = moments(case['specimen'], case['anatomy'])
            for row in case['records']:
                T, D = np.array(row['estimated_transform']), np.array(row['imposed_transform'])
                R = T[:3, :3] @ D[:3, :3]
                t = T[:3, :3] @ D[:3, 3] + T[:3, 3]
                A = R - np.eye(3)
                rms = np.sqrt(max(0., np.trace(A @ covariance @ A.T) + np.sum((A @ mean + t) ** 2)))
                rre = np.rad2deg(Rotation.from_matrix(R).magnitude())
                rte = np.linalg.norm(T[:3, 3] + D[:3, :3].T @ D[:3, 3])
                error = np.abs(np.array([rms, rre, rte]) - np.array([
                    row['virtual_target_rms_displacement_mm'], row['rotation_error_deg'], row['translation_error_mm']]))
                maximum = np.maximum(maximum, error)
                assert error[0] < 1e-8 and error[1] < 1e-4 and error[2] < 1e-8
                total += 1
    result = dict(status='pass', rows_checked=total,
                  independent_formula='trace((R-I) Cov (R-I)^T) + norm((R-I) mean + t)^2',
                  independent_rotation='SciPy rotation-vector magnitude',
                  independent_inverse_translation='-R_D^T t_D',
                  maximum_difference_rms_mm_rotation_deg_translation_mm=maximum.tolist())
    (APP / 'pose_metric_QA.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
