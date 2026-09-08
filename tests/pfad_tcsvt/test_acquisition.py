from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "src"))
from pfad_tcsvt.acquisition import M,conditions,seeded_rng,subset_rays,transform_rays
from pfad_tcsvt.common import experiment_directory


def test_conditions_cover_all_frame_phases_without_duplicate_ids():
    cfg,_ = experiment_directory(ROOT / "configs/pfad/tcsvt/acquisition_robustness_v1.json")
    rows = conditions(cfg)
    assert len(rows)==22
    assert len({r["id"] for r in rows})==22
    for stride in [2,4]:
        assert sorted(r["replicate"] for r in rows if r["family"]=="frames" and r["level"]==stride)==list(range(stride))


def test_record_pose_error_transforms_points_and_rays_consistently():
    rng=np.random.default_rng(3)
    origins=rng.normal(size=(17,3))*50
    directions=rng.normal(size=(17,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    ray=M.Rays(origins,directions,np.arange(17)+2,np.ones(17),np.arange(17),np.arange(17)*4)
    rotation=Rotation.from_rotvec(np.array([.01,-.02,.03])).as_matrix()
    translation=np.array([.2,.5,-.7]);pivot=ray.points_mm.mean(0)
    changed=transform_rays(ray,rotation,translation,pivot)
    np.testing.assert_allclose(changed.points_mm,(ray.points_mm-pivot)@rotation.T+pivot+translation,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(changed.directions,axis=1),1,atol=1e-12)
    np.testing.assert_array_equal(changed.depths_mm,ray.depths_mm)
    thinned=subset_rays(changed,changed.frame_index%4==2)
    np.testing.assert_array_equal(thinned.frame_index,np.array([2,6,10,14]))


def test_replicates_are_deterministic_and_severity_uses_shared_draws():
    a=seeded_rng(20260908,"specimen01","foot","pose",1).normal(size=6)
    b=seeded_rng(20260908,"specimen01","foot","pose",1).normal(size=6)
    c=seeded_rng(20260908,"specimen01","foot","pose",2).normal(size=6)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,c)
