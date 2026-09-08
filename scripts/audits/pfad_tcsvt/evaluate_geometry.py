#!/usr/bin/env python3
"""Evaluate already frozen predictions; fixed original CT-visible reference.

Run with the isolated Python 3.10/Open3D environment. No training or model
selection is imported or called by this evaluator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / "src"))
from pfad_tcsvt.common import ANATOMIES,RESULT,SPECIMENS,dump_json,experiment_directory,load_pfad,sha256
from pfad_tcsvt.acquisition import DATA,limit_tree_threads,load_payload

M=load_pfad()
ENDPOINTS=("chamfer_l1_mm","hd95_mm","fscore_1mm","normal_consistency","precision_1mm","coverage_1mm")


def rays_from_payload(payload):
    return M.Rays(payload["ray_origins_mm"].astype(np.float64),payload["ray_directions"].astype(np.float64),
        payload["ray_depths_mm"].astype(np.float64),payload["ray_confidence"].astype(np.float64),
        payload["ray_record_frame"].astype(np.int64),payload["ray_column"].astype(np.int64))


def sample_existing_surface(path,ct_mesh,reference,scene):
    mesh=o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():raise RuntimeError(f"empty mesh {path}")
    o3d.utility.random.seed(260831)
    samples=np.asarray(mesh.sample_points_uniformly(number_of_points=50000).points)
    return M.point_set_metrics(ct_mesh,samples,reference,24,scene)


def make_comparator_surface(payload,ct_mesh,reference,scene,destination):
    rays=rays_from_payload(payload)
    keep=payload["selection_keep"].astype(bool)
    points=rays.origins_mm+(rays.depths_mm+payload["correction"].astype(np.float64))[:,None]*rays.directions
    record=np.repeat(np.arange(len(payload["counts"])),payload["counts"])
    surface=M.reconstruct_sweep_topology_surface(points[keep],rays.directions[keep],record[keep],
        rays.frame_index[keep],rays.column_index[keep],rays.confidence[keep],
        max_frame_gap=1,max_column_gap_px=64,max_edge_mm=5.0)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(destination),surface):raise IOError(destination)
    o3d.utility.random.seed(260831)
    samples=np.asarray(surface.sample_points_uniformly(number_of_points=50000).points)
    result=M.point_set_metrics(ct_mesh,samples,reference,24,scene)
    result.update({"input_points":int(keep.sum()),"mesh_vertices":len(surface.vertices),"mesh_triangles":len(surface.triangles),
                   "mesh_path":str(destination.relative_to(ROOT))})
    return result


def run_case(config_path,specimen,anatomy):
    config,output=experiment_directory(Path(config_path))
    _,training_root=experiment_directory(ROOT / config["training_config"])
    limit_tree_threads()
    name=f"{specimen}_{anatomy}"
    prediction_manifest=output / "predictions" / name / "complete.json"
    if not prediction_manifest.exists():raise RuntimeError("evaluation before prediction freeze")
    done=output / "metrics" / f"{name}.json"
    if done.exists():return json.loads(done.read_text())
    started=time.monotonic()
    prediction_rows=json.loads(prediction_manifest.read_text())["predictions"]
    original_path=RESULT / "selector_attentive_loocv/nested_policy" / f"{name}.npz"
    original=load_payload(original_path)
    original_rays=rays_from_payload(original)
    mesh_path=DATA / specimen / "CT_bone_segmentations" / f"{anatomy}.stl"
    hit,backend=M.ct_first_intersections(mesh_path,original_rays)
    finite=np.isfinite(hit)
    reference=original_rays.origins_mm[finite]+hit[finite,None]*original_rays.directions[finite]
    if not len(reference):raise RuntimeError("empty visible reference")
    ct_mesh=trimesh.load_mesh(mesh_path,process=True)
    scene=M.build_open3d_surface_scene(ct_mesh)
    if scene is None:raise RuntimeError("Open3D closest-surface backend required")
    np.savez_compressed(output / "metrics" / f"{name}_reference.npz",points_mm=reference)
    rows=[]
    for prediction_row in prediction_rows:
        prediction_path=ROOT / prediction_row["path"]
        if sha256(prediction_path)!=prediction_row["sha256"]:raise RuntimeError("prediction changed after freeze")
        prediction=load_payload(prediction_path)
        feature_path=ROOT / str(prediction["feature_path"].item())
        if sha256(feature_path)!=prediction_row["feature_sha256"]:raise RuntimeError("input changed after freeze")
        feature=load_payload(feature_path)
        rays=rays_from_payload(feature)
        raw=rays.points_mm
        corrected=rays.origins_mm+(rays.depths_mm+prediction["correction"].astype(np.float64))[:,None]*rays.directions
        points={"raw":raw,"cross_sweep_nearest":corrected[prediction["geometry_keep"].astype(bool)],
                "pfad":corrected[prediction["pfad_keep"].astype(bool)],
                "set_transformer":corrected[prediction["set_transformer_keep"].astype(bool)]}
        metrics={method:M.point_set_metrics(ct_mesh,cloud,reference,24,scene) for method,cloud in points.items()}
        rows.append({"condition":prediction_row["condition"],"metrics":metrics,"queries":len(raw),"records":len(feature["records"])})
    legacy_point=json.loads((RESULT / "selector_attentive_loocv/nested_policy" / f"{name}.pointset.json").read_text())
    original_points=original_rays.origins_mm+(original_rays.depths_mm+original["correction"].astype(np.float64))[:,None]*original_rays.directions
    original_metric=M.point_set_metrics(ct_mesh,original_points[original["selection_keep"].astype(bool)],reference,24,scene)
    legacy_surface_path=RESULT / "surface_attentive_nested/sweep_topology_f1_c64_e5" / name
    original_surface=sample_existing_surface(legacy_surface_path / "deployed.ply",ct_mesh,reference,scene)
    st_archive=training_root / "predictions/set_transformer" / f"{name}.npz"
    if not st_archive.exists():
        # The clean sensitivity prediction uses the same selected outer model.
        # Reuse its frozen arrays when evaluating a downloaded model release.
        clean_path=next(ROOT / r["path"] for r in prediction_rows if r["condition"]["family"]=="clean")
        clean_prediction=load_payload(clean_path)
        clean_feature=load_payload(ROOT / str(clean_prediction["feature_path"].item()))
        clean_feature.update(selection_keep=clean_prediction["set_transformer_keep"],correction=clean_prediction["correction"])
        st_archive=output / "predictions" / name / "clean_set_transformer.npz"
        np.savez_compressed(st_archive,**clean_feature)
    st_surface=make_comparator_surface(load_payload(st_archive),ct_mesh,reference,scene,output / "meshes" / name / "set_transformer.ply")
    old_surface=json.loads((legacy_surface_path / "surface.json").read_text())["metrics"]["deployed"]
    clean=next(r for r in rows if r["condition"]["family"]=="clean")
    qa={"visible_reference_count":len(reference),"old_visible_reference_count":legacy_point["visible_reference_points"],
        "raw_metric_abs_differences":{m:abs(clean["metrics"]["raw"][m]-legacy_point["all_points_metrics"]["raw"][m]) for m in ENDPOINTS},
        "original_pfad_metric_abs_differences":{m:abs(original_metric[m]-legacy_point["all_points_metrics"]["selector_refined"][m]) for m in ENDPOINTS},
        "original_surface_metric_abs_differences":{m:abs(original_surface[m]-old_surface[m]) for m in ENDPOINTS}}
    if qa["visible_reference_count"]!=qa["old_visible_reference_count"]:raise RuntimeError("visibility reference drift")
    report={"status":"complete","specimen":specimen,"anatomy":anatomy,"conditions":rows,
            "original_pfad_point_metrics":original_metric,"clean_surface_metrics":{"pfad_original":original_surface,"set_transformer":st_surface},
            "qa":qa,"reference_backend":backend,"seconds":time.monotonic()-started,
            "prediction_manifest_sha256":sha256(prediction_manifest),"CT_sha256":sha256(mesh_path)}
    dump_json(done,report)
    print(json.dumps({"stage":"case_evaluated","case":name,"seconds":report["seconds"],
                      "max_legacy_point_difference":max(qa["original_pfad_metric_abs_differences"].values()),
                      "max_legacy_mesh_difference":max(qa["original_surface_metric_abs_differences"].values())}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT / "configs/pfad/tcsvt/acquisition_robustness_v1.json")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--wait",action="store_true")
    parser.add_argument("--case",action="append")
    args=parser.parse_args()
    _,output=experiment_directory(args.config)
    (output / "metrics").mkdir(parents=True,exist_ok=True)
    code={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT / "scripts/run_first_arrival_refinement.py"]}
    dump_json(output / "evaluation_run.json",{"status":"running","code_sha256":code,"open3d":o3d.__version__,
        "trimesh":trimesh.__version__,"numpy":np.__version__,"reference":"fixed complete original acquisition"})
    tasks=[tuple(c.split("/")) for c in args.case] if args.case else [(s,a) for s in SPECIMENS for a in ANATOMIES]
    submitted=set()
    pending={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as executor:
        while True:
            for s,a in tasks:
                if (s,a) in submitted or (output / "metrics" / f"{s}_{a}.json").exists():continue
                if (output / "predictions" / f"{s}_{a}/complete.json").exists():
                    future=executor.submit(run_case,str(args.config.resolve()),s,a)
                    pending[future]=(s,a);submitted.add((s,a))
            if pending:
                completed,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result();del pending[future]
            complete=sum((output / "metrics" / f"{s}_{a}.json").exists() for s,a in tasks)
            if complete==len(tasks):break
            if not pending:
                if not args.wait:break
                time.sleep(5)
    all_done=all((output / "metrics" / f"{s}_{a}.json").exists() for s in SPECIMENS for a in ANATOMIES)
    dump_json(output / "evaluation_run.json",{"status":"complete" if all_done else "partial","code_sha256":code,
        "open3d":o3d.__version__,"trimesh":trimesh.__version__,"numpy":np.__version__})


if __name__=="__main__":main()
