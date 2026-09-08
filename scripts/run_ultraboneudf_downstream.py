#!/usr/bin/env python3
"""Run UltraBoneUDF downstream of frozen CT-free selector predictions.

This adapter keeps the official UltraBoneUDF network, data preparation, loss,
optimizer schedule, and DualMesh-UDF extraction unchanged.  It changes only
the experiment plumbing that the official entry point hard-codes:

* frozen anatomy-level predictions are split back into acquisition records;
* raw and deployed point clouds are supplied as two matched inputs; and
* both outputs are evaluated against one fixed ray-visible CT reference.

The CT reference is opened only by ``evaluate``.  It is never used while
exporting inputs or fitting the per-record unsigned-distance network.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
import torch
import torch.nn.functional as F
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ROOT = PROJECT_ROOT / "third_party" / "UltraBoneUDF"
DEFAULT_OFFICIAL_CONF = DEFAULT_OFFICIAL_ROOT / "conf" / "conf_UltraBones100k.conf"


def archive_scalar(archive: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(archive[key]).item())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def method_point_sets(
    prediction: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.ndarray], str]:
    origins = prediction["ray_origins_mm"].astype(np.float64)
    directions = prediction["ray_directions"].astype(np.float64)
    depths = prediction["ray_depths_mm"].astype(np.float64)
    correction = prediction["correction"].astype(np.float64)
    raw = origins + depths[:, None] * directions
    refined = origins + (depths + correction)[:, None] * directions

    action = archive_scalar(prediction, "deployment_action")
    if action == "selector_refined":
        keep = prediction["selection_keep"].astype(bool)
        deployed = refined[keep]
    elif action == "selector_raw":
        keep = prediction["selection_keep"].astype(bool)
        deployed = raw[keep]
    elif action == "relaxed_anchor":
        keep = np.ones(len(raw), dtype=bool)
        deployed = refined
    elif action == "raw":
        keep = np.ones(len(raw), dtype=bool)
        deployed = raw
    else:
        raise ValueError(f"unsupported deployment action: {action}")

    return {
        "raw": raw,
        "deployed": deployed,
        "raw_keep": np.ones(len(raw), dtype=bool),
        "deployed_keep": keep,
    }, action


def export_case(args: argparse.Namespace) -> None:
    prediction_path = Path(args.prediction_archive).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    with np.load(prediction_path) as prediction:
        specimen = archive_scalar(prediction, "specimen")
        anatomy = archive_scalar(prediction, "anatomy")
        records = tuple(str(value) for value in prediction["records"])
        counts = prediction["counts"].astype(np.int64)
        if int(np.sum(counts)) != len(prediction["ray_depths_mm"]):
            raise ValueError("record counts do not match frozen ray arrays")
        point_sets, action = method_point_sets(prediction)

        cursor = 0
        for record_index, (record, count_value) in enumerate(
            zip(records, counts, strict=True)
        ):
            count = int(count_value)
            selected = slice(cursor, cursor + count)
            for method in ("raw", "deployed"):
                local_keep = point_sets[f"{method}_keep"][selected]
                local_points = point_sets[method]
                if method == "raw":
                    record_points = local_points[selected]
                else:
                    global_indices = np.flatnonzero(
                        point_sets["deployed_keep"]
                    )
                    record_start = int(np.searchsorted(global_indices, cursor))
                    record_end = int(
                        np.searchsorted(global_indices, cursor + count)
                    )
                    record_points = local_points[record_start:record_end]
                if len(record_points) != int(np.sum(local_keep)):
                    raise RuntimeError(f"point split failed for {record}")
                if len(record_points) < 100:
                    raise RuntimeError(
                        f"{method}/{record} retained fewer than 100 points"
                    )
                relative = Path("inputs") / method / f"{record}.xyz"
                target = output_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(target, record_points, fmt="%.8f")
                cache = target.with_suffix(".pt")
                if cache.exists():
                    cache.unlink()
                entries.append(
                    {
                        "method": method,
                        "record": record,
                        "record_index": record_index,
                        "points": int(len(record_points)),
                        "retained_fraction": float(np.mean(local_keep)),
                        "input_xyz": str(relative),
                    }
                )
            cursor += count

        manifest = {
            "status": "frozen_CT_free_inputs_exported",
            "specimen": specimen,
            "anatomy": anatomy,
            "records": list(records),
            "deployment_action": action,
            "source_prediction": str(prediction_path),
            "evaluation_stride": int(prediction["evaluation_stride"].item()),
            "evaluation_offset": int(prediction["evaluation_offset"].item()),
            "line_mode": archive_scalar(prediction, "line_mode"),
            "protocol": {
                "input_unit": "one_acquisition_record",
                "raw": "all_official_skeleton_samples_in_the_frozen_archive",
                "deployed": "frozen_outer_LOOCV_action_without_CT_at_inference",
                "CT_used_for_export": False,
                "paired_records": True,
            },
            "entries": entries,
        }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


def set_official_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_official_modules(official_root: Path):
    official_root = official_root.resolve()
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    from extensions.chamfer_dist import ChamferDistance
    from models.dataset import Dataset
    from models.networks import UltraBoneUDF
    from utils.DualMeshUDF import extract_mesh_from_udf
    from utils.read_conf import read_confs

    return Dataset, UltraBoneUDF, ChamferDistance, extract_mesh_from_udf, read_confs


def update_learning_rate(
    optimizer: torch.optim.Optimizer,
    iteration: int,
    maximum: int,
    warmup: int,
    initial: float,
) -> float:
    if iteration < warmup:
        factor = iteration / warmup
    else:
        factor = 0.5 * (
            math.cos((iteration - warmup) / (maximum - warmup) * math.pi)
            + 1.0
        )
    learning_rate = factor * initial
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def train_record(
    *,
    input_xyz: Path,
    output_dir: Path,
    official_root: Path,
    official_conf: Path,
    gpu: int,
    seed: int,
    maximum_iterations: int,
    batch_size: int,
    report_frequency: int,
    mesh_depth: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "run.json"
    mesh_path = output_dir / "mesh.obj"
    if result_path.is_file() and mesh_path.is_file() and not overwrite:
        result = read_json(result_path)
        if result.get("status") == "complete":
            print(f"reuse {result_path}", flush=True)
            return result

    if not input_xyz.is_file():
        raise FileNotFoundError(input_xyz)
    if maximum_iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("the official UltraBoneUDF implementation requires CUDA")
    torch.cuda.set_device(gpu)
    set_official_seed(seed)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    (
        Dataset,
        UltraBoneUDF,
        ChamferDistance,
        extract_mesh_from_udf,
        read_confs,
    ) = import_official_modules(official_root)
    conf = read_confs(str(official_conf))
    conf.put("dataset.pcd_file", str(input_xyz.resolve()))

    cache_path = input_xyz.with_suffix(".pt")
    if cache_path.exists() and cache_path.stat().st_mtime < input_xyz.stat().st_mtime:
        cache_path.unlink()

    preprocessing_start = time.perf_counter()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        dataset = Dataset(conf)
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    print(
        f"prepared {input_xyz.name}: {len(dataset.pcd_gt)} points in "
        f"{preprocessing_seconds:.1f}s",
        flush=True,
    )

    device = torch.device("cuda")
    network = UltraBoneUDF(**conf["model.udf_network"]).to(device)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=conf.get_float("train.learning_rate")
    )
    chamfer = ChamferDistance().to(device)
    lambda_nc = conf.get_float("train.lambda_NC")
    lambda_gs = conf.get_float("train.lambda_GS")
    initial_lr = conf.get_float("train.learning_rate")
    warmup = conf.get_int("train.warm_up_end")

    grid = dataset.grid_sparse
    grid_target = dataset.grid_sparse_udf_gt
    valid_grid = grid_target > 0.1
    grid = grid[valid_grid]
    grid_target = grid_target[valid_grid]

    training_start = time.perf_counter()
    final_loss = float("nan")
    final_nc = float("nan")
    final_gs = float("nan")
    for iteration in range(maximum_iterations):
        learning_rate = update_learning_rate(
            optimizer,
            iteration,
            maximum_iterations,
            warmup,
            initial_lr,
        )
        samples, _, _, point_cloud, _ = dataset.np_train_data(batch_size)
        optimizer.zero_grad(set_to_none=True)
        samples.requires_grad_(True)
        gradients = network.gradient(samples).squeeze()
        udf = network.udf(samples)
        gradient_direction = F.normalize(gradients, dim=1)
        moved = samples - gradient_direction * udf
        _, nearest_index, _ = chamfer(
            moved.unsqueeze(0), point_cloud.unsqueeze(0)
        )
        normal_constraint = torch.abs(
            (
                gradient_direction
                * (samples - point_cloud[nearest_index[0]])
            ).sum(dim=1, keepdim=True)
            - udf
        ).mean()
        grid_prediction = network.udf(grid)
        grid_loss = F.mse_loss(grid_prediction, grid_target)
        loss = lambda_nc * normal_constraint + lambda_gs * grid_loss
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        final_nc = float(normal_constraint.detach().cpu())
        final_gs = float(grid_loss.detach().cpu())
        completed = iteration + 1
        if completed % report_frequency == 0 or completed == maximum_iterations:
            elapsed = time.perf_counter() - training_start
            print(
                f"iter {completed}/{maximum_iterations} loss={final_loss:.6f} "
                f"lr={learning_rate:.3e} elapsed={elapsed:.1f}s",
                flush=True,
            )
    training_seconds = time.perf_counter() - training_start

    extraction_start = time.perf_counter()
    vertices, faces = extract_mesh_from_udf(
        network,
        "cuda",
        max_depth=mesh_depth,
        min_UDF_threshold=0.000001,
        max_UDF_threshold=0.01,
        singular_value_threshold=0.2,
    )
    extraction_seconds = time.perf_counter() - extraction_start
    if vertices is None or faces is None or len(vertices) == 0:
        raise RuntimeError("UltraBoneUDF returned an empty mesh")
    world_vertices = (
        np.asarray(vertices, dtype=np.float64) * float(dataset.shape_scale)
        + np.asarray(dataset.shape_center, dtype=np.float64)
    )
    surface = trimesh.Trimesh(
        vertices=world_vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    surface.export(mesh_path)

    result = {
        "status": "complete",
        "input_xyz": str(input_xyz.resolve()),
        "mesh": str(mesh_path.resolve()),
        "official_source_root": str(official_root.resolve()),
        "official_configuration": str(official_conf.resolve()),
        "seed": seed,
        "gpu": gpu,
        "maximum_iterations": maximum_iterations,
        "batch_size": batch_size,
        "mesh_depth": mesh_depth,
        "input_points_after_official_preprocessing": int(len(dataset.pcd_gt)),
        "shape_scale_mm": float(dataset.shape_scale),
        "shape_center_mm": np.asarray(dataset.shape_center).tolist(),
        "final_loss": final_loss,
        "final_normal_constraint": final_nc,
        "final_sparse_grid_loss": final_gs,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
        "extraction_seconds": extraction_seconds,
        "vertices": int(len(surface.vertices)),
        "faces": int(len(surface.faces)),
        "protocol": {
            "network": "official_UltraBoneUDF",
            "loss": "official_normal_constraint_plus_sparse_grid_loss",
            "optimizer": "official_Adam_cosine_schedule",
            "CT_used_during_fit": False,
            "intermediate_CT_validation_removed": True,
        },
    }
    write_json(result_path, result)
    print(result_path, flush=True)
    return result


def train_record_command(args: argparse.Namespace) -> None:
    train_record(
        input_xyz=Path(args.input_xyz).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        official_root=Path(args.official_root).resolve(),
        official_conf=Path(args.official_conf).resolve(),
        gpu=int(args.gpu),
        seed=int(args.seed),
        maximum_iterations=int(args.maxiter),
        batch_size=int(args.batch_size),
        report_frequency=int(args.report_frequency),
        mesh_depth=int(args.mesh_depth),
        overwrite=bool(args.overwrite),
    )


def selected_entries(
    manifest: dict[str, Any],
    methods: Iterable[str] | None,
    records: Iterable[str] | None,
) -> list[dict[str, Any]]:
    method_set = set(methods or ("raw", "deployed"))
    record_set = set(records or manifest["records"])
    entries = [
        entry
        for entry in manifest["entries"]
        if entry["method"] in method_set and entry["record"] in record_set
    ]
    expected = len(method_set) * len(record_set)
    if len(entries) != expected:
        raise ValueError(
            f"manifest resolved {len(entries)} entries, expected {expected}"
        )
    return entries


def run_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    run_root = Path(args.run_root).resolve()
    entries = selected_entries(manifest, args.method, args.record)
    for entry in entries:
        input_xyz = manifest_path.parent / entry["input_xyz"]
        output_dir = run_root / entry["method"] / entry["record"]
        train_record(
            input_xyz=input_xyz,
            output_dir=output_dir,
            official_root=Path(args.official_root).resolve(),
            official_conf=Path(args.official_conf).resolve(),
            gpu=int(args.gpu),
            seed=int(args.seed),
            maximum_iterations=int(args.maxiter),
            batch_size=int(args.batch_size),
            report_frequency=int(args.report_frequency),
            mesh_depth=int(args.mesh_depth),
            overwrite=bool(args.overwrite),
        )


def deterministic_mesh_samples(
    meshes: list[trimesh.Trimesh], count: int, seed: int
) -> np.ndarray:
    triangles = np.concatenate(
        [np.asarray(mesh.triangles, dtype=np.float64) for mesh in meshes],
        axis=0,
    )
    if len(triangles) == 0:
        raise ValueError("cannot sample empty meshes")
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    area = 0.5 * np.linalg.norm(cross, axis=1)
    valid = area > 0.0
    triangles = triangles[valid]
    area = area[valid]
    if len(triangles) == 0:
        raise ValueError("all mesh faces are degenerate")
    rng = np.random.default_rng(seed)
    triangle_index = rng.choice(
        len(triangles), size=count, replace=True, p=area / area.sum()
    )
    selected = triangles[triangle_index]
    first = np.sqrt(rng.random(count))
    second = rng.random(count)
    return (
        (1.0 - first)[:, None] * selected[:, 0]
        + (first * (1.0 - second))[:, None] * selected[:, 1]
        + (first * second)[:, None] * selected[:, 2]
    )


def support_gate_mesh(
    mesh: trimesh.Trimesh,
    support_points: np.ndarray,
    radius_mm: float,
) -> trimesh.Trimesh:
    """Retain UDF faces supported by the frozen CT-free deployed points."""
    if radius_mm <= 0.0:
        raise ValueError("support radius must be positive")
    if len(support_points) < 3:
        raise ValueError("support gating requires at least three points")
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    centroids = triangles.mean(axis=1)
    distance = cKDTree(support_points).query(centroids, workers=-1)[0]
    retained_faces = np.flatnonzero(distance <= radius_mm)
    if len(retained_faces) < 3:
        raise RuntimeError("support gating retained fewer than three faces")
    gated = mesh.submesh([retained_faces], append=True, repair=False)
    if not isinstance(gated, trimesh.Trimesh) or len(gated.faces) < 3:
        raise RuntimeError("support gating returned an invalid mesh")
    return gated


def record_slices(
    prediction: np.lib.npyio.NpzFile,
) -> dict[str, slice]:
    records = tuple(str(value) for value in prediction["records"])
    counts = prediction["counts"].astype(np.int64)
    result: dict[str, slice] = {}
    cursor = 0
    for record, count_value in zip(records, counts, strict=True):
        count = int(count_value)
        result[record] = slice(cursor, cursor + count)
        cursor += count
    if cursor != len(prediction["ray_depths_mm"]):
        raise ValueError("record counts do not cover the frozen rays")
    return result


def evaluate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    prediction_path = Path(manifest["source_prediction"])
    run_root = Path(args.run_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    methods = tuple(args.method or ("raw", "deployed"))
    records = tuple(args.record or manifest["records"])

    scripts_root = PROJECT_ROOT / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    import run_first_arrival_refinement as core

    with np.load(prediction_path) as prediction:
        slices = record_slices(prediction)
        missing = [record for record in records if record not in slices]
        if missing:
            raise ValueError(f"records absent from prediction: {missing}")
        indices = np.concatenate(
            [np.arange(slices[record].start, slices[record].stop) for record in records]
        )
        origins = prediction["ray_origins_mm"].astype(np.float64)[indices]
        directions = prediction["ray_directions"].astype(np.float64)[indices]

    specimen = str(manifest["specimen"])
    anatomy = str(manifest["anatomy"])
    ct_path = (
        dataset_root
        / specimen
        / "CT_bone_segmentations"
        / f"{anatomy}.stl"
    )
    ct_mesh = trimesh.load_mesh(ct_path, process=True)
    if not isinstance(ct_mesh, trimesh.Trimesh):
        raise TypeError(f"expected one CT mesh: {ct_path}")
    reference = core.mesh_first_intersection_points(
        ct_path, origins, directions
    )
    scene = core.build_open3d_surface_scene(ct_mesh)

    metrics: dict[str, Any] = {}
    manifest_entries = {
        (entry["method"], entry["record"]): entry
        for entry in manifest["entries"]
    }
    for method in methods:
        meshes: list[trimesh.Trimesh] = []
        total_vertices = 0
        total_faces = 0
        total_components = 0
        total_input_points = 0
        total_support_points = 0
        total_faces_before_gate = 0
        for record in records:
            reconstructed_method = (
                "raw" if method == "support_gated_raw" else method
            )
            mesh_path = run_root / reconstructed_method / record / "mesh.obj"
            mesh = trimesh.load_mesh(mesh_path, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError(f"expected one reconstructed mesh: {mesh_path}")
            total_faces_before_gate += int(len(mesh.faces))
            if method == "support_gated_raw":
                support_entry = manifest_entries[("deployed", record)]
                support_path = manifest_path.parent / support_entry["input_xyz"]
                support = np.loadtxt(support_path, dtype=np.float64, ndmin=2)
                mesh = support_gate_mesh(
                    mesh, support, float(args.support_radius_mm)
                )
                total_support_points += int(len(support))
            meshes.append(mesh)
            total_vertices += int(len(mesh.vertices))
            total_faces += int(len(mesh.faces))
            total_components += int(mesh.body_count)
            total_input_points += int(
                manifest_entries[(reconstructed_method, record)]["points"]
            )
        sampled = deterministic_mesh_samples(
            meshes, int(args.sample_points), int(args.sample_seed)
        )
        metrics[method] = {
            **core.point_set_metrics(
                ct_mesh,
                sampled,
                reference,
                int(args.normal_neighbours),
                scene,
            ),
            "input_points": total_input_points,
            "vertices": total_vertices,
            "faces": total_faces,
            "connected_components": total_components,
            "records": len(records),
            "faces_before_support_gate": total_faces_before_gate,
            "support_points": total_support_points,
        }

    result = {
        "case": {
            "specimen": specimen,
            "anatomy": anatomy,
            "records": list(records),
        },
        "protocol": {
            "reconstructor": "official_UltraBoneUDF_per_acquisition_record",
            "input_comparison": "raw_versus_frozen_CT_free_deployed",
            "reference": "same_target_CT_first_intersections_for_all_methods",
            "CT_used_during_UDF_fit": False,
            "surface_sampling": "deterministic_area_weighted",
            "sample_points": int(args.sample_points),
            "normal_neighbours": int(args.normal_neighbours),
            "support_gate": (
                f"raw_UDF_face_centroid_within_{float(args.support_radius_mm):g}mm_"
                "of_frozen_deployed_point"
                if "support_gated_raw" in methods
                else "not_evaluated"
            ),
            "scope": "downstream_compatibility_not_cross_protocol_table_ranking",
        },
        "visible_reference_points": int(len(reference)),
        "metrics": metrics,
    }
    output = Path(args.output).resolve()
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    paths = sorted(root.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no evaluations match {args.pattern!r} under {root}")
    cases = [read_json(path) for path in paths]
    specimens = [case["case"]["specimen"] for case in cases]
    if len(specimens) != len(set(specimens)):
        raise ValueError("summary requires one evaluation per specimen")
    if len(cases) != int(args.expected_cases):
        raise ValueError(
            f"found {len(cases)} cases, expected {int(args.expected_cases)}"
        )

    endpoint_direction = {
        "chamfer_l1_mm": "lower",
        "hd95_mm": "lower",
        "fscore_1mm": "higher",
        "normal_consistency": "higher",
    }
    endpoint_results: dict[str, Any] = {}
    raw_method = str(args.raw_method)
    compared_method = str(args.compared_method)
    p_values: dict[str, float] = {}
    rng = np.random.default_rng(int(args.bootstrap_seed))
    for endpoint, direction in endpoint_direction.items():
        raw = np.asarray(
            [case["metrics"][raw_method][endpoint] for case in cases],
            dtype=np.float64,
        )
        compared = np.asarray(
            [case["metrics"][compared_method][endpoint] for case in cases],
            dtype=np.float64,
        )
        gain = raw - compared if direction == "lower" else compared - raw
        bootstrap_index = rng.integers(
            0,
            len(cases),
            size=(int(args.bootstrap_resamples), len(cases)),
        )
        bootstrap_mean = gain[bootstrap_index].mean(axis=1)
        statistic = wilcoxon(gain, alternative="two-sided", method="auto")
        p_value = float(statistic.pvalue)
        p_values[endpoint] = p_value
        endpoint_results[endpoint] = {
            "direction": direction,
            "raw_mean": float(np.mean(raw)),
            "compared_mean": float(np.mean(compared)),
            "mean_gain": float(np.mean(gain)),
            "gain_95ci": [
                float(np.percentile(bootstrap_mean, 2.5)),
                float(np.percentile(bootstrap_mean, 97.5)),
            ],
            "wins": int(np.sum(gain > 0.0)),
            "ties": int(np.sum(gain == 0.0)),
            "losses": int(np.sum(gain < 0.0)),
            "minimum_gain": float(np.min(gain)),
            "wilcoxon_statistic": float(statistic.statistic),
            "wilcoxon_two_sided_p": p_value,
        }
    adjusted = holm_adjust(p_values)
    for endpoint, value in adjusted.items():
        endpoint_results[endpoint]["holm_adjusted_p"] = value

    result = {
        "status": "complete",
        "independent_unit": "specimen",
        "cases": len(cases),
        "specimens": specimens,
        "raw_method": raw_method,
        "compared_method": compared_method,
        "endpoint_results": endpoint_results,
        "protocol": {
            "subset": "lexicographically_first_acquisition_record_per_specimen",
            "bootstrap": "specimen_level_nonparametric_percentile",
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "paired_test": "two_sided_Wilcoxon_signed_rank",
            "multiplicity": "Holm_across_four_primary_geometry_endpoints",
        },
        "case_metrics": {
            specimen: case["metrics"]
            for specimen, case in zip(specimens, cases, strict=True)
        },
    }
    output = Path(args.output).resolve()
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--official-conf", default=str(DEFAULT_OFFICIAL_CONF))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--report-frequency", type=int, default=1_000)
    parser.add_argument("--mesh-depth", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export", help="export matched raw/deployed record point clouds"
    )
    export.add_argument("--prediction-archive", required=True)
    export.add_argument("--output-root", required=True)
    export.set_defaults(function=export_case)

    record = subparsers.add_parser(
        "train-record", help="fit official UltraBoneUDF to one point cloud"
    )
    record.add_argument("--input-xyz", required=True)
    record.add_argument("--output-dir", required=True)
    add_training_arguments(record)
    record.set_defaults(function=train_record_command)

    run = subparsers.add_parser(
        "run-manifest", help="fit selected entries from an exported manifest"
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--method", action="append", choices=("raw", "deployed"))
    run.add_argument("--record", action="append")
    add_training_arguments(run)
    run.set_defaults(function=run_manifest)

    evaluation = subparsers.add_parser(
        "evaluate", help="evaluate reconstructed meshes on one fixed reference"
    )
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--run-root", required=True)
    evaluation.add_argument("--dataset-root", required=True)
    evaluation.add_argument(
        "--method",
        action="append",
        choices=("raw", "deployed", "support_gated_raw"),
    )
    evaluation.add_argument("--record", action="append")
    evaluation.add_argument("--sample-points", type=int, default=100_000)
    evaluation.add_argument("--sample-seed", type=int, default=20260831)
    evaluation.add_argument("--normal-neighbours", type=int, default=24)
    evaluation.add_argument("--support-radius-mm", type=float, default=2.0)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(function=evaluate)

    summary = subparsers.add_parser(
        "summarize", help="summarize specimen-level downstream evaluations"
    )
    summary.add_argument("--root", required=True)
    summary.add_argument(
        "--pattern",
        default="specimen*_fibula/official_i30000_d9/"
        "evaluation_first_record_support_gate_2mm.json",
    )
    summary.add_argument("--expected-cases", type=int, default=14)
    summary.add_argument("--raw-method", default="raw")
    summary.add_argument("--compared-method", default="support_gated_raw")
    summary.add_argument("--bootstrap-resamples", type=int, default=100_000)
    summary.add_argument("--bootstrap-seed", type=int, default=20260831)
    summary.add_argument("--output", required=True)
    summary.set_defaults(function=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
