"""PFAD scientific implementation: tracked rays, privileged targets, support selection and surface geometry."""
from __future__ import annotations


import argparse


import gc


import json


import math


import random


import re


from dataclasses import dataclass


from pathlib import Path


from typing import Iterable, Sequence


import numpy as np


import pandas as pd


from PIL import Image


from scipy.spatial import cKDTree


from scipy.spatial.transform import Rotation


from skimage.morphology import skeletonize


import torch


from torch import nn


from torch.nn import functional as F


from torch.utils.data import DataLoader, TensorDataset


import trimesh


CALIBRATION_TRANSLATION_MM = np.asarray(
    [26.44694442, -0.52572229, 128.00100047], dtype=np.float64
)


CALIBRATION_EULER_DEG = np.asarray(
    [92.48865621, -0.46874914, 179.2277322], dtype=np.float64
)


TIMESTAMP_PATTERN = re.compile(r"^(\d+)_label_pred\.png$")


THRESHOLD = 127


TRAIN_STRIDE = 8


TRAIN_OFFSET = 0


EVALUATION_STRIDE = 32


EVALUATION_OFFSET = 4


NEIGHBOURS_PER_SWEEP = 32


MLS_DISTANCE_OFFSET_MM = 0.5


PLANE_RAY_COSINE_FLOOR = 0.15


TANGENT_RAY_COSINE_FLOOR = 0.35


SUPPORT_SCALE_MM = 1.5


CASE_SUPPORT_FLOOR = 1.0 / 3.0


LEARNED_SHRINKAGE = 0.5


MAX_CORRECTION_MM = 2.0


FEATURE_WIDTH = 82


ANALYTIC_RELAXATION = 0.25


AGREEMENT_SCALE_MM = 1.0


SOURCE_EVIDENCE_NAMES = (
    "plane_delta_mm",
    "nearest_mm",
    "neighbourhood_median_mm",
    "surface_variation",
    "linearity_fraction",
    "plane_ray_cosine",
    "source_confidence",
    "source_query_ray_cosine",
)


ANATOMIES = ("foot", "tibia", "fibula")


TEACHER_SURFACE_TOLERANCE_MM = 2.0


TEACHER_ANATOMY_TEMPERATURE_MM = 1.0


TEACHER_SURFACE_TEMPERATURE_MM = 0.25


STUDENT_SOURCE_WIDTH = len(SOURCE_EVIDENCE_NAMES)


STUDENT_GLOBAL_NAMES = (
    "response_confidence",
    "depth_fraction",
    "column_fraction",
    "frame_fraction",
    "task_x",
    "task_y",
    "task_z",
    "beam_x",
    "beam_y",
    "beam_z",
    "plane_centre",
    "plane_disagreement",
    "agreement",
    "cross_sweep_nearest",
    "cross_sweep_neighbourhood",
    "source_count",
    "anatomy_foot",
    "anatomy_tibia",
    "anatomy_fibula",
)


STUDENT_GLOBAL_WIDTH = len(STUDENT_GLOBAL_NAMES)


SELECTOR_TARGET_RECALL = 0.985


MINIMUM_RECORD_RETENTION = 0.90


@dataclass(frozen=True)
class Rays:
    origins_mm: np.ndarray
    directions: np.ndarray
    depths_mm: np.ndarray
    confidence: np.ndarray
    frame_index: np.ndarray
    column_index: np.ndarray

    def __len__(self) -> int:
        return int(self.depths_mm.shape[0])

    @property
    def points_mm(self) -> np.ndarray:
        return self.origins_mm + self.depths_mm[:, None] * self.directions


@dataclass(frozen=True)
class RecordData:
    specimen: str
    anatomy: str
    record: str
    record_root: Path
    mesh_path: Path
    image_height: int
    image_width: int
    scale_mm: float
    rows: tuple[pd.Series, ...]
    prediction_paths: tuple[Path, ...]


def transform_matrix(
    translation: Sequence[float], euler_xyz_deg: Sequence[float]
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler(
        "xyz", euler_xyz_deg, degrees=True
    ).as_matrix()
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def image_to_world(row: pd.Series) -> np.ndarray:
    pose = transform_matrix(
        row[["x", "y", "z"]].to_numpy(dtype=np.float64),
        row[["euler_x", "euler_y", "euler_z"]].to_numpy(dtype=np.float64),
    )
    calibration = transform_matrix(
        CALIBRATION_TRANSLATION_MM, CALIBRATION_EULER_DEG
    )
    return pose @ calibration


def load_record(
    dataset_root: Path, specimen: str, anatomy: str, record: str
) -> RecordData:
    specimen_root = dataset_root / specimen
    record_root = specimen_root / "ultrasound_records" / anatomy / record
    tracking_path = record_root / "tracking.csv"
    prediction_root = record_root / "Labels_pred"
    mesh_path = specimen_root / "CT_bone_segmentations" / f"{anatomy}.stl"
    for required in (tracking_path, prediction_root, mesh_path):
        if not required.exists():
            raise FileNotFoundError(required)

    frame_table = pd.read_csv(tracking_path)
    by_timestamp = {
        int(float(row.timestamp)): row for _, row in frame_table.iterrows()
    }
    paired: list[tuple[int, Path, pd.Series]] = []
    for path in prediction_root.glob("*_label_pred.png"):
        match = TIMESTAMP_PATTERN.match(path.name)
        if match is None:
            continue
        timestamp = int(match.group(1))
        if timestamp in by_timestamp:
            paired.append((timestamp, path, by_timestamp[timestamp]))
    paired.sort(key=lambda item: item[0])
    if not paired:
        raise RuntimeError(f"no prediction/tracking pairs under {record_root}")

    sample = np.asarray(Image.open(paired[0][1]))
    if sample.ndim != 2:
        raise ValueError(f"expected a single-channel response: {paired[0][1]}")
    height, width = map(int, sample.shape)
    scales = np.asarray([float(item[2].space_factor) for item in paired])
    if not np.allclose(scales, scales[0], rtol=0.0, atol=1e-8):
        raise ValueError(f"pixel spacing changes within {record_root}")
    return RecordData(
        specimen=specimen,
        anatomy=anatomy,
        record=record,
        record_root=record_root,
        mesh_path=mesh_path,
        image_height=height,
        image_width=width,
        scale_mm=float(scales[0]),
        rows=tuple(item[2] for item in paired),
        prediction_paths=tuple(item[1] for item in paired),
    )


def extract_rays(
    data: RecordData,
    *,
    stride: int,
    offset: int,
    threshold: int = THRESHOLD,
    line_mode: str = "skeleton",
) -> Rays:
    if stride <= 0 or not 0 <= offset < stride:
        raise ValueError("require stride > 0 and 0 <= offset < stride")
    if line_mode not in {"top", "midrun", "centroid", "skeleton"}:
        raise ValueError(f"unsupported line mode: {line_mode}")
    columns = np.arange(offset, data.image_width, stride, dtype=np.int64)
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    column_indices: list[np.ndarray] = []
    for frame_index, (row, path) in enumerate(
        zip(data.rows, data.prediction_paths, strict=True)
    ):
        response = np.asarray(Image.open(path), dtype=np.uint8)
        mask = response >= threshold
        if line_mode == "skeleton":
            line = skeletonize(mask)
            sampled_line = line[:, columns]
            selected_rows_int, sampled_column_indices = np.nonzero(sampled_line)
            if len(selected_rows_int) == 0:
                continue
            selected_columns_int = columns[sampled_column_indices]
            selected_rows = selected_rows_int.astype(np.float64)
            peak_values = response[selected_rows_int, selected_columns_int]
        else:
            sampled = response[:, columns]
            peak_rows = np.argmax(sampled, axis=0)
            peak_values_all = sampled[
                peak_rows, np.arange(columns.shape[0])
            ]
            keep = peak_values_all >= threshold
            if not np.any(keep):
                continue
            selected_columns_int = columns[keep]
            peak_values = peak_values_all[keep]
            if line_mode == "top":
                selected_rows = peak_rows[keep].astype(np.float64)
            else:
                selected_rows_list = []
                for column in columns[keep]:
                    rows = np.flatnonzero(mask[:, column])
                    if line_mode == "midrun":
                        selected_rows_list.append(0.5 * (rows[0] + rows[-1]))
                    else:
                        selected_rows_list.append(float(np.mean(rows)))
                selected_rows = np.asarray(selected_rows_list, dtype=np.float64)
        selected_columns = selected_columns_int.astype(np.float64) + 0.5
        image_rows = selected_rows + 0.5
        world = image_to_world(row)
        local_origins = np.column_stack(
            [
                selected_columns * data.scale_mm,
                np.full(selected_columns.shape, 0.5 * data.scale_mm),
                np.zeros(selected_columns.shape),
                np.ones(selected_columns.shape),
            ]
        )
        world_origins = (world @ local_origins.T).T[:, :3]
        direction = world[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
        direction /= np.linalg.norm(direction)
        origins.append(world_origins)
        directions.append(
            np.repeat(direction[None, :], len(selected_rows), axis=0)
        )
        depths.append((image_rows - 0.5) * data.scale_mm)
        confidences.append(peak_values.astype(np.float64) / 255.0)
        frame_indices.append(
            np.full(len(selected_rows), frame_index, dtype=np.int64)
        )
        column_indices.append(selected_columns_int.astype(np.int64))
    if not origins:
        raise RuntimeError(f"no ray survived thresholding in {data.record_root}")
    return Rays(
        origins_mm=np.concatenate(origins),
        directions=np.concatenate(directions),
        depths_mm=np.concatenate(depths),
        confidence=np.concatenate(confidences),
        frame_index=np.concatenate(frame_indices),
        column_index=np.concatenate(column_indices),
    )


class SweepSetSelector(nn.Module):
    """Permutation-invariant student over the other tracked sweeps."""

    def __init__(self) -> None:
        super().__init__()
        self.source_encoder = nn.Sequential(
            nn.Linear(STUDENT_SOURCE_WIDTH, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * 32 + STUDENT_GLOBAL_WIDTH, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 3),
        )

    def forward(
        self,
        source_evidence: torch.Tensor,
        source_mask: torch.Tensor,
        global_evidence: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.source_encoder(source_evidence)
        mask = source_mask.unsqueeze(2)
        mean = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~mask.bool(), -torch.inf).amax(dim=1)
        return self.fusion(torch.cat([mean, maximum, global_evidence], dim=1))


class PooledEvidenceSelector(nn.Module):
    """Capacity-matched MLP on fixed mean/max source summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * STUDENT_SOURCE_WIDTH + STUDENT_GLOBAL_WIDTH, 112),
            nn.LayerNorm(112),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(112, 88),
            nn.GELU(),
            nn.Linear(88, 3),
        )

    def forward(
        self,
        source_evidence: torch.Tensor,
        source_mask: torch.Tensor,
        global_evidence: torch.Tensor,
    ) -> torch.Tensor:
        mask = source_mask.unsqueeze(2)
        mean = (source_evidence * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        maximum = source_evidence.masked_fill(~mask.bool(), -torch.inf).amax(dim=1)
        return self.fusion(torch.cat([mean, maximum, global_evidence], dim=1))


class PointwiseSelector(nn.Module):
    """Capacity-matched pointwise MLP without per-record set evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STUDENT_GLOBAL_WIDTH, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(128, 88),
            nn.GELU(),
            nn.Linear(88, 3),
        )

    def forward(
        self,
        source_evidence: torch.Tensor,
        source_mask: torch.Tensor,
        global_evidence: torch.Tensor,
    ) -> torch.Tensor:
        del source_evidence, source_mask
        return self.network(global_evidence)


class SweepOnlySelector(nn.Module):
    """Set encoder ablation with all global query features removed."""

    def __init__(self) -> None:
        super().__init__()
        self.source_encoder = nn.Sequential(
            nn.Linear(STUDENT_SOURCE_WIDTH, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * 32, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 3),
        )

    def forward(
        self,
        source_evidence: torch.Tensor,
        source_mask: torch.Tensor,
        global_evidence: torch.Tensor,
    ) -> torch.Tensor:
        del global_evidence
        encoded = self.source_encoder(source_evidence)
        mask = source_mask.unsqueeze(2)
        mean = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~mask.bool(), -torch.inf).amax(dim=1)
        return self.fusion(torch.cat([mean, maximum], dim=1))


class AttentiveSweepSelector(nn.Module):
    """Query-conditioned attention over independent acquisition records."""

    def __init__(self) -> None:
        super().__init__()
        self.source_encoder = nn.Sequential(
            nn.Linear(STUDENT_SOURCE_WIDTH, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(STUDENT_GLOBAL_WIDTH, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.attention = nn.Linear(32, 1, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(2 * 32 + STUDENT_GLOBAL_WIDTH, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 3),
        )

    def forward(
        self,
        source_evidence: torch.Tensor,
        source_mask: torch.Tensor,
        global_evidence: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.source_encoder(source_evidence)
        query = self.query_encoder(global_evidence).unsqueeze(1)
        attention_logit = self.attention(torch.tanh(encoded + query)).squeeze(2)
        attention_logit = attention_logit.masked_fill(
            ~source_mask.bool(), -torch.inf
        )
        attention_weight = torch.softmax(attention_logit, dim=1).unsqueeze(2)
        attended = (attention_weight * encoded).sum(dim=1)
        maximum = encoded.masked_fill(
            ~source_mask.bool().unsqueeze(2), -torch.inf
        ).amax(dim=1)
        return self.fusion(
            torch.cat([attended, maximum, global_evidence], dim=1)
        )


SELECTOR_VARIANTS = (
    "sweep_set",
    "attentive_set",
    "pooled_summary",
    "pointwise",
    "sweep_only",
)


def build_sweep_selector(variant: str) -> nn.Module:
    if variant == "sweep_set":
        return SweepSetSelector()
    if variant == "attentive_set":
        return AttentiveSweepSelector()
    if variant == "pooled_summary":
        return PooledEvidenceSelector()
    if variant == "pointwise":
        return PointwiseSelector()
    if variant == "sweep_only":
        return SweepOnlySelector()
    raise ValueError(f"unsupported selector variant: {variant}")


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def archive_scalar(archive: np.lib.npyio.NpzFile, key: str) -> str:
    value = archive[key]
    return str(value.item() if value.ndim == 0 else value[0])


def archive_int(
    archive: np.lib.npyio.NpzFile, key: str, default: int
) -> int:
    if key not in archive.files:
        return int(default)
    value = archive[key]
    return int(value.item() if value.ndim == 0 else value[0])


RAY_ARCHIVE_KEYS = (
    "ray_origins_mm",
    "ray_directions",
    "ray_depths_mm",
    "ray_confidence",
    "ray_record_frame",
    "ray_column",
)


def archived_rays(
    archive: np.lib.npyio.NpzFile, selected: slice
) -> Rays | None:
    if not all(key in archive.files for key in RAY_ARCHIVE_KEYS):
        return None
    return Rays(
        origins_mm=archive["ray_origins_mm"][selected].astype(np.float64),
        directions=archive["ray_directions"][selected].astype(np.float64),
        depths_mm=archive["ray_depths_mm"][selected].astype(np.float64),
        confidence=archive["ray_confidence"][selected].astype(np.float64),
        frame_index=archive["ray_record_frame"][selected].astype(np.int64),
        column_index=archive["ray_column"][selected].astype(np.int64),
    )


def record_index_from_counts(counts: np.ndarray, total: int) -> np.ndarray:
    indices = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    if len(indices) != total:
        raise ValueError("record counts do not match archived rays")
    return indices


def normalized_source_evidence(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[2] != STUDENT_SOURCE_WIDTH:
        raise ValueError(
            "source evidence must have shape [points, sources, "
            f"{STUDENT_SOURCE_WIDTH}]"
        )
    normalized = raw.astype(np.float32, copy=True)
    normalized[:, :, 0] = np.clip(normalized[:, :, 0] / 2.0, -4.0, 4.0)
    normalized[:, :, 1] = np.clip(normalized[:, :, 1] / 5.0, 0.0, 8.0)
    normalized[:, :, 2] = np.clip(normalized[:, :, 2] / 5.0, 0.0, 8.0)
    normalized[:, :, 3:] = np.clip(normalized[:, :, 3:], 0.0, 1.0)
    return normalized


def student_global_evidence(
    archive: np.lib.npyio.NpzFile,
) -> np.ndarray:
    required = (
        "counts",
        "record_image_widths",
        "record_image_heights",
        "record_scale_mm",
        "record_frame_counts",
        "ray_origins_mm",
        "ray_directions",
        "ray_depths_mm",
        "ray_confidence",
        "ray_record_frame",
        "ray_column",
        "plane_centre_mm",
        "plane_disagreement_mm",
        "agreement",
        "cross_sweep_nearest_mm",
        "cross_sweep_neighbourhood_mm",
    )
    missing = [key for key in required if key not in archive.files]
    if missing:
        raise ValueError(f"anchor archive lacks student inputs: {missing}")
    total = len(archive["ray_depths_mm"])
    counts = archive["counts"].astype(np.int64)
    record_index = record_index_from_counts(counts, total)
    widths = archive["record_image_widths"].astype(np.float32)[record_index]
    heights = archive["record_image_heights"].astype(np.float32)[record_index]
    scales = archive["record_scale_mm"].astype(np.float32)[record_index]
    frame_counts = archive["record_frame_counts"].astype(np.float32)[record_index]
    depths = archive["ray_depths_mm"].astype(np.float32)
    origins = archive["ray_origins_mm"].astype(np.float32)
    directions = archive["ray_directions"].astype(np.float32)
    points = origins + depths[:, None] * directions
    centre = np.median(points, axis=0)
    task_scale = np.percentile(points, 90, axis=0) - np.percentile(
        points, 10, axis=0
    )
    task_scale = np.maximum(task_scale, 1.0)
    task_coordinates = np.clip((points - centre) / task_scale, -3.0, 3.0)
    anatomy = archive_scalar(archive, "anatomy")
    anatomy_one_hot = np.zeros((total, len(ANATOMIES)), dtype=np.float32)
    anatomy_one_hot[:, ANATOMIES.index(anatomy)] = 1.0
    source_count = int(archive["source_evidence"].shape[1])
    matrix = np.column_stack(
        [
            archive["ray_confidence"].astype(np.float32),
            depths / np.maximum(heights * scales, 1e-6),
            (archive["ray_column"].astype(np.float32) + 0.5)
            / np.maximum(widths, 1.0),
            archive["ray_record_frame"].astype(np.float32)
            / np.maximum(frame_counts - 1.0, 1.0),
            task_coordinates,
            directions,
            np.clip(
                archive["plane_centre_mm"].astype(np.float32) / 2.0,
                -4.0,
                4.0,
            ),
            np.clip(
                archive["plane_disagreement_mm"].astype(np.float32) / 2.0,
                0.0,
                4.0,
            ),
            archive["agreement"].astype(np.float32),
            np.clip(
                archive["cross_sweep_nearest_mm"].astype(np.float32) / 5.0,
                0.0,
                8.0,
            ),
            np.clip(
                archive["cross_sweep_neighbourhood_mm"].astype(np.float32)
                / 5.0,
                0.0,
                8.0,
            ),
            np.full(total, source_count / 5.0, dtype=np.float32),
            anatomy_one_hot,
        ]
    ).astype(np.float32)
    if matrix.shape[1] != STUDENT_GLOBAL_WIDTH:
        raise RuntimeError(
            f"expected {STUDENT_GLOBAL_WIDTH} global features, got {matrix.shape[1]}"
        )
    return matrix


def load_student_case(
    feature_path: Path,
    max_sources: int,
    label_path: Path | None = None,
) -> dict[str, object]:
    with np.load(feature_path) as feature:
        if "source_evidence" not in feature.files:
            raise ValueError(f"archive predates sweep-set evidence: {feature_path}")
        raw_source = feature["source_evidence"].astype(np.float32)
        if raw_source.shape[1] > max_sources:
            raise ValueError("max_sources is smaller than an archived source set")
        source = np.zeros(
            (len(raw_source), max_sources, STUDENT_SOURCE_WIDTH),
            dtype=np.float32,
        )
        mask = np.zeros((len(raw_source), max_sources), dtype=np.float32)
        source[:, : raw_source.shape[1]] = normalized_source_evidence(raw_source)
        mask[:, : raw_source.shape[1]] = 1.0
        case: dict[str, object] = {
            "feature_path": feature_path,
            "specimen": archive_scalar(feature, "specimen"),
            "anatomy": archive_scalar(feature, "anatomy"),
            "source": source,
            "mask": mask,
            "global": student_global_evidence(feature),
            "counts": feature["counts"].astype(np.int64),
        }
    if label_path is not None:
        with np.load(label_path) as label:
            if archive_scalar(label, "specimen") != case["specimen"]:
                raise ValueError(f"specimen mismatch: {feature_path} and {label_path}")
            if archive_scalar(label, "anatomy") != case["anatomy"]:
                raise ValueError(f"anatomy mismatch: {feature_path} and {label_path}")
            hard = np.column_stack(
                [
                    label["anatomy_target"],
                    label["surface_valid"],
                    label["joint_target"],
                ]
            ).astype(np.float32)
            soft = np.column_stack(
                [
                    label["teacher_anatomy_probability"],
                    label["teacher_surface_probability"],
                    label["teacher_joint_probability"],
                ]
            ).astype(np.float32)
        if len(hard) != len(source):
            raise ValueError(f"label length mismatch: {label_path}")
        case["hard_target"] = hard
        case["soft_target"] = soft
        case["label_path"] = label_path
    return case


def source_sweep_plane_delta(
    tree: cKDTree, source: Rays, query: Rays
) -> np.ndarray:
    """Return one source sweep's local, CT-free evidence for each query ray."""
    neighbour_count = min(NEIGHBOURS_PER_SWEEP, len(source))
    distances, indices = tree.query(
        query.points_mm, k=neighbour_count, workers=-1
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    points = source.points_mm[indices]
    weights = source.confidence[indices] / (
        distances + MLS_DISTANCE_OFFSET_MM
    )
    weight_sum = weights.sum(axis=1)
    centres = np.einsum("nk,nkj->nj", weights, points) / weight_sum[:, None]
    centred = points - centres[:, None, :]
    covariance = np.einsum(
        "nk,nki,nkj->nij", weights, centred, centred
    ) / weight_sum[:, None, None]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    plane_cosine = np.einsum("ni,ni->n", normals, query.directions)
    stable = np.abs(plane_cosine) >= PLANE_RAY_COSINE_FLOOR
    plane_depth = np.einsum(
        "ni,ni->n", normals, centres - query.origins_mm
    ) / np.where(stable, plane_cosine, 1.0)
    plane_delta = np.where(stable, plane_depth - query.depths_mm, 0.0)
    eigenvalue_sum = eigenvalues.sum(axis=1) + 1e-8
    source_directions = source.directions[indices]
    ray_cosine = np.abs(
        np.einsum("nkj,nj->nk", source_directions, query.directions)
    )
    return np.column_stack(
        [
            plane_delta,
            distances[:, 0],
            np.median(distances, axis=1),
            eigenvalues[:, 0] / eigenvalue_sum,
            (eigenvalues[:, 2] - eigenvalues[:, 1])
            / (eigenvalues[:, 2] + 1e-8),
            np.abs(plane_cosine),
            source.confidence[indices].mean(axis=1),
            ray_cosine.mean(axis=1),
        ]
    ).astype(np.float32)


def case_anchor_arrays(
    training_rays: Sequence[Rays], evaluation_rays: Sequence[Rays]
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build the record-balanced analytic anchor for one anatomy."""
    if len(training_rays) < 2 or len(training_rays) != len(evaluation_rays):
        raise ValueError("a case requires at least two matched sweeps")
    trees = [cKDTree(rays.points_mm) for rays in training_rays]
    anchors: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    disagreements: list[np.ndarray] = []
    nearest_support: list[np.ndarray] = []
    neighbourhood_support: list[np.ndarray] = []
    source_evidence: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    counts: list[int] = []
    frame_offset = 0
    for record_index, query in enumerate(evaluation_rays):
        evidence = np.stack(
            [
            source_sweep_plane_delta(trees[index], source, query)
            for index, source in enumerate(training_rays)
            if index != record_index
            ],
            axis=1,
        )
        source_deltas = evidence[:, :, 0]
        source_nearest = evidence[:, :, 1]
        source_neighbourhood = evidence[:, :, 2]
        centre = np.median(source_deltas, axis=1)
        disagreement = np.median(
            np.abs(source_deltas - centre[:, None]), axis=1
        )
        reliability = np.clip(
            1.0 - disagreement / AGREEMENT_SCALE_MM, 0.0, 1.0
        )
        anchor = np.clip(
            centre * reliability,
            -MAX_CORRECTION_MM,
            MAX_CORRECTION_MM,
        )
        anchors.append(anchor.astype(np.float32))
        centres.append(centre.astype(np.float32))
        disagreements.append(disagreement.astype(np.float32))
        nearest_support.append(
            np.median(source_nearest, axis=1).astype(np.float32)
        )
        neighbourhood_support.append(
            np.median(source_neighbourhood, axis=1).astype(np.float32)
        )
        source_evidence.append(evidence)
        frames.append((query.frame_index + frame_offset).astype(np.int32))
        counts.append(len(query))
        frame_offset += int(query.frame_index.max()) + 1
    return (
        np.concatenate(anchors),
        np.concatenate(centres),
        np.concatenate(disagreements),
        np.concatenate(nearest_support),
        np.concatenate(neighbourhood_support),
        np.concatenate(source_evidence),
        np.concatenate(frames),
        np.asarray(counts, dtype=np.int64),
    )


def resolve_records(
    root: Path, specimen: str, anatomy: str, requested: Sequence[str] | None
) -> tuple[str, ...]:
    records = tuple(requested or ())
    if not records:
        record_root = root / specimen / "ultrasound_records" / anatomy
        records = tuple(
            path.name
            for path in sorted(record_root.glob("record*"))
            if path.is_dir()
        )
    if len(records) < 2:
        raise ValueError("multi-sweep correction requires at least two records")
    return records


def save_analytic_anchor(args: argparse.Namespace) -> None:
    """Build only the active zero-parameter analytic state."""
    root = Path(args.dataset_root)
    records = resolve_records(
        root, args.specimen, args.anatomy, args.records
    )
    data = [
        load_record(root, args.specimen, args.anatomy, record)
        for record in records
    ]
    training = [
        extract_rays(
            item,
            stride=TRAIN_STRIDE,
            offset=TRAIN_OFFSET,
            line_mode="skeleton",
        )
        for item in data
    ]
    evaluation = [
        extract_rays(
            item,
            stride=int(args.evaluation_stride),
            offset=int(args.evaluation_offset),
            line_mode="skeleton",
        )
        for item in data
    ]
    (
        anchor,
        centre,
        disagreement,
        nearest_support,
        neighbourhood_support,
        source_evidence,
        frames,
        counts,
    ) = case_anchor_arrays(training, evaluation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        analytic=anchor,
        plane_centre_mm=centre,
        plane_disagreement_mm=disagreement,
        cross_sweep_nearest_mm=nearest_support,
        cross_sweep_neighbourhood_mm=neighbourhood_support,
        source_evidence=source_evidence,
        source_evidence_names=np.asarray(SOURCE_EVIDENCE_NAMES),
        agreement=np.clip(
            1.0 - disagreement / AGREEMENT_SCALE_MM, 0.0, 1.0
        ).astype(np.float32),
        frame=frames,
        counts=counts,
        records=np.asarray(records),
        record_image_widths=np.asarray(
            [item.image_width for item in data], dtype=np.int32
        ),
        record_image_heights=np.asarray(
            [item.image_height for item in data], dtype=np.int32
        ),
        record_scale_mm=np.asarray(
            [item.scale_mm for item in data], dtype=np.float32
        ),
        record_frame_counts=np.asarray(
            [len(item.rows) for item in data], dtype=np.int32
        ),
        specimen=np.asarray(args.specimen),
        anatomy=np.asarray(args.anatomy),
        evaluation_stride=np.asarray(int(args.evaluation_stride)),
        evaluation_offset=np.asarray(int(args.evaluation_offset)),
        line_mode=np.asarray("skeleton"),
        ray_origins_mm=np.concatenate(
            [rays.origins_mm for rays in evaluation]
        ).astype(np.float32),
        ray_directions=np.concatenate(
            [rays.directions for rays in evaluation]
        ).astype(np.float32),
        ray_depths_mm=np.concatenate(
            [rays.depths_mm for rays in evaluation]
        ).astype(np.float32),
        ray_confidence=np.concatenate(
            [rays.confidence for rays in evaluation]
        ).astype(np.float32),
        ray_record_frame=np.concatenate(
            [rays.frame_index for rays in evaluation]
        ).astype(np.int32),
        ray_column=np.concatenate(
            [rays.column_index for rays in evaluation]
        ).astype(np.int32),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rays": int(len(anchor)),
                "records": len(records),
                "median_agreement": float(
                    np.median(
                        np.clip(
                            1.0 - disagreement / AGREEMENT_SCALE_MM,
                            0.0,
                            1.0,
                        )
                    )
                ),
                "median_cross_sweep_nearest_mm": float(
                    np.median(nearest_support)
                ),
                "trainable_parameters": 0,
            },
            indent=2,
        )
    )


def ray_intersector(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        return RayMeshIntersector(mesh), "embree"
    except ImportError:
        from trimesh.ray.ray_triangle import RayMeshIntersector

        return RayMeshIntersector(mesh), "triangle"


def ct_first_intersections(
    mesh_path: Path, rays: Rays, chunk_size: int = 50_000
) -> tuple[np.ndarray, str]:
    """Return the nearest forward CT hit depth for every acquisition ray."""
    try:
        import open3d as o3d

        legacy = o3d.io.read_triangle_mesh(str(mesh_path))
        if legacy.is_empty():
            raise ValueError(f"empty triangle mesh: {mesh_path}")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        depths: list[np.ndarray] = []
        for start in range(0, len(rays), chunk_size):
            packed_rays = np.column_stack(
                [
                    rays.origins_mm[start : start + chunk_size],
                    rays.directions[start : start + chunk_size],
                ]
            ).astype(np.float32)
            depths.append(
                scene.cast_rays(o3d.core.Tensor(packed_rays))["t_hit"]
                .numpy()
                .astype(np.float64)
            )
        return np.concatenate(depths), "open3d"
    except ImportError:
        pass

    mesh = trimesh.load_mesh(mesh_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a triangle mesh: {mesh_path}")
    intersector, backend = ray_intersector(mesh)
    locations, ray_indices, _ = intersector.intersects_location(
        rays.origins_mm, rays.directions, multiple_hits=False
    )
    depths = np.full(len(rays), np.inf, dtype=np.float64)
    if len(ray_indices):
        depths[ray_indices] = np.einsum(
            "ij,ij->i",
            locations - rays.origins_mm[ray_indices],
            rays.directions[ray_indices],
        )
    return depths, backend


def mesh_surface_distance(
    mesh_path: Path, points_mm: np.ndarray, chunk_size: int = 20_000
) -> np.ndarray:
    """Compute exact point-to-triangle distances without retaining CT geometry."""
    try:
        import open3d as o3d

        legacy = o3d.io.read_triangle_mesh(str(mesh_path))
        if legacy.is_empty():
            raise ValueError(f"empty triangle mesh: {mesh_path}")
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        distances = []
        for start in range(0, len(points_mm), chunk_size):
            query = o3d.core.Tensor(
                points_mm[start : start + chunk_size].astype(np.float32)
            )
            distances.append(scene.compute_distance(query).numpy())
        return np.concatenate(distances).astype(np.float32)
    except ImportError:
        pass

    mesh = trimesh.load_mesh(mesh_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a triangle mesh: {mesh_path}")
    distances: list[np.ndarray] = []
    for start in range(0, len(points_mm), chunk_size):
        _, distance, _ = mesh.nearest.on_surface(
            points_mm[start : start + chunk_size]
        )
        distances.append(distance.astype(np.float32))
    return np.concatenate(distances)


def build_privileged_teacher_labels(args: argparse.Namespace) -> None:
    """Open CT only to create training labels for the CT-free selector."""
    source = np.load(args.input_archive)
    missing = [key for key in RAY_ARCHIVE_KEYS if key not in source.files]
    if missing:
        raise ValueError(f"anchor archive is missing ray geometry: {missing}")
    specimen = archive_scalar(source, "specimen")
    anatomy = archive_scalar(source, "anatomy")
    points_mm = (
        source["ray_origins_mm"].astype(np.float64)
        + source["ray_depths_mm"].astype(np.float64)[:, None]
        * source["ray_directions"].astype(np.float64)
    )
    specimen_root = Path(args.dataset_root) / specimen
    surface_distances = np.column_stack(
        [
            mesh_surface_distance(
                specimen_root / "CT_bone_segmentations" / f"{name}.stl",
                points_mm,
            )
            for name in ANATOMIES
        ]
    ).astype(np.float32)
    target_index = ANATOMIES.index(anatomy)
    target_distance = surface_distances[:, target_index]
    nearest_anatomy = np.argmin(surface_distances, axis=1)
    anatomy_target = nearest_anatomy == target_index
    surface_valid = target_distance <= TEACHER_SURFACE_TOLERANCE_MM
    joint_target = anatomy_target & surface_valid

    anatomy_logits = -surface_distances / TEACHER_ANATOMY_TEMPERATURE_MM
    anatomy_logits -= anatomy_logits.max(axis=1, keepdims=True)
    anatomy_weights = np.exp(anatomy_logits)
    anatomy_probability = (
        anatomy_weights[:, target_index] / anatomy_weights.sum(axis=1)
    )
    surface_probability = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (
                    target_distance - TEACHER_SURFACE_TOLERANCE_MM
                )
                / TEACHER_SURFACE_TEMPERATURE_MM,
                -40.0,
                40.0,
            )
        )
    )
    joint_probability = anatomy_probability * surface_probability

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        specimen=np.asarray(specimen),
        anatomy=np.asarray(anatomy),
        anatomy_target=anatomy_target.astype(np.uint8),
        surface_valid=surface_valid.astype(np.uint8),
        joint_target=joint_target.astype(np.uint8),
        teacher_anatomy_probability=anatomy_probability.astype(np.float32),
        teacher_surface_probability=surface_probability.astype(np.float32),
        teacher_joint_probability=joint_probability.astype(np.float32),
        target_surface_distance_mm=target_distance,
        nearest_anatomy=nearest_anatomy.astype(np.uint8),
        anatomy_order=np.asarray(ANATOMIES),
        settings=np.asarray(
            json.dumps(
                {
                    "role": "training_only_privileged_CT_teacher",
                    "target_surface_tolerance_mm": (
                        TEACHER_SURFACE_TOLERANCE_MM
                    ),
                    "anatomy_temperature_mm": (
                        TEACHER_ANATOMY_TEMPERATURE_MM
                    ),
                    "surface_temperature_mm": (
                        TEACHER_SURFACE_TEMPERATURE_MM
                    ),
                    "inference_uses_CT": False,
                }
            )
        ),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "specimen": specimen,
                "anatomy": anatomy,
                "points": int(len(points_mm)),
                "anatomy_target_fraction": float(np.mean(anatomy_target)),
                "surface_valid_fraction": float(np.mean(surface_valid)),
                "joint_target_fraction": float(np.mean(joint_target)),
            },
            indent=2,
        )
    )


def reversed_record_rank_partners(
    cases: Sequence[dict[str, object]], target_name: str
) -> np.ndarray:
    """Pair each candidate with its reverse-rank peer in the same record."""
    total = int(sum(len(case["source"]) for case in cases))
    partners = np.empty(total, dtype=np.int64)
    case_offset = 0
    for case in cases:
        target = np.asarray(case[target_name], dtype=np.float32)[:, 2]
        cursor = 0
        for count_value in np.asarray(case["counts"], dtype=np.int64):
            count = int(count_value)
            local = np.arange(cursor, cursor + count, dtype=np.int64)
            order = local[np.argsort(target[local], kind="stable")]
            partners[case_offset + order] = case_offset + order[::-1]
            cursor += count
        if cursor != len(target):
            raise ValueError("record counts do not match selector targets")
        case_offset += len(target)
    return partners


def train_sweep_set_selector(
    cases: Sequence[dict[str, object]],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    student_variant: str = "sweep_set",
    teacher_target: str = "soft",
    ranking_weight: float = 0.0,
) -> tuple[nn.Module, list[float]]:
    if not cases:
        raise ValueError("at least one labelled training case is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    source = np.concatenate([case["source"] for case in cases])
    mask = np.concatenate([case["mask"] for case in cases])
    global_evidence = np.concatenate([case["global"] for case in cases])
    if teacher_target not in ("soft", "hard"):
        raise ValueError("teacher_target must be 'soft' or 'hard'")
    soft_target = np.concatenate(
        [case[f"{teacher_target}_target"] for case in cases]
    )
    hard_target = np.concatenate([case["hard_target"] for case in cases])
    total = len(source)
    case_weight_parts = [
        np.full(
            len(case["source"]),
            total / (len(cases) * len(case["source"])),
            dtype=np.float32,
        )
        for case in cases
    ]
    case_weight = np.concatenate(case_weight_parts)
    target_weight = np.empty_like(hard_target, dtype=np.float32)
    for index in range(hard_target.shape[1]):
        positive_fraction = float(np.mean(hard_target[:, index]))
        positive_fraction = float(np.clip(positive_fraction, 1e-3, 1.0 - 1e-3))
        target_weight[:, index] = np.where(
            hard_target[:, index] > 0.5,
            0.5 / positive_fraction,
            0.5 / (1.0 - positive_fraction),
        )
    target_weight *= case_weight[:, None]
    target_weight /= target_weight.mean(axis=0, keepdims=True)

    source_tensor = torch.from_numpy(source).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)
    global_tensor = torch.from_numpy(global_evidence).to(device)
    soft_tensor = torch.from_numpy(soft_target).to(device)
    weight_tensor = torch.from_numpy(target_weight).to(device)
    partner_tensor: torch.Tensor | None = None
    if float(ranking_weight) > 0.0:
        partner_tensor = torch.from_numpy(
            reversed_record_rank_partners(
                cases, f"{teacher_target}_target"
            )
        ).to(device)
    model = build_sweep_selector(student_variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    history: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        loss_sum = 0.0
        example_count = 0
        order = torch.randperm(total, device=device)
        for start in range(0, total, int(batch_size)):
            selected = order[start : start + int(batch_size)]
            batch_source = source_tensor[selected]
            batch_mask = mask_tensor[selected]
            batch_global = global_tensor[selected]
            batch_soft = soft_tensor[selected]
            batch_weight = weight_tensor[selected]
            logits = model(batch_source, batch_mask, batch_global)
            losses = F.binary_cross_entropy_with_logits(
                logits, batch_soft, reduction="none"
            )
            loss = torch.mean(losses * batch_weight)
            if partner_tensor is not None:
                partner = partner_tensor[selected]
                partner_logits = model(
                    source_tensor[partner],
                    mask_tensor[partner],
                    global_tensor[partner],
                )
                teacher_gap = (
                    soft_tensor[selected, 2] - soft_tensor[partner, 2]
                )
                gap_weight = torch.abs(teacher_gap)
                direction = torch.sign(teacher_gap)
                factor_log_score = F.logsigmoid(logits[:, :2]).sum(dim=1)
                partner_factor_log_score = F.logsigmoid(
                    partner_logits[:, :2]
                ).sum(dim=1)
                ranking = F.softplus(
                    -direction * (factor_log_score - partner_factor_log_score)
                )
                ranking_loss = torch.sum(ranking * gap_weight) / torch.sum(
                    gap_weight
                ).clamp_min(1e-6)
                loss = loss + float(ranking_weight) * ranking_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_source)
            example_count += len(batch_source)
        history.append(loss_sum / max(example_count, 1))
    return model, history


def predict_sweep_set_selector(
    model: nn.Module,
    case: dict[str, object],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(
        torch.from_numpy(case["source"]),
        torch.from_numpy(case["mask"]),
        torch.from_numpy(case["global"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_source, batch_mask, batch_global in loader:
            logits = model(
                batch_source.to(device, non_blocking=True),
                batch_mask.to(device, non_blocking=True),
                batch_global.to(device, non_blocking=True),
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)


def coverage_constrained_selector_mask(
    probability: np.ndarray,
    threshold: float,
    counts: np.ndarray,
    minimum_record_retention: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a score threshold without erasing the support of any sweep."""
    floor = float(minimum_record_retention)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("minimum record retention must lie in [0, 1]")
    keep = np.asarray(probability) >= float(threshold)
    counts = np.asarray(counts, dtype=np.int64)
    if int(np.sum(counts)) != len(keep):
        raise ValueError("record counts do not match selector probabilities")
    retained_fractions: list[float] = []
    cursor = 0
    for count_value in counts:
        count = int(count_value)
        selected = slice(cursor, cursor + count)
        local_keep = keep[selected]
        required = min(count, max(3, int(math.ceil(floor * count))))
        if int(np.sum(local_keep)) < required:
            local_score = np.asarray(probability[selected])
            rank = count - required
            rank_threshold = float(np.partition(local_score, rank)[rank])
            local_keep |= local_score >= rank_threshold
            keep[selected] = local_keep
        retained_fractions.append(float(np.mean(local_keep)))
        cursor += count
    return keep, np.asarray(retained_fractions, dtype=np.float64)


def save_ranked_selector_prediction(
    case: dict[str, object],
    probability: np.ndarray,
    *,
    score_name: str,
    retention: float,
    output: Path,
    settings: dict[str, object],
) -> dict[str, object]:
    """Freeze a head-derived per-record rank without any held-out CT input."""
    score_parts = {
        "anatomy": probability[:, 0],
        "surface": probability[:, 1],
        "joint": probability[:, 2],
        "factor": probability[:, 0] * probability[:, 1],
    }
    if score_name not in score_parts:
        raise ValueError(f"unsupported selector score: {score_name}")
    action_score = np.clip(score_parts[score_name], 0.0, 1.0)
    with np.load(case["feature_path"]) as feature:
        keep, record_retention = coverage_constrained_selector_mask(
            action_score,
            np.inf,
            feature["counts"],
            retention,
        )
        payload = {key: feature[key] for key in feature.files}
        unrelaxed = feature["analytic"].astype(np.float32)
        payload.update(
            {
                "correction": np.clip(
                    ANALYTIC_RELAXATION * unrelaxed,
                    -MAX_CORRECTION_MM,
                    MAX_CORRECTION_MM,
                ).astype(np.float32),
                "learned": np.zeros_like(unrelaxed),
                "support": action_score.astype(np.float32),
                "active": np.asarray(True),
                "correction_mode": np.asarray(
                    "privileged_sweep_set_selector"
                ),
                "selector_anatomy_probability": probability[:, 0].astype(
                    np.float32
                ),
                "selector_surface_probability": probability[:, 1].astype(
                    np.float32
                ),
                "selector_joint_probability": probability[:, 2].astype(
                    np.float32
                ),
                "selection_probability": action_score.astype(np.float32),
                "selection_threshold_keep": keep.astype(np.uint8),
                "selection_keep": keep.astype(np.uint8),
                "selection_threshold": np.asarray(np.nan),
                "minimum_record_retention": np.asarray(retention),
                "record_retained_fraction": record_retention.astype(
                    np.float32
                ),
                "student_global_names": np.asarray(STUDENT_GLOBAL_NAMES),
                "settings": np.asarray(json.dumps(settings)),
                "deployment_action": np.asarray("selector_refined"),
                "policy_rejected_fraction": np.asarray(
                    float(1.0 - np.mean(keep))
                ),
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **payload)
    return {
        "retained_fraction": float(np.mean(keep)),
        "record_retained_fraction": record_retention.tolist(),
    }


def discover_case_archive(root: Path, specimen: str, anatomy: str) -> Path:
    matches = sorted(root.rglob(f"{specimen}_{anatomy}.npz"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {specimen}/{anatomy} archive under "
            f"{root}, found {len(matches)}"
        )
    return matches[0]


def crossfit_selector_probabilities(
    cases: Sequence[dict[str, object]],
    specimens: Sequence[str],
    *,
    fold_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    progress_prefix: str,
    student_variant: str = "sweep_set",
    teacher_target: str = "soft",
    ranking_weight: float = 0.0,
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict[str, object]]]:
    """Return specimen-disjoint inner probabilities for action selection."""
    if not 2 <= int(fold_count) <= len(specimens):
        raise ValueError("inner fold count must lie between 2 and specimen count")
    oof_probability: dict[tuple[str, str], np.ndarray] = {}
    reports: list[dict[str, object]] = []
    for fold in range(int(fold_count)):
        validation_specimens = {
            specimen
            for index, specimen in enumerate(specimens)
            if index % int(fold_count) == fold
        }
        inner_training = [
            case
            for case in cases
            if str(case["specimen"]) not in validation_specimens
        ]
        inner_validation = [
            case
            for case in cases
            if str(case["specimen"]) in validation_specimens
        ]
        model, history = train_sweep_set_selector(
            inner_training,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            student_variant=student_variant,
            teacher_target=teacher_target,
            ranking_weight=ranking_weight,
        )
        for case in inner_validation:
            key = (str(case["specimen"]), str(case["anatomy"]))
            oof_probability[key] = predict_sweep_set_selector(
                model,
                case,
                batch_size=batch_size,
                device=device,
            )
        report = {
            "fold": fold,
            "validation_specimens": sorted(validation_specimens),
            "training_specimens": sorted(
                {str(case["specimen"]) for case in inner_training}
            ),
            "final_loss": float(history[-1]),
        }
        reports.append(report)
        print(
            json.dumps({"stage": progress_prefix, **report}, sort_keys=True),
            flush=True,
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    expected = {
        (str(case["specimen"]), str(case["anatomy"])) for case in cases
    }
    if set(oof_probability) != expected:
        raise RuntimeError("inner cross-fitting did not predict every training case")
    return oof_probability, reports


def choose_inner_rank_policy(
    cases: Sequence[dict[str, object]],
    oof_probability: dict[tuple[str, str], np.ndarray],
    retentions: Sequence[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Choose one score/retention by specimen-macro joint-label F1."""
    candidates: list[dict[str, object]] = []
    for score_name in ("anatomy", "surface", "joint", "factor"):
        for retention in retentions:
            per_specimen: dict[str, list[dict[str, float]]] = {}
            for case in cases:
                key = (str(case["specimen"]), str(case["anatomy"]))
                probability = oof_probability[key]
                score_parts = {
                    "anatomy": probability[:, 0],
                    "surface": probability[:, 1],
                    "joint": probability[:, 2],
                    "factor": probability[:, 0] * probability[:, 1],
                }
                keep, _ = coverage_constrained_selector_mask(
                    score_parts[score_name],
                    np.inf,
                    np.asarray(case["counts"]),
                    float(retention),
                )
                metrics = binary_selection_metrics(
                    np.asarray(case["hard_target"])[:, 2].astype(bool), keep
                )
                per_specimen.setdefault(key[0], []).append(
                    {
                        "fscore": float(metrics["fscore"]),
                        "specificity": float(metrics["specificity"]),
                        "recall": float(metrics["recall"]),
                    }
                )
            specimen_f1 = np.asarray(
                [
                    np.mean([row["fscore"] for row in rows])
                    for rows in per_specimen.values()
                ],
                dtype=np.float64,
            )
            specimen_balanced_accuracy = np.asarray(
                [
                    np.mean(
                        [
                            0.5 * (row["recall"] + row["specificity"])
                            for row in rows
                        ]
                    )
                    for rows in per_specimen.values()
                ],
                dtype=np.float64,
            )
            candidates.append(
                {
                    "score": score_name,
                    "retention": float(retention),
                    "selection_objective": "specimen_macro_joint_label_f1",
                    "mean_f1": float(np.mean(specimen_f1)),
                    "standard_deviation_f1": float(
                        np.std(specimen_f1, ddof=1)
                    ),
                    "mean_balanced_accuracy": float(
                        np.mean(specimen_balanced_accuracy)
                    ),
                    "specimens": int(len(specimen_f1)),
                }
            )
    score_order = {"anatomy": 0, "surface": 1, "joint": 2, "factor": 3}
    chosen = max(
        candidates,
        key=lambda row: (
            float(row["mean_f1"]),
            float(row["mean_balanced_accuracy"]),
            float(row["retention"]),
            score_order[str(row["score"])],
        ),
    )
    return chosen, candidates


def select_nested_policy_from_frozen_outer(args: argparse.Namespace) -> None:
    """Select each outer action using only cross-fit predictions from its 13 training specimens."""
    input_protocol_path = Path(args.input_protocol)
    source_protocol = json.loads(
        input_protocol_path.read_text(encoding="utf-8")
    )
    if source_protocol.get("status") != "nested_specimen_out_of_fold_predictions_frozen":
        raise ValueError("input protocol is not a frozen outer-LOOCV run")
    specimens = tuple(str(item) for item in source_protocol["specimens"])
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    retentions = tuple(float(value) for value in args.retention)
    if not retentions or any(not 0.0 < value <= 1.0 for value in retentions):
        raise ValueError("every candidate retention must lie in (0, 1]")
    device = choose_device(args.device)
    feature_paths = {
        (specimen, anatomy): discover_case_archive(
            feature_root, specimen, anatomy
        )
        for specimen in specimens
        for anatomy in ANATOMIES
    }
    max_sources = 0
    for path in feature_paths.values():
        with np.load(path) as archive:
            max_sources = max(
                max_sources, int(archive["source_evidence"].shape[1])
            )

    output_folds: list[dict[str, object]] = []
    for fold_index, source_fold in enumerate(source_protocol["folds"]):
        heldout = str(source_fold["heldout_specimen"])
        training_specimens = tuple(
            specimen for specimen in specimens if specimen != heldout
        )
        training_cases = [
            load_student_case(
                feature_paths[(specimen, anatomy)],
                max_sources,
                discover_case_archive(label_root, specimen, anatomy),
            )
            for specimen in training_specimens
            for anatomy in ANATOMIES
        ]
        oof_probability, inner_reports = crossfit_selector_probabilities(
            training_cases,
            training_specimens,
            fold_count=int(args.inner_folds),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
            device=device,
            progress_prefix=f"policy_outer_{fold_index + 1:02d}_{heldout}",
            student_variant=str(args.student_variant),
            teacher_target=str(args.teacher_target),
            ranking_weight=float(args.ranking_weight),
        )
        chosen, candidates = choose_inner_rank_policy(
            training_cases, oof_probability, retentions
        )
        predictions: list[dict[str, object]] = []
        for source_record in source_fold["predictions"]:
            anatomy = str(source_record["anatomy"])
            source_path = Path(source_record["output"])
            if not source_path.exists():
                source_path = input_protocol_path.parent / source_path.name
            with np.load(source_path) as source:
                joint_probability = (
                    source["selector_joint_probability"].astype(np.float32)
                    if "selector_joint_probability" in source.files
                    else source["selection_probability"].astype(np.float32)
                )
                probability = np.column_stack(
                    [
                        source["selector_anatomy_probability"],
                        source["selector_surface_probability"],
                        joint_probability,
                    ]
                ).astype(np.float32)
            case = load_student_case(
                feature_paths[(heldout, anatomy)], max_sources
            )
            output = output_root / f"{heldout}_{anatomy}.npz"
            settings = {
                "method": "nested_training_only_policy_selected_PFAD",
                "evaluation_design": "outer_LOOCV_with_inner_policy_selection",
                "student_variant": str(args.student_variant),
                "teacher_target": str(args.teacher_target),
                "record_ranking_weight": float(args.ranking_weight),
                "outer_heldout_specimen": heldout,
                "outer_training_specimens": list(training_specimens),
                "inner_folds": int(args.inner_folds),
                "selection_objective": chosen["selection_objective"],
                "selected_score": chosen["score"],
                "selected_retention": chosen["retention"],
                "heldout_teacher_label_loaded": False,
                "heldout_CT_opened_before_freeze": False,
                "inference_uses_CT": False,
            }
            retention_report = save_ranked_selector_prediction(
                case,
                probability,
                score_name=str(chosen["score"]),
                retention=float(chosen["retention"]),
                output=output,
                settings=settings,
            )
            predictions.append(
                {
                    "specimen": heldout,
                    "anatomy": anatomy,
                    "output": str(output),
                    "points": int(len(probability)),
                    "deployment_action": "selector_refined",
                    "score": chosen["score"],
                    "retention": chosen["retention"],
                    "heldout_teacher_label_loaded": False,
                    "heldout_CT_used": False,
                    **retention_report,
                }
            )
        output_fold = {
            "outer_fold": fold_index,
            "heldout_specimen": heldout,
            "training_specimens": list(training_specimens),
            "policy_selection": {
                "chosen": chosen,
                "candidates": candidates,
                "candidate_scores": ["anatomy", "surface", "joint", "factor"],
                "candidate_retentions": list(retentions),
                "heldout_labels_used": False,
            },
            "inner_folds": inner_reports,
            "predictions": predictions,
        }
        output_folds.append(output_fold)
        print(
            json.dumps(
                {
                    "stage": "outer_policy_frozen",
                    "heldout_specimen": heldout,
                    "chosen": chosen,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del training_cases, oof_probability
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "status": "nested_specimen_out_of_fold_predictions_frozen",
        "interpretation": (
            "outer-held-out PFAD predictions with score and retention selected "
            "only from specimen-disjoint inner predictions on the other 13 specimens"
        ),
        "specimens": list(specimens),
        "outer_folds": len(specimens),
        "inner_folds": int(args.inner_folds),
        "policy_selection_objective": "specimen_macro_joint_label_f1",
        "student_variant": str(args.student_variant),
        "teacher_target": str(args.teacher_target),
        "record_ranking_weight": float(args.ranking_weight),
        "candidate_scores": ["anatomy", "surface", "joint", "factor"],
        "candidate_retentions": list(retentions),
        "candidate_minimum_retention": float(min(retentions)),
        "deployment_actions": {
            anatomy: "selector_refined" for anatomy in ANATOMIES
        },
        "heldout_teacher_labels_loaded": False,
        "heldout_CT_used_before_prediction_freeze": False,
        "folds": output_folds,
    }
    protocol_output = Path(args.protocol_output)
    protocol_output.parent.mkdir(parents=True, exist_ok=True)
    protocol_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol_output": str(protocol_output),
                "cases": len(specimens) * len(ANATOMIES),
            },
            indent=2,
        )
    )


def selector_ablation_leave_one_specimen_out(args: argparse.Namespace) -> None:
    """Freeze fixed-policy outer-LOOCV predictions for one student ablation."""
    specimens = tuple(dict.fromkeys(args.specimen))
    if len(specimens) < 4:
        raise ValueError("selector ablation requires at least four specimens")
    evaluation_specimens = tuple(
        dict.fromkeys(args.heldout_specimen or specimens)
    )
    unknown = set(evaluation_specimens) - set(specimens)
    if unknown:
        raise ValueError(f"held-out specimens are outside the cohort: {unknown}")
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    student_variant = str(args.student_variant)
    teacher_target = str(args.teacher_target)
    score_name = str(args.score)
    retention = float(args.retention)
    device = choose_device(args.device)
    feature_paths = {
        (specimen, anatomy): discover_case_archive(
            feature_root, specimen, anatomy
        )
        for specimen in specimens
        for anatomy in ANATOMIES
    }
    max_sources = 0
    for path in feature_paths.values():
        with np.load(path) as archive:
            max_sources = max(
                max_sources, int(archive["source_evidence"].shape[1])
            )

    folds: list[dict[str, object]] = []
    for outer_index, heldout_specimen in enumerate(evaluation_specimens):
        training_specimens = tuple(
            specimen for specimen in specimens if specimen != heldout_specimen
        )
        training_cases = [
            load_student_case(
                feature_paths[(specimen, anatomy)],
                max_sources,
                discover_case_archive(label_root, specimen, anatomy),
            )
            for specimen in training_specimens
            for anatomy in ANATOMIES
        ]
        heldout_cases = [
            load_student_case(
                feature_paths[(heldout_specimen, anatomy)], max_sources
            )
            for anatomy in ANATOMIES
        ]
        model, history = train_sweep_set_selector(
            training_cases,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
            device=device,
            student_variant=student_variant,
            teacher_target=teacher_target,
            ranking_weight=float(args.ranking_weight),
        )
        parameter_count = int(sum(item.numel() for item in model.parameters()))
        predictions: list[dict[str, object]] = []
        for case in heldout_cases:
            probability = predict_sweep_set_selector(
                model,
                case,
                batch_size=int(args.batch_size),
                device=device,
            )
            anatomy = str(case["anatomy"])
            output = output_root / f"{heldout_specimen}_{anatomy}.npz"
            settings = {
                "method": "selector_architecture_or_teacher_ablation",
                "evaluation_design": "leave_one_specimen_out_fixed_policy",
                "student_variant": student_variant,
                "teacher_target": teacher_target,
                "record_ranking_weight": float(args.ranking_weight),
                "deployment_score": score_name,
                "minimum_record_retention": retention,
                "outer_heldout_specimen": heldout_specimen,
                "outer_training_specimens": list(training_specimens),
                "heldout_teacher_label_loaded": False,
                "heldout_CT_opened_before_freeze": False,
                "inference_uses_CT": False,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "parameter_count": parameter_count,
            }
            retention_report = save_ranked_selector_prediction(
                case,
                probability,
                score_name=score_name,
                retention=retention,
                output=output,
                settings=settings,
            )
            predictions.append(
                {
                    "specimen": heldout_specimen,
                    "anatomy": anatomy,
                    "output": str(output),
                    "points": int(len(probability)),
                    "deployment_action": "selector_refined",
                    "score": score_name,
                    "retention": retention,
                    "heldout_teacher_label_loaded": False,
                    "heldout_CT_used": False,
                    **retention_report,
                }
            )
        fold = {
            "outer_fold": outer_index,
            "heldout_specimen": heldout_specimen,
            "training_specimens": list(training_specimens),
            "final_training_loss": history,
            "parameter_count": parameter_count,
            "predictions": predictions,
        }
        folds.append(fold)
        print(
            json.dumps(
                {
                    "stage": "ablation_outer_prediction_frozen",
                    "heldout_specimen": heldout_specimen,
                    "student_variant": student_variant,
                    "teacher_target": teacher_target,
                    "final_loss": float(history[-1]),
                    "parameter_count": parameter_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del model, training_cases, heldout_cases
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "status": "nested_specimen_out_of_fold_predictions_frozen",
        "interpretation": (
            "fixed-policy specimen-out-of-fold selector ablation; training-only "
            "CT supervision and CT-free held-out inference"
        ),
        "specimens": list(evaluation_specimens),
        "training_universe": list(specimens),
        "outer_folds": len(evaluation_specimens),
        "inner_folds": 0,
        "minimum_record_retention": retention,
        "deployment_actions": {
            anatomy: "selector_refined" for anatomy in ANATOMIES
        },
        "student_variant": student_variant,
        "teacher_target": teacher_target,
        "record_ranking_weight": float(args.ranking_weight),
        "deployment_score": score_name,
        "heldout_teacher_labels_loaded": False,
        "heldout_CT_used_before_prediction_freeze": False,
        "folds": folds,
    }
    protocol_output = Path(
        args.protocol_output or output_root / "loocv_protocol.json"
    )
    protocol_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol_output": str(protocol_output),
                "cases": len(evaluation_specimens) * len(ANATOMIES),
            },
            indent=2,
        )
    )


def estimate_point_normals(
    points: np.ndarray, neighbours: int, chunk_size: int = 50_000
) -> np.ndarray:
    """Estimate unoriented local PCA normals without changing point positions."""
    if len(points) < 3:
        raise ValueError("normal estimation requires at least three points")
    count = min(max(int(neighbours), 3), len(points))
    tree = cKDTree(points)
    normals: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
        query = points[start : start + chunk_size]
        _, indices = tree.query(query, k=count, workers=-1)
        local = points[indices]
        centred = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centred, centred) / count
        _, eigenvectors = np.linalg.eigh(covariance)
        normals.append(eigenvectors[:, :, 0])
    return np.concatenate(normals)


def build_open3d_surface_scene(mesh: trimesh.Trimesh):
    """Build one reusable exact closest-triangle scene when Open3D is present."""
    try:
        import open3d as o3d
    except ImportError:
        return None
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def closest_surface_query(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    open3d_scene,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    if open3d_scene is None:
        _, distance, triangle_index = mesh.nearest.on_surface(points)
        return distance, triangle_index
    import open3d as o3d

    distances: list[np.ndarray] = []
    triangle_indices: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
        query = o3d.core.Tensor(
            points[start : start + chunk_size].astype(np.float32)
        )
        result = open3d_scene.compute_closest_points(query)
        closest = result["points"].numpy()
        distances.append(
            np.linalg.norm(
                points[start : start + chunk_size] - closest, axis=1
            )
        )
        triangle_indices.append(
            result["primitive_ids"].numpy().astype(np.int64)
        )
    return np.concatenate(distances), np.concatenate(triangle_indices)


def point_set_metrics(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    visible_reference: np.ndarray,
    normal_neighbours: int,
    open3d_scene=None,
) -> dict[str, float | int]:
    """Evaluate a point set against a ray-visible CT surface reference."""
    accuracy, triangle_index = closest_surface_query(
        mesh, points, open3d_scene
    )
    completeness = cKDTree(points).query(visible_reference, workers=-1)[0]
    point_normals = estimate_point_normals(points, normal_neighbours)
    reference_normals = mesh.face_normals[triangle_index]
    normal_cosine = np.clip(
        np.abs(np.einsum("ij,ij->i", point_normals, reference_normals)),
        0.0,
        1.0,
    )
    normal_angle = np.degrees(np.arccos(normal_cosine))
    result: dict[str, float | int] = {
        "points": int(len(points)),
        "accuracy_mean_mm": float(np.mean(accuracy)),
        "accuracy_median_mm": float(np.median(accuracy)),
        "accuracy_p95_mm": float(np.percentile(accuracy, 95)),
        "completeness_mean_mm": float(np.mean(completeness)),
        "completeness_median_mm": float(np.median(completeness)),
        "completeness_p95_mm": float(np.percentile(completeness, 95)),
        "chamfer_l1_mm": float(0.5 * (np.mean(accuracy) + np.mean(completeness))),
        "hd95_mm": float(
            max(np.percentile(accuracy, 95), np.percentile(completeness, 95))
        ),
        "normal_consistency": float(np.mean(normal_cosine)),
        "normal_angle_mean_deg": float(np.mean(normal_angle)),
        "normal_angle_median_deg": float(np.median(normal_angle)),
    }
    for threshold in (1.0, 2.0):
        suffix = str(int(threshold))
        precision = float(np.mean(accuracy <= threshold))
        coverage = float(np.mean(completeness <= threshold))
        fscore = (
            2.0 * precision * coverage / (precision + coverage)
            if precision + coverage > 0.0
            else 0.0
        )
        result[f"precision_{suffix}mm"] = precision
        result[f"coverage_{suffix}mm"] = coverage
        result[f"fscore_{suffix}mm"] = fscore
    return result


def evaluate_point_sets(args: argparse.Namespace) -> None:
    """Evaluate frozen corrections as target-visible 3-D point sets.

    CT is used only here.  The same evaluation-only visibility mask is applied
    to every method: a ray is retained when it has a first intersection with
    the target-anatomy CT mesh.  This must not be used to prepare inputs for a
    downstream reconstruction method.
    """
    prediction = np.load(args.prediction_archive)
    archived_specimen = (
        archive_scalar(prediction, "specimen")
        if "specimen" in prediction.files
        else ""
    )
    archived_anatomy = (
        archive_scalar(prediction, "anatomy")
        if "anatomy" in prediction.files
        else ""
    )
    specimen = str(args.specimen or archived_specimen)
    anatomy = str(args.anatomy or archived_anatomy)
    if not specimen or not anatomy:
        raise ValueError("specimen and anatomy are required for legacy archives")
    if archived_specimen and archived_specimen != specimen:
        raise ValueError("specimen disagrees with prediction archive")
    if archived_anatomy and archived_anatomy != anatomy:
        raise ValueError("anatomy disagrees with prediction archive")

    root = Path(args.dataset_root)
    records = tuple(str(item) for item in prediction["records"])
    counts = prediction["counts"]
    evaluation_stride = archive_int(
        prediction, "evaluation_stride", EVALUATION_STRIDE
    )
    evaluation_offset = archive_int(
        prediction, "evaluation_offset", EVALUATION_OFFSET
    )
    line_mode = (
        archive_scalar(prediction, "line_mode")
        if "line_mode" in prediction.files
        else "top"
    )
    correction = prediction["correction"]
    anchor = prediction["analytic"]
    learned = prediction["learned"]
    support = prediction["support"]
    correction_mode = (
        archive_scalar(prediction, "correction_mode")
        if "correction_mode" in prediction.files
        else "one_sided"
    )

    selector_mode = correction_mode == "privileged_sweep_set_selector"
    pointset_selector_masks: dict[str, np.ndarray] = {}
    if correction_mode == "analytic_only":
        method_names = ("raw", "unrelaxed_anchor", "relaxed_anchor")
    elif selector_mode:
        minimum_record_retention = (
            float(prediction["minimum_record_retention"].item())
            if "minimum_record_retention" in prediction.files
            else MINIMUM_RECORD_RETENTION
        )
        probability = prediction["selection_probability"].astype(np.float64)
        joint_probability = (
            prediction["selector_joint_probability"].astype(np.float64)
            if "selector_joint_probability" in prediction.files
            else probability
        )
        factor_probability = (
            prediction["selector_anatomy_probability"].astype(np.float64)
            * prediction["selector_surface_probability"].astype(np.float64)
        )
        archived_raw_points = (
            prediction["ray_origins_mm"].astype(np.float64)
            + prediction["ray_depths_mm"].astype(np.float64)[:, None]
            * prediction["ray_directions"].astype(np.float64)
        )
        neighbour_count = min(25, len(archived_raw_points))
        neighbour_distance = cKDTree(archived_raw_points).query(
            archived_raw_points, k=neighbour_count, workers=-1
        )[0]
        if neighbour_distance.ndim == 1:
            statistical_outlier_score = -neighbour_distance
        else:
            statistical_outlier_score = -np.mean(
                neighbour_distance[:, 1:], axis=1
            )
        robust_plane_score = -(
            np.abs(prediction["plane_centre_mm"].astype(np.float64))
            + prediction["plane_disagreement_mm"].astype(np.float64)
        )
        threshold_keep = (
            prediction["selector_joint_threshold_keep"].astype(bool)
            if "selector_joint_threshold_keep" in prediction.files
            else (
                prediction["selection_threshold_keep"].astype(bool)
                if "selection_threshold_keep" in prediction.files
                else probability >= float(prediction["selection_threshold"])
            )
        )
        rank_scores = {
            "confidence_screen_refined": prediction["ray_confidence"].astype(
                np.float64
            ),
            "statistical_outlier_screen_refined": statistical_outlier_score,
            "geometry_screen_refined": -prediction[
                "cross_sweep_nearest_mm"
            ].astype(np.float64),
            "robust_plane_screen_refined": robust_plane_score,
            "anatomy_head_screen_refined": prediction[
                "selector_anatomy_probability"
            ].astype(np.float64),
            "surface_head_screen_refined": prediction[
                "selector_surface_probability"
            ].astype(np.float64),
            "factor_screen_refined": factor_probability,
            "joint_rank_screen_refined": joint_probability,
        }
        pointset_selector_masks = {
            name: coverage_constrained_selector_mask(
                score,
                np.inf,
                counts,
                minimum_record_retention,
            )[0]
            for name, score in rank_scores.items()
        }
        pointset_selector_masks[
            "unconstrained_selector_refined"
        ] = threshold_keep
        pointset_selector_masks["selector_raw"] = prediction[
            "selection_keep"
        ].astype(bool)
        pointset_selector_masks[
            "selector_refined"
        ] = pointset_selector_masks["selector_raw"]
        for retention in (0.85, 0.95):
            name = f"factor_top{int(round(100 * retention))}_refined"
            pointset_selector_masks[name] = coverage_constrained_selector_mask(
                factor_probability,
                np.inf,
                counts,
                retention,
            )[0]
        method_names = (
            "raw",
            "relaxed_anchor",
            "confidence_screen_refined",
            "statistical_outlier_screen_refined",
            "geometry_screen_refined",
            "robust_plane_screen_refined",
            "anatomy_head_screen_refined",
            "surface_head_screen_refined",
            "factor_screen_refined",
            "joint_rank_screen_refined",
            "unconstrained_selector_refined",
            "selector_raw",
            "selector_refined",
            "factor_top85_refined",
            "factor_top95_refined",
        )
    else:
        method_names = ("raw", "analytic", "per_ray_network", "hierarchical")
    official_ct_mask: np.ndarray | None = None
    if args.teacher_root:
        teacher_path = discover_case_archive(
            Path(args.teacher_root), specimen, anatomy
        )
        with np.load(teacher_path) as teacher:
            if archive_scalar(teacher, "specimen") != specimen:
                raise ValueError("official CT comparator specimen mismatch")
            if archive_scalar(teacher, "anatomy") != anatomy:
                raise ValueError("official CT comparator anatomy mismatch")
            nearest_anatomy = teacher["nearest_anatomy"].astype(np.int64)
        if len(nearest_anatomy) != int(np.sum(counts)):
            raise ValueError("official CT comparator length mismatch")
        official_ct_mask = np.zeros(len(nearest_anatomy), dtype=bool)
        official_cursor = 0
        for count_value in counts:
            count = int(count_value)
            selected = slice(official_cursor, official_cursor + count)
            local = nearest_anatomy[selected]
            majority_anatomy = int(
                np.argmax(np.bincount(local, minlength=len(ANATOMIES)))
            )
            official_ct_mask[selected] = local == majority_anatomy
            official_cursor += count
        method_names = (
            *method_names,
            "official_CT_filter_raw",
            "official_CT_filter_refined",
        )
    all_points = {name: [] for name in method_names}
    visible_points = {name: [] for name in all_points}
    reference_parts: list[np.ndarray] = []
    cursor = 0
    frame_offset = 0
    backends: set[str] = set()
    export_metadata: list[np.ndarray] = []
    for record_index, record in enumerate(records):
        data = load_record(root, specimen, anatomy, record)
        selected = slice(cursor, cursor + int(counts[record_index]))
        rays = archived_rays(prediction, selected)
        if rays is None:
            rays = extract_rays(
                data,
                stride=evaluation_stride,
                offset=evaluation_offset,
                line_mode=line_mode,
            )
        if len(rays) != int(counts[record_index]):
            raise RuntimeError(f"ray-count mismatch for {record}")
        expected_frames = rays.frame_index + frame_offset
        if not np.array_equal(prediction["frame"][selected], expected_frames):
            raise RuntimeError(f"frame-order mismatch for {record}")
        per_ray = np.clip(
            anchor[selected]
            + LEARNED_SHRINKAGE
            * support[selected]
            * (learned[selected] - anchor[selected]),
            -MAX_CORRECTION_MM,
            MAX_CORRECTION_MM,
        )
        if correction_mode == "analytic_only":
            depths = {
                "raw": rays.depths_mm,
                "unrelaxed_anchor": rays.depths_mm + anchor[selected],
                "relaxed_anchor": rays.depths_mm + correction[selected],
            }
            point_masks = {
                name: np.ones(len(rays), dtype=bool) for name in depths
            }
        elif selector_mode:
            depths = {
                "raw": rays.depths_mm,
                "relaxed_anchor": rays.depths_mm + correction[selected],
                "confidence_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "statistical_outlier_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "geometry_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "robust_plane_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "anatomy_head_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "surface_head_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "factor_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "joint_rank_screen_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "unconstrained_selector_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "selector_raw": rays.depths_mm,
                "selector_refined": rays.depths_mm + correction[selected],
                "factor_top85_refined": (
                    rays.depths_mm + correction[selected]
                ),
                "factor_top95_refined": (
                    rays.depths_mm + correction[selected]
                ),
            }
            point_masks = {
                "raw": np.ones(len(rays), dtype=bool),
                "relaxed_anchor": np.ones(len(rays), dtype=bool),
                **{
                    name: mask[selected]
                    for name, mask in pointset_selector_masks.items()
                },
            }
        else:
            depths = {
                "raw": rays.depths_mm,
                "analytic": rays.depths_mm + anchor[selected],
                "per_ray_network": rays.depths_mm + per_ray,
                "hierarchical": rays.depths_mm + correction[selected],
            }
            point_masks = {
                name: np.ones(len(rays), dtype=bool) for name in depths
            }
        if official_ct_mask is not None:
            local_official_mask = official_ct_mask[selected]
            depths.update(
                {
                    "official_CT_filter_raw": rays.depths_mm,
                    "official_CT_filter_refined": (
                        rays.depths_mm + correction[selected]
                    ),
                }
            )
            point_masks.update(
                {
                    "official_CT_filter_raw": local_official_mask,
                    "official_CT_filter_refined": local_official_mask,
                }
            )
        ct_depths, backend = ct_first_intersections(data.mesh_path, rays)
        backends.add(backend)
        valid = np.isfinite(ct_depths)
        reference_parts.append(
            rays.origins_mm[valid]
            + ct_depths[valid, None] * rays.directions[valid]
        )
        export_metadata.append(
            np.column_stack(
                [
                    np.full(len(rays), record_index, dtype=np.int64),
                    rays.frame_index,
                    rays.column_index,
                ]
            )
        )
        for name, values in depths.items():
            points = rays.origins_mm + values[:, None] * rays.directions
            method_mask = point_masks[name]
            all_points[name].append(points[method_mask])
            visible_points[name].append(points[valid & method_mask])
        cursor += len(rays)
        frame_offset += len(data.rows)
    if cursor != len(correction):
        raise RuntimeError("unused prediction rows remain")

    mesh_path = root / specimen / "CT_bone_segmentations" / f"{anatomy}.stl"
    mesh = trimesh.load_mesh(mesh_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a triangle mesh: {mesh_path}")
    open3d_scene = build_open3d_surface_scene(mesh)
    reference = np.concatenate(reference_parts)
    all_point_metrics = {
        name: point_set_metrics(
            mesh,
            np.concatenate(parts),
            reference,
            int(args.normal_neighbours),
            open3d_scene,
        )
        for name, parts in all_points.items()
    }
    common_visibility_metrics = {
        name: point_set_metrics(
            mesh,
            np.concatenate(parts),
            reference,
            int(args.normal_neighbours),
            open3d_scene,
        )
        for name, parts in visible_points.items()
    }

    if args.export_archive:
        export = Path(args.export_archive)
        export.parent.mkdir(parents=True, exist_ok=True)
        merged = {
            name: np.concatenate(parts).astype(np.float32)
            for name, parts in all_points.items()
        }
        np.savez_compressed(
            export,
            **merged,
            record_frame_column=np.concatenate(export_metadata).astype(np.int32),
            records=np.asarray(records),
            specimen=np.asarray(specimen),
            anatomy=np.asarray(anatomy),
            evaluation_stride=np.asarray(evaluation_stride),
            evaluation_offset=np.asarray(evaluation_offset),
            line_mode=np.asarray(line_mode),
        )

    result = {
        "case": {
            "specimen": specimen,
            "anatomy": anatomy,
            "records": list(records),
            "branch": (
                "relaxed_analytic"
                if correction_mode == "analytic_only"
                else (
                    "privileged_sweep_set_selector"
                    if selector_mode
                    else ("learned" if bool(prediction["active"]) else "anchor")
                )
            ),
        },
        "protocol": {
            "object": "point_set_not_extracted_mesh",
            "reference": "target_CT_first_intersections_on_acquisition_rays",
            "primary_prediction_set": "all_deployed_points_without_CT_filtering",
            "selector_mask": (
                "frozen_CT_free_student_prediction"
                if selector_mode
                else "not_applicable"
            ),
            "official_CT_filter_comparator": (
                "evaluation_only_inference_CT_oracle"
                if official_ct_mask is not None
                else "not_evaluated"
            ),
            "secondary_prediction_set": "common_CT_visible_rays_for_mechanism_control",
            "visibility_mask": "evaluation_only_and_never_applied_to_deployed_input",
            "chamfer": "0.5*(mean_predicted_to_CT_mesh+mean_visible_CT_to_predicted)",
            "hd95": "max_of_directional_95th_percentiles",
            "normal_estimation": f"unoriented_local_PCA_{int(args.normal_neighbours)}NN",
            "evaluation_stride": evaluation_stride,
            "evaluation_offset": evaluation_offset,
            "line_mode": line_mode,
            "independent_unit": "specimen",
        },
        "visible_reference_points": int(len(reference)),
        "all_points_metrics": all_point_metrics,
        "common_visibility_metrics": common_visibility_metrics,
        "intersection_backends": sorted(backends),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


POINTSET_METRIC_DIRECTIONS = {
    "chamfer_l1_mm": "lower",
    "hd95_mm": "lower",
    "fscore_1mm": "higher",
    "normal_consistency": "higher",
}


def descriptive_statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0,
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def paired_endpoint_statistics(
    raw: np.ndarray,
    deployed: np.ndarray,
    *,
    direction: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, object]:
    from scipy.stats import binomtest, wilcoxon

    raw = np.asarray(raw, dtype=np.float64)
    deployed = np.asarray(deployed, dtype=np.float64)
    improvement = raw - deployed if direction == "lower" else deployed - raw
    if np.allclose(improvement, 0.0, atol=1e-12, rtol=0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        method = (
            "exact"
            if not np.any(np.isclose(improvement, 0.0, atol=1e-12))
            else "auto"
        )
        test = wilcoxon(
            improvement,
            alternative="two-sided",
            zero_method="wilcox",
            method=method,
        )
        wilcoxon_statistic = float(test.statistic)
        wilcoxon_p = float(test.pvalue)
    tolerance = 1e-12
    wins = int(np.sum(improvement > tolerance))
    losses = int(np.sum(improvement < -tolerance))
    ties = int(len(improvement) - wins - losses)
    sign_p = float(
        binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue
    ) if wins + losses else 1.0
    generator = np.random.default_rng(int(bootstrap_seed))
    indices = generator.integers(
        0,
        len(improvement),
        size=(int(bootstrap_resamples), len(improvement)),
    )
    bootstrap_mean = np.mean(improvement[indices], axis=1)
    relative = np.divide(
        improvement,
        np.abs(raw),
        out=np.zeros_like(improvement),
        where=np.abs(raw) > 1e-12,
    )
    return {
        "direction": direction,
        "raw": descriptive_statistics(raw),
        "deployed": descriptive_statistics(deployed),
        "improvement": descriptive_statistics(improvement),
        "mean_relative_improvement_percent": float(100.0 * np.mean(relative)),
        "mean_improvement_95ci": [
            float(np.quantile(bootstrap_mean, 0.025)),
            float(np.quantile(bootstrap_mean, 0.975)),
        ],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_two_sided_p": wilcoxon_p,
        "exact_sign_two_sided_p": sign_p,
    }


def add_holm_adjustment(endpoint_summary: dict[str, dict[str, object]]) -> None:
    names = list(endpoint_summary)
    order = sorted(
        range(len(names)),
        key=lambda index: float(
            endpoint_summary[names[index]]["wilcoxon_two_sided_p"]
        ),
    )
    adjusted = np.zeros(len(names), dtype=np.float64)
    running = 0.0
    total = len(names)
    for rank, index in enumerate(order):
        raw_p = float(endpoint_summary[names[index]]["wilcoxon_two_sided_p"])
        running = max(running, min(1.0, (total - rank) * raw_p))
        adjusted[index] = running
    for index, name in enumerate(names):
        endpoint_summary[name]["holm_adjusted_p"] = float(adjusted[index])


def binary_selection_metrics(
    target: np.ndarray, selected: np.ndarray
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    true_positive = int(np.sum(target & selected))
    false_positive = int(np.sum(~target & selected))
    true_negative = int(np.sum(~target & ~selected))
    false_negative = int(np.sum(target & ~selected))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    fscore = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "points": int(len(target)),
        "selected_fraction": float(np.mean(selected)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fscore": float(fscore),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def summarize_nested_loocv(args: argparse.Namespace) -> None:
    """Summarize frozen nested-LOOCV predictions at the specimen level."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "nested_specimen_out_of_fold_predictions_frozen":
        raise ValueError("protocol is not a frozen nested-LOOCV run")
    label_root = Path(args.label_root)
    case_rows: list[dict[str, object]] = []
    for fold in protocol["folds"]:
        heldout = str(fold["heldout_specimen"])
        for prediction_record in fold["predictions"]:
            prediction_path = Path(prediction_record["output"])
            if not prediction_path.exists():
                prediction_path = protocol_path.parent / prediction_path.name
            pointset_path = prediction_path.with_suffix(".pointset.json")
            if not pointset_path.exists():
                raise FileNotFoundError(pointset_path)
            pointset = json.loads(pointset_path.read_text(encoding="utf-8"))
            specimen = str(pointset["case"]["specimen"])
            anatomy = str(pointset["case"]["anatomy"])
            if specimen != heldout:
                raise ValueError("point-set result disagrees with outer fold")
            with np.load(prediction_path) as prediction:
                deployment_action = archive_scalar(
                    prediction, "deployment_action"
                )
                score = prediction["selection_probability"].astype(np.float64)
                threshold_keep = prediction[
                    "selection_threshold_keep"
                ].astype(bool)
                effective_keep = prediction["selection_keep"].astype(bool)
                record_retention = prediction[
                    "record_retained_fraction"
                ].astype(np.float64)
            label_path = discover_case_archive(label_root, specimen, anatomy)
            with np.load(label_path) as label:
                target = label["joint_target"].astype(bool)
            if len(target) != len(score):
                raise ValueError(f"selector/teacher length mismatch: {specimen}/{anatomy}")
            classification = {
                "auroc": float(roc_auc_score(target, score)),
                "average_precision": float(average_precision_score(target, score)),
                "threshold_only": binary_selection_metrics(
                    target, threshold_keep
                ),
                "coverage_constrained": binary_selection_metrics(
                    target, effective_keep
                ),
            }
            case_rows.append(
                {
                    "specimen": specimen,
                    "anatomy": anatomy,
                    "deployment_action": deployment_action,
                    "record_retained_fraction": record_retention.tolist(),
                    "classification": classification,
                    "all_points_metrics": pointset["all_points_metrics"],
                    "common_visibility_metrics": pointset[
                        "common_visibility_metrics"
                    ],
                }
            )
    expected = len(protocol["specimens"]) * len(ANATOMIES)
    keys = {(row["specimen"], row["anatomy"]) for row in case_rows}
    if len(case_rows) != expected or len(keys) != expected:
        raise ValueError("nested summary does not contain every unique case")

    bootstrap_resamples = int(args.bootstrap_resamples)
    bootstrap_seed = int(args.bootstrap_seed)
    deployed_by_anatomy: dict[str, dict[str, object]] = {}
    method_descriptives: dict[str, dict[str, object]] = {}
    classifier_by_anatomy: dict[str, dict[str, object]] = {}
    for anatomy_index, anatomy in enumerate(ANATOMIES):
        rows = [row for row in case_rows if row["anatomy"] == anatomy]
        endpoint_summary: dict[str, dict[str, object]] = {}
        for metric_index, (metric, direction) in enumerate(
            POINTSET_METRIC_DIRECTIONS.items()
        ):
            raw = np.asarray(
                [row["all_points_metrics"]["raw"][metric] for row in rows]
            )
            deployed = np.asarray(
                [
                    row["all_points_metrics"][row["deployment_action"]][metric]
                    for row in rows
                ]
            )
            endpoint_summary[metric] = paired_endpoint_statistics(
                raw,
                deployed,
                direction=direction,
                bootstrap_seed=bootstrap_seed + 100 * anatomy_index + metric_index,
                bootstrap_resamples=bootstrap_resamples,
            )
        add_holm_adjustment(endpoint_summary)
        deployed_by_anatomy[anatomy] = endpoint_summary
        methods = tuple(rows[0]["all_points_metrics"])
        method_descriptives[anatomy] = {
            method: {
                metric: descriptive_statistics(
                    np.asarray(
                        [row["all_points_metrics"][method][metric] for row in rows]
                    )
                )
                for metric in POINTSET_METRIC_DIRECTIONS
            }
            for method in methods
        }
        classifier_by_anatomy[anatomy] = {
            name: descriptive_statistics(
                np.asarray([row["classification"][name] for row in rows])
            )
            for name in ("auroc", "average_precision")
        }
        for selection_name in ("threshold_only", "coverage_constrained"):
            classifier_by_anatomy[anatomy][selection_name] = {
                metric: descriptive_statistics(
                    np.asarray(
                        [
                            row["classification"][selection_name][metric]
                            for row in rows
                        ]
                    )
                )
                for metric in (
                    "selected_fraction",
                    "precision",
                    "recall",
                    "specificity",
                    "fscore",
                )
            }

    specimen_macro: dict[str, dict[str, object]] = {}
    specimens = tuple(str(item) for item in protocol["specimens"])
    for metric_index, (metric, direction) in enumerate(
        POINTSET_METRIC_DIRECTIONS.items()
    ):
        raw_values: list[float] = []
        deployed_values: list[float] = []
        for specimen in specimens:
            rows = [row for row in case_rows if row["specimen"] == specimen]
            raw_values.append(
                float(
                    np.mean(
                        [row["all_points_metrics"]["raw"][metric] for row in rows]
                    )
                )
            )
            deployed_values.append(
                float(
                    np.mean(
                        [
                            row["all_points_metrics"][row["deployment_action"]][metric]
                            for row in rows
                        ]
                    )
                )
            )
        specimen_macro[metric] = paired_endpoint_statistics(
            np.asarray(raw_values),
            np.asarray(deployed_values),
            direction=direction,
            bootstrap_seed=bootstrap_seed + 1000 + metric_index,
            bootstrap_resamples=bootstrap_resamples,
        )
    add_holm_adjustment(specimen_macro)

    result = {
        "status": "nested_loocv_pointset_summary_complete",
        "interpretation": protocol["interpretation"],
        "independent_unit": "specimen",
        "specimens": list(specimens),
        "cases": len(case_rows),
        "deployment_actions": protocol["deployment_actions"],
        "candidate_minimum_retention": protocol[
            "candidate_minimum_retention"
        ],
        "heldout_CT_used_before_prediction_freeze": False,
        "statistics": {
            "paired_test": "two-sided Wilcoxon signed-rank",
            "multiplicity": "Holm adjustment across four endpoints within each analysis",
            "confidence_interval": "specimen-level percentile bootstrap of paired mean improvement",
            "bootstrap_resamples": bootstrap_resamples,
        },
        "deployed_by_anatomy": deployed_by_anatomy,
        "specimen_macro": specimen_macro,
        "method_descriptives_by_anatomy": method_descriptives,
        "selector_classification_by_anatomy": classifier_by_anatomy,
        "case_results": case_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "specimens": len(specimens),
                "cases": len(case_rows),
            },
            indent=2,
        )
    )


def summarize_nested_surfaces(args: argparse.Namespace) -> None:
    """Summarize acquisition-topology surfaces at the specimen level."""
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    specimens = tuple(str(item) for item in protocol["specimens"])
    expected = {(specimen, anatomy) for specimen in specimens for anatomy in ANATOMIES}
    surface_root = Path(args.surface_root)
    rows: list[dict[str, object]] = []
    for path in sorted(surface_root.glob("specimen*/surface.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        specimen = str(result["case"]["specimen"])
        anatomy = str(result["case"]["anatomy"])
        if (specimen, anatomy) not in expected:
            raise ValueError(f"unexpected surface case: {specimen}/{anatomy}")
        if result["protocol"]["CT_used_for_reconstruction"] is not False:
            raise ValueError("surface reconstruction opened CT")
        rows.append(
            {
                "specimen": specimen,
                "anatomy": anatomy,
                "protocol": result["protocol"],
                "metrics": result["metrics"],
            }
        )
    keys = {(row["specimen"], row["anatomy"]) for row in rows}
    if keys != expected or len(rows) != len(expected):
        missing = sorted(expected - keys)
        raise ValueError(f"surface summary is incomplete; missing={missing}")

    bootstrap_resamples = int(args.bootstrap_resamples)
    bootstrap_seed = int(args.bootstrap_seed)
    by_anatomy: dict[str, dict[str, object]] = {}
    topology_by_anatomy: dict[str, dict[str, object]] = {}
    for anatomy_index, anatomy in enumerate(ANATOMIES):
        anatomy_rows = [row for row in rows if row["anatomy"] == anatomy]
        endpoints: dict[str, dict[str, object]] = {}
        for metric_index, (metric, direction) in enumerate(
            POINTSET_METRIC_DIRECTIONS.items()
        ):
            endpoints[metric] = paired_endpoint_statistics(
                np.asarray(
                    [row["metrics"]["raw"][metric] for row in anatomy_rows]
                ),
                np.asarray(
                    [row["metrics"]["deployed"][metric] for row in anatomy_rows]
                ),
                direction=direction,
                bootstrap_seed=bootstrap_seed + 100 * anatomy_index + metric_index,
                bootstrap_resamples=bootstrap_resamples,
            )
        add_holm_adjustment(endpoints)
        by_anatomy[anatomy] = endpoints
        topology_by_anatomy[anatomy] = {
            method: {
                metric: descriptive_statistics(
                    np.asarray(
                        [row["metrics"][method][metric] for row in anatomy_rows]
                    )
                )
                for metric in (
                    "input_points",
                    "mesh_vertices",
                    "mesh_triangles",
                    "mesh_connected_components",
                )
            }
            for method in ("raw", "deployed")
        }

    specimen_macro: dict[str, dict[str, object]] = {}
    for metric_index, (metric, direction) in enumerate(
        POINTSET_METRIC_DIRECTIONS.items()
    ):
        raw_values: list[float] = []
        deployed_values: list[float] = []
        for specimen in specimens:
            specimen_rows = [row for row in rows if row["specimen"] == specimen]
            raw_values.append(
                float(np.mean([row["metrics"]["raw"][metric] for row in specimen_rows]))
            )
            deployed_values.append(
                float(
                    np.mean(
                        [row["metrics"]["deployed"][metric] for row in specimen_rows]
                    )
                )
            )
        specimen_macro[metric] = paired_endpoint_statistics(
            np.asarray(raw_values),
            np.asarray(deployed_values),
            direction=direction,
            bootstrap_seed=bootstrap_seed + 1000 + metric_index,
            bootstrap_resamples=bootstrap_resamples,
        )
    add_holm_adjustment(specimen_macro)

    protocol_signatures = set()
    for row in rows:
        comparable = dict(row["protocol"])
        comparable.pop("deployment_action", None)
        protocol_signatures.add(json.dumps(comparable, sort_keys=True))
    if len(protocol_signatures) != 1:
        raise ValueError("surface cases use inconsistent reconstruction protocols")
    common_surface_protocol = dict(rows[0]["protocol"])
    common_surface_protocol.pop("deployment_action", None)
    result = {
        "status": "nested_loocv_surface_summary_complete",
        "independent_unit": "specimen",
        "specimens": list(specimens),
        "cases": len(rows),
        "prediction_protocol": protocol["status"],
        "surface_protocol": common_surface_protocol,
        "statistics": {
            "paired_test": "two-sided Wilcoxon signed-rank",
            "multiplicity": "Holm adjustment across four endpoints within each analysis",
            "confidence_interval": "specimen-level percentile bootstrap of paired mean improvement",
            "bootstrap_resamples": bootstrap_resamples,
        },
        "deployed_by_anatomy": by_anatomy,
        "specimen_macro": specimen_macro,
        "topology_by_anatomy": topology_by_anatomy,
        "case_results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "specimens": len(specimens),
                "cases": len(rows),
            },
            indent=2,
        )
    )


def reconstruct_poisson_surface(
    points: np.ndarray,
    beam_directions: np.ndarray,
    *,
    depth: int,
    density_quantile: float,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
):
    """Reconstruct one deterministic screened-Poisson surface with Open3D."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=3.0, max_nn=32
        )
    )
    normals = np.asarray(cloud.normals)
    flip = np.einsum("ij,ij->i", normals, -beam_directions) < 0.0
    normals[flip] *= -1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=int(depth),
        linear_fit=True,
    )
    density = np.asarray(densities)
    if density_quantile > 0.0 and len(density):
        cutoff = float(np.quantile(density, density_quantile))
        mesh.remove_vertices_by_mask(density < cutoff)
    bounds = o3d.geometry.AxisAlignedBoundingBox(crop_min, crop_max)
    mesh = mesh.crop(bounds)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced no triangles")
    return mesh


def reconstruct_ball_pivoting_surface(
    points: np.ndarray,
    beam_directions: np.ndarray,
    *,
    radii_mm: Sequence[float],
):
    """Reconstruct an open surface with deterministic ball pivoting."""
    import open3d as o3d

    radii = tuple(float(value) for value in radii_mm)
    if not radii or any(value <= 0.0 for value in radii):
        raise ValueError("ball-pivoting radii must be positive")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(3.0, 2.0 * max(radii)), max_nn=32
        )
    )
    normals = np.asarray(cloud.normals)
    flip = np.einsum("ij,ij->i", normals, -beam_directions) < 0.0
    normals[flip] *= -1.0
    cloud.normals = o3d.utility.Vector3dVector(normals)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector(radii)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise RuntimeError("ball pivoting produced no triangles")
    return mesh


def reconstruct_sweep_topology_surface(
    points: np.ndarray,
    beam_directions: np.ndarray,
    record_index: np.ndarray,
    frame_index: np.ndarray,
    column_index: np.ndarray,
    confidence: np.ndarray,
    *,
    max_frame_gap: int,
    max_column_gap_px: int,
    max_edge_mm: float,
):
    """Triangulate each tracked sweep as an acquisition-ordered open sheet."""
    import open3d as o3d
    from scipy.spatial import Delaunay, QhullError

    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    vertex_offset = 0
    for record in np.unique(record_index):
        selected = np.flatnonzero(record_index == record)
        if len(selected) < 3:
            continue
        order = np.lexsort(
            (
                -confidence[selected],
                column_index[selected],
                frame_index[selected],
            )
        )
        ordered = selected[order]
        coordinates = np.column_stack(
            [frame_index[ordered], column_index[ordered]]
        )
        _, first = np.unique(coordinates, axis=0, return_index=True)
        unique = ordered[np.sort(first)]
        if len(unique) < 3:
            continue
        parameters = np.column_stack(
            [frame_index[unique], column_index[unique]]
        ).astype(np.float64)
        try:
            candidate = Delaunay(parameters).simplices.astype(np.int64)
        except QhullError:
            continue
        candidate_frames = frame_index[unique][candidate]
        candidate_columns = column_index[unique][candidate]
        keep = (
            np.ptp(candidate_frames, axis=1) <= int(max_frame_gap)
        ) & (
            np.ptp(candidate_columns, axis=1) <= int(max_column_gap_px)
        )
        candidate = candidate[keep]
        if not len(candidate):
            continue
        local_points = points[unique]
        triangle_points = local_points[candidate]
        edges = np.stack(
            [
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 1],
                triangle_points[:, 0] - triangle_points[:, 2],
            ],
            axis=1,
        )
        edge_length = np.linalg.norm(edges, axis=2)
        area_vector = np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        )
        keep = (
            np.max(edge_length, axis=1) <= float(max_edge_mm)
        ) & (np.linalg.norm(area_vector, axis=1) > 1e-8)
        candidate = candidate[keep]
        area_vector = area_vector[keep]
        if not len(candidate):
            continue
        mean_beam = np.mean(beam_directions[unique][candidate], axis=1)
        flip = np.einsum("ij,ij->i", area_vector, -mean_beam) < 0.0
        candidate[flip, 1], candidate[flip, 2] = (
            candidate[flip, 2].copy(),
            candidate[flip, 1].copy(),
        )
        vertices.append(local_points)
        triangles.append(candidate + vertex_offset)
        vertex_offset += len(local_points)
    if not triangles:
        raise RuntimeError("sweep-topology reconstruction produced no triangles")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.concatenate(vertices)),
        o3d.utility.Vector3iVector(np.concatenate(triangles)),
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise RuntimeError("sweep-topology cleanup produced no triangles")
    return mesh


def evaluate_reconstructed_surfaces(args: argparse.Namespace) -> None:
    """Reconstruct and evaluate actual meshes from frozen point sets."""
    import open3d as o3d

    prediction = np.load(args.prediction_archive)
    specimen = archive_scalar(prediction, "specimen")
    anatomy = archive_scalar(prediction, "anatomy")
    records = tuple(str(item) for item in prediction["records"])
    root = Path(args.dataset_root)

    reference_parts: list[np.ndarray] = []
    cursor = 0
    for index, record in enumerate(records):
        data = load_record(root, specimen, anatomy, record)
        selected = slice(cursor, cursor + int(prediction["counts"][index]))
        rays = archived_rays(prediction, selected)
        if rays is None:
            raise ValueError("surface evaluation requires archived rays")
        ct_depths, _ = ct_first_intersections(data.mesh_path, rays)
        valid = np.isfinite(ct_depths)
        reference_parts.append(
            rays.origins_mm[valid]
            + ct_depths[valid, None] * rays.directions[valid]
        )
        cursor += len(rays)
    reference = np.concatenate(reference_parts)
    if cursor != len(prediction["ray_depths_mm"]):
        raise RuntimeError("unused archived rays remain")

    origins = prediction["ray_origins_mm"].astype(np.float64)
    directions = prediction["ray_directions"].astype(np.float64)
    depths = prediction["ray_depths_mm"].astype(np.float64)
    correction = prediction["correction"].astype(np.float64)
    point_index = np.arange(len(depths), dtype=np.int64)
    record_index = np.repeat(
        np.arange(len(prediction["counts"]), dtype=np.int64),
        prediction["counts"].astype(np.int64),
    )
    frame_index = prediction["ray_record_frame"].astype(np.int64)
    column_index = prediction["ray_column"].astype(np.int64)
    confidence = prediction["ray_confidence"].astype(np.float64)
    raw_points = origins + depths[:, None] * directions
    refined_points = origins + (depths + correction)[:, None] * directions
    selector_mode = (
        archive_scalar(prediction, "correction_mode")
        == "privileged_sweep_set_selector"
    )
    if selector_mode:
        selector_keep = prediction["selection_keep"].astype(bool)
        available = {
            "raw": (raw_points, directions, point_index),
            "relaxed_anchor": (refined_points, directions, point_index),
            "selector_raw": (
                raw_points[selector_keep],
                directions[selector_keep],
                point_index[selector_keep],
            ),
            "selector_refined": (
                refined_points[selector_keep],
                directions[selector_keep],
                point_index[selector_keep],
            ),
        }
        deployment_action = archive_scalar(prediction, "deployment_action")
    else:
        available = {
            "raw": (raw_points, directions, point_index),
            "relaxed_anchor": (refined_points, directions, point_index),
        }
        deployment_action = "relaxed_anchor"
    requested = tuple(args.method or ("raw", "deployed"))
    resolved: list[tuple[str, str]] = []
    for name in requested:
        underlying = deployment_action if name == "deployed" else name
        if underlying not in available:
            raise ValueError(f"surface method {name!r} is unavailable")
        if name not in {item[0] for item in resolved}:
            resolved.append((name, underlying))

    mesh_root = Path(args.mesh_dir)
    mesh_root.mkdir(parents=True, exist_ok=True)
    mesh_path = root / specimen / "CT_bone_segmentations" / f"{anatomy}.stl"
    ct_mesh = trimesh.load_mesh(mesh_path, process=True)
    if not isinstance(ct_mesh, trimesh.Trimesh):
        raise TypeError(f"expected a triangle mesh: {mesh_path}")
    open3d_scene = build_open3d_surface_scene(ct_mesh)

    metrics: dict[str, dict[str, float | int | bool | str]] = {}
    for output_name, method_name in resolved:
        points, method_directions, method_index = available[method_name]
        input_points = int(len(points))
        if str(args.outlier_mode) == "statistical":
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            _, retained_indices = cloud.remove_statistical_outlier(
                nb_neighbors=int(args.outlier_neighbours),
                std_ratio=float(args.outlier_std_ratio),
            )
            retained = np.asarray(retained_indices, dtype=np.int64)
            points = points[retained]
            method_directions = method_directions[retained]
            method_index = method_index[retained]
        if len(points) < 100:
            raise RuntimeError(f"{output_name} retained fewer than 100 points")
        margin = float(args.crop_margin_mm)
        crop_min = points.min(axis=0) - margin
        crop_max = points.max(axis=0) + margin
        if args.surface_method == "poisson":
            surface = reconstruct_poisson_surface(
                points,
                method_directions,
                depth=int(args.poisson_depth),
                density_quantile=float(args.density_quantile),
                crop_min=crop_min,
                crop_max=crop_max,
            )
        elif args.surface_method == "ball_pivoting":
            surface = reconstruct_ball_pivoting_surface(
                points,
                method_directions,
                radii_mm=args.ball_radii_mm,
            )
        else:
            surface = reconstruct_sweep_topology_surface(
                points,
                method_directions,
                record_index[method_index],
                frame_index[method_index],
                column_index[method_index],
                confidence[method_index],
                max_frame_gap=int(args.topology_max_frame_gap),
                max_column_gap_px=int(args.topology_max_column_gap_px),
                max_edge_mm=float(args.topology_max_edge_mm),
            )
        output_mesh = mesh_root / f"{output_name}.ply"
        if not o3d.io.write_triangle_mesh(str(output_mesh), surface):
            raise IOError(f"failed to write {output_mesh}")
        o3d.utility.random.seed(260831)
        sampled = surface.sample_points_uniformly(
            number_of_points=int(args.sample_points)
        )
        sampled_points = np.asarray(sampled.points)
        surface_metrics = point_set_metrics(
            ct_mesh,
            sampled_points,
            reference,
            int(args.normal_neighbours),
            open3d_scene,
        )
        triangle_clusters, cluster_triangles, _ = (
            surface.cluster_connected_triangles()
        )
        surface_metrics.update(
            {
                "underlying_method": method_name,
                "input_points": input_points,
                "reconstruction_points": int(len(points)),
                "mesh_vertices": int(len(surface.vertices)),
                "mesh_triangles": int(len(surface.triangles)),
                "mesh_connected_components": int(len(cluster_triangles)),
                "mesh_edge_manifold": bool(surface.is_edge_manifold()),
                "mesh_vertex_manifold": bool(surface.is_vertex_manifold()),
                "mesh_watertight": bool(surface.is_watertight()),
                "mesh_path": str(output_mesh),
            }
        )
        metrics[output_name] = surface_metrics

    result = {
        "case": {"specimen": specimen, "anatomy": anatomy},
        "protocol": {
            "surface": str(args.surface_method),
            "poisson_depth": int(args.poisson_depth),
            "density_quantile": float(args.density_quantile),
            "ball_radii_mm": [float(value) for value in args.ball_radii_mm],
            "topology": f"method_specific_{args.surface_method}",
            "normal_orientation": "toward_acquisition_probe",
            "requested_methods": list(requested),
            "deployment_action": deployment_action,
            "outlier_mode": str(args.outlier_mode),
            "outlier_neighbours": int(args.outlier_neighbours),
            "outlier_std_ratio": float(args.outlier_std_ratio),
            "crop_margin_mm": float(args.crop_margin_mm),
            "topology_max_frame_gap": int(args.topology_max_frame_gap),
            "topology_max_column_gap_px": int(
                args.topology_max_column_gap_px
            ),
            "topology_max_edge_mm": float(args.topology_max_edge_mm),
            "surface_samples": int(args.sample_points),
            "CT_used_for_reconstruction": False,
            "reference": "CT_first_intersections_on_acquisition_rays",
        },
        "metrics": metrics,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    anchor = subparsers.add_parser('anchor', help='build the active zero-parameter analytic anchor')
    anchor.add_argument('--dataset-root', required=True)
    anchor.add_argument('--specimen', required=True)
    anchor.add_argument('--anatomy', choices=('foot', 'tibia', 'fibula'), required=True)
    anchor.add_argument('--records', nargs='+', help='records to use; default: every record for this anatomy')
    anchor.add_argument('--output', required=True)
    anchor.add_argument('--evaluation-stride', type=int, default=EVALUATION_STRIDE)
    anchor.add_argument('--evaluation-offset', type=int, default=EVALUATION_OFFSET)
    anchor.set_defaults(function=save_analytic_anchor)
    teacher = subparsers.add_parser('privileged-label', help='create training-only CT teacher targets for one anchor archive')
    teacher.add_argument('--dataset-root', required=True)
    teacher.add_argument('--input-archive', required=True)
    teacher.add_argument('--output', required=True)
    teacher.set_defaults(function=build_privileged_teacher_labels)
    nested_policy = subparsers.add_parser('select-nested-policy', help='select score and retention from inner cross-fit training cases, then apply them to already frozen outer predictions')
    nested_policy.add_argument('--input-protocol', required=True)
    nested_policy.add_argument('--feature-root', required=True)
    nested_policy.add_argument('--label-root', required=True)
    nested_policy.add_argument('--output-dir', required=True)
    nested_policy.add_argument('--protocol-output', required=True)
    nested_policy.add_argument('--inner-folds', type=int, default=3)
    nested_policy.add_argument('--student-variant', choices=SELECTOR_VARIANTS, default='sweep_set')
    nested_policy.add_argument('--teacher-target', choices=('soft', 'hard'), default='soft')
    nested_policy.add_argument('--ranking-weight', type=float, default=0.0)
    nested_policy.add_argument('--retention', type=float, action='append', default=[])
    nested_policy.add_argument('--epochs', type=int, default=12)
    nested_policy.add_argument('--batch-size', type=int, default=8192)
    nested_policy.add_argument('--learning-rate', type=float, default=0.001)
    nested_policy.add_argument('--seed', type=int, default=17)
    nested_policy.add_argument('--device', default='auto')
    nested_policy.set_defaults(function=select_nested_policy_from_frozen_outer)
    selector_ablation = subparsers.add_parser('selector-ablation-loocv', help='freeze a fixed-rank outer-LOOCV architecture or teacher-target ablation')
    selector_ablation.add_argument('--feature-root', required=True)
    selector_ablation.add_argument('--label-root', required=True)
    selector_ablation.add_argument('--specimen', action='append', required=True)
    selector_ablation.add_argument('--heldout-specimen', action='append')
    selector_ablation.add_argument('--output-dir', required=True)
    selector_ablation.add_argument('--protocol-output')
    selector_ablation.add_argument('--student-variant', choices=SELECTOR_VARIANTS, default='sweep_set')
    selector_ablation.add_argument('--teacher-target', choices=('soft', 'hard'), default='soft')
    selector_ablation.add_argument('--ranking-weight', type=float, default=0.0)
    selector_ablation.add_argument('--score', choices=('anatomy', 'surface', 'joint', 'factor'), default='factor')
    selector_ablation.add_argument('--retention', type=float, default=0.9)
    selector_ablation.add_argument('--epochs', type=int, default=12)
    selector_ablation.add_argument('--batch-size', type=int, default=8192)
    selector_ablation.add_argument('--learning-rate', type=float, default=0.001)
    selector_ablation.add_argument('--seed', type=int, default=17)
    selector_ablation.add_argument('--device', default='auto')
    selector_ablation.set_defaults(function=selector_ablation_leave_one_specimen_out)
    point_set = subparsers.add_parser('point-set-evaluate', help='evaluate frozen corrections as CT-visible 3-D point sets')
    point_set.add_argument('--dataset-root', required=True)
    point_set.add_argument('--prediction-archive', required=True)
    point_set.add_argument('--teacher-root', help='evaluation-only CT teacher root for the official inference-CT filter comparator')
    point_set.add_argument('--specimen')
    point_set.add_argument('--anatomy', choices=('foot', 'tibia', 'fibula'))
    point_set.add_argument('--normal-neighbours', type=int, default=24)
    point_set.add_argument('--export-archive')
    point_set.add_argument('--output')
    point_set.set_defaults(function=evaluate_point_sets)
    summary = subparsers.add_parser('summarize-nested-loocv', help='summarize frozen nested-LOOCV point-set and selector results')
    summary.add_argument('--protocol', required=True)
    summary.add_argument('--label-root', required=True)
    summary.add_argument('--output', required=True)
    summary.add_argument('--bootstrap-resamples', type=int, default=100000)
    summary.add_argument('--bootstrap-seed', type=int, default=20260831)
    summary.set_defaults(function=summarize_nested_loocv)
    surface_summary = subparsers.add_parser('summarize-nested-surfaces', help='summarize frozen nested-LOOCV open-surface results')
    surface_summary.add_argument('--protocol', required=True)
    surface_summary.add_argument('--surface-root', required=True)
    surface_summary.add_argument('--output', required=True)
    surface_summary.add_argument('--bootstrap-resamples', type=int, default=100000)
    surface_summary.add_argument('--bootstrap-seed', type=int, default=20260831)
    surface_summary.set_defaults(function=summarize_nested_surfaces)
    surface = subparsers.add_parser('surface-evaluate', help='reconstruct and evaluate actual open surfaces')
    surface.add_argument('--dataset-root', required=True)
    surface.add_argument('--prediction-archive', required=True)
    surface.add_argument('--mesh-dir', required=True)
    surface.add_argument('--output', required=True)
    surface.add_argument('--method', action='append', choices=('raw', 'relaxed_anchor', 'selector_raw', 'selector_refined', 'deployed'), help='surface input to reconstruct; default: raw and deployed')
    surface.add_argument('--surface-method', choices=('sweep_topology', 'ball_pivoting', 'poisson'), default='sweep_topology')
    surface.add_argument('--ball-radii-mm', type=float, nargs='+', default=(0.5, 1.0, 2.0, 4.0))
    surface.add_argument('--poisson-depth', type=int, default=8)
    surface.add_argument('--density-quantile', type=float, default=0.02)
    surface.add_argument('--outlier-mode', choices=('none', 'statistical'), default='none')
    surface.add_argument('--outlier-neighbours', type=int, default=24)
    surface.add_argument('--outlier-std-ratio', type=float, default=2.0)
    surface.add_argument('--crop-margin-mm', type=float, default=2.0)
    surface.add_argument('--topology-max-frame-gap', type=int, default=1)
    surface.add_argument('--topology-max-column-gap-px', type=int, default=64)
    surface.add_argument('--topology-max-edge-mm', type=float, default=5.0)
    surface.add_argument('--sample-points', type=int, default=100000)
    surface.add_argument('--normal-neighbours', type=int, default=24)
    surface.set_defaults(function=evaluate_reconstructed_surfaces)
    return parser

def main():
    args = build_parser().parse_args()
    args.function(args)
