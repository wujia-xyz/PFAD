"""Rebuild ultrasound-only inputs under fixed acquisition perturbations."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .common import ROOT, RESULT, dump_json, load_pfad

M = load_pfad()
DATA = Path(__import__("os").environ.get("PFAD_DATA_ROOT", str(ROOT / "data/UltraBones100k")))
RAY_FIELDS = ("origins_mm", "directions", "depths_mm", "confidence", "frame_index", "column_index")


class LimitedTree(cKDTree):
    def query(self, *args, **kwargs):
        kwargs["workers"] = 2
        return super().query(*args, **kwargs)


def limit_tree_threads():
    M.cKDTree = LimitedTree


def subset_rays(rays, selected):
    return M.Rays(**{name: getattr(rays, name)[selected] for name in RAY_FIELDS})


def seeded_rng(seed, specimen, anatomy, family, replicate):
    value = f"{seed}/{specimen}/{anatomy}/{family}/{replicate}".encode()
    number = int.from_bytes(hashlib.sha256(value).digest()[:8], "little")
    return np.random.default_rng(number)


def conditions(config):
    result = [{"id":"clean", "family":"clean", "level":0, "replicate":0}]
    for fraction in config["sweep_reduction"]["retained_fractions"]:
        for repeat in range(config["replicates"]):
            result.append({"id":f"sweeps_{int(fraction*100)}_r{repeat}", "family":"sweeps", "level":fraction, "replicate":repeat})
    for stride in config["frame_thinning"]["strides"]:
        for phase in range(stride):
            result.append({"id":f"frames_s{stride}_p{phase}", "family":"frames", "level":stride, "replicate":phase})
    for angle, distance in zip(config["pose_error"]["rotation_angle_deg"],config["pose_error"]["translation_norm_mm"],strict=True):
        for repeat in range(config["replicates"]):
            result.append({"id":f"pose_{str(angle).replace('.','p')}_r{repeat}", "family":"pose", "level":angle, "translation_mm":distance, "replicate":repeat})
    return result


def load_payload(path):
    with np.load(path) as data:
        return {key:data[key] for key in data.files}


def save_payload(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def extract_original_rays(specimen, anatomy, template, destination):
    """One skeletonization pass supplies both original column sampling grids."""
    if destination.exists():
        with np.load(destination) as archive:
            source = [M.Rays(**{field:archive[f"s{i}_{field}"] for field in RAY_FIELDS}) for i in range(len(template["records"]))]
            query = [M.Rays(**{field:archive[f"q{i}_{field}"] for field in RAY_FIELDS}) for i in range(len(template["records"]))]
        return source,query
    source,query = [],[]
    cursor = 0
    for index,record in enumerate(template["records"]):
        data = M.load_record(DATA,specimen,anatomy,str(record))
        combined = M.extract_rays(data,stride=4,offset=0,line_mode="skeleton")
        s = subset_rays(combined,combined.column_index % 8 == 0)
        q = subset_rays(combined,combined.column_index % 32 == 4)
        count = int(template["counts"][index])
        if len(q) != count:
            raise RuntimeError(f"query count changed in {specimen}/{anatomy}/{record}")
        mapping = {"origins_mm":"ray_origins_mm", "directions":"ray_directions", "depths_mm":"ray_depths_mm",
                   "confidence":"ray_confidence", "frame_index":"ray_record_frame", "column_index":"ray_column"}
        for field, archived_field in mapping.items():
            old = template[archived_field][cursor:cursor+count]
            regenerated = getattr(q,field).astype(old.dtype)
            if not np.array_equal(regenerated,old):
                raise RuntimeError(f"raw query replay differs in {field}, {specimen}/{anatomy}/{record}")
        source.append(s)
        query.append(q)
        cursor += count
    payload = {f"{kind}{i}_{field}":getattr(rays,field)
               for kind,sets in [("s",source),("q",query)] for i,rays in enumerate(sets) for field in RAY_FIELDS}
    save_payload(destination,payload)
    dump_json(destination.with_suffix(".json"),{"specimen":specimen,"anatomy":anatomy,
        "query_replay_float32_bitwise_equal":True,"source_points":sum(map(len,source)),"query_points":sum(map(len,query)),
        "CT_content_read":False,"extraction":"original skeletonization, source column stride8/offset0, query32/offset4; shared stride4 union pass"})
    return source,query


def transform_rays(rays, rotation, translation, pivot):
    return M.Rays(
        origins_mm=(rays.origins_mm-pivot) @ rotation.T+pivot+translation,
        directions=rays.directions @ rotation.T,
        depths_mm=rays.depths_mm.copy(),confidence=rays.confidence.copy(),
        frame_index=rays.frame_index.copy(),column_index=rays.column_index.copy())


def build_payload(template, sources, queries, retained_records, original_indices, condition):
    arrays = M.case_anchor_arrays(sources,queries)
    anchor,centre,disagreement,nearest,neighborhood,evidence,_,counts = arrays
    payload = {k:np.array(v,copy=True) for k,v in template.items() if not k.startswith("ray_")}
    for key in ("records","record_image_widths","record_image_heights","record_scale_mm","record_frame_counts"):
        payload[key] = template[key][retained_records]
    payload.update({"analytic":anchor,"plane_centre_mm":centre,"plane_disagreement_mm":disagreement,
        "cross_sweep_nearest_mm":nearest,"cross_sweep_neighbourhood_mm":neighborhood,"source_evidence":evidence,
        "agreement":np.clip(1-disagreement/M.AGREEMENT_SCALE_MM,0,1).astype(np.float32),"counts":counts,
        "original_query_index":original_indices.astype(np.int64),"condition":np.asarray(json.dumps(condition))})
    for source_field,target_field in [("origins_mm","ray_origins_mm"),("directions","ray_directions"),
             ("depths_mm","ray_depths_mm"),("confidence","ray_confidence"),
             ("frame_index","ray_record_frame"),("column_index","ray_column")]:
        dtype = np.int32 if source_field in ("frame_index","column_index") else np.float32
        payload[target_field] = np.concatenate([getattr(q,source_field) for q in queries]).astype(dtype)
    offsets = np.cumsum(np.r_[0,payload["record_frame_counts"][:-1]])
    payload["frame"] = np.concatenate([q.frame_index+offset for q,offset in zip(queries,offsets,strict=True)]).astype(np.int32)
    return payload


def reduced_sweep_payload(template, retained_records, condition):
    """Drop both query and source records; retained within-source features are exact."""
    starts = np.cumsum(np.r_[0,template["counts"]])
    all_records = range(len(template["counts"]))
    pieces,original_indices = [],[]
    for record in retained_records:
        original_indices.append(np.arange(starts[record],starts[record+1]))
        original_sources = [s for s in all_records if s != record]
        columns = [original_sources.index(s) for s in retained_records if s != record]
        pieces.append(template["source_evidence"][starts[record]:starts[record+1]][:,columns])
    indices = np.concatenate(original_indices)
    evidence = np.concatenate(pieces)
    payload = {}
    n = len(template["ray_depths_mm"])
    record_fields = ("records","record_image_widths","record_image_heights","record_scale_mm","record_frame_counts","counts")
    for key,value in template.items():
        if key in record_fields:
            payload[key] = value[retained_records]
        elif value.ndim and len(value)==n:
            payload[key] = value[indices]
        else:
            payload[key] = np.array(value,copy=True)
    centre = np.median(evidence[:,:,0],axis=1)
    disagreement = np.median(np.abs(evidence[:,:,0]-centre[:,None]),axis=1)
    reliability = np.clip(1-disagreement/M.AGREEMENT_SCALE_MM,0,1)
    payload.update({"source_evidence":evidence,"plane_centre_mm":centre.astype(np.float32),
        "plane_disagreement_mm":disagreement.astype(np.float32),"agreement":reliability.astype(np.float32),
        "analytic":np.clip(centre*reliability,-M.MAX_CORRECTION_MM,M.MAX_CORRECTION_MM).astype(np.float32),
        "cross_sweep_nearest_mm":np.median(evidence[:,:,1],axis=1).astype(np.float32),
        "cross_sweep_neighbourhood_mm":np.median(evidence[:,:,2],axis=1).astype(np.float32),
        "original_query_index":indices,"condition":np.asarray(json.dumps(condition))})
    cursor = 0
    frames = []
    frame_offset = 0
    for count,length in zip(payload["counts"],payload["record_frame_counts"],strict=True):
        frames.append(payload["ray_record_frame"][cursor:cursor+count]+frame_offset)
        cursor += int(count)
        frame_offset += int(length)
    payload["frame"] = np.concatenate(frames).astype(np.int32)
    return payload


def perturb(template,sources,queries,specimen,anatomy,condition,config):
    count = len(queries)
    all_records = np.arange(count)
    starts = np.cumsum(np.r_[0,template["counts"]])
    if condition["family"] == "sweeps":
        rng = seeded_rng(config["seed"],specimen,anatomy,"sweeps",condition["replicate"])
        number = max(config["sweep_reduction"]["minimum_records"],int(math.ceil(count*condition["level"])))
        retained = np.sort(rng.permutation(count)[:number])
        return reduced_sweep_payload(template,retained,condition),{"retained_record_indices":retained.tolist(),"changed":number<count}
    if condition["family"] == "frames":
        stride,phase = int(condition["level"]),condition["replicate"]
        new_sources = [subset_rays(s,s.frame_index % stride==phase) for s in sources]
        selections = [q.frame_index % stride==phase for q in queries]
        new_queries = [subset_rays(q,selected) for q,selected in zip(queries,selections,strict=True)]
        original_indices = np.concatenate([np.arange(starts[i],starts[i+1])[selected] for i,selected in enumerate(selections)])
        if min(map(len,new_sources+new_queries)) < 3:
            raise RuntimeError("thinning leaves an unusable record; report the failed condition")
        return build_payload(template,new_sources,new_queries,all_records,original_indices,condition),{"frame_stride":stride,"phase":phase,"changed":True}
    if condition["family"] == "pose":
        rng = seeded_rng(config["seed"],specimen,anatomy,"pose",condition["replicate"])
        new_sources,new_queries,transforms = [],[],[]
        for source,query in zip(sources,queries,strict=True):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            rotation = Rotation.from_rotvec(axis*np.deg2rad(condition["level"])).as_matrix()
            translation = direction*condition["translation_mm"]
            pivot = query.points_mm.mean(axis=0)
            new_sources.append(transform_rays(source,rotation,translation,pivot))
            new_queries.append(transform_rays(query,rotation,translation,pivot))
            transforms.append({"rotation":rotation.tolist(),"translation_mm":translation.tolist(),"pivot_mm":pivot.tolist()})
        return build_payload(template,new_sources,new_queries,all_records,np.arange(starts[-1]),condition),{"transforms":transforms,"changed":True}
    raise ValueError(condition)


class ArrayArchive(dict):
    @property
    def files(self):
        return list(self)


def student_case(payload):
    evidence = M.normalized_source_evidence(payload["source_evidence"])
    n,source_count,_ = evidence.shape
    source = np.zeros((n,5,8),np.float32)
    mask = np.zeros((n,5),np.float32)
    source[:,:source_count] = evidence
    mask[:,:source_count] = 1
    return {"source":source,"mask":mask,"global":M.student_global_evidence(ArrayArchive(payload)),
            "counts":payload["counts"],"specimen":str(payload["specimen"].item()),"anatomy":str(payload["anatomy"].item())}
