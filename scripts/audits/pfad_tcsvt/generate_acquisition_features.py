#!/usr/bin/env python3
"""Generate every predeclared ultrasound perturbation without opening CT content."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / "src"))
from pfad_tcsvt.common import ANATOMIES,RESULT,SPECIMENS,dump_json,experiment_directory,load_pfad,sha256
from pfad_tcsvt.acquisition import conditions,extract_original_rays,limit_tree_threads,load_payload,perturb,save_payload,student_case


def run_case(config_path,specimen,anatomy):
    config,output = experiment_directory(Path(config_path))
    limit_tree_threads()
    module = load_pfad()
    name = f"{specimen}_{anatomy}"
    destination = output / "features" / name
    destination.mkdir(parents=True,exist_ok=True)
    done = destination / "complete.json"
    if done.exists():
        return json.loads(done.read_text())
    start = time.monotonic()
    feature_path = module.discover_case_archive(RESULT / "features",specimen,anatomy)
    template = load_payload(feature_path)
    source,query = extract_original_rays(specimen,anatomy,template,output / "raw_rays" / f"{name}.npz")
    rows = []
    for condition in conditions(config):
        path = destination / f"{condition['id']}.npz"
        info_path = path.with_suffix(".json")
        if info_path.exists() and path.exists():
            rows.append(json.loads(info_path.read_text()))
            continue
        item_start = time.monotonic()
        if condition["family"] == "clean":
            payload = dict(template)
            payload["original_query_index"] = np.arange(len(template["ray_depths_mm"]))
            payload["condition"] = np.asarray(json.dumps(condition))
            extra = {"changed":False}
        else:
            payload,extra = perturb(template,source,query,specimen,anatomy,condition,config)
        case = student_case(payload)
        if not np.isfinite(case["source"]).all() or not np.isfinite(case["global"]).all():
            raise RuntimeError(f"nonfinite model input: {name}/{condition['id']}")
        if not np.allclose(np.linalg.norm(payload["ray_directions"],axis=1),1,atol=2e-6):
            raise RuntimeError("invalid beam direction")
        save_payload(path,payload)
        row = {"condition":condition,"feature_path":str(path.relative_to(ROOT)),"sha256":sha256(path),
               "queries":len(payload["ray_depths_mm"]),"records":len(payload["records"]),
               "seconds":time.monotonic()-item_start,"CT_content_read":False,**extra}
        dump_json(info_path,row)
        rows.append(row)
        print(json.dumps({"case":name,"condition":condition["id"],"queries":row["queries"],"seconds":round(row["seconds"],2)}),flush=True)
    report = {"status":"complete","specimen":specimen,"anatomy":anatomy,"conditions":rows,
              "seconds":time.monotonic()-start,"original_feature_sha256":sha256(feature_path)}
    dump_json(done,report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT / "configs/pfad/tcsvt/acquisition_robustness_v1.json")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--case",action="append",help="specimen01/foot")
    args = parser.parse_args()
    config,output = experiment_directory(args.config)
    output.mkdir(parents=True,exist_ok=True)
    code = {str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT / "src/pfad_tcsvt/acquisition.py",ROOT / "scripts/run_first_arrival_refinement.py"]}
    run_file = output / "feature_generation.json"
    if run_file.exists() and json.loads(run_file.read_text())["code_sha256"] != code:
        raise RuntimeError("feature implementation changed after generation started")
    dump_json(run_file,{"status":"running","config":config,"code_sha256":code,"CT_content_read":False})
    tasks = [tuple(c.split("/")) for c in args.case] if args.case else [(s,a) for s in SPECIMENS for a in ANATOMIES]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(run_case,str(args.config.resolve()),s,a):(s,a) for s,a in tasks}
        for future in as_completed(futures):
            report = future.result()
            print(json.dumps({"stage":"case_complete","specimen":report["specimen"],"anatomy":report["anatomy"],"seconds":report["seconds"]}),flush=True)
    all_done = all((output / "features" / f"{s}_{a}/complete.json").exists() for s in SPECIMENS for a in ANATOMIES)
    dump_json(run_file,{"status":"complete" if all_done else "partial","config":config,"code_sha256":code,"CT_content_read":False})


if __name__ == "__main__":
    main()
