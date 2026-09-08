#!/usr/bin/env python3
"""Freeze every perturbed prediction with its clean-training policy; no CT I/O."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / "src"))
from pfad_tcsvt.common import ANATOMIES,SPECIMENS,dump_json,experiment_directory,load_pfad,sha256
from pfad_tcsvt.acquisition import load_payload,save_payload,student_case
from pfad_tcsvt.models import SetTransformerSelector

M=load_pfad()


def score_probability(probability,kind):
    return {"anatomy":probability[:,0],"surface":probability[:,1],"joint":probability[:,2],
            "factor":probability[:,0]*probability[:,1]}[kind]


def process_case(output,training_root,specimen,anatomy,model_cache,device,batch_size):
    case_name=f"{specimen}_{anatomy}"
    directory=output / "predictions" / case_name
    directory.mkdir(parents=True,exist_ok=True)
    done=directory / "complete.json"
    if done.exists():return False
    features_file=output / "features" / case_name / "complete.json"
    fold=training_root / "folds" / specimen
    released=ROOT / "checkpoints" / specimen
    use_released=not (fold / "comparison_fold.json").exists()
    if not features_file.exists():return False
    if use_released and not (released / "provenance.json").exists():return False
    if model_cache.get("specimen")!=specimen:
        model_cache.clear()
        gc.collect()
        if device.type=="cuda":torch.cuda.empty_cache()
        pfad=M.AttentiveSweepSelector().to(device)
        pfad_checkpoint=(released if use_released else fold) / "pfad.pt"
        pfad.load_state_dict(torch.load(pfad_checkpoint,map_location=device,weights_only=True))
        policy=json.loads((released / "set_transformer_policy.json").read_text()) if use_released else json.loads((fold / "chosen_policy.json").read_text())["chosen"]
        checkpoint=released / "set_transformer.pt" if use_released else fold / "outer" / f"epoch{policy['epochs']}.pt"
        st=SetTransformerSelector().to(device)
        st.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        pfad_policy=json.loads((released / "pfad_policy.json").read_text()) if use_released else json.loads((fold / "pfad_replay.json").read_text())["policy"]
        model_cache.update({"specimen":specimen,"pfad":pfad.eval(),"set_transformer":st.eval(),
            "policies":{"pfad":pfad_policy,"set_transformer":policy},
            "checkpoints":{"pfad":sha256(pfad_checkpoint),"set_transformer":sha256(checkpoint)}})
    source=json.loads(features_file.read_text())
    outputs=[]
    start=time.monotonic()
    for feature in source["conditions"]:
        condition=feature["condition"]
        path=directory / f"{condition['id']}.npz"
        metadata=path.with_suffix(".json")
        if metadata.exists() and path.exists():
            outputs.append(json.loads(metadata.read_text()))
            continue
        payload=load_payload(ROOT / feature["feature_path"])
        case=student_case(payload)
        prediction={}
        for name in ["pfad","set_transformer"]:
            probability=M.predict_sweep_set_selector(model_cache[name],case,batch_size=batch_size,device=device)
            policy=model_cache["policies"][name]
            score=score_probability(probability,policy["score"])
            keep,_=M.coverage_constrained_selector_mask(score,np.inf,case["counts"],policy["retention"])
            if not np.isfinite(score).all():raise RuntimeError("nonfinite score")
            prediction[name+"_score"]=score.astype(np.float32)
            prediction[name+"_keep"]=keep.astype(np.uint8)
        geometry_score=-payload["cross_sweep_nearest_mm"].astype(np.float32)
        geometry_keep,_=M.coverage_constrained_selector_mask(geometry_score,np.inf,case["counts"],model_cache["policies"]["pfad"]["retention"])
        prediction.update({"geometry_keep":geometry_keep.astype(np.uint8),"geometry_score":geometry_score,
            "correction":np.clip(M.ANALYTIC_RELAXATION*payload["analytic"],-M.MAX_CORRECTION_MM,M.MAX_CORRECTION_MM).astype(np.float32),
            "feature_path":np.asarray(feature["feature_path"]),"condition":np.asarray(json.dumps(condition)),
            "specimen":np.asarray(specimen),"anatomy":np.asarray(anatomy)})
        save_payload(path,prediction)
        report={"condition":condition,"path":str(path.relative_to(ROOT)),"sha256":sha256(path),
                "feature_sha256":feature["sha256"],"queries":len(payload["ray_depths_mm"]),
                "retained":{name:int(prediction[name+"_keep"].sum()) for name in ["pfad","set_transformer","geometry"]},
                "CT_content_read":False,"heldout_labels_opened":False,"policy_reselected":False}
        dump_json(metadata,report)
        outputs.append(report)
    dump_json(done,{"status":"all_case_conditions_frozen","specimen":specimen,"anatomy":anatomy,
                   "model_checkpoint_sha256":model_cache["checkpoints"],"policies":model_cache["policies"],
                   "predictions":outputs,"seconds":time.monotonic()-start,"CT_content_read":False})
    print(json.dumps({"stage":"case_predictions_frozen","case":case_name,"conditions":len(outputs),"seconds":time.monotonic()-start}),flush=True)
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT / "configs/pfad/tcsvt/acquisition_robustness_v1.json")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--wait",action="store_true")
    parser.add_argument("--case",action="append")
    args=parser.parse_args()
    config,output=experiment_directory(args.config)
    training_config,training_root=experiment_directory(ROOT / config["training_config"])
    torch.set_num_threads(2)
    device=M.choose_device(args.device)
    tasks=[tuple(c.split("/")) for c in args.case] if args.case else [(s,a) for s in SPECIMENS for a in ANATOMIES]
    code_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT / "src/pfad_tcsvt/acquisition.py",ROOT / "src/pfad_tcsvt/models.py"]}
    run_file=output / "prediction_run.json"
    if run_file.exists() and json.loads(run_file.read_text())["code_sha256"]!=code_hashes:
        raise RuntimeError("prediction implementation changed")
    dump_json(run_file,{"status":"running","code_sha256":code_hashes,"heldout_CT_or_labels_read":False})
    cache={}
    last_message=0.0
    while True:
        progress=False
        for specimen,anatomy in tasks:
            progress=process_case(output,training_root,specimen,anatomy,cache,device,training_config["batch_size"]) or progress
        complete=sum((output / "predictions" / f"{s}_{a}/complete.json").exists() for s,a in tasks)
        if complete==len(tasks) or not args.wait:break
        if time.monotonic()-last_message>45:
            print(json.dumps({"stage":"awaiting_inputs_or_frozen_models","complete_cases":complete,"expected_cases":len(tasks)}),flush=True)
            last_message=time.monotonic()
        if not progress:time.sleep(5)
    all_done=all((output / "predictions" / f"{s}_{a}/complete.json").exists() for s in SPECIMENS for a in ANATOMIES)
    dump_json(run_file,{"status":"complete" if all_done else "partial","code_sha256":code_hashes,"heldout_CT_or_labels_read":False})


if __name__=="__main__":main()
