#!/usr/bin/env python3
"""Complete-case, specimen-level paired summaries of the TCSVT extension."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / "src"))
from pfad_tcsvt.common import ANATOMIES,SPECIMENS,dump_json,experiment_directory,load_pfad,sha256

M=load_pfad()
PRIMARY=("chamfer_l1_mm","hd95_mm","fscore_1mm","normal_consistency")
METRICS=PRIMARY+("precision_1mm","coverage_1mm")
METHODS=("raw","cross_sweep_nearest","pfad","set_transformer")
CONDITIONS=(("clean",0),("sweeps",.75),("sweeps",.5),("frames",2),("frames",4),("pose",.5),("pose",1.),("pose",2.))


def condition_key(condition):
    return condition["family"],condition["level"]


def specimen_vector(cases,family,level,method,metric):
    result=[]
    for specimen in SPECIMENS:
        anatomy_means=[]
        for anatomy in ANATOMIES:
            case=cases[(specimen,anatomy)]
            rows=[r for r in case["conditions"] if condition_key(r["condition"])==(family,level)]
            expected=1 if family=="clean" else int(level) if family=="frames" else 3
            if len(rows)!=expected:raise RuntimeError(f"incomplete replicates: {specimen}/{anatomy}/{family}/{level}")
            values=[r["metrics"][method][metric] for r in rows]
            anatomy_means.append(float(np.mean(values)))
        result.append(float(np.mean(anatomy_means)))
    return np.asarray(result)


def descriptive(values):
    rng=np.random.default_rng(20260908)
    index=rng.integers(0,len(values),size=(100000,len(values)))
    means=values[index].mean(axis=1)
    return {"mean":float(values.mean()),"sd":float(values.std(ddof=1)),"specimen_values":values.tolist(),
            "mean_95ci":np.quantile(means,[.025,.975]).tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT / "configs/pfad/tcsvt/acquisition_robustness_v1.json")
    args=parser.parse_args()
    config,output=experiment_directory(args.config)
    _,training_root=experiment_directory(ROOT / config["training_config"])
    paths=[output / "metrics" / f"{s}_{a}.json" for s in SPECIMENS for a in ANATOMIES]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:raise RuntimeError(f"No aggregate result: {len(missing)} cases are missing")
    cases={}
    for p in paths:
        case=json.loads(p.read_text())
        if case["status"]!="complete":raise RuntimeError("incomplete case")
        cases[(case["specimen"],case["anatomy"])]=case
    if set(cases)!={(s,a) for s in SPECIMENS for a in ANATOMIES}:raise RuntimeError("wrong case coverage")
    summary={}
    vectors={}
    for family,level in CONDITIONS:
        key=f"{family}:{level}"
        summary[key]={}
        for method in METHODS:
            summary[key][method]={}
            for metric in METRICS:
                values=specimen_vector(cases,family,level,method,metric)
                if not np.isfinite(values).all():raise RuntimeError("nonfinite aggregate")
                vectors[(key,method,metric)]=values
                summary[key][method][metric]=descriptive(values)
    clean={"points":{},"meshes":{}}
    clean_vectors={}
    for representation in ["points","meshes"]:
        for method in ["pfad","set_transformer"]:
            clean[representation][method]={}
            for metric in METRICS:
                values=[]
                for specimen in SPECIMENS:
                    part=[]
                    for anatomy in ANATOMIES:
                        case=cases[(specimen,anatomy)]
                        if representation=="meshes":
                            part.append(case["clean_surface_metrics"]["pfad_original" if method=="pfad" else method][metric])
                        elif method=="pfad":
                            part.append(case["original_pfad_point_metrics"][metric])
                        else:
                            part.append(next(r for r in case["conditions"] if r["condition"]["family"]=="clean")["metrics"][method][metric])
                    values.append(np.mean(part))
                array=np.asarray(values)
                clean_vectors[(representation,method,metric)]=array
                clean[representation][method][metric]=descriptive(array)
    comparator_tests={}
    for representation in ["points","meshes"]:
        entries={}
        for metric in PRIMARY:
            entries[metric]=M.paired_endpoint_statistics(clean_vectors[(representation,"set_transformer",metric)],
                clean_vectors[(representation,"pfad",metric)],direction=M.POINTSET_METRIC_DIRECTIONS[metric],
                bootstrap_seed=20260908,bootstrap_resamples=100000)
            entries[metric].update({"reference_method":"set_transformer","candidate_method":"pfad","positive_gain_favors":"pfad"})
        M.add_holm_adjustment(entries)
        comparator_tests[representation]=entries
    robustness_tests={}
    for family,level in CONDITIONS[1:]:
        key=f"{family}:{level}"
        for reference in ["raw","cross_sweep_nearest","set_transformer"]:
            for metric in PRIMARY:
                name=f"{key}/{reference}/{metric}"
                robustness_tests[name]=M.paired_endpoint_statistics(vectors[(key,reference,metric)],vectors[(key,"pfad",metric)],
                    direction=M.POINTSET_METRIC_DIRECTIONS[metric],bootstrap_seed=20260908,bootstrap_resamples=100000)
                robustness_tests[name].update({"reference_method":reference,"candidate_method":"pfad_replay","positive_gain_favors":"pfad"})
    if len(robustness_tests)!=84:raise RuntimeError("wrong multiplicity family")
    M.add_holm_adjustment(robustness_tests)
    replay=[json.loads((training_root / "folds" / s / "pfad_replay.json").read_text()) for s in SPECIMENS]
    folds=[json.loads((training_root / "folds" / s / "comparison_fold.json").read_text()) for s in SPECIMENS]
    quality={"cases":len(cases),"condition_replicate_rows":sum(len(c["conditions"]) for c in cases.values()),
       "legacy_point_metric_max_abs_difference":max(max(c["qa"]["original_pfad_metric_abs_differences"].values()) for c in cases.values()),
       "legacy_raw_metric_max_abs_difference":max(max(c["qa"]["raw_metric_abs_differences"].values()) for c in cases.values()),
       "legacy_mesh_metric_max_abs_difference":max(max(c["qa"]["original_surface_metric_abs_differences"].values()) for c in cases.values()),
       "min_replay_selection_agreement":min(r["selection_agreement"] for fold in replay for r in fold["cases"]),
       "max_replay_probability_difference":max(r["probability_max_abs_difference"] for fold in replay for r in fold["cases"]),
       "outer_fits":len(folds),"inner_fits":sum(len(f["inner_fits"]) for f in folds),
       "pfad_parameters":sum(p.numel() for p in M.AttentiveSweepSelector().parameters()),
       "set_transformer_parameters":folds[0]["outer_fit"]["parameters"]}
    destination=output / "tables"
    destination.mkdir(exist_ok=True)
    with (destination / "specimen_condition_metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["condition","method","specimen",*METRICS])
        writer.writeheader()
        for key in summary:
            for method in METHODS:
                for i,specimen in enumerate(SPECIMENS):
                    writer.writerow({"condition":key,"method":method,"specimen":specimen,
                                     **{m:summary[key][method][m]["specimen_values"][i] for m in METRICS}})
    result={"status":"complete","config":config,"quality_checks":quality,"clean_comparator":clean,
       "clean_comparator_tests":comparator_tests,"acquisition":summary,"acquisition_tests":robustness_tests,
       "baseline_selected_policies":{f["heldout"]:f["chosen"] for f in folds},
       "source_metric_sha256":{p.name:sha256(p) for p in paths}}
    dump_json(output / "summary_complete.json",result)
    print(json.dumps({"status":"complete","quality":quality,"summary":str(output / "summary_complete.json")}),flush=True)


if __name__=="__main__":main()
