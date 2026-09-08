#!/usr/bin/env python3
"""Nested Set Transformer comparator and original-protocol PFAD model replay.

This runner never opens an outer held-out label archive. Geometry evaluation
is a separate process operating on the already frozen prediction files.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pfad_tcsvt.common import (
    ANATOMIES, RESULT, SPECIMENS, dump_json, experiment_directory,
    frozen_policies, load_pfad, sha256,
)
from pfad_tcsvt.models import SetTransformerSelector

M = load_pfad()


def train_with_checkpoints(cases, validation, config, device, destination, epochs):
    """Identical weighted BCE/AdamW schedule, with predeclared checkpoint epochs."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    source = np.concatenate([c["source"] for c in cases])
    mask = np.concatenate([c["mask"] for c in cases])
    query = np.concatenate([c["global"] for c in cases])
    soft = np.concatenate([c["soft_target"] for c in cases])
    hard = np.concatenate([c["hard_target"] for c in cases])
    total = len(source)
    case_weight = np.concatenate([
        np.full(len(c["source"]), total / (len(cases) * len(c["source"])), np.float32)
        for c in cases
    ])
    weights = np.empty_like(hard, dtype=np.float32)
    for head in range(3):
        fraction = np.clip(hard[:, head].mean(), 1e-3, 1 - 1e-3)
        weights[:, head] = np.where(hard[:, head] > .5, .5 / fraction, .5 / (1 - fraction))
    weights *= case_weight[:, None]
    weights /= weights.mean(axis=0, keepdims=True)
    tensors = [torch.from_numpy(a).to(device) for a in [source, mask, query, soft, weights]]
    del source, mask, query, soft, hard, weights, case_weight
    model = SetTransformerSelector(config["hidden"], config["heads"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history = []
    destination.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    for epoch in range(1, max(epochs) + 1):
        model.train()
        order = torch.randperm(total, device=device)
        loss_sum = 0.0
        for start in range(0, total, config["batch_size"]):
            selection = order[start:start + config["batch_size"]]
            s, m, g, target, weight = [t[selection] for t in tensors]
            logits = model(s, m, g)
            loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * len(selection)
        history.append(loss_sum / total)
        print(json.dumps({"stage": "epoch", "fit": str(destination.relative_to(ROOT)), "epoch": epoch,
                          "loss": history[-1], "seconds": round(time.monotonic() - start_time, 2)}), flush=True)
        if epoch in epochs:
            weights_path = destination / f"epoch{epoch}.pt"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights_path)
            predictions = {}
            for case in validation:
                key = case["specimen"] + "_" + case["anatomy"]
                predictions[key] = M.predict_sweep_set_selector(model, case,
                    batch_size=config["batch_size"], device=device)
                if not np.isfinite(predictions[key]).all():
                    raise RuntimeError("nonfinite probabilities")
            np.savez_compressed(destination / f"epoch{epoch}_predictions.npz", **predictions)
    report = {"status": "complete", "training_specimens": sorted({c["specimen"] for c in cases}),
              "validation_specimens": sorted({c["specimen"] for c in validation}), "training_cases": len(cases),
              "training_queries": total, "epochs": list(epochs), "loss_history": history,
              "parameters": sum(p.numel() for p in model.parameters()),
              "seconds": time.monotonic() - start_time}
    dump_json(destination / "fit.json", report)
    del model, optimizer, tensors
    gc.collect()
    torch.cuda.empty_cache()
    return report


def load_probabilities(path):
    with np.load(path) as data:
        return {tuple(k.rsplit("_", 1)): data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pfad/tcsvt/learning_comparator_v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--specimen", action="append")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    config, output = experiment_directory(args.config)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = M.choose_device(args.device)
    code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [
        Path(__file__), ROOT / "src/pfad_tcsvt/models.py", ROOT / "scripts/run_first_arrival_refinement.py"]}
    run_file = output / "run.json"
    if run_file.exists():
        previous = json.loads(run_file.read_text())
        if previous["code_sha256"] != code_hashes:
            raise RuntimeError("Runner or model changed after a run started; create a versioned protocol")
    else:
        dump_json(run_file, {"status": "running", "configuration": config, "code_sha256": code_hashes,
                            "torch": torch.__version__, "device": str(device),
                            "outer_test_labels_opened": False, "config_sha256": sha256(args.config)})
    paths = {(s, a): M.discover_case_archive(RESULT / "features", s, a)
             for s in SPECIMENS for a in ANATOMIES}
    max_sources = 5
    unlabeled = {k: M.load_student_case(p, max_sources) for k, p in paths.items()}
    if any(not np.isfinite(c["source"]).all() or not np.isfinite(c["global"]).all()
           or not c["mask"].any(axis=1).all() for c in unlabeled.values()):
        raise ValueError("invalid clean input")
    original_policies = frozen_policies()
    print(json.dumps({"output": str(output), "loaded_unlabeled_cases": len(unlabeled),
                      "parameters": sum(p.numel() for p in SetTransformerSelector(config["hidden"],config["heads"]).parameters())}), flush=True)
    for heldout in args.specimen or SPECIMENS:
        if heldout not in SPECIMENS:
            raise ValueError(heldout)
        fold_root = output / "folds" / heldout
        fold_root.mkdir(parents=True, exist_ok=True)
        train_specimens = [s for s in SPECIMENS if s != heldout]
        train = [M.load_student_case(paths[(s, a)], max_sources,
                    M.discover_case_archive(RESULT / "teacher_labels", s, a))
                 for s in train_specimens for a in ANATOMIES]
        test = [unlabeled[(heldout, a)] for a in ANATOMIES]
        replay_report = fold_root / "pfad_replay.json"
        if not replay_report.exists():
            start = time.monotonic()
            model, history = M.train_sweep_set_selector(train, epochs=12,
                batch_size=config["batch_size"], learning_rate=config["learning_rate"],
                seed=config["seed"], device=device, student_variant="attentive_set", teacher_target="soft")
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, fold_root / "pfad.pt")
            rows = []
            chosen = original_policies[heldout]
            for case in test:
                probabilities = M.predict_sweep_set_selector(model, case, batch_size=config["batch_size"], device=device)
                path = output / "predictions/pfad_replay" / f"{heldout}_{case['anatomy']}.npz"
                M.save_ranked_selector_prediction(case, probabilities, score_name=chosen["score"],
                    retention=chosen["retention"], output=path,
                    settings={"model": "pfad_replay", "training_specimens": train_specimens,
                              "seed": config["seed"], "epochs":12, "original_policy":chosen})
                with np.load(RESULT / "selector_attentive_loocv/nested_policy" / path.name) as old, np.load(path) as new:
                    old_prob = np.column_stack([old["selector_anatomy_probability"],old["selector_surface_probability"],old["selector_joint_probability"]])
                    rows.append({"anatomy":case["anatomy"],"probability_max_abs_difference":float(np.max(np.abs(old_prob-probabilities))),
                                 "probability_mean_abs_difference":float(np.mean(np.abs(old_prob-probabilities))),
                                 "selection_agreement":float(np.mean(old["selection_keep"]==new["selection_keep"]))})
            dump_json(replay_report, {"status":"complete","training_specimens":train_specimens,"policy":chosen,
                                      "loss_history":history,"seconds":time.monotonic()-start,"cases":rows})
            print(json.dumps({"stage":"pfad_replay","heldout":heldout,"seconds":time.monotonic()-start,"cases":rows}),flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if args.replay_only:
            del train
            continue
        complete_file = fold_root / "comparison_fold.json"
        if complete_file.exists():
            del train
            continue
        inner_reports = []
        for fold in range(config["inner_folds"]):
            val_specimens = {s for index,s in enumerate(train_specimens) if index % config["inner_folds"] == fold}
            inner_train = [c for c in train if c["specimen"] not in val_specimens]
            validation = [unlabeled[(s,a)] for s in train_specimens if s in val_specimens for a in ANATOMIES]
            if heldout in val_specimens or {c["specimen"] for c in inner_train} & val_specimens:
                raise RuntimeError("specimen leakage")
            destination = fold_root / f"inner{fold}"
            if not (destination / "fit.json").exists():
                report = train_with_checkpoints(inner_train, validation, config, device, destination, config["epoch_candidates"])
            else:
                report = json.loads((destination / "fit.json").read_text())
            inner_reports.append(report)
        candidates = []
        score_order = {"anatomy":0,"surface":1,"joint":2,"factor":3}
        for epoch in config["epoch_candidates"]:
            probabilities = {}
            for fold in range(config["inner_folds"]):
                probabilities.update(load_probabilities(fold_root / f"inner{fold}/epoch{epoch}_predictions.npz"))
            if set(probabilities) != {(s,a) for s in train_specimens for a in ANATOMIES}:
                raise RuntimeError("incomplete inner predictions")
            _, entries = M.choose_inner_rank_policy(train, probabilities, config["retentions"])
            candidates.extend([dict(entry, epochs=epoch) for entry in entries])
        chosen = max(candidates,key=lambda c:(c["mean_f1"],c["mean_balanced_accuracy"],c["retention"],score_order[c["score"]],-c["epochs"]))
        dump_json(fold_root / "chosen_policy.json", {"chosen":chosen,"candidates":candidates,"outer_test_labels_used":False})
        destination = fold_root / "outer"
        if not (destination / "fit.json").exists():
            outer_report = train_with_checkpoints(train, test, config, device, destination, [chosen["epochs"]])
        else:
            outer_report = json.loads((destination / "fit.json").read_text())
        test_prob = load_probabilities(destination / f"epoch{chosen['epochs']}_predictions.npz")
        outputs = []
        for case in test:
            path = output / "predictions/set_transformer" / f"{heldout}_{case['anatomy']}.npz"
            retention = M.save_ranked_selector_prediction(case,test_prob[(heldout,case["anatomy"])],
                score_name=chosen["score"],retention=chosen["retention"],output=path,
                settings={"model":"set_transformer_record_selector","training_specimens":train_specimens,
                          "chosen_policy":chosen,"upstream_commit":"73432c640ac78140496d6738416c54d32c686d65"})
            outputs.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path),**retention})
        dump_json(complete_file,{"status":"predictions_frozen","heldout":heldout,"training_specimens":train_specimens,
                                "inner_fits":inner_reports,"outer_fit":outer_report,"chosen":chosen,
                                "predictions":outputs,"outer_test_labels_opened":False})
        print(json.dumps({"stage":"fold_complete","heldout":heldout,"chosen":chosen}),flush=True)
        del train,test_prob,probabilities
        gc.collect()
    complete = all((output / "folds" / s / "comparison_fold.json").exists() for s in SPECIMENS)
    if complete:
        run = json.loads(run_file.read_text())
        run["status"] = "all_42_outer_predictions_frozen"
        dump_json(run_file,run)
    print(json.dumps({"stage":"runner_finished","all_folds_complete":complete,"output":str(output)}),flush=True)


if __name__ == "__main__":
    main()
