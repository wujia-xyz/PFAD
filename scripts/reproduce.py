#!/usr/bin/env python3
"""Prepare inputs, reproduce the nested study, and evaluate frozen predictions."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/first_arrival_refinement/rebuild_v2"
SPECIMENS = [f"specimen{i:02}" for i in range(1, 15)]
ANATOMIES = ["foot", "tibia", "fibula"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "comparator", "sensitivity", "registration", "plot"])
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("PFAD_DATA_ROOT", ROOT / "data/UltraBones100k")))
    parser.add_argument("--specimen", choices=SPECIMENS, action="append")
    parser.add_argument("--anatomy", choices=ANATOMIES, action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.specimen or args.anatomy) and args.stage not in {"prepare", "evaluate"}:
        parser.error("Case selection is supported by prepare and evaluate; other stages use the complete study protocol.")
    specimens, anatomies = args.specimen or SPECIMENS, args.anatomy or ANATOMIES
    dataset = args.dataset_root.resolve()
    env = os.environ.copy()
    env["PFAD_DATA_ROOT"] = str(dataset)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    core = [sys.executable, str(ROOT / "scripts/run_first_arrival_refinement.py")]

    def run(command):
        print(shlex.join([str(v) for v in command]), flush=True)
        if not args.dry_run:
            subprocess.run([str(v) for v in command], cwd=ROOT, env=env, check=True)

    def script(path):
        run([sys.executable, ROOT / path])

    if args.stage == "prepare":
        for specimen in specimens:
            for anatomy in anatomies:
                name = f"{specimen}_{anatomy}.npz"
                feature, label = RESULT / "features" / name, RESULT / "teacher_labels" / name
                if not feature.exists():
                    run(core + ["anchor", "--dataset-root", dataset, "--specimen", specimen, "--anatomy", anatomy, "--output", feature])
                if not label.exists():
                    run(core + ["privileged-label", "--dataset-root", dataset, "--input-archive", feature, "--output", label])
    elif args.stage == "train":
        if args.specimen or args.anatomy:
            parser.error("The paper training protocol uses all 14 specimens and all three anatomies.")
        outer = RESULT / "selector_attentive_loocv"
        common = ["--feature-root", RESULT / "features", "--label-root", RESULT / "teacher_labels", "--student-variant", "attentive_set", "--teacher-target", "soft", "--epochs", "12", "--seed", "17", "--device", args.device]
        split = [v for specimen in SPECIMENS for v in ["--specimen", specimen]]
        run(core + ["selector-ablation-loocv"] + common + split + ["--output-dir", outer, "--protocol-output", outer / "loocv_protocol.json"])
        run(core + ["select-nested-policy"] + common + ["--input-protocol", outer / "loocv_protocol.json", "--output-dir", outer / "nested_policy", "--protocol-output", outer / "nested_policy/loocv_protocol.json", "--inner-folds", "3", "--retention", "0.85", "--retention", "0.90", "--retention", "0.95"])
    elif args.stage == "evaluate":
        prediction_root = RESULT / "selector_attentive_loocv/nested_policy"
        for specimen in specimens:
            for anatomy in anatomies:
                name = f"{specimen}_{anatomy}"
                prediction = prediction_root / f"{name}.npz"
                if not args.dry_run and not prediction.exists():
                    raise FileNotFoundError("Train PFAD or download the original prediction release before evaluation.")
                run(core + ["point-set-evaluate", "--dataset-root", dataset, "--prediction-archive", prediction, "--output", prediction_root / f"{name}.pointset.json"])
                mesh_root = RESULT / "surface_attentive_nested/sweep_topology_f1_c64_e5" / name
                run(core + ["surface-evaluate", "--dataset-root", dataset, "--prediction-archive", prediction, "--mesh-dir", mesh_root, "--output", mesh_root / "surface.json", "--surface-method", "sweep_topology", "--sample-points", "50000"])
    elif args.stage == "comparator":
        run([sys.executable, ROOT / "scripts/comparators/pfad_tcsvt/run_learning_comparator.py", "--device", args.device])
    elif args.stage == "sensitivity":
        for name in ["generate_acquisition_features", "predict_acquisition", "evaluate_geometry", "summarize"]:
            command = [sys.executable, ROOT / f"scripts/audits/pfad_tcsvt/{name}.py"]
            if name == "predict_acquisition":
                command += ["--device", args.device]
            run(command)
    elif args.stage == "registration":
        destination = ROOT / "artifacts/pfad_registration_application_20260906/protocol.json"
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(ROOT / "configs/pfad/registration/base_v1.json", destination)
        script("scripts/audits/pfad_registration_application/run.py")
        script("scripts/pilots/pfad_registration/run_convergence.py")
        script("scripts/pilots/pfad_registration/run_backends.py")
        run([sys.executable, ROOT / "scripts/audits/pfad_registration_application/summarize_extensions.py", "--study", "convergence"])
    else:
        script("scripts/plot_acquisition.py")


if __name__ == "__main__":
    main()
