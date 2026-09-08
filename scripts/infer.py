#!/usr/bin/env python3
"""Apply a released support predictor and optionally export an open surface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pfad import core
from pfad_tcsvt.models import SetTransformerSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fold", choices=[f"specimen{i:02}" for i in range(1, 15)], required=True)
    parser.add_argument("--model", choices=["pfad", "set_transformer"], default="pfad")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    torch.set_num_threads(4)
    fold = args.checkpoint_root / args.fold
    with np.load(args.features, allow_pickle=False) as payload:
        max_sources = max(len(payload["counts"]) - 1, 1)
    case = core.load_student_case(args.features, max_sources)
    if case["specimen"] in {f"specimen{i:02}" for i in range(1, 15)} and case["specimen"] != args.fold:
        raise ValueError("For the study specimens, use the model that held out this specimen.")
    device = core.choose_device(args.device)
    if args.model == "pfad":
        model = core.AttentiveSweepSelector()
        checkpoint = fold / "pfad.pt"
        policy = json.loads((fold / "pfad_policy.json").read_text())
    else:
        model = SetTransformerSelector(128, 4)
        policy = json.loads((fold / "set_transformer_policy.json").read_text())
        checkpoint = fold / "set_transformer.pt"
    if not checkpoint.exists():
        raise FileNotFoundError("Download the model release first: python scripts/fetch_release.py models")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    probability = core.predict_sweep_set_selector(model, case, batch_size=args.batch_size, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    retained = core.save_ranked_selector_prediction(
        case, probability, score_name=policy["score"], retention=policy["retention"], output=args.output,
        settings={"model": args.model, "heldout_specimen": args.fold, "checkpoint": checkpoint.name, "CT_used_at_inference": False},
    )
    if args.mesh:
        import open3d as o3d
        with np.load(args.output, allow_pickle=False) as prediction:
            keep = prediction["selection_keep"].astype(bool)
            directions = prediction["ray_directions"].astype(np.float64)
            points = prediction["ray_origins_mm"].astype(np.float64) + (prediction["ray_depths_mm"].astype(np.float64) + prediction["correction"].astype(np.float64))[:, None] * directions
            record = np.repeat(np.arange(len(prediction["counts"])), prediction["counts"])
            surface = core.reconstruct_sweep_topology_surface(
                points[keep], directions[keep], record[keep], prediction["ray_record_frame"][keep],
                prediction["ray_column"][keep], prediction["ray_confidence"][keep],
                max_frame_gap=1, max_column_gap_px=64, max_edge_mm=5.0,
            )
        args.mesh.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(args.mesh), surface):
            raise IOError(f"Could not write {args.mesh}")
    print(json.dumps({"prediction": str(args.output), "model": args.model, "fold": args.fold, "retention": retained, "CT_used_at_inference": False}, indent=2))


if __name__ == "__main__":
    main()
