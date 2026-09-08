#!/usr/bin/env python3
"""Print the learned comparison from the complete stored study summary."""
import json
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "paper_results/acquisition_summary.json"
    data = json.loads(path.read_text())
    if data["status"] != "complete" or data["quality_checks"]["cases"] != 42:
        raise RuntimeError("The complete cohort summary is required.")
    print("Means across 14 specimens after equal anatomy averaging")
    print("representation,method,chamfer_l1_mm,hd95_mm,fscore_1mm,normal_consistency")
    for representation in ["points", "meshes"]:
        for method in ["pfad", "set_transformer"]:
            row = data["clean_comparator"][representation][method]
            numbers = [row[m]["mean"] for m in ["chamfer_l1_mm", "hd95_mm", "fscore_1mm", "normal_consistency"]]
            print(",".join([representation, method] + [f"{v:.6f}" for v in numbers]))


if __name__ == "__main__":
    main()
