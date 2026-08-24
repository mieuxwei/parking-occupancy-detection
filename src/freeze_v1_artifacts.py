"""Create a SHA-256 registry for completed V1 artifacts without copying them."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluate_cross_domain import file_sha256


V1_ARTIFACTS = {
    "checkpoint": [
        "models/simple_cnn_baseline.pt",
        "models/resnet18_transfer.pt",
        "models/resnet18_pklot_finetuned.pt",
    ],
    "result": [
        "results/simple_cnn_baseline.json",
        "results/resnet18_transfer.json",
        "results/cross_domain_evaluation.json",
        "results/pklot_finetuning.json",
        "results/error_analysis.json",
    ],
    "local_manifest": [
        "data/processed/pklot_adaptation_manifest.csv.gz",
        "data/processed/pklot_error_manifest.csv.gz",
    ],
    "documentation": [
        "docs/DATASET_RESEARCH.md",
        "docs/DATA_SPLIT.md",
        "docs/DATA_STORAGE.md",
        "docs/EDA_REPORT.md",
        "docs/SIMPLE_CNN_BASELINE.md",
        "docs/RESNET18_TRANSFER.md",
        "docs/PKLOT_PREPARATION.md",
        "docs/CROSS_DOMAIN_EVALUATION.md",
        "docs/PKLOT_ADAPTATION_SPLIT.md",
        "docs/PKLOT_FINETUNING.md",
        "docs/ERROR_ANALYSIS.md",
        "docs/DEMO.md",
    ],
    "demo_application": [
        "app/app.py",
        "src/inference.py",
        ".streamlit/config.toml",
        "requirements.txt",
    ],
    "demo_artifact": [
        "images/demo_initial.png",
        "images/demo_inference_result.png",
        "images/demo_sample_occupied.jpg",
        "images/parking_occupancy_demo.gif",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        registry = json.loads(args.output.read_text(encoding="utf-8"))
        mismatches = []
        for record in registry.get("artifacts", []):
            path = root / record["path"]
            if (
                not path.is_file()
                or path.stat().st_size != record["size_bytes"]
                or file_sha256(path) != record["sha256"]
            ):
                mismatches.append(record["path"])
        if mismatches:
            raise ValueError(f"Frozen V1 artifact mismatch: {mismatches}")
        print(json.dumps({"status": "verified", "artifacts": len(registry["artifacts"])}, indent=2))
        return
    records = []
    for artifact_type, relative_paths in V1_ARTIFACTS.items():
        for relative_path in relative_paths:
            path = root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Required V1 artifact is missing: {relative_path}")
            records.append(
                {
                    "path": relative_path,
                    "type": artifact_type,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    registry = {
        "status": "v1_frozen_read_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "overwrite_existing_v1_artifacts": False,
            "reinterpret_completed_v1_metrics": False,
            "use_v1_error_manifest_for_v2_training_or_selection": False,
            "streamlit_v1_changed_for_v2_protocol": False,
        },
        "artifacts": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
