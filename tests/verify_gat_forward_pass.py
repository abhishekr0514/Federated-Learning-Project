"""Direct two-modality GAT forward pass (128-d embeddings, no HTTP)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.inference.fusion_runtime import get_fusion_inference_runtime  # noqa: E402


def main() -> None:
    model_dir = REPO / "models" / "gat_ecg_report"
    runtime = get_fusion_inference_runtime(model_dir, inference_enabled=True)
    readiness = runtime.assess_readiness()
    print("=== GAT Fusion Runtime Readiness ===")
    for k in (
        "manifest_present",
        "checkpoint_present",
        "checkpoint_trained",
        "model_loaded",
        "ready",
    ):
        print(f"  {k:22} = {getattr(readiness, k)}")
    assert readiness.manifest_present and readiness.checkpoint_present
    assert readiness.checkpoint_trained is True

    ecg = [0.01 * (i % 17 - 8) for i in range(128)]
    text = [0.02 * (i % 13 - 6) for i in range(128)]
    result = runtime.run_proxy_inference(ecg=ecg, clinical_text=text)
    print("\n=== GAT Forward Pass ===")
    print(f"  confidence          = {result['confidence_uncalibrated']:.6f}")
    print(f"  predicted_class     = {result['predicted_class_index']} ({result['predicted_class_label']})")
    print(f"  architecture_id     = {result['architecture_id']}")

    assert 0.0 <= result["confidence_uncalibrated"] <= 1.0
    assert result["predicted_class_index"] in (0, 1)
    assert result["architecture_id"] == "medshield.diagnostic_net.gat_ecg_report"
    print("\n[PASS] GAT forward pass verified.")


if __name__ == "__main__":
    main()
