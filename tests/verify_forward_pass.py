"""
Direct ECG + Clinical Report forward pass verification.
Runs against the real trained checkpoint without HTTP layer or authentication.
Uses a synthetic (non-clinical) ECG signal generated with numpy.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.inference.ecg_report_runtime import get_ecg_report_runtime

MODEL_DIR = Path("models/ecg_report_fusion")


def main():
    runtime = get_ecg_report_runtime(MODEL_DIR, enabled=True)

    # Confirm readiness first
    readiness = runtime.assess_readiness(verify_forward_pass=False)
    print("=== ECG+Report Runtime Readiness ===")
    print(f"  manifest_present      = {readiness.manifest_present}")
    print(f"  checkpoint_present    = {readiness.checkpoint_present}")
    print(f"  checkpoint_trained    = {readiness.checkpoint_trained}")
    print(f"  normalization_present = {readiness.normalization_present}")
    print(f"  text_model_loaded     = {readiness.text_model_loaded}")
    print(f"  model_loaded          = {readiness.model_loaded}")

    # Synthetic ECG: mild sinusoidal signal (12 leads, 1000 samples @ 100 Hz)
    rng = np.random.default_rng(42)
    signal = rng.normal(0.0, 0.3, size=(12, 1000)).astype(np.float32)
    t = np.linspace(0, 10, 1000, dtype=np.float32)
    for lead_i in range(12):
        signal[lead_i] += 0.1 * np.sin(2 * np.pi * 1.0 * t)

    report = (
        "SYNTHETIC TEST FIXTURE. Normal sinus rhythm. "
        "No ischemic changes. Research verification only. Not clinical data."
    )

    print("\n=== Running Real Forward Pass ===")
    result = runtime.run_inference(signal, report)

    print(f"  probability         = {result['probability']:.6f}")
    print(f"  predicted_class     = {result['predicted_class_index']} ({result['predicted_class_label']})")
    print(f"  architecture_id     = {result['architecture_id']}")
    print(f"  model_version_tag   = {result['model_version_tag']}")
    print(f"  is_research_only    = {result['is_research_only']}")
    print(f"  disclaimer          = {result['disclaimer'][:80]}...")

    # Assertions
    assert 0.0 <= result["probability"] <= 1.0, f"Probability out of range: {result['probability']}"
    assert result["predicted_class_index"] in (0, 1), f"Bad class index: {result['predicted_class_index']}"
    assert result["predicted_class_label"] in ("NORM", "MI/STTC"), f"Bad label: {result['predicted_class_label']}"
    assert result["is_research_only"] is True, "Must be research only"
    assert result["architecture_id"] == "medshield.ecg_report_fusion.v1", "Architecture ID mismatch"
    assert "EXPERIMENTAL" in result["disclaimer"], "Disclaimer missing EXPERIMENTAL marker"
    assert result["probability"] == result["probability"], "NaN probability detected"

    print("\n[PASS] All assertions passed. Real model forward pass VERIFIED.")
    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)
