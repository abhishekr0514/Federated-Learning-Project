"""Readiness gate for PTB-XL in-cohort ECG + report two-modality GAT training/inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PTbxl = REPO_ROOT / "data" / "raw" / "ptb-xl"
DEFAULT_ENCODER_MANIFEST = REPO_ROOT / "models" / "ecg_report_fusion" / "manifest.json"


@dataclass
class ECGReportGATGateResult:
    allowed: bool
    verdict: str
    blockers: list[str]
    paired_record_count: int = 0
    unique_patients: int = 0

    def required_to_unblock(self) -> list[str]:
        return list(self.blockers)


def evaluate_ecg_report_gat_gate(
    ptbxl_root: Path | None = None,
    encoder_manifest: Path | None = None,
) -> ECGReportGATGateResult:
    blockers: list[str] = []
    root = ptbxl_root or DEFAULT_PTbxl
    manifest = encoder_manifest or DEFAULT_ENCODER_MANIFEST

    if not root.is_dir():
        blockers.append("PTB-XL root missing for ECG/report pairing")
        return ECGReportGATGateResult(False, "BLOCKED", blockers)

    try:
        from ml.ptbxl.ecg_report_index import build_ecg_report_index

        records, stats = build_ecg_report_index(root)
        n = stats.get("included_with_report", len(records))
        patients = len({r["patient_id"] for r in records})
        if n < 100:
            blockers.append("Insufficient paired ECG/report records")
    except Exception as exc:
        blockers.append(f"Failed to build ECG/report index: {type(exc).__name__}")
        n, patients = 0, 0

    if not manifest.is_file():
        blockers.append("ECG-report encoder manifest missing")
    else:
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
        ckpt_rel = data.get("checkpoint_path")
        if not ckpt_rel:
            blockers.append("Encoder manifest missing checkpoint_path")
        else:
            ckpt = REPO_ROOT / ckpt_rel.replace("/", "\\")
            if not ckpt.is_file():
                blockers.append("ECG-report encoder checkpoint file missing")
            if not data.get("trained"):
                blockers.append("ECG-report encoder not marked trained")

    allowed = len(blockers) == 0
    return ECGReportGATGateResult(
        allowed=allowed,
        verdict="ALLOWED" if allowed else "BLOCKED",
        blockers=blockers,
        paired_record_count=n,
        unique_patients=patients,
    )
