"""
Research-only ECG quality gate (Prompt 12A).

Not wired to production API routes. Does not enable model_inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

from ml.preprocessing.ecg_signal_quality import ECGQualityReport

MSG_QUALITY_INSUFFICIENT = "ECG quality insufficient"
MSG_REQUIRES_REVIEW = "Prediction requires review/retry"


@dataclass(frozen=True)
class ECGQualityGateOutcome:
    quality_passed: bool
    probability: float | None
    user_messages: tuple[str, ...] = field(default_factory=tuple)
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    diagnosis_eligible: bool = False


def _validate_probability(probability: float) -> bool:
    if probability is None:
        return False
    if not math.isfinite(probability):
        return False
    return 0.0 <= probability <= 1.0


def apply_ecg_quality_gate(
    quality: ECGQualityReport,
    research_probability: float | None = None,
) -> ECGQualityGateOutcome:
    """
    Fail-safe gate for research workflows.

    On quality failure: probability is None; no positive/negative diagnosis outcome.
    On quality success: passes through ``research_probability`` if finite and in [0, 1].
    Passing the gate does not imply clinical validity or calibration.
    """
    if not quality.passed:
        return ECGQualityGateOutcome(
            quality_passed=False,
            probability=None,
            user_messages=(MSG_QUALITY_INSUFFICIENT, MSG_REQUIRES_REVIEW),
            reason_codes=tuple(c.value for c in quality.reason_codes),
            diagnosis_eligible=False,
        )

    if research_probability is None:
        return ECGQualityGateOutcome(
            quality_passed=True,
            probability=None,
            user_messages=(),
            reason_codes=(),
            diagnosis_eligible=False,
        )

    if not _validate_probability(research_probability):
        return ECGQualityGateOutcome(
            quality_passed=True,
            probability=None,
            user_messages=(MSG_QUALITY_INSUFFICIENT, MSG_REQUIRES_REVIEW),
            reason_codes=("invalid_research_probability",),
            diagnosis_eligible=False,
        )

    return ECGQualityGateOutcome(
        quality_passed=True,
        probability=float(research_probability),
        user_messages=(),
        reason_codes=(),
        diagnosis_eligible=False,
    )
