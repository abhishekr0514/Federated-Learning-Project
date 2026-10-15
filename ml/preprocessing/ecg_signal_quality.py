"""ECG signal quality assessment for offline research loading (Prompt 12A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from ml.ptbxl.labels import LEAD_ORDER

EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 1000


class QualityReasonCode(str, Enum):
    EMPTY_SIGNAL = "empty_signal"
    INVALID_DIMENSIONS = "invalid_dimensions"
    WRONG_LEAD_COUNT = "wrong_lead_count"
    WRONG_SAMPLE_COUNT = "wrong_sample_count"
    NON_FINITE_VALUES = "non_finite_values"
    NOT_NUMERIC = "not_numeric"
    SATURATION_HEURISTIC = "saturation_heuristic"
    FLATLINE_HEURISTIC = "flatline_heuristic"


@dataclass(frozen=True)
class QualityThresholds:
    """
    Heuristic thresholds — not clinically validated.

    flatline_all_leads_std_below_mv: if set, fail when every lead's std (mV) is below
        this value (likely zero/dead signal). Default None disables the check.
    saturation_abs_mv: fail when any |sample| exceeds this (likely clipping/corruption).
        Default 100 mV — conservative for typical 10 s 12-lead mV traces.
    """

    flatline_all_leads_std_below_mv: float | None = None
    saturation_abs_mv: float = 100.0
    enable_saturation_check: bool = True


@dataclass
class ECGQualityReport:
    passed: bool
    reason_codes: list[QualityReasonCode] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    checks_performed: list[str] = field(default_factory=list)
    unavailable_checks: list[str] = field(default_factory=list)
    heuristic_warnings: list[str] = field(default_factory=list)

    def fail(self, code: QualityReasonCode, description: str) -> None:
        self.passed = False
        self.reason_codes.append(code)
        self.descriptions.append(description)


DEFAULT_THRESHOLDS = QualityThresholds()


def assess_ecg_signal(
    signal: np.ndarray,
    *,
    lead_order: Sequence[str] = LEAD_ORDER,
    thresholds: QualityThresholds = DEFAULT_THRESHOLDS,
) -> ECGQualityReport:
    """
    Assess a single ECG array shaped (n_leads, n_samples) in ``lead_order`` channel order.

    ``lead_order`` is recorded for traceability; channel order must match loader metadata.
    """
    report = ECGQualityReport(passed=True)
    report.checks_performed.extend(
        [
            "array_present",
            "numeric_dtype",
            "shape_2d",
            f"lead_count_{EXPECTED_LEADS}",
            f"samples_per_lead_{EXPECTED_SAMPLES}",
            "finite_values",
        ]
    )
    report.unavailable_checks.append(
        "lead_name_verification_per_channel (requires WFDB metadata; done in load_ecg_mv)"
    )

    if lead_order is not LEAD_ORDER and tuple(lead_order) != tuple(LEAD_ORDER):
        report.heuristic_warnings.append("Non-default lead_order tuple supplied for assessment.")

    if signal is None:
        report.fail(QualityReasonCode.EMPTY_SIGNAL, "Signal is None.")
        return report

    if not isinstance(signal, np.ndarray):
        report.fail(QualityReasonCode.NOT_NUMERIC, "Signal is not a numpy.ndarray.")
        return report

    if signal.size == 0:
        report.fail(QualityReasonCode.EMPTY_SIGNAL, "Signal array is empty.")
        return report

    if not np.issubdtype(signal.dtype, np.number):
        report.fail(QualityReasonCode.NOT_NUMERIC, f"Non-numeric dtype: {signal.dtype}.")
        return report

    if signal.ndim != 2:
        report.fail(
            QualityReasonCode.INVALID_DIMENSIONS,
            f"Expected 2D array (leads, samples); got ndim={signal.ndim}.",
        )
        return report

    n_leads, n_samples = signal.shape
    if n_leads != EXPECTED_LEADS:
        report.fail(
            QualityReasonCode.WRONG_LEAD_COUNT,
            f"Expected {EXPECTED_LEADS} leads; got {n_leads}.",
        )
    if n_leads != len(lead_order):
        report.fail(
            QualityReasonCode.WRONG_LEAD_COUNT,
            f"Lead count {n_leads} does not match lead_order length {len(lead_order)}.",
        )
    if n_samples != EXPECTED_SAMPLES:
        report.fail(
            QualityReasonCode.WRONG_SAMPLE_COUNT,
            f"Expected {EXPECTED_SAMPLES} samples per lead; got {n_samples}.",
        )

    if not np.isfinite(signal).all():
        report.fail(QualityReasonCode.NON_FINITE_VALUES, "Signal contains NaN or infinite values.")
        return report

    if thresholds.enable_saturation_check and thresholds.saturation_abs_mv is not None:
        report.checks_performed.append(f"saturation_heuristic_abs_mv_{thresholds.saturation_abs_mv}")
        peak = float(np.max(np.abs(signal)))
        if peak > thresholds.saturation_abs_mv:
            report.fail(
                QualityReasonCode.SATURATION_HEURISTIC,
                f"Peak |amplitude| {peak:.4f} mV exceeds heuristic limit "
                f"{thresholds.saturation_abs_mv} mV (possible clipping; not a clinical rule).",
            )

    if thresholds.flatline_all_leads_std_below_mv is not None:
        report.checks_performed.append(
            f"flatline_heuristic_all_leads_std_below_{thresholds.flatline_all_leads_std_below_mv}"
        )
        per_lead_std = signal.std(axis=1)
        if np.all(per_lead_std < thresholds.flatline_all_leads_std_below_mv):
            report.fail(
                QualityReasonCode.FLATLINE_HEURISTIC,
                "All leads have std below flatline heuristic threshold (possible dead signal).",
            )

    return report
