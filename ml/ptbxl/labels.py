"""PTB-XL binary label mapping per DATA_CONTRACT and protocol v1."""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

PROTOCOL_VERSION = "1.0.0"

LEAD_ORDER = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


class ExclusionReason(str, Enum):
    EMPTY_SUPERCLASS = "empty_superclass"
    NORM_PATHOLOGY_CONFLICT = "norm_pathology_conflict"
    NON_BINARY_SUPERCLASS = "non_binary_superclass"


@dataclass(frozen=True)
class LabelDecision:
    included: bool
    label: int | None = None
    exclusion: ExclusionReason | None = None
    superclasses: frozenset[str] | None = None


def load_diagnostic_class_map(scp_statements_path: Path) -> dict[str, str]:
    df = pd.read_csv(scp_statements_path, index_col=0)
    diag = df[df["diagnostic"] == 1.0]
    return {str(k): str(v) for k, v in diag["diagnostic_class"].items()}


def superclasses_from_scp_codes(scp_codes: str, class_map: dict[str, str]) -> frozenset[str]:
    try:
        parsed = ast.literal_eval(scp_codes)
    except (SyntaxError, ValueError):
        return frozenset()
    if not isinstance(parsed, dict):
        return frozenset()
    out: set[str] = set()
    for key in parsed:
        if key in class_map:
            out.add(class_map[key])
    return frozenset(out)


def assign_binary_label(superclasses: frozenset[str]) -> LabelDecision:
    if not superclasses:
        return LabelDecision(False, exclusion=ExclusionReason.EMPTY_SUPERCLASS, superclasses=superclasses)
    has_norm = "NORM" in superclasses
    has_path = "MI" in superclasses or "STTC" in superclasses
    if has_norm and has_path:
        return LabelDecision(
            False,
            exclusion=ExclusionReason.NORM_PATHOLOGY_CONFLICT,
            superclasses=superclasses,
        )
    if superclasses == frozenset({"NORM"}):
        return LabelDecision(True, label=0, superclasses=superclasses)
    if has_path and not has_norm:
        return LabelDecision(True, label=1, superclasses=superclasses)
    return LabelDecision(
        False,
        exclusion=ExclusionReason.NON_BINARY_SUPERCLASS,
        superclasses=superclasses,
    )


def build_labeled_index(
    ptbxl_root: Path,
    *,
    database_csv: str = "ptbxl_database.csv",
    scp_statements_csv: str = "scp_statements.csv",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return included records and exclusion statistics (no waveforms loaded)."""
    class_map = load_diagnostic_class_map(ptbxl_root / scp_statements_csv)
    included: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "total_metadata_rows": 0,
        "included": 0,
        "excluded": 0,
        "exclusion_counts": {r.value: 0 for r in ExclusionReason},
        "class_counts": {"0": 0, "1": 0},
        "patient_counts": {"0": set(), "1": set()},
    }

    with open(ptbxl_root / database_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stats["total_metadata_rows"] += 1
            supers = superclasses_from_scp_codes(row["scp_codes"], class_map)
            decision = assign_binary_label(supers)
            if not decision.included:
                stats["excluded"] += 1
                assert decision.exclusion is not None
                stats["exclusion_counts"][decision.exclusion.value] += 1
                continue
            assert decision.label is not None
            stats["included"] += 1
            lbl = str(decision.label)
            stats["class_counts"][lbl] += 1
            stats["patient_counts"][lbl].add(row["patient_id"])
            included.append(
                {
                    "ecg_id": int(row["ecg_id"]),
                    "patient_id": row["patient_id"],
                    "strat_fold": int(float(row["strat_fold"])),
                    "filename_lr": row["filename_lr"],
                    "label": decision.label,
                    "superclasses": sorted(decision.superclasses or ()),
                }
            )

    stats["unique_patients_included"] = len(
        stats["patient_counts"]["0"] | stats["patient_counts"]["1"]
    )
    stats["patient_counts"] = {
        "0": len(stats["patient_counts"]["0"]),
        "1": len(stats["patient_counts"]["1"]),
    }
    return included, stats
