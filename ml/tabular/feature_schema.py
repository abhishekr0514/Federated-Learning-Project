"""
Versioned tabular feature schema for MedShield-FL.

Source of truth for the 10 features: docs/ARCHITECTURE.md (Structured Tabular Modality)
lists age, sex, systolic BP, diastolic BP, cholesterol, glucose, BMI, smoking, alcohol,
physical activity. Mapped to cardio_train.csv columns for the standalone tabular experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from ml.tabular.errors import TabularRowValidationError, TabularSchemaError

SCHEMA_ID = "medshield.tabular.cardio_survey"
SCHEMA_VERSION = "1.0.0"

EXCLUDED_FROM_FEATURES = frozenset({"id", "cardio"})


class FeatureKind(str, Enum):
    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    BINARY = "binary"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source_column: str
    kind: FeatureKind
    description: str
    valid_range: tuple[float, float] | None = None
    ordinal_allowed: frozenset[int] | None = None
    missing_policy: str = "reject"


# Fixed order — index 0..9 → model input column 0..9
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "age_years",
        "age",
        FeatureKind.CONTINUOUS,
        "Age in years (converted from source age in days / 365.25).",
        valid_range=(18.0, 100.0),
    ),
    FeatureSpec(
        "gender_female",
        "gender",
        FeatureKind.BINARY,
        "1 if female (source gender code 1), else 0 (source code 2 = male).",
        ordinal_allowed=frozenset({1, 2}),
    ),
    FeatureSpec(
        "ap_hi",
        "ap_hi",
        FeatureKind.CONTINUOUS,
        "Systolic blood pressure (mmHg).",
        valid_range=(80.0, 250.0),
    ),
    FeatureSpec(
        "ap_lo",
        "ap_lo",
        FeatureKind.CONTINUOUS,
        "Diastolic blood pressure (mmHg).",
        valid_range=(40.0, 150.0),
    ),
    FeatureSpec(
        "cholesterol_level",
        "cholesterol",
        FeatureKind.ORDINAL,
        "Survey ordinal 1=normal, 2=above normal, 3=well above normal.",
        ordinal_allowed=frozenset({1, 2, 3}),
    ),
    FeatureSpec(
        "glucose_level",
        "gluc",
        FeatureKind.ORDINAL,
        "Survey ordinal 1=normal, 2=above normal, 3=well above normal.",
        ordinal_allowed=frozenset({1, 2, 3}),
    ),
    FeatureSpec(
        "bmi",
        "_derived_bmi",
        FeatureKind.CONTINUOUS,
        "BMI kg/m² from height (cm) and weight (kg).",
        valid_range=(10.0, 60.0),
    ),
    FeatureSpec(
        "smoke",
        "smoke",
        FeatureKind.BINARY,
        "Smoking indicator (0/1).",
        ordinal_allowed=frozenset({0, 1}),
    ),
    FeatureSpec(
        "alco",
        "alco",
        FeatureKind.BINARY,
        "Alcohol intake indicator (0/1).",
        ordinal_allowed=frozenset({0, 1}),
    ),
    FeatureSpec(
        "active",
        "active",
        FeatureKind.BINARY,
        "Physical activity indicator (0/1).",
        ordinal_allowed=frozenset({0, 1}),
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)
N_TABULAR_FEATURES = len(FEATURE_SPECS)


def validate_feature_name_list(names: Sequence[str], *, schema_version: str = SCHEMA_VERSION) -> None:
    if schema_version != SCHEMA_VERSION:
        raise TabularSchemaError(f"Unsupported schema version: {schema_version}")
    if len(names) != N_TABULAR_FEATURES:
        raise TabularSchemaError(f"Expected {N_TABULAR_FEATURES} features, got {len(names)}.")
    if len(set(names)) != len(names):
        raise TabularSchemaError("Duplicate feature names are not allowed.")
    if list(names) != list(FEATURE_NAMES):
        raise TabularSchemaError("Feature names or order do not match the registered schema.")


def validate_raw_record_columns(columns: Sequence[str]) -> None:
    required = {s.source_column for s in FEATURE_SPECS if not s.source_column.startswith("_")} | {"id", "cardio"}
    required.add("height")
    required.add("weight")
    col_set = set(columns)
    missing = required - col_set
    if missing:
        raise TabularSchemaError(f"Missing required source columns: {sorted(missing)}")
    forbidden_in_features = col_set & EXCLUDED_FROM_FEATURES
    if forbidden_in_features and any(f in FEATURE_NAMES for f in forbidden_in_features):
        raise TabularSchemaError("Target or ID columns must not appear as model features.")


def extract_feature_vector_from_row(row: dict[str, Any]) -> dict[str, float]:
    """Map a cardio_train.csv row to named features (pre-normalization)."""
    validate_raw_record_columns(row.keys())
    age_days = float(row["age"])
    age_years = age_days / 365.25
    gender = int(float(row["gender"]))
    height_cm = float(row["height"])
    weight_kg = float(row["weight"])
    bmi = weight_kg / ((height_cm / 100.0) ** 2)

    values: dict[str, float] = {
        "age_years": age_years,
        "gender_female": 1.0 if gender == 1 else 0.0,
        "ap_hi": float(row["ap_hi"]),
        "ap_lo": float(row["ap_lo"]),
        "cholesterol_level": float(row["cholesterol"]),
        "glucose_level": float(row["gluc"]),
        "bmi": bmi,
        "smoke": float(row["smoke"]),
        "alco": float(row["alco"]),
        "active": float(row["active"]),
    }

    for spec in FEATURE_SPECS:
        val = values[spec.name]
        if spec.kind == FeatureKind.ORDINAL and int(val) not in (spec.ordinal_allowed or set()):
            raise TabularRowValidationError(f"Invalid ordinal for {spec.name}: {val}")
        if spec.kind == FeatureKind.BINARY and int(val) not in (spec.ordinal_allowed or set()):
            raise TabularRowValidationError(f"Invalid value for {spec.name}: {val}")

    if values["ap_lo"] > values["ap_hi"]:
        raise TabularRowValidationError("ap_lo greater than ap_hi (invalid blood pressure pair).")

    return values


def validate_feature_dict(values: dict[str, float]) -> None:
    keys = set(values.keys())
    expected = set(FEATURE_NAMES)
    if keys != expected:
        missing = expected - keys
        extra = keys - expected
        raise TabularSchemaError(
            f"Feature dict mismatch. Missing: {sorted(missing)}. Unexpected: {sorted(extra)}."
        )


def vectorize_ordered(values: dict[str, float]) -> list[float]:
    validate_feature_dict(values)
    return [float(values[name]) for name in FEATURE_NAMES]
