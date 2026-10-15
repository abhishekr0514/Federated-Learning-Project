"""
Training-only tabular preprocessing (fit on train split by record id).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ml.tabular.errors import TabularSchemaError, TabularTransformError
from ml.tabular.feature_schema import (
    FEATURE_NAMES,
    N_TABULAR_FEATURES,
    SCHEMA_ID,
    SCHEMA_VERSION,
    vectorize_ordered,
)

SCALE_FEATURE_NAMES = frozenset(
    {"age_years", "ap_hi", "ap_lo", "bmi", "cholesterol_level", "glucose_level"}
)
SCALE_INDICES = [i for i, n in enumerate(FEATURE_NAMES) if n in SCALE_FEATURE_NAMES]
PASS_INDICES = [i for i, n in enumerate(FEATURE_NAMES) if n not in SCALE_FEATURE_NAMES]


@dataclass
class OutlierPolicy:
    """
    Outlier handling (does not silently delete rows unless reject_invalid_bp=True).

    Audit reference (cardio_train.csv): ap_lo>ap_hi in 1234 rows; extreme BP/anthropometrics
    documented in docs/data-audit/DATASET_AUDIT_REPORT.md.
    """

    reject_invalid_bp: bool = True
    flag_out_of_range: bool = True
    clip_to_train_quantiles: bool = False
    quantile_low: float = 0.001
    quantile_high: float = 0.999


@dataclass
class TabularPreprocessorState:
    schema_id: str = SCHEMA_ID
    schema_version: str = SCHEMA_VERSION
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    mean: list[float] = field(default_factory=lambda: [0.0] * N_TABULAR_FEATURES)
    std: list[float] = field(default_factory=lambda: [1.0] * N_TABULAR_FEATURES)
    impute_median: list[float] = field(default_factory=lambda: [0.0] * N_TABULAR_FEATURES)
    quantile_low: list[float] | None = None
    quantile_high: list[float] | None = None
    fitted: bool = False
    outlier_policy: dict[str, Any] = field(default_factory=dict)
    fit_row_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> TabularPreprocessorState:
        data = json.loads(raw)
        return cls(**data)


class TabularPreprocessor:
    def __init__(self, policy: OutlierPolicy | None = None) -> None:
        self.policy = policy or OutlierPolicy()
        self.state = TabularPreprocessorState(outlier_policy=asdict(self.policy))

    @property
    def fitted(self) -> bool:
        return self.state.fitted

    def fit(self, feature_rows: Iterable[dict[str, float]]) -> TabularPreprocessorState:
        rows = [vectorize_ordered(r) for r in feature_rows]
        if not rows:
            raise TabularSchemaError("Cannot fit preprocessor on empty training set.")
        arr = np.asarray(rows, dtype=np.float64)
        mean = np.zeros(N_TABULAR_FEATURES, dtype=np.float64)
        std = np.ones(N_TABULAR_FEATURES, dtype=np.float64)
        imputed = arr.copy()
        for j in SCALE_INDICES:
            col = arr[:, j]
            med = float(np.nanmedian(col))
            mean[j] = med
            bad = ~np.isfinite(col)
            if bad.any():
                imputed[bad, j] = med
            s = float(imputed[:, j].std())
            std[j] = s if s >= 1e-8 else 1.0
        self.state.mean = mean.tolist()
        self.state.std = std.tolist()
        self.state.impute_median = mean.tolist()
        if self.policy.clip_to_train_quantiles:
            self.state.quantile_low = np.quantile(imputed[:, SCALE_INDICES], self.policy.quantile_low, axis=0).tolist()
            self.state.quantile_high = np.quantile(imputed[:, SCALE_INDICES], self.policy.quantile_high, axis=0).tolist()
        self.state.fitted = True
        self.state.fit_row_count = len(rows)
        return self.state

    def transform(self, feature_dict: dict[str, float], *, allow_update_stats: bool = False) -> np.ndarray:
        if not self.state.fitted:
            raise TabularSchemaError("Preprocessor must be fit on training data before transform.")
        if allow_update_stats:
            raise TabularSchemaError("Validation/test transform must not update fitted parameters.")
        vec = np.asarray(vectorize_ordered(feature_dict), dtype=np.float64)
        out = vec.copy()
        mean = np.asarray(self.state.mean)
        std = np.asarray(self.state.std)
        for j in SCALE_INDICES:
            out[j] = (vec[j] - mean[j]) / std[j]
        if self.policy.clip_to_train_quantiles and self.state.quantile_low is not None:
            lo = np.asarray(self.state.quantile_low)
            hi = np.asarray(self.state.quantile_high)
            for k, j in enumerate(SCALE_INDICES):
                out[j] = np.clip(out[j], lo[k], hi[k])
        if not np.all(np.isfinite(out)):
            raise TabularTransformError("Non-finite values after transformation.")
        return out.astype(np.float32)

    def transform_batch(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        return np.stack([self.transform(r) for r in rows], axis=0)

    def save(self, path: Path) -> None:
        path.write_text(self.state.to_json(), encoding="utf-8")

    def load(self, path: Path) -> None:
        self.state = TabularPreprocessorState.from_json(path.read_text(encoding="utf-8"))
        if self.state.schema_version != SCHEMA_VERSION:
            raise TabularSchemaError(f"State schema version mismatch: {self.state.schema_version}")
