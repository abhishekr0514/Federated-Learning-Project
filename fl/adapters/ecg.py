"""ECG federation adapter interface (PTB-XL proxy task)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fl.errors import ClientLocalDataError, FederationConfigError
from fl.registry import ECG_PTbxl_V1

DEFAULT_PTbxl_ROOT = Path("data/raw/ptb-xl")


def ecg_federation_readiness(ptbxl_root: Path | str = DEFAULT_PTbxl_ROOT) -> dict[str, Any]:
    root = Path(ptbxl_root)
    issues: list[str] = []
    if not root.is_dir():
        issues.append(f"PTB-XL root not present: {root}")
    try:
        import torch  # noqa: F401
        from ml.models.ecg_encoder import ECGEncoder  # noqa: F401
    except ImportError as exc:
        issues.append(f"ECG model dependencies unavailable: {exc}")
    return {
        "federation_id": ECG_PTbxl_V1.federation_id,
        "ready_for_task_experiment": not issues,
        "issues": issues,
        "note": "Interface boundary only; federated ECG training not started in Prompt 20.",
    }


class ECGFederationAdapterStub:
    """Placeholder type documenting future ECGEncoder-based client adapter."""

    federation_id = ECG_PTbxl_V1.federation_id

    def __init__(self, ptbxl_root: Path | str = DEFAULT_PTbxl_ROOT) -> None:
        self.ptbxl_root = Path(ptbxl_root)
        if not self.ptbxl_root.is_dir():
            raise ClientLocalDataError(f"PTB-XL root missing: {self.ptbxl_root}")

    def fit(self, config: dict[str, Any]) -> None:
        raise FederationConfigError("ECG federated training deferred in Prompt 20.")
