"""Memory-conscious PTB-XL ECG dataset (lazy WFDB load)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.preprocessing.ecg_signal_quality import assess_ecg_signal
from ml.ptbxl.labels import LEAD_ORDER


def load_ecg_mv(ptbxl_root: Path, filename_lr: str) -> np.ndarray:
    """Load (12, 1000) mV array in contracted lead order."""
    import wfdb

    record_path = ptbxl_root / filename_lr
    signal, meta = wfdb.rdsamp(str(record_path))
    if signal.shape != (1000, 12):
        raise ValueError(f"Unexpected shape {signal.shape} for {filename_lr}")
    if list(meta["sig_name"]) != list(LEAD_ORDER):
        raise ValueError(f"Lead order mismatch for {filename_lr}: {meta['sig_name']}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite values in {filename_lr}")
    # WFDB returns (T, C); convert to (12, 1000)
    arr = signal.astype(np.float32).T
    quality = assess_ecg_signal(arr, lead_order=LEAD_ORDER)
    if not quality.passed:
        raise ValueError(
            f"ECG quality assessment failed for {filename_lr}: "
            + "; ".join(quality.descriptions)
        )
    return arr


class PTBXLECGDataset(Dataset):
    """Lazy dataset over pre-built index entries."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        ptbxl_root: Path,
        normalization=None,
    ) -> None:
        self.records = records
        self.ptbxl_root = Path(ptbxl_root)
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        rec = self.records[idx]
        signal = load_ecg_mv(self.ptbxl_root, rec["filename_lr"])
        if self.normalization is not None:
            signal = self.normalization.transform(signal)
        return {
            "signal": torch.from_numpy(signal),
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "ecg_id": int(rec["ecg_id"]),
        }
