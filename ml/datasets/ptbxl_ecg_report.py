"""Lazy PTB-XL ECG + report text dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ml.datasets.ptbxl_ecg import load_ecg_mv


class PTBXLECGReportDataset(Dataset):
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        signal = load_ecg_mv(self.ptbxl_root, rec["filename_lr"])
        if self.normalization is not None:
            signal = self.normalization.transform(signal)
        return {
            "signal": torch.from_numpy(signal),
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "report": rec["report"],
            "ecg_id": int(rec["ecg_id"]),
            "patient_id": str(rec["patient_id"]),
        }
