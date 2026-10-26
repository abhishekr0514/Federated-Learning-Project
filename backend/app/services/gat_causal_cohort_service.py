"""Build GAT-grounded research cohort from frozen embeddings + trained two-modality GAT."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

COHORT_ID = "gat_ecg_report_ptbxl_research_v1"
COHORT_VERSION = "1.0.0"

# Model-derived (descriptive / predictive only — not valid causal treatment/outcome).
DESCRIPTIVE_VARIABLES: Dict[str, str] = {
    "gat_predicted_class": "int",
    "gat_prob_class_1": "float",
    "gat_confidence": "float",
    "research_proxy_label": "int",
    "data_split": "str",
}

# Sourced from PTB-XL metadata when aligned (descriptive / documentation only — not auto-approved for causal claims).
METADATA_VARIABLES: Dict[str, str] = {
    "sex": "int",
    "age_years": "float",
    "research_proxy_label": "int",
}

COHORT_CAP_METHOD = "deterministic_prefix_after_train_alignment"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _default_embedding_cache_path() -> Optional[Path]:
    configured = (settings.CAUSAL_GAT_EMBEDDING_CACHE_PATH or "").strip()
    if configured:
        candidate = _resolve_path(configured)
        if candidate.is_file():
            return candidate
    repo = _repo_root()
    search_roots = [
        repo / "data" / "processed" / "federated" / "ecg_report_experiments" / "full_gat_s42_bs16",
        repo / "data" / "processed" / "federated" / "ecg_report_experiments" / "gat_smoke_s42",
        repo / "data" / "processed" / "gat_ecg_report" / "runs",
    ]
    found: list[Path] = []
    for root in search_roots:
        if root.is_file() and root.name.startswith("embedding_cache"):
            found.append(root)
        elif root.is_dir():
            found.extend(root.rglob("embedding_cache_train.pt"))
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def _ptbxl_root() -> Path:
    raw = (settings.CAUSAL_PTBNXL_ROOT or "").strip()
    if raw:
        return _resolve_path(raw)
    return _repo_root() / "data" / "raw" / "ptb-xl"


def _load_ptbxl_demographics(ptbxl_root: Path) -> Dict[int, Dict[str, Any]]:
    db = ptbxl_root / "ptbxl_database.csv"
    if not db.is_file():
        return {}
    import csv

    out: Dict[int, Dict[str, Any]] = {}
    with db.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ecg_id = int(row["ecg_id"])
            try:
                age = float(row["age"])
            except (KeyError, TypeError, ValueError):
                age = None
            try:
                sex_raw = row.get("sex")
                sex = int(float(sex_raw)) if sex_raw not in (None, "") else None
            except (TypeError, ValueError):
                sex = None
            out[ecg_id] = {"age_years": age, "sex": sex}
    return out


def _pseudonymous_id(ecg_id: int, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{ecg_id}".encode("utf-8")).hexdigest()
    return f"rs_{digest[:16]}"


def _align_train_records(ptbxl_root: Path, expected_rows: int) -> Tuple[List[dict], Dict[str, Any]]:
    from ml.ptbxl.ecg_report_index import build_ecg_report_index
    from ml.ptbxl.splits import SPLIT_POLICY_V1, verify_patient_disjoint
    from ml.training.train_ecg_report_fusion import partition_records

    if not ptbxl_root.is_dir():
        return [], {"error": "ptbxl_root_missing", "ptbxl_root": str(ptbxl_root)}
    paired, stats = build_ecg_report_index(ptbxl_root)
    train = partition_records(paired)["train"]
    disjoint = verify_patient_disjoint(paired)
    alignment = {
        "expected_embedding_rows": expected_rows,
        "train_records": len(train),
        "aligned": len(train) == expected_rows,
        "split_policy": SPLIT_POLICY_V1["protocol_version"],
        "cohort_split": "train",
        "train_folds": SPLIT_POLICY_V1["train_folds"],
        "test_fold_excluded": True,
        "patient_disjoint_across_splits": disjoint.get("patient_disjoint"),
        "patient_split_conflicts": disjoint.get("conflict_count"),
        "label_source": "ml.ptbxl.labels.build_labeled_index (SCP superclasses → binary proxy)",
        "label_independent_of_gat": True,
        "ptbxl_index_stats": {
            "included_with_report": stats.get("included_with_report"),
            "included_missing_report": stats.get("included_missing_report"),
            "excluded_masking_failed": stats.get("excluded_masking_failed"),
            "protocol_version": stats.get("protocol_version"),
            "target_definition": stats.get("target_definition"),
        },
    }
    if len(train) != expected_rows:
        logger.warning(
            "GAT cohort alignment mismatch: train records=%s embedding rows=%s",
            len(train),
            expected_rows,
        )
        return [], alignment
    return train, alignment


def _compute_descriptive_stats(df: pd.DataFrame, *, label_mismatch_count: int) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "n_rows": len(df),
        "label_mismatch_vs_ptbxl_records": label_mismatch_count,
    }
    if "research_proxy_label" in df.columns:
        vc = df["research_proxy_label"].value_counts(dropna=False).to_dict()
        stats["class_balance_proxy_label"] = {str(k): int(v) for k, v in vc.items()}
    for col in ("sex", "age_years"):
        if col in df.columns:
            stats[f"missing_{col}"] = int(df[col].isna().sum())
            stats[f"present_{col}"] = int(df[col].notna().sum())
    if "gat_predicted_class" in df.columns and "research_proxy_label" in df.columns:
        agree = (df["gat_predicted_class"] == df["research_proxy_label"]).mean()
        stats["gat_accuracy_vs_proxy_label"] = float(agree) if len(df) else None
    return stats


def _run_gat_on_embeddings(
    ecg_emb,
    text_emb,
    model_dir: Path,
) -> Tuple[List[int], List[float], List[float], str]:
    import torch
    import torch.nn.functional as F

    from ml.models.gat_ecg_report_fusion import GATECGReportInput, load_gat_ecg_report_checkpoint

    manifest_path = model_dir / "medshield_fusion_manifest.json"
    ckpt_name = "best_gat_checkpoint.pt"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ckpt_name = manifest.get("checkpoint_relative_path", ckpt_name)
        model_version = manifest.get("model_version_tag", "unknown")
    else:
        model_version = "unknown"

    ckpt = model_dir / ckpt_name
    if not ckpt.is_file():
        raise FileNotFoundError(f"GAT checkpoint missing at {ckpt}")

    device = torch.device("cpu")
    model = load_gat_ecg_report_checkpoint(ckpt, map_location=device).to(device)

    preds: List[int] = []
    probs: List[float] = []
    confidences: List[float] = []
    batch = 256
    n = ecg_emb.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch):
            ecg = ecg_emb[start : start + batch].to(device)
            text = text_emb[start : start + batch].to(device)
            logits = model(GATECGReportInput(ecg, text))
            prob = F.softmax(logits, dim=-1)
            p1 = prob[:, 1].cpu().numpy()
            pred = prob.argmax(dim=-1).cpu().numpy()
            conf = prob.max(dim=-1).values.cpu().numpy()
            preds.extend(int(x) for x in pred)
            probs.extend(float(x) for x in p1)
            confidences.extend(float(x) for x in conf)
    return preds, probs, confidences, model_version


@dataclass
class GATCohortBuildResult:
    ok: bool
    blockers: List[str]
    cohort_mode: str  # DESCRIPTIVE | UNAVAILABLE
    metadata_aligned: bool
    frame: Optional[pd.DataFrame]
    provenance: Dict[str, Any]
    descriptive_stats: Dict[str, Any]


class GATCausalCohortService:
    @staticmethod
    def get_gat_model_dir() -> Path:
        return _resolve_path(settings.FUSION_MODEL_DIR)

    @staticmethod
    def probe_cohort_availability() -> GATCohortBuildResult:
        """Fast path for capability registry (no embedding load or GAT inference)."""
        blockers: List[str] = []
        if not _torch_available():
            blockers.append("PyTorch not installed")
        if not GATCausalCohortService.gat_checkpoint_ready():
            blockers.append("GAT checkpoint or manifest missing under FUSION_MODEL_DIR")
        cache_path = _default_embedding_cache_path()
        if cache_path is None or not cache_path.is_file():
            blockers.append("No GAT embedding cache found (set CAUSAL_GAT_EMBEDDING_CACHE_PATH)")
        ok = len(blockers) == 0
        provenance: Dict[str, Any] = {
            "cohort_id": COHORT_ID,
            "probe_only": True,
            "embedding_cache": (
                str(cache_path.relative_to(_repo_root())).replace("\\", "/") if cache_path else None
            ),
        }
        return GATCohortBuildResult(
            ok=ok,
            blockers=blockers,
            cohort_mode="DESCRIPTIVE" if ok else "UNAVAILABLE",
            metadata_aligned=False,
            frame=None,
            provenance=provenance,
            descriptive_stats={},
        )

    @staticmethod
    def gat_checkpoint_ready() -> bool:
        model_dir = GATCausalCohortService.get_gat_model_dir()
        manifest = model_dir / "medshield_fusion_manifest.json"
        ckpt = model_dir / "best_gat_checkpoint.pt"
        return manifest.is_file() and ckpt.is_file()

    @staticmethod
    def build_cohort(*, max_rows: Optional[int] = None) -> GATCohortBuildResult:
        blockers: List[str] = []
        if not _torch_available():
            blockers.append("PyTorch not installed")
        if not GATCausalCohortService.gat_checkpoint_ready():
            blockers.append("GAT checkpoint or manifest missing under FUSION_MODEL_DIR")

        cache_path = _default_embedding_cache_path()
        if cache_path is None or not cache_path.is_file():
            blockers.append("No GAT embedding cache found (set CAUSAL_GAT_EMBEDDING_CACHE_PATH)")

        if blockers:
            return GATCohortBuildResult(
                ok=False,
                blockers=blockers,
                cohort_mode="UNAVAILABLE",
                metadata_aligned=False,
                frame=None,
                provenance={},
                descriptive_stats={},
            )

        import torch

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_ecg, train_text, train_y = payload["train"]
        encoder_manifest = payload.get("encoder_manifest") or {}
        n_rows = int(train_y.shape[0])

        ptbxl_root = _ptbxl_root()
        train_records, alignment = _align_train_records(ptbxl_root, expected_rows=n_rows)

        limit = max_rows or settings.CAUSAL_ANALYSIS_MAX_ROWS
        if n_rows > limit:
            train_ecg = train_ecg[:limit]
            train_text = train_text[:limit]
            train_y = train_y[:limit]
            train_records = train_records[:limit] if train_records else []
            n_rows = limit
            alignment = {
                **alignment,
                "truncated_to_max_rows": limit,
                "cohort_cap_method": COHORT_CAP_METHOD,
                "cohort_cap_deterministic": True,
            }
        demographics = _load_ptbxl_demographics(ptbxl_root) if ptbxl_root.is_dir() else {}

        try:
            preds, probs, confidences, gat_version = _run_gat_on_embeddings(
                train_ecg,
                train_text,
                GATCausalCohortService.get_gat_model_dir(),
            )
        except Exception as exc:
            logger.warning("GAT cohort inference failed: %s", type(exc).__name__)
            return GATCohortBuildResult(
                ok=False,
                blockers=[f"GAT inference on embedding cache failed: {type(exc).__name__}"],
                cohort_mode="UNAVAILABLE",
                metadata_aligned=False,
                frame=None,
                provenance={"embedding_cache": str(cache_path.relative_to(_repo_root())).replace("\\", "/")},
                descriptive_stats={},
            )

        salt = settings.CAUSAL_COHORT_PSEUDONYM_SALT
        label_mismatches = 0
        rows: List[Dict[str, Any]] = []
        for idx in range(n_rows):
            label = int(train_y[idx].item())
            if train_records and idx < len(train_records):
                rec_label = int(train_records[idx]["label"])
                if rec_label != label:
                    label_mismatches += 1
            row: Dict[str, Any] = {
                "research_sample_id": _pseudonymous_id(
                    train_records[idx]["ecg_id"] if train_records else idx,
                    salt,
                ),
                "research_proxy_label": label,
                "gat_predicted_class": preds[idx],
                "gat_prob_class_1": probs[idx],
                "gat_confidence": confidences[idx],
                "data_split": "train",
            }
            if train_records and idx < len(train_records):
                demo = demographics.get(int(train_records[idx]["ecg_id"]), {})
                if demo.get("sex") is not None:
                    row["sex"] = int(demo["sex"])
                if demo.get("age_years") is not None:
                    row["age_years"] = float(demo["age_years"])
            rows.append(row)

        df = pd.DataFrame(rows)
        metadata_aligned = bool(
            alignment.get("aligned")
            and label_mismatches == 0
            and alignment.get("patient_disjoint_across_splits") is True
        )
        if label_mismatches > 0:
            blockers.append(
                f"Embedding cache labels mismatch PTB-XL train records ({label_mismatches} rows)."
            )
        cohort_ok = metadata_aligned and len(df) > 0
        cohort_mode = "DESCRIPTIVE" if cohort_ok else "UNAVAILABLE"
        if not alignment.get("aligned"):
            blockers.append(
                "PTB-XL metadata alignment unavailable; cohort is descriptive/predictive only."
            )

        descriptive_stats = _compute_descriptive_stats(df, label_mismatch_count=label_mismatches)

        provenance = {
            "cohort_id": COHORT_ID,
            "cohort_version": COHORT_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "embedding_cache": str(cache_path.relative_to(_repo_root())).replace("\\", "/"),
            "encoder_manifest": {
                k: encoder_manifest.get(k)
                for k in ("architecture_id", "model_version_tag", "checkpoint_path")
                if isinstance(encoder_manifest, dict)
            },
            "gat_checkpoint_dir": str(
                GATCausalCohortService.get_gat_model_dir().relative_to(_repo_root())
            ).replace("\\", "/"),
            "gat_model_version": gat_version,
            "target_definition": (
                "PTB-XL proxy: NORM (0) vs MI/STTC (1). GAT outputs are model-derived features, "
                "not ground truth and not causal treatments."
            ),
            "alignment": alignment,
            "sample_count": len(df),
            "cohort_mode": cohort_mode,
            "metadata_aligned": metadata_aligned,
            "descriptive_statistics": descriptive_stats,
        }
        return GATCohortBuildResult(
            ok=cohort_ok,
            blockers=blockers,
            cohort_mode=cohort_mode,
            metadata_aligned=metadata_aligned,
            frame=df if cohort_ok else None,
            provenance=provenance,
            descriptive_stats=descriptive_stats,
        )

    @staticmethod
    def list_cohort_metadata() -> Dict[str, Any]:
        from backend.app.services.causal_design_review import assess_gat_ptbxl_cohort_design

        build = GATCausalCohortService.build_cohort()
        variables = list(DESCRIPTIVE_VARIABLES.keys())
        if build.metadata_aligned:
            variables = sorted(set(variables) | set(METADATA_VARIABLES.keys()))
        design_review = assess_gat_ptbxl_cohort_design(
            metadata_aligned=build.metadata_aligned,
            provenance=build.provenance,
        )
        return {
            "cohort_id": COHORT_ID,
            "cohort_version": COHORT_VERSION,
            "description": (
                "GAT-grounded PTB-XL ECG+report research cohort from frozen encoder embeddings "
                "and trained two-modality GAT inference."
            ),
            "variables": variables,
            "descriptive_variables": list(DESCRIPTIVE_VARIABLES.keys()),
            "cohort_available": build.ok,
            "cohort_mode": build.cohort_mode,
            "metadata_aligned": build.metadata_aligned,
            "sample_count": build.provenance.get("sample_count"),
            "descriptive_statistics": build.descriptive_stats,
            "provenance": build.provenance if build.ok else {},
            "blockers": build.blockers,
            "causal_estimation_supported": False,
            "causal_design_review": design_review,
        }
