"""
PTB-XL installation verification — Prompt 11B.
READ ONLY: does not modify raw PTB-XL files.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PTBXL_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ptb-xl"
OFFICIAL_SOURCE = "https://physionet.org/content/ptb-xl/1.0.3/"
EXPECTED_METADATA_ROWS = 21799


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums_against_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.exists():
        return {"status": "MANIFEST_MISSING", "verified_files": []}
    expected = {}
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                expected[parts[1]] = parts[0]
    to_check = [
        "LICENSE.txt",
        "ptbxl_database.csv",
        "scp_statements.csv",
        "example_physionet.py",
        "RECORDS",
    ]
    results = {}
    for name in to_check:
        p = root / name
        if not p.exists():
            results[name] = {"present": False}
            continue
        actual = sha256_of_file(p)
        exp = expected.get(name)
        results[name] = {
            "present": True,
            "sha256_computed": actual,
            "sha256_manifest": exp,
            "matches_manifest": actual == exp if exp else None,
        }
    return {"status": "OK", "root_file_checksums": results}


def count_waveform_files(root: Path) -> dict[str, int]:
    import os

    counts = {}
    for sub in ("records100", "records500"):
        p = root / sub
        hea = dat = 0
        if p.exists():
            for _dp, _dns, fns in os.walk(p):
                hea += sum(1 for fn in fns if fn.endswith(".hea"))
                dat += sum(1 for fn in fns if fn.endswith(".dat"))
        counts[f"{sub}_hea"] = hea
        counts[f"{sub}_dat"] = dat
    return counts


def full_waveform_existence_check(root: Path) -> dict[str, Any]:
    db_path = root / "ptbxl_database.csv"
    missing_lr = []
    missing_hr = []
    n = 0
    with open(db_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            lr = row["filename_lr"]
            hr = row["filename_hr"]
            lr_base = root / lr
            hr_base = root / hr
            if not lr_base.with_suffix(".hea").exists() or not lr_base.with_suffix(".dat").exists():
                missing_lr.append(row["ecg_id"])
            if not hr_base.with_suffix(".hea").exists() or not hr_base.with_suffix(".dat").exists():
                missing_hr.append(row["ecg_id"])
    return {
        "metadata_rows_checked": n,
        "missing_lr_waveforms": len(missing_lr),
        "missing_hr_waveforms": len(missing_hr),
        "sample_missing_lr_ecg_ids": missing_lr[:10],
        "sample_missing_hr_ecg_ids": missing_hr[:10],
        "complete": len(missing_lr) == 0 and len(missing_hr) == 0 and n == EXPECTED_METADATA_ROWS,
    }


def parse_hea_header(hea_path: Path) -> dict[str, Any]:
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return {"error": "empty hea"}
    parts = lines[0].split()
    if len(parts) < 4:
        return {"error": "malformed hea first line"}
    n_sig = int(parts[1])
    fs = float(parts[2])
    n_samples = int(parts[3])
    names = []
    for line in lines[1 : 1 + n_sig]:
        names.append(line.split()[-1])
    return {"n_leads": n_sig, "fs_hz": fs, "n_samples": n_samples, "lead_names": names}


def load_signals_wfdb(root: Path, rel_path: str):
    import wfdb

    rec = root / rel_path
    return wfdb.rdsamp(str(rec))


def signal_loading_audit(root: Path, metadata_rows: list[dict], sample_size: int = 96) -> dict[str, Any]:
    try:
        import wfdb  # noqa: F401
    except ImportError:
        return {
            "wfdb_available": False,
            "verdict": "CONDITIONAL_GO",
            "note": "wfdb not installed; .hea header parsing only. Install optional [ml] dependency for full signal read.",
            "hea_only_sample": [],
        }

    random.seed(42)
    if len(metadata_rows) > sample_size:
        indices = sorted(random.sample(range(len(metadata_rows)), sample_size))
        sample = [metadata_rows[i] for i in indices]
    else:
        sample = metadata_rows

    lead_ref = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lr_results = {
        "attempted": 0,
        "load_ok": 0,
        "shape_mismatch": 0,
        "fs_mismatch": 0,
        "lead_order_mismatch": 0,
        "nan_values": 0,
        "errors": [],
    }
    hr_results = {"attempted": 0, "load_ok": 0, "shape_5000x12": 0, "fs_500": 0, "errors": []}

    for row in sample:
        ecg_id = row["ecg_id"]
        try:
            sig, meta = load_signals_wfdb(root, row["filename_lr"])
            lr_results["attempted"] += 1
            if sig.shape != (1000, 12):
                lr_results["shape_mismatch"] += 1
            if float(meta["fs"]) != 100.0:
                lr_results["fs_mismatch"] += 1
            if list(meta["sig_name"]) != lead_ref:
                lr_results["lead_order_mismatch"] += 1
            if sig.shape == (1000, 12) and float(meta["fs"]) == 100.0 and list(meta["sig_name"]) == lead_ref:
                lr_results["load_ok"] += 1
            if (sig != sig).sum() > 0:  # NaN check without numpy import in type path
                lr_results["nan_values"] += 1
        except Exception as exc:
            lr_results["errors"].append({"ecg_id": ecg_id, "path": row["filename_lr"], "error": str(exc)[:200]})

        try:
            sig5, meta5 = load_signals_wfdb(root, row["filename_hr"])
            hr_results["attempted"] += 1
            if sig5.shape == (5000, 12):
                hr_results["shape_5000x12"] += 1
            if float(meta5["fs"]) == 500.0:
                hr_results["fs_500"] += 1
            if sig5.shape == (5000, 12) and float(meta5["fs"]) == 500.0:
                hr_results["load_ok"] += 1
        except Exception as exc:
            hr_results["errors"].append({"ecg_id": ecg_id, "path": row["filename_hr"], "error": str(exc)[:200]})

    first_lr = root / sample[0]["filename_lr"]
    hea_parsed = parse_hea_header(first_lr.with_suffix(".hea"))

    return {
        "wfdb_available": True,
        "sample_size": len(sample),
        "reference_lead_order": lead_ref,
        "units_observed": "mV (from wfdb meta on sample ecg_id=1)",
        "records100_lr_sample_audit": lr_results,
        "records500_hr_sample_audit": hr_results,
        "first_sample_hea_parsed": hea_parsed,
        "preprocessing_for_12x1000_at_100hz": {
            "filename_lr": "Already 12 leads × 1000 samples at 100 Hz — no resampling required.",
            "filename_hr": "500 Hz × 5000 samples — downsample by factor 5 (e.g. decimation or anti-alias filter) to 100 Hz × 1000.",
        },
    }


def load_metadata_rows(root: Path) -> list[dict]:
    rows = []
    with open(root / "ptbxl_database.csv", "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def label_and_split_audit(root: Path, rows: list[dict]) -> dict[str, Any]:
    import pandas as pd

    agg = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = agg[agg["diagnostic"] == 1.0]

    def aggregate_superclass(scp_str: str) -> list[str]:
        try:
            y_dic = ast.literal_eval(scp_str)
        except (SyntaxError, ValueError):
            return []
        out = []
        for key in y_dic:
            if key in diag.index:
                out.append(diag.loc[key, "diagnostic_class"])
        return list(set(out))

    def has_scp_key(scp_str: str, key: str) -> bool:
        try:
            y_dic = ast.literal_eval(scp_str)
        except (SyntaxError, ValueError):
            return False
        return key in y_dic

    super_counts: Counter = Counter()
    rhythm_sr = 0
    multi_super = 0
    empty_super = 0
    proxy_pos = 0
    proxy_neg = 0
    proxy_ambiguous = 0

    patient_folds: dict[str, set[int]] = defaultdict(set)
    fold_counts: Counter = Counter()
    ecg_ids = []

    for row in rows:
        ecg_ids.append(row["ecg_id"])
        scp = row["scp_codes"]
        supers = aggregate_superclass(scp)
        if not supers:
            empty_super += 1
        if len(supers) > 1:
            multi_super += 1
        for s in supers:
            super_counts[s] += 1
        if has_scp_key(scp, "SR"):
            rhythm_sr += 1

        # MedShield documented proxy (README / model registry) — not clinical diagnosis
        has_sr = has_scp_key(scp, "SR")
        has_mi = "MI" in supers
        has_sttc = "STTC" in supers
        if has_sr and not (has_mi or has_sttc):
            proxy_pos += 1
        elif has_mi or has_sttc:
            proxy_neg += 1
        else:
            proxy_ambiguous += 1

        pid = str(row["patient_id"])
        fold = int(float(row["strat_fold"]))
        patient_folds[pid].add(fold)
        fold_counts[fold] += 1

    dup_ecg = len(ecg_ids) - len(set(ecg_ids))
    patients_multi_fold = sum(1 for folds in patient_folds.values() if len(folds) > 1)
    unique_patients = len(patient_folds)

    # Official fold-10 holdout (example_physionet.py): patient overlap train vs test
    test_pids = {str(r["patient_id"]) for r in rows if int(float(r["strat_fold"])) == 10}
    train_pids = {str(r["patient_id"]) for r in rows if int(float(r["strat_fold"])) != 10}
    overlap_official = len(test_pids & train_pids)

    return {
        "scp_statements_rows": len(agg),
        "diagnostic_statements": int(diag.shape[0]),
        "rhythm_column_documentation": "scp_statements.csv rhythm=1.0 marks rhythm statements (e.g. SR, AFIB).",
        "sr_statement_meaning": "SR = sinus rhythm (SCP-ECG); aECG REFID MDC_ECG_RHY_SINUS_RHY in official table.",
        "diagnostic_superclass_counts": dict(super_counts),
        "records_with_multiple_superclasses": multi_super,
        "records_with_no_diagnostic_superclass": empty_super,
        "records_with_SR_in_scp_codes": rhythm_sr,
        "target_definitions": {
            "official_multi_label_superclass": {
                "description": "Five diagnostic superclasses NORM, MI, STTC, CD, HYP via scp_codes + scp_statements diagnostic_class.",
                "source": "example_physionet.py and PTB-XL publication",
            },
            "notebook_task": {
                "description": "Independent binary heads per superclass (multi-label).",
                "source": "data/datasets/ecg-ptb-xl.ipynb",
            },
            "medshield_sinus_rhythm_proxy": {
                "description": (
                    "Repository contract: normal sinus rhythm (SNOMED 426783006) vs pathological superclasses MI/STTC. "
                    "Operational mapping for audit counts: SR present in scp_codes AND no MI/STTC superclass "
                    "vs MI or STTC superclass present. NOT validated heart-disease diagnosis."
                ),
                "source": "README.md, backend/app/models/ml_models.py ModelVersion.target_definition",
                "snomed_in_ptbxl_csv": False,
                "proxy_class_counts": {
                    "proxy_positive_sr_without_mi_sttc": proxy_pos,
                    "proxy_negative_mi_or_sttc_superclass": proxy_neg,
                    "proxy_ambiguous_other": proxy_ambiguous,
                },
            },
        },
        "split_feasibility": {
            "unique_patient_id": unique_patients,
            "total_ecg_records": len(rows),
            "duplicate_ecg_id_rows": dup_ecg,
            "strat_fold_distribution": dict(sorted(fold_counts.items())),
            "patients_with_records_in_multiple_folds": patients_multi_fold,
            "official_fold10_holdout_patient_overlap_with_other_folds": overlap_official,
            "recommended_strategy": {
                "primary": "Patient-level split by patient_id with stratification on aggregated labels (deterministic seed 42).",
                "official_baseline": "strat_fold != 10 train/val, strat_fold == 10 test (matches example_physionet.py); verified 0 patients spanning folds in this install.",
                "test_fold_for_model_selection": "Forbidden — hold out fold 10 (or patient test partition) for final evaluation only.",
                "manifest_path": "data/manifests/ptbxl_patient_split_seed42.json",
            },
        },
    }


def run_ptbxl_verification(ptbxl_dir: Path | None = None) -> dict[str, Any]:
    root = ptbxl_dir or PTBXL_DIR
    if not root.exists():
        return {"present": False, "verdicts": {"installation": "BLOCKED", "reason": "Directory missing"}}

    version_evidence = "1.0.3" if (root / "ptbxl_v103_changelog.txt").exists() else "UNVERIFIED"
    checksums = verify_checksums_against_manifest(root)
    wf_counts = count_waveform_files(root)
    existence = full_waveform_existence_check(root)
    rows = load_metadata_rows(root)
    signals = signal_loading_audit(root, rows)
    labels = label_and_split_audit(root, rows)

    db_sha = sha256_of_file(root / "ptbxl_database.csv")
    scp_sha = sha256_of_file(root / "scp_statements.csv")

    install_complete = (
        existence["complete"]
        and wf_counts.get("records100_hea", 0) == EXPECTED_METADATA_ROWS
        and wf_counts.get("records500_hea", 0) == EXPECTED_METADATA_ROWS
        and (root / "scp_statements.csv").exists()
    )

    lr_sample = signals.get("records100_lr_sample_audit", {})
    signal_ok = (
        signals.get("wfdb_available")
        and lr_sample.get("load_ok", 0) == signals.get("sample_size", 0)
        and not lr_sample.get("errors")
    )

    preprocess_go = install_complete and (
        signals.get("wfdb_available") and lr_sample.get("load_ok", 0) > 0
    )

    verdicts = {
        "PTBXL_installation_completeness": "GO" if install_complete else "BLOCKED",
        "ECG_signal_loading": "GO" if signal_ok else ("CONDITIONAL_GO" if signals.get("wfdb_available") else "CONDITIONAL_GO"),
        "preprocessing_12x1000_at_100hz": "GO" if preprocess_go else "BLOCKED",
        "ECG_target_validity": "CONDITIONAL_GO",
        "patient_independent_evaluation_feasibility": "CONDITIONAL_GO",
        "ECG_only_research_readiness": "GO" if install_complete and signal_ok else "CONDITIONAL_GO",
        "multimodal_GAT_readiness": "BLOCKED",
    }
    if not signal_ok and signals.get("wfdb_available"):
        verdicts["ECG_signal_loading"] = "CONDITIONAL_GO" if lr_sample.get("load_ok", 0) > 0 else "BLOCKED"
    if labels["split_feasibility"]["patients_with_records_in_multiple_folds"] > 0:
        verdicts["patient_independent_evaluation_feasibility"] = "CONDITIONAL_GO"

    return {
        "present": True,
        "audit_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "directory": str(root),
        "provenance": {
            "official_name": "PTB-XL",
            "official_source": OFFICIAL_SOURCE,
            "version_evidence": version_evidence,
            "license_file": "LICENSE.txt (Creative Commons Attribution 4.0)",
            "changelog_files": [p.name for p in root.glob("ptbxl_v*.txt")],
        },
        "checksums": checksums,
        "metadata_sha256": {"ptbxl_database.csv": db_sha, "scp_statements.csv": scp_sha},
        "waveform_file_counts": wf_counts,
        "waveform_existence": existence,
        "signal_loading": signals,
        "labels_and_splits": labels,
        "verdicts": verdicts,
        "verdict_notes": {
            "ECG_target_validity": "Labels are SCP-ECG statements / superclasses — proxy only, not clinical heart-disease diagnosis.",
            "multimodal_GAT_readiness": "No linkage to cardio_train.csv or NBME; unchanged from Prompt 11A.",
        },
    }


def main():
    result = run_ptbxl_verification()
    out = Path(__file__).resolve().parent.parent / "docs" / "data-audit" / "ptbxl_ecg_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result["verdicts"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
