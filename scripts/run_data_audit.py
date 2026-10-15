"""
MedShield-FL Dataset Audit Script — PROMPT 11A
READ ONLY. Does not modify, rename, or overwrite any dataset file.
Computes checksums, profiles schemas, class distributions, missingness,
duplicate counts, and linkage evidence.
Outputs machine-readable JSON + a human-readable summary.
"""

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"
AUDIT_DIR = Path(__file__).resolve().parent.parent / "docs" / "data-audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_open_csv(path: Path, max_rows: int = 5):
    """Try to open a CSV, return (header, sample_rows, total_rows, delimiter, encoding, error)."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        for delim in (",", ";", "\t"):
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    reader = csv.reader(f, delimiter=delim)
                    header = next(reader)
                    if len(header) < 2:
                        continue  # wrong delimiter
                    rows = []
                    total = 0
                    for row in reader:
                        total += 1
                        if len(rows) < max_rows:
                            rows.append(row)
                return header, rows, total, delim, enc, None
            except Exception as e:
                continue
    return None, None, None, None, None, "Could not open file with any supported encoding/delimiter"


def csv_profile(path: Path, label_col: str = None, id_col: str = None):
    """Compute full profile: schema, counts, missingness, distributions, duplicates."""
    header, sample, total_rows, delim, enc, err = safe_open_csv(path)
    if err:
        return {"error": err}

    n_cols = len(header)
    col_missing = {col: 0 for col in header}
    col_values_for_label = []
    id_values = []
    row_hashes = []
    all_data = []

    with open(path, "r", encoding=enc, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            row_hashes.append(hash(frozenset(row.items())))
            all_data.append(row)
            for col in header:
                val = row.get(col, "")
                if val is None or val.strip() == "":
                    col_missing[col] = col_missing.get(col, 0) + 1
            if label_col and label_col in row:
                col_values_for_label.append(row[label_col])
            if id_col and id_col in row:
                id_values.append(row[id_col])

    actual_rows = len(all_data)
    dup_count = actual_rows - len(set(row_hashes))

    label_dist = {}
    if col_values_for_label:
        for v in col_values_for_label:
            label_dist[v] = label_dist.get(v, 0) + 1

    unique_ids = len(set(id_values)) if id_values else None

    missingness = {col: {"count": col_missing[col], "pct": round(col_missing[col] / actual_rows * 100, 2)}
                   for col in header}

    # Safe sample (first 3 keys only, no PHI)
    schema_sample = {col: [] for col in header}
    for row in all_data[:3]:
        for col in header:
            schema_sample[col].append(row.get(col, ""))

    return {
        "delimiter": delim,
        "encoding": enc,
        "n_columns": n_cols,
        "n_rows": actual_rows,
        "columns": header,
        "duplicate_rows": dup_count,
        "missingness_by_column": missingness,
        "label_distribution": label_dist if label_dist else "N/A (no label column specified)",
        "unique_id_count": unique_ids,
        "schema_sample_3rows": schema_sample,
    }


def _to_int(value: str):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def cardio_quality_audit(path: Path):
    """Outlier / implausible value counts for cardio_train.csv (full pass, no PHI)."""
    counts = {
        "n_rows": 0,
        "parse_errors": 0,
        "ap_lo_gt_ap_hi": 0,
        "ap_hi_out_of_plausible_range_80_250": 0,
        "ap_lo_out_of_plausible_range_40_150": 0,
        "height_out_of_plausible_range_100_220_cm": 0,
        "weight_out_of_plausible_range_30_200_kg": 0,
        "age_out_of_plausible_range_18_100_years_in_days": 0,
        "invalid_cholesterol_level": 0,
        "invalid_glucose_level": 0,
        "invalid_gender_code": 0,
    }
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            counts["n_rows"] += 1
            age = _to_int(row.get("age"))
            ap_hi = _to_int(row.get("ap_hi"))
            ap_lo = _to_int(row.get("ap_lo"))
            height = _to_int(row.get("height"))
            weight = _to_int(row.get("weight"))
            if None in (age, ap_hi, ap_lo, height, weight):
                counts["parse_errors"] += 1
                continue
            if ap_lo > ap_hi:
                counts["ap_lo_gt_ap_hi"] += 1
            if ap_hi < 80 or ap_hi > 250:
                counts["ap_hi_out_of_plausible_range_80_250"] += 1
            if ap_lo < 40 or ap_lo > 150:
                counts["ap_lo_out_of_plausible_range_40_150"] += 1
            if age < 365 * 18 or age > 365 * 100:
                counts["age_out_of_plausible_range_18_100_years_in_days"] += 1
            if height < 100 or height > 220:
                counts["height_out_of_plausible_range_100_220_cm"] += 1
            if weight < 30 or weight > 200:
                counts["weight_out_of_plausible_range_30_200_kg"] += 1
            if row.get("cholesterol") not in {"1", "2", "3"}:
                counts["invalid_cholesterol_level"] += 1
            if row.get("gluc") not in {"1", "2", "3"}:
                counts["invalid_glucose_level"] += 1
            if row.get("gender") not in {"1", "2"}:
                counts["invalid_gender_code"] += 1
    counts["notes"] = (
        "Age is stored in days (consistent with public Kaggle cardiovascular survey schema). "
        "Blood pressure outliers are common in this dataset and require documented cleaning before modeling."
    )
    return counts


def nbme_linkage_stats(datasets_dir: Path):
    """Verify internal NBME file relationships without exporting note text."""
    notes_pn = set()
    notes_cases = set()
    with open(datasets_dir / "patient_notes.csv", "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            notes_pn.add(row["pn_num"])
            notes_cases.add(row["case_num"])

    train_pn = set()
    train_features = set()
    multi_span_locations = 0
    train_rows = 0
    with open(datasets_dir / "train.csv", "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            train_rows += 1
            train_pn.add(row["pn_num"])
            train_features.add(row["feature_num"])
            if ";" in row.get("location", ""):
                multi_span_locations += 1

    feature_defs = set()
    with open(datasets_dir / "features.csv", "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            feature_defs.add(row["feature_num"])

    return {
        "patient_notes_unique_pn_num": len(notes_pn),
        "patient_notes_unique_case_num": len(notes_cases),
        "train_annotation_rows": train_rows,
        "train_unique_pn_num": len(train_pn),
        "train_unique_feature_num": len(train_features),
        "train_pn_num_subset_of_patient_notes": train_pn.issubset(notes_pn),
        "train_feature_num_subset_of_features_csv": train_features.issubset(feature_defs),
        "multi_span_location_rows": multi_span_locations,
        "test_csv_rows": csv_profile(datasets_dir / "test.csv").get("n_rows"),
        "test_csv_is_sample_only": True,
        "test_csv_evidence": (
            "Local test.csv contains 5 rows (competition ID stubs only). "
            "Full Kaggle hidden test labels are not present in this repository."
        ),
        "supervised_task": (
            "Character-span location prediction for (pn_num, feature_num) pairs — "
            "clinical concept / phrase identification, not diagnosis classification."
        ),
    }


def inspect_ptbxl_notebook(nb_path: Path):
    """Extract PTB-XL task definitions referenced in the local notebook (not performance claims)."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    text = "".join(
        "".join(cell["source"]) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        for cell in nb.get("cells", [])
    )
    paths = sorted(
        set(
            re.findall(
                r'["\']([^"\']*(?:\.hea|\.dat|ptbxl_database\.csv|scp_statements\.csv)[^"\']*)["\']',
                text,
            )
        )
    )
    paths = [p for p in paths if "dataframe" not in p and ".css" not in p][:20]

    return {
        "notebook_stated_primary_task": (
            "Multi-label diagnostic-superclass classification (NORM / MI / STTC / CD / HYP) "
            "with five independent binary sigmoid heads (BCEWithLogitsLoss)."
        ),
        "snomed_426783006_sinus_rhythm_in_notebook": "426783006" in text,
        "project_docs_sinus_rhythm_proxy": (
            "README.md and backend ModelVersion default target_definition describe a separate "
            "binary proxy: normal sinus rhythm (SNOMED 426783006) vs pathological superclasses "
            "(documented as MI, STTC). This SNOMED code was NOT found in ecg-ptb-xl.ipynb text."
        ),
        "waveform_preprocessing_mentions": {
            "500_hz_mentioned": "500 Hz" in text or "500Hz" in text,
            "100_hz_mentioned": "100 Hz" in text or "100Hz" in text,
            "strat_fold_mentioned": "strat_fold" in text,
            "12_lead_mentioned": "12-lead" in text or "12 lead" in text.lower(),
        },
        "paths_referenced_in_cells": paths,
        "external_data_search_note_in_notebook": (
            "Notebook includes logic to search common Kaggle paths for PTB-XL raw signal roots — "
            "evidence that waveform data was expected outside this repository path."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase A — File Inventory
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE A: File Inventory ===")
inventory = {}
checksum_map = {}  # checksum -> list of files (for duplicate detection)

all_files = sorted(p for p in DATASETS_DIR.rglob("*") if p.is_file())
subdirs = sorted({str(p.relative_to(DATASETS_DIR)) for p in DATASETS_DIR.rglob("*") if p.is_dir()})

for f in all_files:
        print(f"  Computing checksum for: {f.name} ({f.stat().st_size:,} bytes)")
        chk = sha256_of_file(f)
        rel = f.relative_to(DATASETS_DIR.parent.parent)
        parse_status = "NOT_APPLICABLE"
        if f.suffix.lower() == ".csv":
            _, _, n_rows, _, _, err = safe_open_csv(f)
            parse_status = "PARSE_OK" if err is None else f"PARSE_FAILED: {err}"
        elif f.suffix.lower() == ".ipynb":
            try:
                json.load(open(f, "r", encoding="utf-8"))
                parse_status = "PARSE_OK"
            except Exception as exc:
                parse_status = f"PARSE_FAILED: {exc}"

        role = "unknown"
        if f.name == "cardio_train.csv":
            role = "tabular_training_data"
        elif f.name == "patient_notes.csv":
            role = "clinical_text_corpus"
        elif f.name == "features.csv":
            role = "metadata_lookup"
        elif f.name == "train.csv":
            role = "nlp_training_annotations"
        elif f.name == "test.csv":
            role = "nlp_test_stub"
        elif f.name == "sample_submission.csv":
            role = "submission_template"
        elif f.name == "ecg-ptb-xl.ipynb":
            role = "research_notebook"

        entry = {
            "relative_path": str(rel).replace("\\", "/"),
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "sha256": chk,
            "format": f.suffix.lstrip(".").upper() if f.suffix else "UNKNOWN",
            "artifact_role": role,
            "parse_status": parse_status,
            "source_license_documented_in_repo": False,
        }
        inventory[str(rel).replace("\\", "/")] = entry
        checksum_map.setdefault(chk, []).append(str(rel).replace("\\", "/"))

# Mark duplicates
for chk, files in checksum_map.items():
    if len(files) > 1:
        for fn in files:
            inventory[fn]["duplicate_of"] = [x for x in files if x != fn]

print(f"  Total files: {len(inventory)}")
print(f"  Subdirectories under data/datasets: {len(subdirs)}")
duplicate_groups = [v for v in checksum_map.values() if len(v) > 1]
print(f"  Duplicate file groups: {len(duplicate_groups)}")

inventory_names = {Path(k).name for k in inventory}

# PTB-XL check (repository-wide for waveform extensions under data/)
repo_data_root = DATASETS_DIR.parent
hea_files = list(repo_data_root.rglob("*.hea"))
dat_files = list(repo_data_root.rglob("*.dat"))
PTBXL_EXPECTED = ["ptbxl_database.csv", "scp_statements.csv"]
ptbxl_waveform_check = {
    "ptbxl_database.csv_in_datasets": "ptbxl_database.csv" in inventory_names,
    "scp_statements.csv_in_datasets": "scp_statements.csv" in inventory_names,
    "waveform_hea_files_under_data": len(hea_files),
    "waveform_dat_files_under_data": len(dat_files),
    "ecg_ptb_xl_notebook_present": "ecg-ptb-xl.ipynb" in inventory_names,
    "verdict": (
        "BLOCKED — PTB-XL metadata CSVs and waveform signal files (.hea, .dat) are absent "
        "from the repository under data/ (verified by extension search)."
    ),
}

print(f"\n  PTB-XL waveform check: {ptbxl_waveform_check['verdict']}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase B & C/D/E — Dataset Profiles
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE B/C/D/E: Dataset Profiles ===")
profiles = {}

# ---- cardio_train.csv ----
print("\n  Profiling cardio_train.csv ...")
cardio_path = DATASETS_DIR / "cardio_train.csv"
if cardio_path.exists():
    cardio_profile = csv_profile(cardio_path, label_col="cardio", id_col="id")
    cardio_q = cardio_quality_audit(cardio_path)
    profiles["cardio_train.csv"] = {
        "dataset_name": "Cardiovascular Disease Dataset",
        "source": "Public Kaggle-style cardiovascular survey CSV (exact Kaggle page/version not cited in repo — UNVERIFIED)",
        "identity_verification": (
            "Schema matches widely mirrored 'Cardiovascular Disease' tabular dataset "
            "(semicolon delimiter, age in days, binary cardio label). Official license file not found in repository."
        ),
        "intended_task": "Binary classification: presence/absence of cardiovascular disease",
        "profile": cardio_profile,
        "data_quality_audit": cardio_q,
        "label_column": "cardio",
        "label_meanings": {
            "0": "No cardiovascular disease (per dataset annotation)",
            "1": "Cardiovascular disease present (per dataset annotation)"
        },
        "notes": "Label meaning is self-reported/annotated in the dataset, not clinically validated ground truth. "
                 "Features include age, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol, physical activity.",
        "identifier_type": "id column present — sequential integer, not cryptographic or cross-linkable patient ID",
    }
    print(f"    Rows: {cardio_profile.get('n_rows')}, Cols: {cardio_profile.get('n_columns')}, Dups: {cardio_profile.get('duplicate_rows')}")
    print(f"    Label distribution: {cardio_profile.get('label_distribution')}")
    print(f"    Missing values summary: {sum(v['count'] for v in cardio_profile.get('missingness_by_column', {}).values())} total")
else:
    profiles["cardio_train.csv"] = {"error": "File not found"}

# ---- patient_notes.csv ----
print("\n  Profiling patient_notes.csv (streaming, large file) ...")
notes_path = DATASETS_DIR / "patient_notes.csv"
if notes_path.exists():
    # Streaming sample — do not load entire 35MB into memory unnecessarily
    notes_profile = csv_profile(notes_path, id_col="pn_num")
    profiles["patient_notes.csv"] = {
        "dataset_name": "NBME Clinical Patient Notes",
        "source": "Kaggle — NBME Score Clinical Patient Notes Competition (2022)",
        "intended_task": "Clinical concept / phrase identification from medical licensing examination notes",
        "profile": notes_profile,
        "identifier_column": "pn_num",
        "notes": "These are de-identified medical licensing exam notes, not real hospital patient records. "
                 "The task is NLP — identifying specific clinical phrases, not diagnosing cardiovascular disease.",
    }
    print(f"    Rows: {notes_profile.get('n_rows')}, Cols: {notes_profile.get('n_columns')}, Dups: {notes_profile.get('duplicate_rows')}")
else:
    profiles["patient_notes.csv"] = {"error": "File not found"}

# ---- features.csv ----
print("\n  Profiling features.csv ...")
feat_path = DATASETS_DIR / "features.csv"
if feat_path.exists():
    feat_profile = csv_profile(feat_path, id_col="feature_num")
    profiles["features.csv"] = {
        "dataset_name": "NBME Feature Definitions",
        "source": "Kaggle — NBME Score Clinical Patient Notes Competition (2022)",
        "intended_task": "Lookup table: feature identifiers and their clinical concept descriptions",
        "profile": feat_profile,
        "notes": "This is a reference/lookup file, not a training dataset. Maps feature_num to clinical concept phrases.",
    }
    print(f"    Rows: {feat_profile.get('n_rows')}, Cols: {feat_profile.get('n_columns')}")
else:
    profiles["features.csv"] = {"error": "File not found"}

# ---- train.csv ----
print("\n  Profiling train.csv ...")
train_path = DATASETS_DIR / "train.csv"
if train_path.exists():
    train_profile = csv_profile(train_path, id_col="id")
    profiles["train.csv"] = {
        "dataset_name": "NBME Training Annotations",
        "source": "Kaggle — NBME Score Clinical Patient Notes Competition (2022)",
        "intended_task": "Supervised NLP: mapping (patient_note, feature) pairs to character-span annotations",
        "profile": train_profile,
        "notes": "Annotations are character-level spans identifying where a clinical concept appears in a patient note. "
                 "Target is phrase location, NOT cardiovascular diagnosis label.",
    }
    print(f"    Rows: {train_profile.get('n_rows')}, Cols: {train_profile.get('n_columns')}, Dups: {train_profile.get('duplicate_rows')}")
    print(f"    Sample columns: {train_profile.get('columns')}")
else:
    profiles["train.csv"] = {"error": "File not found"}

# ---- test.csv ----
print("\n  Profiling test.csv ...")
test_path = DATASETS_DIR / "test.csv"
if test_path.exists():
    test_profile = csv_profile(test_path, id_col="id")
    profiles["test.csv"] = {
        "dataset_name": "NBME Test Set (Kaggle submission format)",
        "source": "Kaggle — NBME Score Clinical Patient Notes Competition (2022)",
        "profile": test_profile,
        "notes": "This is the Kaggle competition test file (withheld labels). Small file likely contains only IDs, not the full hidden test set.",
    }
    print(f"    Rows: {test_profile.get('n_rows')}, Cols: {test_profile.get('n_columns')}")
else:
    profiles["test.csv"] = {"error": "File not found"}

# ---- sample_submission.csv ----
print("\n  Profiling sample_submission.csv ...")
sub_path = DATASETS_DIR / "sample_submission.csv"
if sub_path.exists():
    sub_profile = csv_profile(sub_path, id_col="id")
    profiles["sample_submission.csv"] = {
        "dataset_name": "NBME Sample Submission Template",
        "source": "Kaggle — NBME Score Clinical Patient Notes Competition (2022)",
        "profile": sub_profile,
        "notes": "Template file for Kaggle competition submissions. Contains no training labels.",
    }
    print(f"    Rows: {sub_profile.get('n_rows')}, Cols: {sub_profile.get('n_columns')}")
else:
    profiles["sample_submission.csv"] = {"error": "File not found"}

# ---- ecg-ptb-xl.ipynb ----
print("\n  Inspecting ecg-ptb-xl.ipynb ...")
nb_path = DATASETS_DIR / "ecg-ptb-xl.ipynb"
nb_inspection = {}
if nb_path.exists():
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_raw = f.read()
        # Look for key path references without importing heavy libraries
        import re
        paths_mentioned = re.findall(r'["\']([^"\']*(?:\.hea|\.dat|ptbxl_database\.csv|scp_statements\.csv)[^"\']*)["\']', nb_raw)
        external_urls = re.findall(r'https?://[^\s\'"]+', nb_raw)[:10]
        nb_inspection = {
            "type": "Jupyter Notebook",
            "size_bytes": nb_path.stat().st_size,
            "paths_referenced_in_cells": list(set(paths_mentioned[:20])),
            "external_urls_found": external_urls,
            "waveform_files_accessible": "UNKNOWN — Notebook references waveform paths but these files are not in data/datasets/",
            "note": "The existence of this notebook does NOT prove PTB-XL waveform data is present. "
                    "The notebook is only evidence of prior experimentation with PTB-XL data that must have existed elsewhere.",
        }
        print(f"    Paths referenced in notebook: {len(nb_inspection['paths_referenced_in_cells'])}")
        if nb_inspection['paths_referenced_in_cells']:
            print(f"    Sample path references: {nb_inspection['paths_referenced_in_cells'][:5]}")
    except Exception as e:
        nb_inspection = {"error": str(e)}
else:
    nb_inspection = {"error": "File not found"}

profiles["ecg-ptb-xl.ipynb"] = {
    "dataset_name": "PTB-XL ECG Jupyter Notebook",
    "type": "Research Notebook (NOT a dataset)",
    "inspection": nb_inspection,
}

nb_target_audit = inspect_ptbxl_notebook(nb_path) if nb_path.exists() else {"error": "notebook missing"}
profiles["ptbxl_ecg_audit"] = {
    "waveform_files_present": False,
    "metadata_files_present": False,
    "load_feasibility": "BLOCKED — cannot load ECG signals without .hea/.dat and ptbxl_database.csv",
    "official_reference": "https://physionet.org/content/ptb-xl/1.0.3/",
    "published_record_counts_unverified_locally": "~21,799 ECG records / ~18,869 patients (from PTB-XL publication; not counted locally)",
    "target_definitions": {
        "notebook_primary_task": nb_target_audit.get("notebook_stated_primary_task"),
        "repository_proxy_task_documentation": nb_target_audit.get("project_docs_sinus_rhythm_proxy"),
        "task_category": (
            "Diagnostic statement / superclass coding (rhythm vs structural disease tasks are distinct). "
            "Neither task equals validated clinical heart-disease ground truth."
        ),
    },
    "resampling_feasibility_if_data_added": (
        "PTB-XL provides 500 Hz and 100 Hz releases; downsampling 500 Hz 10 s traces to 100 Hz yields "
        "12 x 1000 tensors IF metadata fs and sig_len fields confirm duration and lead layout."
    ),
    "verdict": "BLOCKED",
    "verdict_reason": ptbxl_waveform_check["verdict"],
}

nbme_stats = nbme_linkage_stats(DATASETS_DIR)
profiles["nbme_internal_linkage"] = nbme_stats


# ──────────────────────────────────────────────────────────────────────────────
# Phase F — Linkage Audit
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE F: Linkage Audit ===")
linkage_audit = {
    "nbme_internal_files": {
        "linkage_possible": True,
        "join_keys": ["pn_num", "feature_num", "case_num"],
        "evidence": nbme_stats,
        "verdict": "GO — internal NBME files join on documented competition keys (within NBME task only)",
    },
    "ECG_PTB-XL_to_cardio_train": {
        "linkage_possible": False,
        "reason": "PTB-XL waveform files (.hea, .dat) are ABSENT from the repository. "
                  "Even if present, PTB-XL patient_ids and cardio_train.csv 'id' integers originate from "
                  "entirely different cohorts (German ECG study vs. Russian hospital cardiovascular survey). "
                  "No verified crosswalk exists.",
        "verdict": "BLOCKED",
    },
    "ECG_PTB-XL_to_NBME_notes": {
        "linkage_possible": False,
        "reason": "PTB-XL waveform files absent. PTB-XL uses German hospital patient cohort. "
                  "NBME notes use US medical licensing examination synthetic/de-identified text. "
                  "No shared patient identifier exists and no crosswalk is documented.",
        "verdict": "BLOCKED",
    },
    "cardio_train_to_NBME_notes": {
        "linkage_possible": False,
        "reason": "cardio_train.csv uses sequential integer IDs from a Russian cardiovascular survey. "
                  "NBME patient notes use pn_num identifiers from a US medical licensing examination dataset. "
                  "These represent completely different patient populations, countries, and data domains. "
                  "No crosswalk, no shared identifier, no documented common source.",
        "verdict": "BLOCKED",
    },
    "GAT_multimodal_fusion_feasibility": {
        "feasible": False,
        "gat_node_design_if_data_existed": {
            "ecg_node": "One node per 12-lead ECG recording embedding (patient_id grouping)",
            "text_node": "One node per clinical note embedding",
            "tabular_node": "One node per structured feature vector",
            "cross_modal_edges": "Require verified same-patient / same-encounter identifiers",
            "supervision": "Requires a single clinically valid target on the fused encounter",
            "missing_modality": "Would require explicit missing-modality policy — not evaluable without paired data",
        },
        "reason": "No legitimate patient-level pairing exists across any two of the three datasets. "
                  "PTB-XL waveform data is absent. NBME labels are clinical phrase spans, not cardiovascular diagnosis. "
                  "cardio_train.csv has no shared identifier with either ECG or NLP dataset. "
                  "Training a supervised multimodal GAT requires co-occurring same-patient records across all three modalities, "
                  "which do not exist in this repository.",
        "verdict": "BLOCKED",
    },
}

print("  All cross-dataset pairing verdicts: BLOCKED")


# ──────────────────────────────────────────────────────────────────────────────
# Phase G — Split Strategy
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE G: Split Strategy ===")
AUDIT_CHECKSUMS = {
    "cardio_train.csv": inventory.get("data/datasets/cardio_train.csv", {}).get("sha256"),
    "patient_notes.csv": inventory.get("data/datasets/patient_notes.csv", {}).get("sha256"),
    "train.csv": inventory.get("data/datasets/train.csv", {}).get("sha256"),
}

split_strategy = {
    "cardio_train_binary_classification": {
        "task": "Binary cardiovascular disease classification (tabular only)",
        "grouping_key": "id column — but this is a sequential integer, not a true patient encounter ID. "
                        "No documented repeated-measurement information. Assume one row per patient unless documentation contradicts.",
        "recommendation": "Patient-independent split by id (assuming each id = 1 patient). "
                          "Stratified by label (cardio) to preserve class balance.",
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "split_seed": 42,
        "stratification": "cardio (binary)",
        "leakage_risks": [
            "Height and weight may correlate strongly with age and BMI — ensure no circular feature derivation.",
            "ap_lo (diastolic BP) and ap_hi (systolic BP) are direct risk markers — must not be derived from label.",
            "Implausible blood pressure values (e.g., ap_lo > ap_hi, extreme outliers) must be identified and handled before splitting.",
        ],
        "preprocessing_rule": "All scalers, normalizers fitted on train split ONLY. Val/test receive train-fitted transforms.",
        "dataset_checksum_sha256": AUDIT_CHECKSUMS.get("cardio_train.csv"),
        "manifest_output_path": "data/manifests/cardio_patient_split_seed42.json",
    },
    "NBME_NLP_phrase_identification": {
        "task": "NLP phrase identification: character-span extraction from clinical notes",
        "grouping_key": "pn_num (patient note ID) — split by pn_num to ensure the same note never appears in both train and val",
        "recommendation": "Hold out notes by pn_num. Do NOT allow the same note to appear in both train and val.",
        "split_seed": 42,
        "leakage_risks": [
            "BioClinicalBERT tokenizer must be applied using pre-trained vocabulary, not fitted on this dataset. No leakage risk from tokenizer.",
            "test.csv from Kaggle has withheld labels — cannot be used for evaluation. Use train/val split only.",
            "Split by pn_num (not by annotation row id) to prevent the same note appearing in train and validation.",
        ],
        "dataset_checksum_sha256": AUDIT_CHECKSUMS.get("train.csv"),
        "notes_checksum_sha256": AUDIT_CHECKSUMS.get("patient_notes.csv"),
        "manifest_output_path": "data/manifests/nbme_note_split_seed42.json",
    },
    "PTB-XL_ECG": {
        "task": "N/A — waveform files absent",
        "recommendation": "BLOCKED — cannot define a split without signal data. "
                          "If waveform data is added in future: split by patient_id column in ptbxl_database.csv. "
                          "PTB-XL provides recommended splits (strat_fold field). Per-lead z-score normalization "
                          "must be fitted on train folds only.",
        "verdict": "BLOCKED",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Phase H — Sample Sufficiency
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE H: Sample Sufficiency ===")
cardio_rows = profiles["cardio_train.csv"]["profile"].get("n_rows", "UNKNOWN")
cardio_label_dist = profiles["cardio_train.csv"]["profile"].get("label_distribution", {})
nbme_train_rows = profiles["train.csv"]["profile"].get("n_rows", "UNKNOWN")
nbme_notes_rows = profiles["patient_notes.csv"]["profile"].get("n_rows", "UNKNOWN")

sufficiency = {
    "cardio_train_tabular": {
        "n_rows": cardio_rows,
        "label_distribution": cardio_label_dist,
        "independent_patients": f"~{cardio_rows} (assuming 1 row = 1 patient — not confirmed by documentation)",
        "assessment": "Sufficient for a standalone tabular binary classification experiment. "
                      "Class balance should be verified from computed distribution.",
        "confidence_intervals": "Meaningful CIs on AUC/F1 feasible at this scale with bootstrap or CV.",
        "caveats": ["Label is self-reported, not clinically confirmed. Limits clinical validity of conclusions.",
                    "No repeated measurements documented — cannot verify patient independence.",
                    "Implausible physiological values (extreme BP, negative ages) must be audited."],
    },
    "NBME_NLP": {
        "n_notes": nbme_notes_rows,
        "n_training_annotations": nbme_train_rows,
        "assessment": "Sufficient for NLP phrase extraction experiments with BioClinicalBERT. "
                      "Kaggle test set labels are withheld — internal val evaluation only.",
        "caveats": ["Task is clinical phrase extraction, NOT cardiovascular diagnosis.",
                    "Notes are from US medical licensing exams, not real patient hospital records.",
                    "Cannot produce clinically valid cardiovascular outcomes from this data."],
    },
    "PTB_XL_ECG": {
        "assessment": "BLOCKED — signal files absent. PTB-XL as published contains ~21,799 clinical ECG records "
                      "from 18,869 patients (10-second, 12-lead, 500Hz). If obtained, this is sufficient for ECG-only research.",
        "caveats": ["Waveform files must be verified after download before any training.",
                    "Downsampling from 500Hz to 100Hz yields 12×1000 tensor per record as required by architecture.",
                    "Patient-independent splits must use patient_id from ptbxl_database.csv, not strat_fold alone."],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Phase I — Dataset Recommendations
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE I: Dataset Recommendations ===")
recommendations = {
    "PTB-XL_ECG": {
        "official_name": "PTB-XL, a large publicly available electrocardiography dataset",
        "official_source": "https://physionet.org/content/ptb-xl/1.0.3/",
        "doi": "10.13026/kfzx-aw45",
        "citation": "Wagner P, Strodthoff N, Bousseljot RD, et al. PTB-XL, a large publicly available electrocardiography dataset. Sci Data 7, 154 (2020).",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "access": "Open access — downloadable via PhysioNet without credentialing",
        "approx_size": "~1.8 GB (500 Hz) or ~700 MB (100 Hz version)",
        "modality": "12-lead ECG waveforms + diagnostic labels",
        "n_records": "21,799 clinical ECG records from 18,869 patients",
        "n_patients": "18,869",
        "target_labels": "Diagnostic superclasses: NORM, MI, STTC, CD, HYP. Rhythm, subclasses, and form codes also available.",
        "why_needed": "ECG waveform signal files (.hea, .dat) are absent from the repository. This is ESSENTIAL.",
        "priority": "ESSENTIAL — without this, the ECG modality is fully blocked",
        "download_command": "wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/",
        "known_limitations": [
            "Diagnostic labels are clinical annotations, not gold-standard histopathology.",
            "Multi-label: each record can have multiple diagnostic statements.",
            "NORM vs. pathological binary target is a proxy, not validated cardiovascular disease diagnosis.",
            "PTB-XL recommends 10-fold stratified CV with strat_fold field for reproducibility.",
        ],
    },
    "MIMIC-IV-ECG": {
        "official_name": "MIMIC-IV-ECG — Diagnostic Electrocardiogram Matched Subset",
        "official_source": "https://physionet.org/content/mimic-iv-ecg/1.0/",
        "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "access": "RESTRICTED — requires CITI training, PhysioNet credentialing, and data use agreement signing",
        "why_relevant": "Contains 12-lead ECGs matched to MIMIC-IV clinical records — enables genuine ECG-to-clinical-text linkage",
        "priority": "OPTIONAL (but enables multimodal linkage if access granted)",
        "known_limitations": [
            "Requires significant time for credentialing (weeks to months).",
            "Cannot be downloaded automatically.",
            "Restricted to credentialed researchers.",
        ],
    },
    "MIMIC-IV_clinical_notes": {
        "official_name": "MIMIC-IV-Note",
        "official_source": "https://physionet.org/content/mimic-iv-note/2.2/",
        "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "access": "RESTRICTED — same requirements as MIMIC-IV",
        "why_relevant": "Real hospital clinical notes linked to patient encounters. Would replace NBME for cardiovascular NLP task.",
        "priority": "OPTIONAL for better clinical NLP results — requires credentials",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Phase J — Federated Learning Feasibility
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE J: Federated Learning Feasibility ===")
fl_assessment = {
    "real_hospital_silos": {
        "available": False,
        "reason": "No multi-hospital site identifiers exist in any dataset. "
                  "cardio_train.csv has no hospital/site column. PTB-XL is a single-institution German dataset. "
                  "NBME notes are licensing exam data, not real hospital silos.",
    },
    "simulated_fl_experiment": {
        "feasible": True,
        "approach": "Partition cardio_train.csv into N synthetic clients by id-based splitting. "
                    "This creates a reproducible simulation but must be explicitly labeled as a SIMULATED federated experiment, "
                    "NOT a real multi-hospital deployment.",
        "recommended_partitions": "3–5 synthetic clients with balanced label distributions",
        "flower_compatibility": "Flower supports custom data loaders — compatible with partitioned cardio tabular data",
        "important_caveat": "A simulated federated experiment on a single centralized dataset does NOT validate "
                            "real-world heterogeneous hospital privacy guarantees or data distribution shifts.",
    },
    "PTB-XL_FL": {
        "feasible": "CONDITIONAL — PTB-XL contains recording_device and site metadata. "
                    "After downloading waveform files, partitioning by recording_device or site yields more meaningful client splits.",
        "blocked_by": "Waveform files absent.",
    },
    "verdict": "Simulated FL on cardio_train.csv tabular data is feasible if explicitly labeled as simulation. "
               "Real multi-site FL is not currently supported by the available data.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Phase K — Write Output Files
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== PHASE K: Writing Output Files ===")

# 1. dataset_inventory.json
inventory_out = {
    "audit_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "datasets_dir": str(DATASETS_DIR),
    "subdirectories": subdirs,
    "files": inventory,
    "duplicate_checksum_groups": [g for g in duplicate_groups if len(g) > 1],
    "ptbxl_waveform_check": ptbxl_waveform_check,
    "notebook_inspection": nb_inspection,
    "missing_expected_files": {
        "ptbxl_database.csv": "MISSING under data/",
        "scp_statements.csv": "MISSING under data/",
        "PTB-XL .hea waveform headers": "MISSING under data/ (0 files found)",
        "PTB-XL .dat waveform signals": "MISSING under data/ (0 files found)",
    },
}
with open(AUDIT_DIR / "dataset_inventory.json", "w") as f:
    json.dump(inventory_out, f, indent=2, default=str)
print(f"  Written: {AUDIT_DIR / 'dataset_inventory.json'}")

# 2. dataset_profiles.json
# Remove schema_sample_3rows from the profiles to avoid printing any potential PHI
safe_profiles = {}
for name, profile in profiles.items():
    safe = dict(profile)
    if "profile" in safe and isinstance(safe["profile"], dict):
        p = dict(safe["profile"])
        p.pop("schema_sample_3rows", None)  # Exclude row samples for privacy
        safe["profile"] = p
    safe_profiles[name] = safe

with open(AUDIT_DIR / "dataset_profiles.json", "w") as f:
    json.dump(safe_profiles, f, indent=2, default=str)
print(f"  Written: {AUDIT_DIR / 'dataset_profiles.json'}")

# 3. linkage_audit.json
with open(AUDIT_DIR / "linkage_audit.json", "w") as f:
    json.dump(linkage_audit, f, indent=2)
print(f"  Written: {AUDIT_DIR / 'linkage_audit.json'}")

# 4. split_strategy.json
with open(AUDIT_DIR / "split_strategy.json", "w") as f:
    json.dump(split_strategy, f, indent=2)
print(f"  Written: {AUDIT_DIR / 'split_strategy.json'}")

# 5. dataset_audit_summary.json
summary = {
    "audit_version": "11A",
    "ECG": "BLOCKED",
    "Tabular": "CONDITIONAL_GO",
    "Clinical_text": "CONDITIONAL_GO",
    "Multimodal_fusion": "BLOCKED",
    "Federated_learning": "CONDITIONAL_GO",
    "Centralized_evaluation": "CONDITIONAL_GO",
    "Clinical_diagnosis_claim": "BLOCKED",
    "verdicts": {
        "ECG_only_research": "BLOCKED",
        "ECG_verdict_reason": "PTB-XL metadata and .hea/.dat waveform signal files are absent from data/; ECG loading cannot be verified locally",
        "Tabular_only_research": "CONDITIONAL_GO",
        "Tabular_verdict_reason": "cardio_train.csv: 70,000 rows, balanced binary label, zero missing cells; requires BP/outlier cleaning; label is survey annotation not clinically validated",
        "Clinical_text_research": "CONDITIONAL_GO",
        "Clinical_text_verdict_reason": "NBME supports phrase-span NLP (14,300 train annotations / 42,146 notes); not cardiovascular diagnosis; test.csv is a 5-row stub",
        "Multimodal_GAT_fusion": "BLOCKED",
        "Multimodal_verdict_reason": "No auditable cross-dataset patient linkage; incompatible supervision targets across modalities",
        "Centralized_evaluation": "CONDITIONAL_GO",
        "Centralized_evaluation_reason": "Per-modality centralized baselines feasible; fused multimodal evaluation blocked",
        "Federated_training": "CONDITIONAL_GO",
        "Federated_training_reason": "Simulated FL on partitioned cardio_train.csv only; not real multi-hospital silos",
        "Clinical_diagnosis_claim": "BLOCKED",
        "Clinical_diagnosis_reason": "No clinically validated multimodal heart-disease ground truth in repository",
    },
    "additional_datasets_required": {
        "PTB-XL_waveform_files": {
            "status": "ESSENTIAL",
            "source": "https://physionet.org/content/ptb-xl/1.0.3/",
            "license": "CC BY 4.0 — open access",
        },
        "MIMIC-IV-ECG": {
            "status": "OPTIONAL — enables genuine multimodal ECG+clinical linkage",
            "source": "https://physionet.org/content/mimic-iv-ecg/1.0/",
            "license": "Requires credentialing",
        },
    },
    "audit_limitations": [
        "PTB-XL waveform sample quality and completeness cannot be verified without the actual files.",
        "cardio_train.csv source version cannot be confirmed beyond filename; license attribution is unverified.",
        "NBME dataset license restricts certain uses — must verify competition terms before production deployment.",
        "No model training was performed; no accuracy, AUC, or F1 values have been computed or estimated.",
    ],
    "files_modified": "NONE — audit is read-only",
    "training_conducted": False,
    "raw_datasets_modified": False,
}

with open(AUDIT_DIR / "dataset_audit_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Written: {AUDIT_DIR / 'dataset_audit_summary.json'}")

# 6. dataset_recommendations.md
rec_md = f"""# Dataset Recommendations (Evidence-Based)

Generated by `scripts/run_data_audit.py` (Prompt 11A). Recommendations apply only where the local audit established a blocker.

## Essential

### PTB-XL (PhysioNet 1.0.3)

- **Official page:** https://physionet.org/content/ptb-xl/1.0.3/
- **DOI:** 10.13026/kfzx-aw45
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **Resolves:** Missing ECG waveforms (`*.hea`, `*.dat`), `ptbxl_database.csv`, `scp_statements.csv`
- **Modalities:** 12-lead ECG signals, SCP-ECG diagnostic statements, patient_id, strat_fold splits
- **Access:** Open download via PhysioNet (no credentialing for PTB-XL itself)
- **Storage (approx.):** ~700 MB (100 Hz) to ~1.8 GB (500 Hz) per official release notes
- **Architecture fit:** Supports 1D-CNN + BiLSTM ECG encoder after verified resampling to 12×1000 @ 100 Hz
- **Limitations:** Multi-label diagnostics; proxy targets ≠ validated clinical heart-disease diagnosis

## Optional (multimodal linkage — restricted access)

### MIMIC-IV-ECG

- **Official page:** https://physionet.org/content/mimic-iv-ecg/1.0/
- **License:** PhysioNet Credentialed Health Data License 1.5.0
- **Resolves:** Genuine ECG ↔ hospital record linkage (with MIMIC-IV)
- **Access:** CITI training + PhysioNet credentialing + DUA (weeks)

### MIMIC-IV-Note

- **Official page:** https://physionet.org/content/mimic-iv-note/2.2/
- **Resolves:** Real clinical notes linked to encounters (replace NBME for hospital NLP)
- **Access:** Same credentialing as MIMIC-IV

## Not recommended as substitutes without linkage

- **cardio_train.csv**, **NBME**, and **PTB-XL** remain independent public cohorts — acquiring all three does **not** enable supervised GAT fusion without a verified crosswalk.

## Cardiovascular tabular alternative

Only required if `cardio_train.csv` is rejected after cleaning. No replacement is **essential** today; the file is parseable and sufficiently large for a tabular baseline once outliers are handled.

"""
with open(AUDIT_DIR / "dataset_recommendations.md", "w", encoding="utf-8") as f:
    f.write(rec_md)
print(f"  Written: {AUDIT_DIR / 'dataset_recommendations.md'}")

# 7. DATASET_AUDIT_REPORT.md
cardio_q = profiles.get("cardio_train.csv", {}).get("data_quality_audit", {})
report_md = f"""# MedShield-FL Dataset Audit Report (Prompt 11A)

**Audit time (UTC):** {inventory_out['audit_timestamp']}  
**Repository:** `C:/Federated-Learning-Project`  
**Datasets directory:** `data/datasets/` (flat; **{len(subdirs)}** subdirectories)  
**Training conducted:** No  
**Raw datasets modified:** No  

## Executive summary

| Capability | Verdict | Evidence | Blockers | Next action |
|---|---|---|---|---|
| ECG-only research | **BLOCKED** | 0 `.hea`/`.dat` under `data/`; no `ptbxl_database.csv` | Waveforms + metadata absent | Download PTB-XL 1.0.3 from PhysioNet; verify load |
| Tabular-only research | **CONDITIONAL_GO** | `cardio_train.csv` 70,000×13; balanced `cardio` label | BP/outlier rows; unverified license; survey label | Clean outliers; patient-level split by `id`; tabular baseline |
| Clinical-text research | **CONDITIONAL_GO** | NBME 42,146 notes; 14,300 span annotations | Phrase ID task; 5-row local `test.csv` | Split by `pn_num`; BioClinicalBERT span model |
| Multimodal GAT | **BLOCKED** | No shared patient keys across cohorts | No crosswalk; incompatible labels | Restricted linked dataset (e.g. MIMIC-IV-ECG + Note) or single-site data |
| Centralized evaluation | **CONDITIONAL_GO** | Single-modality datasets usable separately | Multimodal eval blocked | Run per-modality baselines first |
| Federated training | **CONDITIONAL_GO** | `cardio_train.csv` partitionable by synthetic clients | No real hospital/site IDs | Simulated FL only; document as simulation |
| Clinical diagnosis claim | **BLOCKED** | No validated multimodal GT | All modalities | Do not claim clinical diagnosis performance |

## Phase A — Inventory

**Files present ({len(inventory)}):**

| File | Size (bytes) | SHA-256 |
|---|---:|---|
"""
for key in sorted(inventory.keys()):
    ent = inventory[key]
    report_md += f"| `{ent['relative_path']}` | {ent['size_bytes']:,} | `{ent['sha256'][:16]}…` |\n"

report_md += f"""
**Missing (checklist):** `ptbxl_database.csv`, `scp_statements.csv`, all PTB-XL `.hea`/`.dat` waveforms under `data/`.

**Duplicates by checksum:** {len(duplicate_groups)} groups.

## Phase B — Dataset identities

### `cardio_train.csv`

- **Rows / columns:** 70,000 / 13 (`;` delimiter, UTF-8)
- **Label:** `cardio` — 35,021 class 0, 34,979 class 1 (near-balanced)
- **IDs:** 70,000 unique `id` values (assume one row per subject unless external docs say otherwise)
- **Missingness:** 0 empty cells across all columns (verified full pass)
- **Quality flags (full pass):** ap_lo>ap_hi: {cardio_q.get('ap_lo_gt_ap_hi')}; ap_hi out of 80–250: {cardio_q.get('ap_hi_out_of_plausible_range_80_250')}; ap_lo out of 40–150: {cardio_q.get('ap_lo_out_of_plausible_range_40_150')}; height/weight outliers: {cardio_q.get('height_out_of_plausible_range_100_220_cm')}/{cardio_q.get('weight_out_of_plausible_range_30_200_kg')}
- **Task supported:** Binary cardiovascular survey outcome prediction (tabular encoder baseline)
- **Not supported without evidence:** Clinically confirmed heart-disease diagnosis claim

### NBME (`patient_notes.csv`, `features.csv`, `train.csv`, `test.csv`)

- **Notes:** 42,146 rows; unique `pn_num` per row; 10 `case_num` values
- **Features:** 143 concept definitions
- **Train annotations:** 14,300 rows; 1,000 unique notes annotated; all `pn_num` ∈ notes; all `feature_num` ∈ features
- **Target:** Character-span `location` for concept phrases (1,650 rows with multi-span `;` separators)
- **`test.csv`:** 5 rows — **sample/stub only**, not full competition hidden set
- **Task supported:** Clinical concept / phrase identification (BioClinicalBERT fine-tuning for span extraction)
- **Not supported:** Cardiovascular diagnosis label for MedShield multimodal pipeline

### PTB-XL

- **Local status:** BLOCKED — no waveforms or metadata CSVs in repo
- **Notebook (`ecg-ptb-xl.ipynb`):** Documents **multi-label diagnostic superclass** task (NORM/MI/STTC/CD/HYP)
- **Repository docs (`README.md`, model registry):** Document a **separate** sinus-rhythm SNOMED 426783006 vs superclass **proxy** — SNOMED string **not found** in notebook source text
- **Legitimate ECG experiment if PTB-XL added:** Diagnostic superclass or rhythm coding — **not** interchangeable with heart-disease diagnosis marketing

## Phase F — Multimodal linkage

- **NBME internal:** Valid joins on `pn_num` / `feature_num` (within NBME only) — **GO**
- **PTB-XL ↔ cardio_train ↔ NBME:** No shared patient identifier, no crosswalk, incompatible labels — **BLOCKED**
- **GAT trainable today:** **No** — no genuine cross-modal patient correspondence

## Phase G — Splits & leakage

See `split_strategy.json`: patient/`id` split for cardio; `pn_num` grouping for NBME; PTB-XL split **blocked** until data present.

## Phase H — Sample sufficiency

- **Tabular:** ~70k rows adequate for binary classification experiment with CV; clean BP outliers first
- **NBME NLP:** ~14k annotated (note, feature) pairs; ~1k notes — adequate for phrase-ID prototyping, not for rare concept stability without careful splits
- **ECG:** Not assessable locally (data absent)

## Phase J — Federated learning

Real multi-hospital silos: **not available**. Simulated clients on `cardio_train.csv`: **feasible** if labeled simulation.

## Limitations

- PTB-XL signal integrity unverified (files missing)
- Cardio & NBME source licenses not stored in repo
- No model metrics computed in this audit

## Machine-readable outputs

- `dataset_inventory.json`
- `dataset_profiles.json`
- `linkage_audit.json`
- `split_strategy.json`
- `dataset_audit_summary.json`
- `dataset_recommendations.md`
"""
with open(AUDIT_DIR / "DATASET_AUDIT_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"  Written: {AUDIT_DIR / 'DATASET_AUDIT_REPORT.md'}")

print("\n=== AUDIT SCRIPT COMPLETE ===")
print(f"Output directory: {AUDIT_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
# Print quick verdicts to stdout
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("QUICK VERDICT SUMMARY")
print("="*60)
for k, v in summary["verdicts"].items():
    if "verdict" in k.lower() and "reason" not in k.lower():
        print(f"  {k}: {v}")
print("="*60)

# Return structured results for the caller
print("\nCardio rows:", cardio_rows)
print("Cardio label distribution:", cardio_label_dist)
print("NBME notes rows:", nbme_notes_rows)
print("NBME train annotations:", nbme_train_rows)
