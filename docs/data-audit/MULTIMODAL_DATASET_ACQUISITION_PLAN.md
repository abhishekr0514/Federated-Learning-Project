# Multimodal Dataset Acquisition Plan (Prompt 16)

Purpose: obtain **verifiable** patient- or encounter-level linkage and a **scientifically defensible shared target** for future multimodal research. This plan does **not** authorize downloads from this repository automation; human credentialing steps are required.

---

## Current gap

Local data supports **three independent research tracks** only. **Supervised three-modality GAT training is blocked** until a linked cohort is installed and re-audited.

---

## Primary recommended path: MIMIC-IV ecosystem (restricted)

| Step | Action | Owner | Output artifact |
|------|--------|-------|-----------------|
| 1 | Complete CITI “Data or Specimens Only Research” (or institution equivalent) | Researcher | Training certificate PDF (store outside git if required) |
| 2 | Apply for PhysioNet credentialed access | Researcher | Approved account |
| 3 | Sign DUA for **MIMIC-IV** (v2.2+) | Researcher | Executed DUA |
| 4 | Request **MIMIC-IV-ECG** (v1.0) access | Researcher | Download authorization |
| 5 | Request **MIMIC-IV-Note** (v2.2) access | Researcher | Download authorization |
| 6 | Install to hospital-local or secure research path (e.g. `data/raw/mimic-iv-ecg/` — **gitignored**, never commit raw) | Researcher | Install manifest with file counts only |
| 7 | Run **Prompt 16B** linkage install audit | Engineering | Updated `multimodal_linkage_audit.json` with verified keys |
| 8 | Define shared target ADR (e.g. encounter-level outcome with temporal window) | Clinical/scientific lead | ADR + protocol doc |
| 9 | Patient-level split manifest | Engineering | `data/manifests/mimic_multimodal_split_seed42.json` (IDs hashed if stored) |

### Expected linkage (documentation-level — verify locally after install)

Per PhysioNet dataset descriptions (cited in `dataset_recommendations.md`):

- ECG records linked to hospital stays via MIMIC identifiers  
- Clinical notes linked via compatible encounter/subject keys  
- Structured vitals/labs/chart events in MIMIC-IV core  

**Do not assume** key names or join SQL until the local schema audit completes.

---

## Alternative paths (only with equal evidence)

| Option | When to use | Requirements |
|--------|-------------|--------------|
| Single-site hospital feed | Real FL deployment | IRB, DUA, on-prem storage, documented cross-modal EMR keys |
| Public linked subset (future) | New PhysioNet release | Same audit pipeline as MIMIC |
| Within-PTB-XL ECG + report | Two-modality NLP+ECG prototype | **Not** a substitute for MedShield triad; separate target protocol |

---

## Explicitly prohibited shortcuts

- Join PTB-XL `patient_id` to `cardio_train.id` by numeric equality  
- Pair NBME notes with random ECG indices  
- Use softmax outputs or synthetic embeddings as “linked” training rows  
- Store raw MIMIC/PHI in PostgreSQL central DB  

---

## Success criteria (before GAT training gate opens)

1. Verified join rate and key semantics documented (no PHI in repo).  
2. Shared target protocol approved (ADR + DATA_CONTRACT addendum if needed).  
3. Patient-independent split manifest published.  
4. Leakage review (temporal + label leakage).  
5. `three_modality_gat_training_readiness` updated to **CONDITIONAL_GO** or **GO** in linkage JSON.  

Until then, remain on **Path A**: independent modality baselines and untrained GAT architecture tests only.
