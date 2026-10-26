"""
Scientific review of whether a causal effect claim is justified for the GAT PTB-XL cohort.

DoWhy completing identify/estimate/refute is **not** the same as a defensible causal design.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Hypothetical observational spec kept for documentation only — not auto-approved.
DOCUMENTED_CANDIDATE_QUESTION = (
    "Candidate (not approved): cross-sectional association of recorded sex with PTB-XL SCP-derived "
    "proxy label (NORM vs MI/STTC) among train-split ECG+report pairs, adjusting for age."
)

DOCUMENTED_CANDIDATE_DAG = """
digraph {
    age_years -> research_proxy_label;
    sex -> research_proxy_label;
}
"""


def assess_gat_ptbxl_cohort_design(*, metadata_aligned: bool, provenance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns review artifacts. `causal_effect_estimation_permitted` stays False unless
    CAUSAL_EFFECT_ESTIMATION_ALLOWED is set explicitly for controlled research/testing.
    """
    limitations: List[str] = [
        "Cross-sectional ECG records: no temporal ordering from sex assignment to outcome onset.",
        "Outcome (research_proxy_label) is derived from SCP superclasses on the same ECG; not independent "
        "of waveform/report content and not a clinical endpoint adjudicated separately from the recording.",
        "Sex is a static attribute, not a manipulable treatment; associational language only if ever estimated.",
        "Selection: cohort requires non-empty machine report and passing privacy mask — induces selection bias "
        "relative to all PTB-XL labeled records.",
        "Plausible unmeasured confounding: comorbidities, medication use, indication for ECG, lifestyle factors.",
        "Documented DAG omits edges from latent cardiovascular risk to both sex and SCP label assignment.",
        "Positivity/variation in sex and label is necessary but not sufficient for causal identifiability.",
        "Refutation tests in DoWhy do not validate causal claims.",
    ]
    if not metadata_aligned:
        limitations.append("PTB-XL row alignment or demographics join incomplete — metadata-backed covariates unreliable.")

    alignment = provenance.get("alignment") or {}
    if alignment.get("aligned") is False:
        limitations.append("Embedding cache row count does not match train partition — cohort alignment failed.")

    defensible = False
    rationale = (
        "Sex → NORM vs MI/STTC on this GAT train cohort is **not** treated as a scientifically defensible "
        "causal effect question for automated estimation. The platform exposes a descriptive GAT-derived cohort "
        "and optional DoWhy machinery only under explicit research override; successful DoWhy execution does "
        "not set causal_claim_validated."
    )

    return {
        "candidate_question": DOCUMENTED_CANDIDATE_QUESTION,
        "candidate_dag": DOCUMENTED_CANDIDATE_DAG,
        "target_population": (
            "PTB-XL train-split ECG+report pairs included in the embedding cache used for GAT training "
            "(strat folds 1–7 per SPLIT_POLICY_V1); not all PTB-XL patients or all indications."
        ),
        "temporal_assumption": "None verified — single time-point ECG and contemporaneous SCP/report labels.",
        "treatment_definition": "Recorded sex field (not an intervention).",
        "outcome_definition": "Protocol proxy label from SCP superclasses (NORM vs MI/STTC), independent of GAT outputs.",
        "causal_question_defensible": defensible,
        "rationale": rationale,
        "limitations": limitations,
        "positivity_note": "Binary sex and label variation may hold in aligned samples; not proof of identifiability.",
        "selection_bias_note": "Report availability and masking exclusions restrict generalizability.",
    }


def causal_estimand_identifiable_for_platform(
    *, metadata_aligned: bool, provenance: Dict[str, Any]
) -> bool:
    """Platform policy: never True by default from data variation alone."""
    review = assess_gat_ptbxl_cohort_design(metadata_aligned=metadata_aligned, provenance=provenance)
    return bool(review["causal_question_defensible"])
