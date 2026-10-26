"""GAT-grounded cohort DoWhy workflow (no default cardio_train.csv)."""

from __future__ import annotations

import importlib.util
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.services.causal_design_review import (
    DOCUMENTED_CANDIDATE_DAG,
    DOCUMENTED_CANDIDATE_QUESTION,
    assess_gat_ptbxl_cohort_design,
    causal_estimand_identifiable_for_platform,
)
from backend.app.services.gat_causal_cohort_service import (
    COHORT_ID,
    DESCRIPTIVE_VARIABLES,
    GATCausalCohortService,
    METADATA_VARIABLES,
)

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "RESEARCH-ONLY descriptive cohort and optional DoWhy machinery. Automated refutation does not "
    "validate causal claims. GAT predictions are not treatments or validated diagnoses. "
    "Not for clinical triage."
)

FORBIDDEN_CAUSAL_PREFIXES = ("gat_",)

CANDIDATE_CAUSAL_SPEC = {
    "treatment": "sex",
    "outcome": "research_proxy_label",
    "confounders": ["age_years"],
    "causal_graph": DOCUMENTED_CANDIDATE_DAG,
    "causal_question": DOCUMENTED_CANDIDATE_QUESTION,
}


def _dowhy_available() -> bool:
    return importlib.util.find_spec("dowhy") is not None


def _metadata_variables_available(cohort_mode: str, metadata_aligned: bool) -> Dict[str, str]:
    if cohort_mode == "DESCRIPTIVE" and metadata_aligned:
        return dict(METADATA_VARIABLES)
    return {}


class CausalAnalysisService:
    @staticmethod
    def get_capability_status() -> str:
        return CausalAnalysisService.get_readiness()["summary"]

    @staticmethod
    def _compute_readiness_flags(build) -> Dict[str, Any]:
        cohort_available = build.ok
        metadata_aligned = getattr(build, "metadata_aligned", False)
        design_review = assess_gat_ptbxl_cohort_design(
            metadata_aligned=metadata_aligned,
            provenance=build.provenance if cohort_available else {},
        )
        pipeline = (
            settings.CAUSAL_ANALYSIS_ENABLED
            and _dowhy_available()
            and GATCausalCohortService.gat_checkpoint_ready()
            and cohort_available
        )
        estimand_identifiable = causal_estimand_identifiable_for_platform(
            metadata_aligned=metadata_aligned,
            provenance=build.provenance if cohort_available else {},
        )
        return {
            "causal_pipeline_available": pipeline,
            "causal_estimand_identifiable": estimand_identifiable,
            "causal_claim_validated": False,
            "causal_design_review": design_review,
            "metadata_aligned": metadata_aligned,
        }

    @staticmethod
    def get_readiness(*, build_cohort: bool = False) -> Dict[str, Any]:
        blockers: List[str] = []
        if not settings.CAUSAL_ANALYSIS_ENABLED:
            blockers.append("CAUSAL_ANALYSIS_ENABLED is false")
        if not _dowhy_available():
            blockers.append("dowhy package not installed (install medshield-fl[ml])")

        gat_ready = GATCausalCohortService.gat_checkpoint_ready()
        if not gat_ready:
            blockers.append("GAT checkpoint not ready under FUSION_MODEL_DIR")

        build = (
            GATCausalCohortService.build_cohort()
            if build_cohort
            else GATCausalCohortService.probe_cohort_availability()
        )
        flags = CausalAnalysisService._compute_readiness_flags(build)
        cohort_available = build.ok
        if not cohort_available:
            blockers.extend(build.blockers)

        cohort_mode = build.cohort_mode if cohort_available else "UNAVAILABLE"

        if not settings.CAUSAL_ANALYSIS_ENABLED:
            summary = "DISABLED_BY_CONFIGURATION"
        elif not _dowhy_available():
            summary = "MISSING_DEPENDENCY"
        elif not gat_ready:
            summary = "GAT_MODEL_NOT_READY"
        elif not cohort_available:
            summary = "COHORT_NOT_AVAILABLE"
        elif not flags["causal_estimand_identifiable"]:
            summary = "CAUSAL_DESIGN_REQUIRED"
        else:
            summary = "AVAILABLE"

        reason_parts: List[str] = []
        if blockers:
            reason_parts.append("; ".join(blockers))
        elif summary == "CAUSAL_DESIGN_REQUIRED":
            reason_parts.append(flags["causal_design_review"]["rationale"])
        else:
            reason_parts.append("Approved causal design (not enabled in default platform policy).")

        meta_vars = list(_metadata_variables_available(cohort_mode, flags["metadata_aligned"]).keys())

        return {
            "summary": summary,
            "reason": " ".join(reason_parts),
            "blockers": blockers,
            "cohort_id": COHORT_ID,
            "cohort_available": cohort_available,
            "cohort_mode": cohort_mode,
            "metadata_aligned": flags["metadata_aligned"],
            "causal_pipeline_available": flags["causal_pipeline_available"],
            "causal_estimand_identifiable": flags["causal_estimand_identifiable"],
            "causal_claim_validated": flags["causal_claim_validated"],
            "causal_estimation_supported": False,
            "causal_effect_estimation_allowed": settings.CAUSAL_EFFECT_ESTIMATION_ALLOWED,
            "causal_design_review": flags["causal_design_review"],
            "descriptive_statistics": build.descriptive_stats if cohort_available else {},
            "gat_checkpoint_ready": gat_ready,
            "dowhy_installed": _dowhy_available(),
            "analysis_enabled": settings.CAUSAL_ANALYSIS_ENABLED,
            "descriptive_variables": list(DESCRIPTIVE_VARIABLES.keys()),
            "metadata_variables": meta_vars,
            "approved_variables": [],
            "disclaimer": DISCLAIMER,
            "provenance": build.provenance if cohort_available else {},
        }

    @staticmethod
    def get_descriptive_summary() -> Dict[str, Any]:
        build = GATCausalCohortService.build_cohort()
        if not build.ok:
            raise ValueError("GAT research cohort is not available.")
        return {
            "cohort_id": COHORT_ID,
            "descriptive_statistics": build.descriptive_stats,
            "provenance": build.provenance,
            "metadata_aligned": build.metadata_aligned,
        }

    @staticmethod
    def list_cohorts() -> Dict[str, Any]:
        readiness = CausalAnalysisService.get_readiness()
        meta = GATCausalCohortService.list_cohort_metadata()
        return {
            "status": readiness["summary"],
            "cohorts": [meta],
            "readiness": readiness,
        }

    @staticmethod
    def _load_cohort_frame():
        build = GATCausalCohortService.build_cohort()
        if not build.ok or build.frame is None:
            raise ValueError("GAT research cohort is not available.")
        return build, build.frame

    @staticmethod
    def _pick_estimator(*, treatment_series, outcome_series) -> str:
        t_unique = treatment_series.nunique()
        o_unique = outcome_series.nunique()
        if t_unique == 2 and o_unique == 2:
            return "backdoor.generalized_linear_model"
        if t_unique == 2:
            return "backdoor.linear_regression"
        return "backdoor.linear_regression"

    @staticmethod
    def run_cohort_analysis(
        *,
        treatment: str,
        outcome: str,
        confounders: List[str],
        effect_modifiers: Optional[List[str]] = None,
        causal_graph: Optional[str] = None,
    ) -> Dict[str, Any]:
        readiness = CausalAnalysisService.get_readiness()
        if not settings.CAUSAL_EFFECT_ESTIMATION_ALLOWED:
            raise RuntimeError(
                "Causal effect estimation is disabled (CAUSAL_EFFECT_ESTIMATION_ALLOWED=false). "
                "Descriptive cohort statistics remain available via GET /causal/cohorts."
            )
        if not readiness["causal_pipeline_available"]:
            raise RuntimeError(readiness["reason"])

        approved = _metadata_variables_available("DESCRIPTIVE", readiness["metadata_aligned"])
        for name in (treatment, outcome, *confounders, *(effect_modifiers or [])):
            if name.startswith(FORBIDDEN_CAUSAL_PREFIXES):
                raise ValueError(f"Model-derived variable '{name}' cannot be used in causal estimation.")
            if name not in approved:
                raise ValueError(f"Variable '{name}' is not in the metadata variable contract.")

        if treatment == outcome:
            raise ValueError("Treatment and outcome must differ.")

        build, df = CausalAnalysisService._load_cohort_frame()
        df = df.dropna(subset=[treatment, outcome, *confounders])
        if len(df) < settings.CAUSAL_ANALYSIS_MIN_SAMPLE_SIZE:
            raise ValueError(
                f"Insufficient eligible observations ({len(df)} < {settings.CAUSAL_ANALYSIS_MIN_SAMPLE_SIZE})."
            )
        if df[treatment].nunique() < 2:
            raise ValueError("Treatment variable lacks variation in the eligible cohort.")
        if df[outcome].nunique() < 2:
            raise ValueError("Outcome variable lacks variation in the eligible cohort.")

        graph = (causal_graph or CANDIDATE_CAUSAL_SPEC["causal_graph"]).strip()
        method = CausalAnalysisService._pick_estimator(
            treatment_series=df[treatment],
            outcome_series=df[outcome],
        )

        from dowhy import CausalModel

        model = CausalModel(
            data=df[[treatment, outcome, *confounders]],
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders if "digraph" not in graph else None,
            effect_modifiers=effect_modifiers or None,
            graph=graph if "digraph" in graph else None,
        )

        design_review = readiness["causal_design_review"]

        try:
            identified = model.identify_effect(proceed_when_unidentifiable=False)
        except Exception as exc:
            return {
                "status": "NOT_IDENTIFIABLE",
                "message": str(exc)[:500],
                "sample_size": len(df),
                "cohort_id": COHORT_ID,
                "causal_claim_validated": False,
                "causal_estimand_identifiable": readiness["causal_estimand_identifiable"],
                "disclaimer": DISCLAIMER,
            }

        try:
            estimate = model.estimate_effect(identified, method_name=method, test_significance=True)
        except Exception:
            if method != "backdoor.linear_regression":
                try:
                    estimate = model.estimate_effect(
                        identified,
                        method_name="backdoor.linear_regression",
                        test_significance=True,
                    )
                    method = "backdoor.linear_regression"
                except Exception as exc2:
                    logger.warning("Causal estimation failed: %s", type(exc2).__name__)
                    return {
                        "status": "ESTIMATION_FAILED",
                        "message": "Effect estimation failed for the specified model and cohort.",
                        "sample_size": len(df),
                        "cohort_id": COHORT_ID,
                        "causal_claim_validated": False,
                        "disclaimer": DISCLAIMER,
                    }
            else:
                return {
                    "status": "ESTIMATION_FAILED",
                    "message": "Effect estimation failed for the specified model and cohort.",
                    "sample_size": len(df),
                    "cohort_id": COHORT_ID,
                    "causal_claim_validated": False,
                    "disclaimer": DISCLAIMER,
                }

        refutation_results: List[Dict[str, Any]] = []
        refute_ok = False
        try:
            refute = model.refute_estimate(estimate, method_name="random_common_cause")
            refute_ok = True
            refutation_results.append(
                {
                    "method": "random_common_cause",
                    "new_effect": getattr(refute, "new_effect", None),
                    "refutation_result": str(getattr(refute, "refutation_result", refute)),
                }
            )
        except Exception as exc:
            refutation_results.append({"method": "random_common_cause", "error": type(exc).__name__})

        return {
            "status": "ESTIMATED",
            "cohort_id": COHORT_ID,
            "cohort_mode": build.cohort_mode,
            "cohort_version": build.provenance.get("cohort_version"),
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "effect_modifiers": effect_modifiers or [],
            "estimand": str(identified),
            "effect_estimate": float(estimate.value),
            "estimation_method": method,
            "sample_size": len(df),
            "refutation": refutation_results,
            "refutation_passed": refute_ok,
            "causal_graph_supplied": bool(causal_graph),
            "causal_question": design_review.get("candidate_question"),
            "assumptions": design_review.get("limitations"),
            "causal_pipeline_available": True,
            "causal_estimand_identifiable": readiness["causal_estimand_identifiable"],
            "causal_claim_validated": False,
            "provenance": build.provenance,
            "disclaimer": DISCLAIMER,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
