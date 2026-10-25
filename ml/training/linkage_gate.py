"""Gate multimodal fusion training on documented linkage audit verdicts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_PATH = REPO_ROOT / "docs" / "data-audit" / "multimodal_linkage_audit.json"

REQUIRED_VERDICT_KEY = "three_modality_gat_training_readiness"
ALLOWED_VERDICT = "CONDITIONAL_GO"


@dataclass(frozen=True)
class LinkageGateResult:
    allowed: bool
    verdict: str
    evidence: str
    audit_path: Path
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "audit_path": str(self.audit_path.relative_to(REPO_ROOT)),
            "blockers": list(self.blockers),
        }


def load_linkage_audit(audit_path: Path | None = None) -> dict[str, Any]:
    path = audit_path or DEFAULT_AUDIT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Linkage audit not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_three_modality_gate(audit_path: Path | None = None) -> LinkageGateResult:
    """Return whether supervised three-modality GAT training is permitted by project audit."""
    path = audit_path or DEFAULT_AUDIT_PATH
    audit = load_linkage_audit(path)
    matrix = audit.get("decision_matrix", {})
    entry = matrix.get(REQUIRED_VERDICT_KEY, {})
    verdict = str(entry.get("verdict", "BLOCKED"))
    evidence = str(entry.get("evidence", ""))

    blockers: list[str] = []
    if verdict == "BLOCKED":
        blockers.append(f"{REQUIRED_VERDICT_KEY}: BLOCKED — {evidence}")
    elif verdict != ALLOWED_VERDICT:
        blockers.append(f"Unexpected verdict {verdict!r}; expected BLOCKED or {ALLOWED_VERDICT}")

    allowed = verdict == ALLOWED_VERDICT

    return LinkageGateResult(
        allowed=allowed,
        verdict=verdict,
        evidence=evidence,
        audit_path=path,
        blockers=blockers if not allowed else [],
    )


def assert_three_modality_training_allowed(audit_path: Path | None = None) -> LinkageGateResult:
    result = evaluate_three_modality_gate(audit_path)
    if not result.allowed:
        raise RuntimeError(
            "Three-modality GAT training is BLOCKED by linkage audit. "
            f"Verdict={result.verdict}. Blockers: {'; '.join(result.blockers)}"
        )
    return result
