"""Print inference readiness and setup steps (safe for logs; no secrets)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.diagnosis_inference_service import DiagnosisInferenceService  # noqa: E402


def main() -> int:
    report = DiagnosisInferenceService.get_readiness()
    print(json.dumps(report, indent=2))
    return 0 if report.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
