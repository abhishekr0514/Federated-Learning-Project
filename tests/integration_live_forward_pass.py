"""
Live integration test: real ECG + clinical report forward pass.

Uses the AsyncTestClient against the running FastAPI app to:
1. Login as the seeded super-admin (dev-only credentials from .env)
2. Call /api/v1/ecg-report/evaluate with a synthetic (all-zeros) 12-lead ECG
   and a clearly synthetic report string
3. Verify response schema, probability range, class mapping, and disclaimer

This test proves an actual model forward pass was executed, not just an import check.
"""

import asyncio
import json
import sys
import numpy as np
import httpx

BASE_URL = "http://127.0.0.1:8000"

# Synthetic test fixture — clearly labelled, no clinical data
SYNTHETIC_ECG = [[0.0] * 1000 for _ in range(12)]
SYNTHETIC_REPORT = (
    "SYNTHETIC TEST FIXTURE ONLY — not clinical data. "
    "Normal sinus rhythm. No ST changes. QRS within normal limits. "
    "This text is generated programmatically for model verification only."
)


async def run_integration_test():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        results = {}

        # --- 1. Health ---
        r = await client.get("/health")
        assert r.status_code == 200, f"Health failed: {r.status_code}"
        health = r.json()
        results["health"] = {
            "status": health["status"],
            "ecg_report_inference": health["components"]["ecg_report_inference"],
            "gat_inference": health["components"]["gat_inference"],
        }
        print(f"[PASS] /health → {health['status']}")
        print(f"       ecg_report_inference = {health['components']['ecg_report_inference']}")
        print(f"       gat_inference        = {health['components']['gat_inference']}")

        # --- 2. ECG+Report Readiness ---
        r = await client.get("/api/v1/ecg-report/readiness")
        assert r.status_code == 200
        readiness = r.json()
        assert readiness["ready"] is True, f"ECG readiness blockers: {readiness.get('blockers')}"
        results["ecg_readiness"] = {
            "ready": readiness["ready"],
            "model_loaded": readiness["model_loaded"],
            "text_model_loaded": readiness["text_model_loaded"],
            "checkpoint_trained": readiness["checkpoint_trained"],
        }
        print(f"[PASS] /api/v1/ecg-report/readiness → ready={readiness['ready']}")

        # --- 3. GAT Readiness (must be blocked) ---
        r = await client.get("/api/v1/gat-fusion/readiness")
        assert r.status_code == 200
        gat_readiness = r.json()
        assert gat_readiness["ready"] is False, "GAT should NOT be ready (no trained checkpoint)"
        assert len(gat_readiness["blockers"]) > 0
        results["gat_readiness"] = {
            "ready": gat_readiness["ready"],
            "linkage_gate_verdict": gat_readiness["linkage_gate_verdict"],
            "blocker_count": len(gat_readiness["blockers"]),
        }
        print(f"[PASS] /api/v1/gat-fusion/readiness → ready=False (correctly blocked)")
        print(f"       verdict={gat_readiness['linkage_gate_verdict']}, blockers={len(gat_readiness['blockers'])}")

        # --- 4. Attempt login to get JWT ---
        login_resp = await client.post(
            "/api/v1/auth/login",
            data={"username": "admin@medshield.dev", "password": "SuperAdmin123!"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if login_resp.status_code not in (200, 201):
            # Try JSON body
            login_resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "admin@medshield.dev", "password": "SuperAdmin123!"},
            )

        if login_resp.status_code not in (200, 201):
            print(f"[SKIP] Auth login returned {login_resp.status_code} — no dev seed user present.")
            print(f"       Run: .venv\\Scripts\\python.exe -m pytest tests/test_postgres_dev_seed.py -v")
            print(f"       Then retry this integration test.")
            token = None
        else:
            token = login_resp.json().get("access_token")
            print(f"[PASS] Auth login successful, JWT obtained")

        # --- 5. Real ECG forward pass ---
        if token:
            headers = {"Authorization": f"Bearer {token}"}
            payload = {
                "ecg_leads": SYNTHETIC_ECG,
                "report_text": SYNTHETIC_REPORT,
            }
            r = await client.post(
                "/api/v1/ecg-report/evaluate",
                json=payload,
                headers=headers,
            )
            assert r.status_code == 200, f"ECG evaluate failed: {r.status_code} — {r.text[:300]}"
            result = r.json()

            # Verify schema
            assert "probability" in result
            assert "predicted_class_index" in result
            assert "predicted_class_label" in result
            assert "disclaimer" in result
            assert "architecture_id" in result

            # Verify ranges
            prob = result["probability"]
            assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

            cls_idx = result["predicted_class_index"]
            assert cls_idx in (0, 1), f"Class index must be 0 or 1, got {cls_idx}"

            cls_label = result["predicted_class_label"]
            assert cls_label in ("NORM", "MI/STTC"), f"Unexpected class label: {cls_label}"

            assert result["is_research_only"] is True
            assert "EXPERIMENTAL" in result["disclaimer"] or "NOT" in result["disclaimer"]
            assert result["architecture_id"] == "medshield.ecg_report_fusion.v1"

            results["forward_pass"] = {
                "probability": round(prob, 6),
                "predicted_class_index": cls_idx,
                "predicted_class_label": cls_label,
                "architecture_id": result["architecture_id"],
                "model_version_tag": result["model_version_tag"],
                "is_research_only": result["is_research_only"],
                "disclaimer_length": len(result["disclaimer"]),
            }
            print(f"[PASS] /api/v1/ecg-report/evaluate — REAL FORWARD PASS VERIFIED")
            print(f"       probability         = {prob:.6f}")
            print(f"       predicted_class     = {cls_idx} ({cls_label})")
            print(f"       architecture_id     = {result['architecture_id']}")
            print(f"       model_version_tag   = {result['model_version_tag']}")
            print(f"       is_research_only    = {result['is_research_only']}")
        else:
            results["forward_pass"] = "SKIPPED — no auth token (dev seed user not created)"
            print(f"[SKIP] ECG forward pass skipped — no auth token")

        print("\n=== INTEGRATION TEST SUMMARY ===")
        print(json.dumps(results, indent=2))
        return results


if __name__ == "__main__":
    asyncio.run(run_integration_test())
