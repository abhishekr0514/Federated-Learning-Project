"""Non-secret end-to-end API verification for MedShield-FL integration audit."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
FRONTEND = "http://localhost:5173"
DEV_PASSWORD = "ValidPassword123!"


async def main() -> int:
    results: dict[str, str] = {}
    async with httpx.AsyncClient(base_url=BASE, timeout=120.0) as client:
        # Health
        try:
            h = (await client.get("/health")).json()
            results["backend_health"] = "PASS" if h.get("status") == "healthy" else "FAIL"
            results["db_component"] = h.get("components", {}).get("database", "unknown")
        except Exception as exc:
            results["backend_health"] = f"FAIL ({type(exc).__name__})"
            print(json.dumps(results, indent=2))
            return 1

        # Login envelope
        for label, email in (
            ("superadmin_login", "admin@medshield.org"),
            ("hospital_admin_login", "admin_a@metro.org"),
        ):
            bad = await client.post("/api/v1/auth/login", json={"email": email, "password": "wrong"})
            ok = await client.post("/api/v1/auth/login", json={"email": email, "password": DEV_PASSWORD})
            if bad.status_code != 401 or ok.status_code != 200:
                results[label] = f"FAIL bad={bad.status_code} ok={ok.status_code}"
                continue
            body = ok.json()
            if "data" not in body or "access_token" not in body.get("data", {}):
                results[label] = "FAIL missing envelope.data.access_token"
                continue
            tok = body["data"]["access_token"]
            me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
            results[label] = "PASS" if me.status_code == 200 else f"FAIL me={me.status_code}"
            if label == "superadmin_login" and me.status_code == 200:
                super_token = tok

        # Protected hospital route
        ha = await client.post(
            "/api/v1/auth/login", json={"email": "admin_a@metro.org", "password": DEV_PASSWORD}
        )
        ha_tok = ha.json()["data"]["access_token"]
        nodes = await client.get(
            "/api/v1/hospital/accessible-nodes", headers={"Authorization": f"Bearer {ha_tok}"}
        )
        hosp_id = None
        if nodes.status_code == 200:
            hospitals = nodes.json().get("hospitals") or []
            if hospitals:
                hosp_id = hospitals[0].get("id")
        ov_path = "/api/v1/hospital/overview"
        if hosp_id:
            ov_path = f"{ov_path}?hospital_id={hosp_id}"
        ov = await client.get(ov_path, headers={"Authorization": f"Bearer {ha_tok}"})
        results["hospital_overview"] = "PASS" if ov.status_code == 200 else f"FAIL {ov.status_code}"

        # ECG readiness + inference (research, no patient)
        ecg_r = (await client.get("/api/v1/ecg-report/readiness")).json()
        results["ecg_readiness"] = "PASS" if ecg_r.get("ready") else f"FAIL blockers={ecg_r.get('blockers')}"

        # Minimal valid payload: 12x1000 zeros + short masked text
        signal = [[0.0] * 1000 for _ in range(12)]
        payload = {
            "ecg_leads": signal,
            "report_text": "Patient stable. No acute ST changes. NORM.",
        }
        inf = await client.post(
            "/api/v1/ecg-report/evaluate",
            headers={"Authorization": f"Bearer {ha_tok}"},
            json=payload,
        )
        if inf.status_code == 200:
            body = inf.json()
            has_pred = "probability" in body or "probability" in str(body.get("data", {}))
            disclaimer = body.get("disclaimer") or body.get("data", {}).get("disclaimer")
            results["ecg_inference"] = (
                "PASS"
                if has_pred and disclaimer
                else f"FAIL fields={list(body.keys())[:8]}"
            )
        else:
            results["ecg_inference"] = f"FAIL HTTP {inf.status_code} {inf.text[:200]}"

        gat_r = (await client.get("/api/v1/gat-fusion/readiness")).json()
        results["gat_readiness"] = "PASS" if gat_r.get("ready") else f"NOT_READY blockers={gat_r.get('blockers')}"
        if gat_r.get("ready"):
            embed = await client.post(
                "/api/v1/ecg-report/embed",
                headers={"Authorization": f"Bearer {ha_tok}"},
                json=payload,
            )
            if embed.status_code != 200:
                results["gat_inference"] = f"FAIL embed HTTP {embed.status_code}"
            else:
                emb = embed.json()
                gat_body = {
                    "ecg_embedding": emb["ecg_embedding"],
                    "text_embedding": emb["text_embedding"],
                }
                gat_eval = await client.post(
                    "/api/v1/gat-fusion/evaluate",
                    headers={"Authorization": f"Bearer {ha_tok}"},
                    json=gat_body,
                )
                if gat_eval.status_code == 200:
                    gb = gat_eval.json()
                    results["gat_inference"] = (
                        "PASS"
                        if gb.get("disclaimer") and "confidence_uncalibrated" in gb
                        else "FAIL missing disclaimer/confidence"
                    )
                else:
                    results["gat_inference"] = f"FAIL {gat_eval.status_code}"
            results["gat_encoder_embed"] = "PASS" if embed.status_code == 200 else f"FAIL {embed.status_code}"
        else:
            results["gat_inference"] = "SKIP (not ready)"

        priv = await client.post(
            "/api/v1/privacy/mask-text",
            headers={"Authorization": f"Bearer {ha_tok}"},
            json={"raw_text": "John Doe MRN 12345 chest pain"},
        )
        results["privacy_masking"] = "PASS" if priv.status_code == 200 else f"FAIL {priv.status_code}"

        caps = await client.get("/api/v1/capabilities", headers={"Authorization": f"Bearer {super_token}"})
        if caps.status_code == 200:
            c = caps.json()["capabilities"]
            results["capabilities_ecg"] = c.get("ecg_report_inference", "?")
            results["capabilities_gat"] = c.get("multimodal_gat_fusion", "?")
            results["capabilities_flower"] = c.get("flower_federated", "?")
            results["capabilities_dice"] = c.get("dice_counterfactuals", "?")
        else:
            results["capabilities"] = f"FAIL {caps.status_code}"

        fed = await client.post("/api/v1/federated/start", headers={"Authorization": f"Bearer {super_token}"})
        results["federated_start"] = f"HTTP {fed.status_code} (expect non-success if not orchestrated)"

    # Frontend probe
    try:
        async with httpx.AsyncClient(timeout=5.0) as fe:
            fr = await fe.get(FRONTEND)
            results["frontend_5173"] = "PASS" if fr.status_code == 200 else f"FAIL {fr.status_code}"
    except Exception as exc:
        results["frontend_5173"] = f"FAIL ({type(exc).__name__})"

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
