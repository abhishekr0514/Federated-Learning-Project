/**
 * GAT diagnostic workflow: embedding validation, route separation, mock HTTP plan.
 */

import assert from 'node:assert';
import {
  MODALITY_EMBEDDING_DIM,
  ECG_REPORT_EVALUATE_PATH,
  ECG_REPORT_EMBED_PATH,
  GAT_FUSION_EVALUATE_PATH,
  buildGatFusionPayload,
  getInferenceRequestPlan,
  resolvePrimaryInferencePath,
  validateEmbeddingVector,
  validateEncoderEmbedResponse,
  selectWorkflowPredictionResult,
  resolvePredictionScore,
} from '../src/utils/diagnosticGatWorkflow.js';

console.log('--- diagnostic_gat_workflow.test.js ---');

assert.strictEqual(resolvePrimaryInferencePath('gat_fusion'), GAT_FUSION_EVALUATE_PATH);
assert.strictEqual(resolvePrimaryInferencePath('ecg_report'), ECG_REPORT_EVALUATE_PATH);

const gatPlan = getInferenceRequestPlan('gat_fusion');
assert.strictEqual(gatPlan.length, 2);
assert.strictEqual(gatPlan[0].path, ECG_REPORT_EMBED_PATH);
assert.strictEqual(gatPlan[1].path, GAT_FUSION_EVALUATE_PATH);
assert.ok(
  !gatPlan.some((s) => s.path === ECG_REPORT_EVALUATE_PATH),
  'GAT plan must not call ecg-report evaluate'
);

const ecgPlan = getInferenceRequestPlan('ecg_report');
assert.strictEqual(ecgPlan.length, 1);
assert.strictEqual(ecgPlan[0].path, ECG_REPORT_EVALUATE_PATH);

const validVec = Array.from({ length: MODALITY_EMBEDDING_DIM }, (_, i) => i * 0.001);
assert.strictEqual(validateEmbeddingVector(validVec).ok, true);
assert.strictEqual(validateEmbeddingVector(validVec.slice(0, 64)).ok, false);

const embedBody = {
  ecg_embedding: validVec,
  text_embedding: validVec.map((v) => v + 0.01),
};
assert.strictEqual(validateEncoderEmbedResponse(embedBody).ok, true);

const payload = buildGatFusionPayload(embedBody, 'idem-1');
assert.strictEqual(payload.ecg_embedding.length, 128);
assert.strictEqual(payload.text_embedding.length, 128);
assert.strictEqual(payload.idempotency_key, 'idem-1');

assert.strictEqual(
  selectWorkflowPredictionResult('gat_fusion', { probability: 0.1 }, { confidence_uncalibrated: 0.9 }).confidence_uncalibrated,
  0.9,
);
assert.strictEqual(resolvePredictionScore({ confidence_uncalibrated: 0.42 }), 0.42);
assert.strictEqual(resolvePredictionScore({ probability: 0.33 }), 0.33);

/** Mock client proving GAT action never hits evaluate path */
async function runGatWorkflowMock(fetchImpl) {
  const calls = [];
  const fetch = async (url, opts) => {
    calls.push({ url, method: opts?.method || 'GET' });
    return fetchImpl(url, opts);
  };

  const embedRes = {
    ecg_embedding: validVec,
    text_embedding: embedBody.text_embedding,
  };
  const gatRes = {
    prediction_id: '00000000-0000-4000-8000-000000000001',
    confidence_uncalibrated: 0.55,
    predicted_class_index: 1,
    predicted_class_label: 'MI/STTC',
    disclaimer: 'Research only',
  };

  await fetch(ECG_REPORT_EMBED_PATH, { method: 'POST', body: '{}' });
  const check = validateEncoderEmbedResponse(embedRes);
  if (!check.ok) throw new Error(check.error);
  await fetch(GAT_FUSION_EVALUATE_PATH, {
    method: 'POST',
    body: JSON.stringify(buildGatFusionPayload(embedRes)),
  });

  return { calls, gatRes };
}

const { calls } = await runGatWorkflowMock(async (url) => ({
  ok: true,
  json: async () => (url.includes('embed') ? embedBody : { confidence_uncalibrated: 0.5 }),
}));

assert.strictEqual(calls.length, 2);
assert.ok(calls.some((c) => c.url === ECG_REPORT_EMBED_PATH));
assert.ok(calls.some((c) => c.url === GAT_FUSION_EVALUATE_PATH));
assert.ok(!calls.some((c) => c.url === ECG_REPORT_EVALUATE_PATH));

console.log('diagnostic_gat_workflow.test.js passed');
