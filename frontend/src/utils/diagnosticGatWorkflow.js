/** GAT fusion workflow helpers (encoder → fusion API, no fabricated embeddings). */

export const MODALITY_EMBEDDING_DIM = 128;

export const ECG_REPORT_EVALUATE_PATH = '/api/v1/ecg-report/evaluate';
export const ECG_REPORT_EMBED_PATH = '/api/v1/ecg-report/embed';
export const GAT_FUSION_EVALUATE_PATH = '/api/v1/gat-fusion/evaluate';

/**
 * @param {'ecg_report'|'gat_fusion'} workflow
 */
export function resolvePrimaryInferencePath(workflow) {
  if (workflow === 'gat_fusion') {
    return GAT_FUSION_EVALUATE_PATH;
  }
  return ECG_REPORT_EVALUATE_PATH;
}

export function validateEmbeddingVector(values, label = 'embedding') {
  if (!Array.isArray(values)) {
    return { ok: false, error: `${label} must be an array` };
  }
  if (values.length !== MODALITY_EMBEDDING_DIM) {
    return {
      ok: false,
      error: `${label} must have ${MODALITY_EMBEDDING_DIM} dimensions, got ${values.length}`,
    };
  }
  for (let i = 0; i < values.length; i += 1) {
    const n = Number(values[i]);
    if (!Number.isFinite(n)) {
      return { ok: false, error: `${label} contains non-finite value at index ${i}` };
    }
  }
  return { ok: true };
}

export function validateEncoderEmbedResponse(body) {
  if (!body || typeof body !== 'object') {
    return { ok: false, error: 'Encoder response missing' };
  }
  const ecgCheck = validateEmbeddingVector(body.ecg_embedding, 'ecg_embedding');
  if (!ecgCheck.ok) {
    return ecgCheck;
  }
  const textCheck = validateEmbeddingVector(body.text_embedding, 'text_embedding');
  if (!textCheck.ok) {
    return textCheck;
  }
  return { ok: true };
}

export function buildGatFusionPayload(embedBody, idempotencyKey) {
  const check = validateEncoderEmbedResponse(embedBody);
  if (!check.ok) {
    throw new Error(check.error);
  }
  const payload = {
    ecg_embedding: embedBody.ecg_embedding,
    text_embedding: embedBody.text_embedding,
  };
  if (idempotencyKey && String(idempotencyKey).trim()) {
    payload.idempotency_key = String(idempotencyKey).trim();
  }
  return payload;
}

/**
 * Ordered HTTP steps for a workflow (for tests and orchestration).
 * @param {'ecg_report'|'gat_fusion'} workflow
 */
export function getInferenceRequestPlan(workflow) {
  if (workflow === 'gat_fusion') {
    return [
      { method: 'POST', path: ECG_REPORT_EMBED_PATH, purpose: 'encoder' },
      { method: 'POST', path: GAT_FUSION_EVALUATE_PATH, purpose: 'gat_fusion' },
    ];
  }
  return [{ method: 'POST', path: ECG_REPORT_EVALUATE_PATH, purpose: 'ecg_report_fusion' }];
}

export function selectWorkflowPredictionResult(workflow, ecgReportResult, gatFusionResult) {
  return workflow === 'gat_fusion' ? gatFusionResult : ecgReportResult;
}

export function resolvePredictionScore(result) {
  if (!result || typeof result !== 'object') {
    return null;
  }
  if (typeof result.confidence_uncalibrated === 'number') {
    return result.confidence_uncalibrated;
  }
  if (typeof result.probability === 'number') {
    return result.probability;
  }
  return null;
}
