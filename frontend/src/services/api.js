import { parseAuthTokenEnvelope } from '../utils/integrationStatus.js';

/**
 * Centralized API Client for MedShield-FL Frontend
 * Handles HTTP requests, Bearer token attachment, X-Request-ID propagation,
 * FormData upload handling, timeout aborts, token refresh, and structured errors.
 */

export const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) || '';

let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

/**
 * Generate a unique v4 UUID for X-Request-ID header
 */
export function generateUUID() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Perform custom fetch with automatic auth header, timeout, and refresh handling
 */
export async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const accessToken = typeof sessionStorage !== 'undefined' ? sessionStorage.getItem('medshield_access_token') : null;

  const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;

  const headers = {
    'X-Request-ID': generateUUID(),
    ...(options.headers || {}),
  };

  if (!isFormData && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  if (accessToken && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  // Timeout handling (default 45s for deep ML inference)
  const timeoutMs = options.timeout || 45000;
  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timeoutId = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;

  const config = {
    ...options,
    headers,
    signal: options.signal || (controller ? controller.signal : undefined),
  };

  try {
    let response = await fetch(url, config);

    if (timeoutId) clearTimeout(timeoutId);

    // If HTTP 401 Unauthorized occurs on protected route, attempt refresh
    if (response.status === 401 && !endpoint.includes('/auth/login') && !endpoint.includes('/auth/refresh')) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((newAccessToken) => {
          config.headers['Authorization'] = `Bearer ${newAccessToken}`;
          return fetch(url, config).then(handleResponse);
        });
      }

      isRefreshing = true;
      const refreshToken = typeof sessionStorage !== 'undefined' ? sessionStorage.getItem('medshield_refresh_token') : null;

      if (!refreshToken) {
        isRefreshing = false;
        processQueue(new Error('No refresh token available'), null);
        throw new ApiError('Session expired. Please log in again.', 401, 'UNAUTHENTICATED');
      }

      try {
        const refreshResponse = await fetch(`${API_BASE_URL}/api/v1/auth/refresh`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Request-ID': generateUUID(),
          },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (!refreshResponse.ok) {
          throw new Error('Refresh failed');
        }

        const refreshData = await refreshResponse.json();
        const { access_token: newAccess, refresh_token: newRefresh } = parseAuthTokenEnvelope(refreshData);

        if (typeof sessionStorage !== 'undefined') {
          sessionStorage.setItem('medshield_access_token', newAccess);
          sessionStorage.setItem('medshield_refresh_token', newRefresh);
        }

        isRefreshing = false;
        processQueue(null, newAccess);

        // Retry original request with new access token
        config.headers['Authorization'] = `Bearer ${newAccess}`;
        response = await fetch(url, config);
      } catch (refreshErr) {
        isRefreshing = false;
        processQueue(refreshErr, null);
        if (typeof sessionStorage !== 'undefined') {
          sessionStorage.removeItem('medshield_access_token');
          sessionStorage.removeItem('medshield_refresh_token');
        }
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('medshield:auth_failed'));
        }
        throw new ApiError('Session expired. Please login again.', 401, 'UNAUTHENTICATED');
      }
    }

    return await handleResponse(response);
  } catch (error) {
    if (timeoutId) clearTimeout(timeoutId);
    if (error instanceof ApiError) {
      throw error;
    }
    if (error.name === 'AbortError') {
      throw new ApiError(`Request timed out after ${timeoutMs / 1000}s`, 408, 'TIMEOUT_ERROR');
    }
    throw new ApiError(error.message || 'Network connection failure', 0, 'NETWORK_ERROR');
  }
}

/**
 * Standardize response envelope extraction
 */
async function handleResponse(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    // Empty or non-JSON body
  }

  if (!response.ok) {
    const errorMsg = body?.error?.message || body?.message || body?.detail || `HTTP ${response.status} Error`;
    const errorCode = body?.error?.code || 'UNKNOWN_ERROR';
    const metadata = body?.metadata || {};
    throw new ApiError(errorMsg, response.status, errorCode, metadata);
  }

  return body;
}

/**
 * Standardized API Error Class
 */
export class ApiError extends Error {
  constructor(message, status, code = 'GENERIC_ERROR', metadata = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.metadata = metadata;
  }
}

// ---------------------------------------------------------------------------
// Dedicated Domain API Helpers
// ---------------------------------------------------------------------------

/** Health and Readiness */
export const apiGetHealth = () => apiRequest('/health');
export const apiGetPlatformCapabilities = () => apiRequest('/api/v1/capabilities', { timeout: 30000 });
export const apiGetEcgReportReadiness = ({ verify = false } = {}) => {
  const q = verify ? '?verify_forward_pass=true' : '';
  return apiRequest(`/api/v1/ecg-report/readiness${q}`);
};

/** Encoder embeddings for GAT (server-side privacy mask; no raw text in response). */
export const apiEncodeEcgReportEmbeddings = (payload) =>
  apiRequest('/api/v1/ecg-report/embed', {
    method: 'POST',
    body: JSON.stringify({
      ecg_leads: payload.ecg_leads,
      report_text: payload.report_text,
    }),
    timeout: 120000,
  });
export const apiGetGatFusionReadiness = () => apiRequest('/api/v1/gat-fusion/readiness');

/** Privacy Masking Validation */
export const apiValidatePrivacyMasking = (rawText) =>
  apiRequest('/api/v1/privacy/mask-text', {
    method: 'POST',
    body: JSON.stringify({ raw_text: rawText }),
  });
export const apiGetPrivacyStatus = () => apiRequest('/api/v1/privacy/status');

/** Inference Evaluation */
export const apiEvaluateEcgReport = (payload, patientId = null) => {
  const query = patientId ? `?patient_id=${encodeURIComponent(patientId)}` : '';
  return apiRequest(`/api/v1/ecg-report/evaluate${query}`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
};

export const apiGetCausalReadiness = () => apiRequest('/api/v1/causal/readiness');
export const apiListCausalCohorts = () => apiRequest('/api/v1/causal/cohorts');
export const apiRunCausalAnalysis = (payload) =>
  apiRequest('/api/v1/causal/analyze', {
    method: 'POST',
    body: JSON.stringify(payload),
    timeout: 180000,
  });

export const apiEvaluateGatFusion = (payload, patientId = null) => {
  const query = patientId ? `?patient_id=${encodeURIComponent(patientId)}` : '';
  return apiRequest(`/api/v1/gat-fusion/evaluate${query}`, {
    method: 'POST',
    body: JSON.stringify(payload),
    timeout: 120000,
  });
};

/** Federated Learning */
export const apiGetLatestTabularExperiment = (runId = null) => {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : '';
  return apiRequest(`/api/v1/federated/experiments/tabular/latest${query}`);
};

export const apiGetFlOrchestrationStatus = () =>
  apiRequest('/api/v1/federated/status');

export const apiStartFlCampaign = (numRounds = null) =>
  apiRequest('/api/v1/federated/start', {
    method: 'POST',
    body: JSON.stringify(numRounds != null ? { num_rounds: numRounds } : {}),
  });

export const apiStopFlCampaign = () =>
  apiRequest('/api/v1/federated/stop', { method: 'POST' });

export const apiGetFlRounds = () =>
  apiRequest('/api/v1/federated/rounds');

export const apiGetModelsStatus = () =>
  apiRequest('/api/v1/models/status');

/** Clinician APIs */
export const apiGetClinicianOverview = () =>
  apiRequest('/api/v1/clinician/overview');

export const apiGetAssignedPatients = (limit = 20, offset = 0) =>
  apiRequest(`/api/v1/clinician/patients?limit=${limit}&offset=${offset}`);

export const apiGetEligiblePatients = (limit = 20, offset = 0) =>
  apiRequest(`/api/v1/clinician/patients/eligible?limit=${limit}&offset=${offset}`);

export const apiAssignPatient = (patientId) =>
  apiRequest(`/api/v1/clinician/patients/${patientId}/assignment`, { method: 'POST' });

export const apiGetPatientDetail = (patientId) =>
  apiRequest(`/api/v1/clinician/patients/${patientId}`);

export const apiGetClinicalDocuments = (patientId = null, limit = 20, offset = 0) => {
  const q = patientId ? `&patient_id=${encodeURIComponent(patientId)}` : '';
  return apiRequest(`/api/v1/clinician/documents?limit=${limit}&offset=${offset}${q}`);
};

export const apiGetDocumentDetail = (docId) =>
  apiRequest(`/api/v1/clinician/documents/${docId}`);

export const apiUploadClinicalDocument = (formData) =>
  apiRequest('/api/v1/clinician/documents/upload', {
    method: 'POST',
    body: formData,
  });

export const apiProcessClinicalDocument = (docId) =>
  apiRequest(`/api/v1/clinician/documents/${docId}/process`, { method: 'POST' });

export const apiGetClinicianDiagnosis = () =>
  apiRequest('/api/v1/clinician/diagnosis');

/** Hospital Administration APIs */
export const apiGetAccessibleNodes = () =>
  apiRequest('/api/v1/hospital/accessible-nodes');

export const apiGetHospitalOverview = () =>
  apiRequest('/api/v1/hospital/overview');

export const apiGetHospitalSystemHealth = () =>
  apiRequest('/api/v1/hospital/system-health');

export const apiGetHospitalClinicians = () =>
  apiRequest('/api/v1/hospital/clinicians');

export const apiGetHospitalPatients = (limit = 20, offset = 0) =>
  apiRequest(`/api/v1/hospital/patients?limit=${limit}&offset=${offset}`);
