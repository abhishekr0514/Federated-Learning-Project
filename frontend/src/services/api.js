/**
 * Centralized API Client for MedShield-FL Frontend
 * Handles HTTP requests, Bearer token attachment, X-Request-ID propagation,
 * automatic 401 token refresh queuing, and structured error responses.
 */

const API_BASE_URL = '';

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
 * Perform custom fetch with automatic auth header and refresh handling
 */

export async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const accessToken = sessionStorage.getItem('medshield_access_token');

  const headers = {
    'Content-Type': 'application/json',
    'X-Request-ID': generateUUID(),
    ...(options.headers || {}),
  };

  if (accessToken && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  const config = {
    ...options,
    headers,
  };

  try {
    let response = await fetch(url, config);

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
      const refreshToken = sessionStorage.getItem('medshield_refresh_token');

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
        const newAccess = refreshData.data.access_token;
        const newRefresh = refreshData.data.refresh_token;

        sessionStorage.setItem('medshield_access_token', newAccess);
        sessionStorage.setItem('medshield_refresh_token', newRefresh);

        isRefreshing = false;
        processQueue(null, newAccess);

        // Retry original request with new access token
        config.headers['Authorization'] = `Bearer ${newAccess}`;
        response = await fetch(url, config);
      } catch (refreshErr) {
        isRefreshing = false;
        processQueue(refreshErr, null);
        sessionStorage.removeItem('medshield_access_token');
        sessionStorage.removeItem('medshield_refresh_token');
        window.dispatchEvent(new CustomEvent('medshield:auth_failed'));
        throw new ApiError('Session expired. Please login again.', 401, 'UNAUTHENTICATED');
      }
    }

    return await handleResponse(response);
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
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
    const errorMsg = body?.error?.message || body?.message || `HTTP ${response.status} Error`;
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
