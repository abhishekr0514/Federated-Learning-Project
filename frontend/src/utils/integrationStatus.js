/**
 * Maps backend capability / operational status strings to UI badge labels.
 * Keeps purple "NOT IMPLEMENTED" only for genuinely unbuilt subsystems.
 */

export function normalizeIntegrationStatus(raw) {
  const norm = String(raw || 'NO_DATA').toUpperCase();
  switch (norm) {
    case 'AVAILABLE':
    case 'SUCCESS':
    case 'ACTIVE':
    case 'OPERATIONAL':
      return { status: 'AVAILABLE', label: 'OPERATIONAL' };
    case 'INFRASTRUCTURE_READY':
    case 'READY':
      return { status: 'READY', label: 'READY' };
    case 'NO_DATA':
      return { status: 'NO_DATA', label: 'NO DATA' };
    case 'NOT_ENABLED':
      return { status: 'NOT_ENABLED', label: 'NOT ENABLED' };
    case 'NOT_MEASURED':
      return { status: 'NOT_MEASURED', label: 'NOT MEASURED' };
    case 'MISSING_DEPENDENCY':
    case 'CONFIGURATION_REQUIRED':
      return { status: 'CONFIGURATION_REQUIRED', label: 'CONFIG REQUIRED' };
    case 'UNAVAILABLE':
    case 'ERROR':
      return { status: 'UNAVAILABLE', label: 'UNAVAILABLE' };
    case 'NOT_IMPLEMENTED':
      return { status: 'NOT_IMPLEMENTED', label: 'NOT IMPLEMENTED' };
    default:
      return { status: norm, label: norm.replace(/_/g, ' ') };
  }
}

export function parseAuthTokenEnvelope(body) {
  if (!body || typeof body !== 'object') {
    return { access_token: null, refresh_token: null };
  }
  const data = body.data && typeof body.data === 'object' ? body.data : body;
  return {
    access_token: data.access_token || null,
    refresh_token: data.refresh_token || null,
  };
}
