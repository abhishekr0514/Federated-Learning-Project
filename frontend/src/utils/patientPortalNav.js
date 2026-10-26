/**
 * Patient portal sidebar badges — plain language (avoid "Offline" for disabled features).
 */

export function patientPortalNavBadge(status) {
  const key = String(status || 'NOT_ENABLED').toUpperCase();
  const labels = {
    AVAILABLE: 'Ready',
    RESULTS_ON_FILE: 'Results',
    NO_DATA: 'No results',
    NOT_ENABLED: 'Not enabled',
    NOT_IMPLEMENTED: 'Not built',
    CONFIGURATION_REQUIRED: 'Setup needed',
    MISSING_DEPENDENCY: 'Deps missing',
    UNAVAILABLE: 'Unavailable',
  };
  return labels[key] || key.replace(/_/g, ' ').slice(0, 14);
}

export function patientMessagingNavBadge(capabilities) {
  const msg = capabilities?.patient_messaging || capabilities?.clinician_messaging;
  if (msg === 'NOT_IMPLEMENTED') return 'Not built';
  return patientPortalNavBadge(msg);
}
