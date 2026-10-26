/**
 * Map canonical backend capability statuses to compact sidebar badges.
 * Never use HTTP status codes (e.g. 501) as capability labels.
 */

const SHORT_LABELS = {
  AVAILABLE: 'READY',
  CONFIGURATION_REQUIRED: 'CONFIG',
  NOT_ENABLED: 'OFF',
  NOT_IMPLEMENTED: 'N/A',
  MISSING_DEPENDENCY: 'DEPS',
  DISABLED_BY_CONFIGURATION: 'OFF',
  CAUSAL_DESIGN_REQUIRED: 'DESIGN',
  COHORT_NOT_AVAILABLE: 'NO COHORT',
  GAT_MODEL_NOT_READY: 'NO GAT',
  INFRASTRUCTURE_READY: 'INFRA',
  WAITING_FOR_CLIENTS: 'WAIT',
  STARTING: 'START',
  STOPPING: 'STOP',
  SERVER_STOPPED: 'IDLE',
  RUNNING: 'RUN',
  COMPLETED: 'DONE',
  FAILED: 'FAIL',
  SECURITY_NOT_CONFIGURED: 'TLS',
};

export function capabilityNavBadge(status) {
  if (!status) return null;
  const key = String(status).toUpperCase();
  return SHORT_LABELS[key] || key.replace(/_/g, ' ').slice(0, 12);
}

export function isGatDiagnosticWorkflowEnabled(capabilities, capabilityDetails) {
  const gat = capabilities?.multimodal_gat_fusion;
  const ecg = capabilities?.ecg_report_inference;
  const gatReady = gat === 'AVAILABLE' || capabilityDetails?.multimodal_gat_fusion?.ready === true;
  const ecgReady = ecg === 'AVAILABLE' || capabilityDetails?.ecg_report_inference?.ready === true;
  return gatReady || ecgReady;
}
