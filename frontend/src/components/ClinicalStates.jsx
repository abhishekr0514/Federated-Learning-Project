import React from 'react';
import {
  HelpCircle,
  AlertTriangle,
  Lock,
  Search,
  AlertCircle,
  Loader2,
  CheckCircle2,
} from 'lucide-react';

/**
 * Canonical Operational Clinical Status Badge
 */
export function ClinicalStatusBadge({ status }) {
  const norm = (status || '').toUpperCase();
  let badgeClass = 'badge badge-muted';
  let label = norm || 'UNKNOWN';

  switch (norm) {
    case 'AVAILABLE':
    case 'SUCCESS':
    case 'ACTIVE':
    case 'SANITIZED':
    case 'COMPLETED':
      badgeClass = 'badge badge-success';
      break;
    case 'NO_DATA':
      badgeClass = 'badge badge-muted';
      label = 'NO DATA';
      break;
    case 'NOT_ENABLED':
      badgeClass = 'badge badge-warning';
      label = 'NOT ENABLED';
      break;
    case 'NOT_MEASURED':
      badgeClass = 'badge badge-info';
      label = 'NOT MEASURED';
      break;
    case 'OPERATIONAL':
      badgeClass = 'badge badge-success';
      label = 'OPERATIONAL';
      break;
    case 'READY':
    case 'INFRASTRUCTURE_READY':
      badgeClass = 'badge badge-info';
      label = 'READY';
      break;
    case 'CONFIGURATION_REQUIRED':
    case 'MISSING_DEPENDENCY':
      badgeClass = 'badge badge-warning';
      label = 'CONFIG REQUIRED';
      break;
    case 'UNAVAILABLE':
      badgeClass = 'badge badge-danger';
      label = 'UNAVAILABLE';
      break;
    case 'NOT_IMPLEMENTED':
      badgeClass = 'badge badge-purple';
      label = 'NOT IMPLEMENTED';
      break;
    case 'FORBIDDEN':
    case 'ERROR':
    case 'CRITICAL':
    case 'DENIED':
    case 'FAILED':
      badgeClass = 'badge badge-danger';
      break;
    default:
      badgeClass = 'badge badge-muted';
  }

  return <span className={badgeClass}>{label}</span>;
}

/**
 * Reusable Honest Empty State
 */
export function ClinicalEmptyState({
  title = 'No Records Available',
  description = 'There are no clinical records stored for this query.',
  icon: Icon = HelpCircle,
  status = 'NO_DATA',
}) {
  return (
    <div className="honest-state-box">
      <div
        className="honest-state-icon"
        style={{ background: 'rgba(156, 163, 175, 0.1)', color: 'var(--text-muted)' }}
      >
        <Icon size={32} />
      </div>
      <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', maxWidth: '500px', margin: '0 auto 1rem' }}>
        {description}
      </p>
      <ClinicalStatusBadge status={status} />
    </div>
  );
}

/**
 * Clinical Subsystem Not Implemented State Card
 */
export function ClinicalNotImplemented({
  title = 'Subsystem Not Implemented',
  moduleName = 'Clinical Subsystem',
  description = 'This capability is intentionally pending implementation. Never simulated or fabricated.',
}) {
  return (
    <div className="card" style={{ borderLeft: '4px solid #8b5cf6', background: 'rgba(139, 92, 246, 0.03)' }}>
      <div className="card-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="card-icon" style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#8b5cf6' }}>
            <AlertTriangle size={20} />
          </div>
          <div>
            <div className="card-title">{title}</div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{moduleName}</span>
          </div>
        </div>
        <ClinicalStatusBadge status="NOT_IMPLEMENTED" />
      </div>
      <div style={{ marginTop: '0.75rem' }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>{description}</p>
        <div
          style={{
            marginTop: '0.75rem',
            padding: '0.5rem 0.75rem',
            background: 'var(--bg-secondary)',
            borderRadius: '6px',
            fontSize: '0.8125rem',
            color: 'var(--text-muted)',
            fontStyle: 'italic',
          }}
        >
          Zero mock predictions contract active: Risk scores, model confidences, and clinical recommendations are never simulated.
        </div>
      </div>
    </div>
  );
}

/**
 * Clinical Subsystem Not Enabled State Card
 */
export function ClinicalNotEnabled({
  title = 'Subsystem Not Enabled',
  moduleName = 'Background Subsystem',
  description = 'This background service is currently disabled in the server deployment.',
}) {
  return (
    <div className="card" style={{ borderLeft: '4px solid var(--warning)', background: 'rgba(245, 158, 11, 0.03)' }}>
      <div className="card-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="card-icon text-warning">
            <AlertTriangle size={20} />
          </div>
          <div>
            <div className="card-title">{title}</div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{moduleName}</span>
          </div>
        </div>
        <ClinicalStatusBadge status="NOT_ENABLED" />
      </div>
      <div style={{ marginTop: '0.75rem' }}>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>{description}</p>
      </div>
    </div>
  );
}

/**
 * Clinical Forbidden Access State
 */
export function ClinicalForbidden({
  title = 'Access Forbidden',
  description = 'You do not possess the required clinician authorization or institutional role to access this resource.',
}) {
  return (
    <div className="honest-state-box" style={{ borderLeftColor: 'var(--danger)' }}>
      <div
        className="honest-state-icon"
        style={{ background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)' }}
      >
        <Lock size={32} />
      </div>
      <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', maxWidth: '500px', margin: '0 auto 1rem' }}>
        {description}
      </p>
      <ClinicalStatusBadge status="FORBIDDEN" />
    </div>
  );
}

/**
 * Clinical Not Found State (Resource outside clinician scope)
 */
export function ClinicalNotFound({
  title = 'Clinical Resource Not Found',
  description = 'The requested patient reference or clinical dossier is not assigned to your clinician account or does not exist.',
}) {
  return (
    <div className="honest-state-box">
      <div
        className="honest-state-icon"
        style={{ background: 'rgba(156, 163, 175, 0.1)', color: 'var(--text-muted)' }}
      >
        <Search size={32} />
      </div>
      <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', maxWidth: '500px', margin: '0 auto 1rem' }}>
        {description}
      </p>
      <ClinicalStatusBadge status="NO_DATA" />
    </div>
  );
}

/**
 * Clinical Error State
 */
export function ClinicalError({
  title = 'Clinical Operation Error',
  message = 'An unexpected error occurred while communicating with the hospital backend.',
}) {
  return (
    <div className="honest-state-box" style={{ borderLeftColor: 'var(--danger)' }}>
      <div
        className="honest-state-icon"
        style={{ background: 'rgba(239, 68, 68, 0.1)', color: 'var(--danger)' }}
      >
        <AlertCircle size={32} />
      </div>
      <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', maxWidth: '500px', margin: '0 auto 1rem' }}>
        {message}
      </p>
      <ClinicalStatusBadge status="ERROR" />
    </div>
  );
}

/**
 * Clinical Loading State
 */
export function ClinicalLoading({ message = 'Loading clinical telemetry...' }) {
  return (
    <div className="honest-state-box">
      <div
        className="honest-state-icon spinner"
        style={{ background: 'rgba(59, 130, 246, 0.1)', color: 'var(--primary)' }}
      >
        <Loader2 size={32} />
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem' }}>{message}</p>
    </div>
  );
}
