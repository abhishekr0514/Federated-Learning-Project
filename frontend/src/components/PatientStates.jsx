import React from 'react';
import {
  ShieldAlert,
  AlertTriangle,
  FileQuestion,
  Clock,
  Stethoscope,
  Inbox,
  Loader2,
  CheckCircle,
  HelpCircle,
  ShieldCheck,
} from 'lucide-react';

export function PatientStatusBadge({ status }) {
  const normalized = (status || 'NO_DATA').toUpperCase();

  const config = {
    AVAILABLE: { label: 'Available', className: 'badge-success' },
    NO_DATA: { label: 'No Records', className: 'badge-neutral' },
    NOT_ENABLED: { label: 'Not Available', className: 'badge-warning' },
    NOT_IMPLEMENTED: { label: 'In Development', className: 'badge-secondary' },
    FORBIDDEN: { label: 'Restricted', className: 'badge-danger' },
    ERROR: { label: 'Error', className: 'badge-danger' },
  };

  const current = config[normalized] || { label: normalized, className: 'badge-neutral' };

  return <span className={`badge ${current.className}`}>{current.label}</span>;
}

export function PatientSafetyDisclaimer({ wording }) {
  const text = wording || 'AI-assisted assessment is not currently available. Please discuss your health questions with your qualified healthcare professional.';
  return (
    <div
      style={{
        background: 'rgba(56, 189, 248, 0.08)',
        border: '1px solid rgba(56, 189, 248, 0.25)',
        borderRadius: '8px',
        padding: '1rem',
        margin: '1rem 0',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.75rem',
      }}
    >
      <ShieldCheck size={20} style={{ color: 'var(--color-primary, #38bdf8)', flexShrink: 0, marginTop: '2px' }} />
      <div style={{ fontSize: '0.875rem', lineHeight: 1.5, color: 'var(--text-main, #e2e8f0)' }}>
        <strong>Medical Information Notice:</strong>
        <p style={{ margin: '0.25rem 0 0 0', color: 'var(--text-muted, #94a3b8)' }}>{text}</p>
      </div>
    </div>
  );
}

export function PatientLoadingState({ message = 'Loading your health information...' }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
      <Loader2
        size={36}
        className="animate-spin"
        style={{ margin: '0 auto 1rem', color: 'var(--color-primary, #38bdf8)' }}
      />
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{message}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem' }}>
        Retrieving your verified personal records securely.
      </p>
    </div>
  );
}

export function PatientEmptyState({
  title = 'No Records Available',
  message = 'No health records are currently available.',
  hint = 'When your healthcare team adds or completes records, they will appear here.',
}) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
      <Inbox size={42} style={{ margin: '0 auto 1rem', color: 'var(--text-muted)' }} />
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem', maxWidth: '480px', margin: '0.5rem auto 0' }}>
        {message}
      </p>
      {hint && (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.5rem', opacity: 0.8 }}>
          {hint}
        </p>
      )}
    </div>
  );
}

export function PatientNotEnabled({
  title = 'AI-Assisted Assessment Not Currently Available',
  message = 'AI-assisted clinical assessment is not currently active for your records.',
  disclaimer = 'Please discuss your health questions with your qualified healthcare professional.',
}) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
      <Stethoscope size={42} style={{ margin: '0 auto 1rem', color: '#f59e0b' }} />
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem', maxWidth: '520px', margin: '0.5rem auto 0' }}>
        {message}
      </p>
      <PatientSafetyDisclaimer wording={disclaimer} />
    </div>
  );
}

export function PatientNotImplemented({
  title = 'Feature In Development',
  message = 'This feature is still being developed.',
}) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
      <Clock size={42} style={{ margin: '0 auto 1rem', color: 'var(--text-muted)' }} />
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem', maxWidth: '480px', margin: '0.5rem auto 0' }}>
        {message}
      </p>
    </div>
  );
}

export function PatientForbidden({ message = 'You do not have access to view this information.' }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
      <ShieldAlert size={42} style={{ margin: '0 auto 1rem', color: '#ef4444' }} />
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>Access Restricted</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem' }}>{message}</p>
    </div>
  );
}

export function PatientNotFound({ message = 'The requested health record could not be found.' }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
      <FileQuestion size={42} style={{ margin: '0 auto 1rem', color: 'var(--text-muted)' }} />
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>Record Not Found</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem' }}>{message}</p>
    </div>
  );
}

export function PatientError({
  message = 'We encountered an issue loading your information. Please try again later.',
  onRetry,
}) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
      <AlertTriangle size={42} style={{ margin: '0 auto 1rem', color: '#ef4444' }} />
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>Unable to Load Information</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem' }}>{message}</p>
      {onRetry && (
        <button className="btn btn-secondary" onClick={onRetry} style={{ marginTop: '1rem' }}>
          Try Again
        </button>
      )}
    </div>
  );
}
