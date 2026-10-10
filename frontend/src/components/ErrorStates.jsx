import React from 'react';
import { AlertTriangle, Lock, FileQuestion, Cpu, Server, ShieldAlert } from 'lucide-react';

export function LoadingSpinner({ message = 'Loading session...' }) {
  return (
    <div className="flex-center p-8">
      <div className="spinner"></div>
      <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>{message}</p>
    </div>
  );
}

export function Unauthorized401() {
  return (
    <div className="error-card">
      <div className="error-icon text-amber">
        <Lock size={36} />
      </div>
      <h2>401 — Session Unauthenticated</h2>
      <p>Your session has expired or requires valid authentication credentials.</p>
      <a href="/login" className="btn btn-primary" style={{ marginTop: '1rem' }}>
        Return to Login
      </a>
    </div>
  );
}

export function Forbidden403({ requiredRole }) {
  return (
    <div className="error-card">
      <div className="error-icon text-danger">
        <ShieldAlert size={36} />
      </div>
      <h2>403 — Access Forbidden</h2>
      <p>
        Your account does not possess the required authorization context ({requiredRole || 'insufficient role'}) to access this institutional resource.
      </p>
      <p className="text-muted" style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
        Backend security guards enforce institutional isolation. Attempting to bypass URL checks is strictly audited.
      </p>
    </div>
  );
}

export function NotFound404() {
  return (
    <div className="error-card">
      <div className="error-icon text-muted">
        <FileQuestion size={36} />
      </div>
      <h2>404 — Resource Not Accessible</h2>
      <p>The requested page or endpoint does not exist or is unavailable within your authorized institutional scope.</p>
    </div>
  );
}

export function FeatureNotImplemented({ title, description, moduleName }) {
  return (
    <div className="card feature-501-card">
      <div className="card-header">
        <div className="card-icon text-info">
          <Cpu size={22} />
        </div>
        <div>
          <div className="card-title">{title || 'Feature Pending ML Implementation'}</div>
          <div className="badge badge-info" style={{ marginTop: '4px' }}>
            HTTP 501 Not Implemented
          </div>
        </div>
      </div>
      <div className="card-body" style={{ marginTop: '1rem' }}>
        <p style={{ color: 'var(--text-muted)', lineHeight: '1.5' }}>
          {description ||
            `The ${moduleName || 'requested ML'} module is intentionally unexposed until actual model training and scientific validation are complete.`}
        </p>
        <div className="notice-banner" style={{ marginTop: '1rem' }}>
          <strong>Zero Mock Data Contract:</strong> MedShield-FL forbids fabricating synthetic diagnostic scores, fake ROC-AUC curves, or mock clinical metrics. System endpoints strictly return HTTP 501 until model weights are verified.
        </div>
      </div>
    </div>
  );
}

export function ServerError500({ error }) {
  return (
    <div className="error-card">
      <div className="error-icon text-danger">
        <Server size={36} />
      </div>
      <h2>500 — System Error</h2>
      <p>{error?.message || 'An unexpected backend error occurred.'}</p>
    </div>
  );
}
