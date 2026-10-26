import React, { useState, useEffect } from 'react';
import {
  Shield,
  AlertCircle,
  Lock,
  Mail,
  Activity,
  Server,
  UserCheck,
  Building2,
  Heart,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { AuthLayout } from '../../layouts/AuthLayout';
import { apiGetHealth } from '../../services/api';
import { SUPERLINK_DEMO_LOGIN_HINT } from '../../components/FederatedDemoAccessGate';

export function LoginPage() {
  const { login, error: authError } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [demoHint, setDemoHint] = useState(null);

  // Backend connection indicator
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking' | 'connected' | 'disconnected'
  const [healthInfo, setHealthInfo] = useState(null);

  const checkBackend = async () => {
    setBackendStatus('checking');
    try {
      const data = await apiGetHealth();
      setHealthInfo(data);
      setBackendStatus('connected');
    } catch {
      setBackendStatus('disconnected');
    }
  };

  useEffect(() => {
    checkBackend();
    if (sessionStorage.getItem(SUPERLINK_DEMO_LOGIN_HINT) === '1') {
      sessionStorage.removeItem(SUPERLINK_DEMO_LOGIN_HINT);
      setDemoHint('SuperLink demo: sign in with Super Admin below (admin@medshield.org).');
      setEmail('admin@medshield.org');
    }
  }, []);

  const handleSubmit = async (e) => {
    e?.preventDefault?.();
    if (!email || !password) {
      setLocalError('Please enter both email address and password.');
      return;
    }

    setLocalError(null);
    setIsSubmitting(true);
    try {
      await login(email, password);
    } catch (err) {
      console.error('Login submit error:', err);
      setLocalError(err.message || 'Invalid email or password');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleQuickLogin = (quickEmail, quickPassword) => {
    setEmail(quickEmail);
    setPassword(quickPassword);
    setLocalError(null);
    setIsSubmitting(true);
    login(quickEmail, quickPassword)
      .catch((err) => {
        setLocalError(err.message || 'Quick login failed');
      })
      .finally(() => {
        setIsSubmitting(false);
      });
  };

  return (
    <AuthLayout>
      <div className="auth-card" style={{ maxWidth: '480px' }}>
        <div className="auth-header">
          <div className="brand flex-center" style={{ marginBottom: '0.5rem' }}>
            <Shield size={38} className="brand-icon" style={{ color: '#06b6d4' }} />
          </div>
          <h1 className="auth-title">MedShield-FL Console</h1>
          <p className="auth-subtitle">
            Privacy-Preserving Federated Learning & Multimodal Diagnostic Intelligence
          </p>

          {/* Backend Health Badge */}
          <div style={{ marginTop: '0.75rem', display: 'flex', justifyContent: 'center' }}>
            {backendStatus === 'connected' ? (
              <span className="badge badge-success" style={{ fontSize: '0.7rem', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                <span className="status-indicator"></span> Backend Connected ({healthInfo?.service || 'FastAPI'} v{healthInfo?.version || '0.1.0'})
              </span>
            ) : backendStatus === 'checking' ? (
              <span className="badge" style={{ fontSize: '0.7rem', background: '#1e293b' }}>
                <RefreshCw size={10} className="spin" style={{ marginRight: '4px' }} /> Checking Backend...
              </span>
            ) : (
              <span className="badge badge-danger" style={{ fontSize: '0.7rem', cursor: 'pointer' }} onClick={checkBackend}>
                <AlertCircle size={10} style={{ marginRight: '4px' }} /> Backend Offline (Click to Retry)
              </span>
            )}
          </div>
        </div>

        {demoHint && (
          <div className="notice-banner" style={{ borderLeftColor: '#3b82f6', marginBottom: '1.25rem', background: '#0f172a' }}>
            <Shield size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }} />
            <span>{demoHint}</span>
          </div>
        )}

        {(localError || authError) && (
          <div className="notice-banner" style={{ borderLeftColor: 'var(--danger)', marginBottom: '1.25rem' }}>
            <AlertCircle size={16} className="text-danger" style={{ display: 'inline', marginRight: '6px' }} />
            <span>{localError || authError}</span>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="email-input">
              Account Email
            </label>
            <div style={{ position: 'relative' }}>
              <input
                id="email-input"
                type="email"
                className="form-input"
                placeholder="dr_metro@metro.org"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password-input">
              Password
            </label>
            <input
              id="password-input"
              type="password"
              className="form-input"
              placeholder="••••••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center', marginTop: '1rem' }}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <RefreshCw size={14} className="spin" style={{ marginRight: '6px' }} />
                Authenticating Session...
              </>
            ) : (
              'Sign In to Console'
            )}
          </button>
        </form>

        {/* Quick Demo Access Bar */}
        <div style={{ marginTop: '1.5rem', paddingTop: '1.25rem', borderTop: '1px solid #1e293b' }}>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '8px', textAlign: 'center' }}>
            One-Click Verified Demo Access:
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => handleQuickLogin('admin@medshield.org', 'ValidPassword123!')}
              disabled={isSubmitting}
              style={{ justifyContent: 'flex-start', fontSize: '0.75rem', gridColumn: '1 / -1' }}
            >
              <Shield size={13} style={{ marginRight: '6px', flexShrink: 0 }} />
              <span>Super Admin — SuperLink demo</span>
            </button>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => handleQuickLogin('dr_metro@metro.org', 'ValidPassword123!')}
              disabled={isSubmitting}
              style={{ justifyContent: 'flex-start', fontSize: '0.75rem' }}
            >
              <UserCheck size={13} className="text-info" style={{ marginRight: '6px', flexShrink: 0 }} />
              <span>Dr. Metro (Clinician)</span>
            </button>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => handleQuickLogin('admin_a@metro.org', 'ValidPassword123!')}
              disabled={isSubmitting}
              style={{ justifyContent: 'flex-start', fontSize: '0.75rem' }}
            >
              <Building2 size={13} className="text-accent" style={{ marginRight: '6px', flexShrink: 0 }} />
              <span>Hospital Admin</span>
            </button>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => handleQuickLogin('patient_alice@example.com', 'ValidPassword123!')}
              disabled={isSubmitting}
              style={{ justifyContent: 'flex-start', fontSize: '0.75rem' }}
            >
              <Heart size={13} className="text-danger" style={{ marginRight: '6px', flexShrink: 0 }} />
              <span>Patient Alice</span>
            </button>
          </div>
        </div>

        <div className="notice-banner" style={{ marginTop: '1.25rem', fontSize: '0.7rem', padding: '0.6rem 0.8rem' }}>
          <strong>Zero Trust Medical Privacy Boundary:</strong>
          <p style={{ marginTop: '2px', color: 'var(--text-muted)', margin: 0 }}>
            Authentication issues signed short-lived JWT tokens with rotating refresh cookies. Sensitive patient records remain locked inside institutional hospital boundaries.
          </p>
        </div>
      </div>
    </AuthLayout>
  );
}
