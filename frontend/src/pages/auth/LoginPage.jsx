import React, { useState } from 'react';
import { Shield, AlertCircle, Lock, Mail } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { AuthLayout } from '../../layouts/AuthLayout';

export function LoginPage() {
  const { login, error: authError } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [localError, setLocalError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
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

  return (
    <AuthLayout>
      <div className="auth-card">
        <div className="auth-header">
          <div className="brand flex-center" style={{ marginBottom: '0.5rem' }}>
            <Shield size={36} className="brand-icon" />
          </div>
          <h1 className="auth-title">MedShield-FL Console</h1>
          <p className="auth-subtitle">
            Privacy Masked Federated Learning for Heart Disease Diagnosis
          </p>
        </div>

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
                placeholder="admin@medshield.org"
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
            {isSubmitting ? 'Authenticating...' : 'Sign In'}
          </button>
        </form>

        <div className="notice-banner" style={{ marginTop: '1.5rem', fontSize: '0.75rem' }}>
          <strong>Zero Trust Security Notice:</strong>
          <p style={{ marginTop: '4px', color: 'var(--text-muted)' }}>
            Authentication credentials are encrypted and validated server-side. Failed attempts are recorded in security event audit logs.
          </p>
        </div>
      </div>
    </AuthLayout>
  );
}
