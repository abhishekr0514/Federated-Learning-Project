import React, { useState } from 'react';
import { AlertCircle, LogOut, Shield } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const SUPERLINK_DEMO_LOGIN_HINT = 'medshield_superlink_demo_login';

/**
 * Shown when the current session lacks federated:monitor (Clinician / Hospital Admin).
 * Does not call Flower orchestration APIs.
 */
export function FederatedDemoAccessGate() {
  const { user, roles, logout } = useAuth();
  const [busy, setBusy] = useState(false);

  const handleSwitchToSuperAdmin = async () => {
    setBusy(true);
    try {
      sessionStorage.setItem(SUPERLINK_DEMO_LOGIN_HINT, '1');
      await logout();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" style={{ border: '1px solid #334155', marginBottom: '1.25rem' }}>
      <div style={{ padding: '1.25rem' }}>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
          <AlertCircle size={22} style={{ color: '#f97316', flexShrink: 0, marginTop: 2 }} />
          <div>
            <h3 style={{ margin: 0, fontSize: '1rem', color: '#f8fafc' }}>
              SuperLink demo requires a Super Admin session
            </h3>
            <p style={{ margin: '0.5rem 0 0', fontSize: '0.8125rem', color: '#cbd5e1', lineHeight: 1.5 }}>
              You are signed in as <strong>{user?.email}</strong> ({roles.join(', ') || 'no role'}). Flower
              orchestration endpoints require the <code>federated:monitor</code> permission, which is granted to{' '}
              <code>admin@medshield.org</code> (Super Admin) only — not Clinician or Hospital Admin accounts.
            </p>
            <p style={{ margin: '0.75rem 0 0', fontSize: '0.75rem', color: '#94a3b8' }}>
              Sign out, then on the login screen choose <strong>Super Admin</strong> under verified demo access (dev
              password from seed: <code>ValidPassword123!</code> when seeded with <code>--non-interactive</code>).
            </p>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              style={{ marginTop: '1rem' }}
              disabled={busy}
              onClick={handleSwitchToSuperAdmin}
            >
              <LogOut size={14} style={{ marginRight: 6 }} />
              {busy ? 'Signing out…' : 'Sign out to switch to Super Admin'}
            </button>
          </div>
          <Shield size={28} style={{ color: '#3b82f6', opacity: 0.35, flexShrink: 0 }} />
        </div>
      </div>
    </div>
  );
}
