import React, { useCallback, useEffect, useState } from 'react';
import { apiRequest, ApiError } from '../../services/api';
import { StatusBadge } from '../../pages/hospital/HospitalAdminDashboard';
import { AlertTriangle, Cpu, RefreshCw, Shield, BarChart3 } from 'lucide-react';

function formatAuroc(value) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(3);
}

export function TabularFederatedResearchPanel() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [payload, setPayload] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const envelope = await apiRequest('/api/v1/federated/experiments/tabular/latest');
      setPayload(envelope?.data ?? envelope);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError('Session expired or missing. Sign in again as Super Admin.');
      } else if (err instanceof ApiError && err.status === 403) {
        setError('Insufficient permission (requires federated:monitor).');
      } else {
        setError(err?.message || 'Failed to load federated experiment summary.');
      }
      setPayload(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) {
    return (
      <div className="card" style={{ padding: '1.5rem' }}>
        <RefreshCw className="spin" size={20} style={{ marginRight: '0.5rem' }} />
        Loading tabular federated research summary…
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ borderColor: 'var(--danger)', padding: '1.25rem' }}>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
          <AlertTriangle size={22} className="text-danger" />
          <div>
            <strong>Could not load experiment status</strong>
            <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>{error}</p>
            <button type="button" className="btn btn-secondary" onClick={load} style={{ marginTop: '0.75rem' }}>
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  const status = payload?.status || 'UNKNOWN';

  if (status === 'NO_RUNS') {
    return (
      <div className="card">
        <StatusBadge status="NO_DATA" />
        <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>
          No saved tabular federated runs on this server. Run the documented CLI experiment locally
          (requires explicit <code>--confirm-run</code>).
        </p>
      </div>
    );
  }

  const report = payload?.report || {};
  const cfg = report.config_summary || {};
  const evalSum = report.evaluation_summary || {};
  const baseline = report.baseline_comparison || {};
  const verify = payload?.checkpoint_verification || {};
  const rounds = report.round_history || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div
        className="card"
        style={{
          borderLeft: '4px solid var(--warning)',
          background: 'rgba(255, 193, 7, 0.06)',
          padding: '1rem 1.25rem',
        }}
      >
        <strong>Research demonstration only</strong>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
          {report.simulation_disclaimer ||
            'Simulated clients are not verified independent hospitals.'}{' '}
          Metrics use the survey cardio target on cardio_train.csv — not clinical deployment.
          {payload?.inference_enabled === false ? ' Clinical inference remains disabled.' : ''}
        </p>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <Cpu size={20} />
            </div>
            <div className="card-title">Federation</div>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{cfg.federation_id}</div>
          <div style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>{cfg.task_id}</div>
          <div style={{ marginTop: '0.75rem' }}>
            <StatusBadge status={status === 'AVAILABLE' ? 'AVAILABLE' : status} />
          </div>
          <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
            Run: {payload?.run_id || '—'}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <BarChart3 size={20} />
            </div>
            <div className="card-title">Participation</div>
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{cfg.num_clients ?? '—'}</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>simulated clients</div>
          <div style={{ marginTop: '0.5rem' }}>
            {rounds.length} round(s) logged · {cfg.local_epochs ?? '—'} local epoch(s) each
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <BarChart3 size={20} />
            </div>
            <div className="card-title">Test AUROC</div>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Federated global</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{formatAuroc(evalSum.test_auroc)}</div>
          <div style={{ fontSize: '0.75rem', marginTop: '0.5rem', color: 'var(--text-muted)' }}>
            Centralized baseline (same train rows): {formatAuroc(baseline.centralized_test_auroc)}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <Shield size={20} />
            </div>
            <div className="card-title">Checkpoint</div>
          </div>
          <StatusBadge status={verify.ok ? 'AVAILABLE' : 'NOT_MEASURED'} />
          <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', color: 'var(--text-muted)' }}>
            Trained: {String(payload?.checkpoint_summary?.trained ?? '—')} · Round{' '}
            {payload?.checkpoint_summary?.round_number ?? '—'}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-icon text-info">
            <Shield size={20} />
          </div>
          <div className="card-title">Privacy boundaries</div>
        </div>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
          {payload?.privacy_limitations || report.privacy_note}
        </p>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          FedAvg is not differential privacy. No live Flower server is exposed from this dashboard.
        </p>
      </div>

      <button type="button" className="btn btn-secondary" onClick={load}>
        <RefreshCw size={16} style={{ marginRight: '0.35rem' }} />
        Refresh
      </button>
    </div>
  );
}
