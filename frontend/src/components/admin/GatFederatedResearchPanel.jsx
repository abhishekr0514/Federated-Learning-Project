import React, { useCallback, useEffect, useState } from 'react';
import { apiRequest, ApiError } from '../../services/api';
import { StatusBadge } from '../../pages/hospital/HospitalAdminDashboard';
import { AlertTriangle, Cpu, RefreshCw, Shield, BarChart3, Network } from 'lucide-react';

function formatAuroc(value, reason) {
  if (value == null || Number.isNaN(value)) {
    return reason ? `N/A (${reason})` : 'N/A';
  }
  return Number(value).toFixed(3);
}

export function GatFederatedResearchPanel() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [payload, setPayload] = useState(null);
  const [runIdQuery, setRunIdQuery] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const q = runIdQuery.trim() ? `?run_id=${encodeURIComponent(runIdQuery.trim())}` : '';
      const envelope = await apiRequest(`/api/v1/federated/experiments/gat/latest${q}`);
      setPayload(envelope?.data ?? envelope);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError('Session expired or missing. Sign in again as Super Admin.');
      } else if (err instanceof ApiError && err.status === 403) {
        setError('Insufficient permission (requires federated:monitor).');
      } else {
        setError(err?.message || 'Failed to load GAT federated experiment summary.');
      }
      setPayload(null);
    } finally {
      setLoading(false);
    }
  }, [runIdQuery]);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) {
    return (
      <div className="card" style={{ padding: '1.5rem' }}>
        <RefreshCw className="spin" size={20} style={{ marginRight: '0.5rem' }} />
        Loading federated GAT research summary…
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
          No saved GAT federated runs yet. Build the embedding bundle (
          <code>scripts/build_gat_fl_embedding_bundle.py</code>), then start a SuperLink run from the panel above.
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
  const scope = report.federated_training_scope || 'gat_fusion_head_over_frozen_embeddings';

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
            'Two simulated hospitals on disjoint PTB-XL train embeddings — not independent institutions.'}{' '}
          {report.target_definition} Metrics are proxy research AUROC, not clinical performance.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          type="text"
          className="form-input"
          placeholder="Filter by Flower run ID (optional)"
          value={runIdQuery}
          onChange={(e) => setRunIdQuery(e.target.value)}
          style={{ maxWidth: '320px', fontSize: '0.8125rem' }}
        />
        <button type="button" className="btn btn-secondary btn-sm" onClick={load}>
          Load run
        </button>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <Network size={20} />
            </div>
            <div className="card-title">GAT federation</div>
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{cfg.architecture_id}</div>
          <div style={{ fontSize: '0.75rem', marginTop: '0.35rem' }}>Scope: {scope}</div>
          <div style={{ marginTop: '0.75rem' }}>
            <StatusBadge status={status === 'AVAILABLE' ? 'AVAILABLE' : status} />
          </div>
          <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
            Run: {payload?.flower_run_id || payload?.run_id || '—'}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <Cpu size={20} />
            </div>
            <div className="card-title">Participation</div>
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{cfg.num_clients ?? 2}</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>simulated hospital clients</div>
          <div style={{ marginTop: '0.5rem' }}>
            {rounds.length} round(s) · {cfg.local_epochs ?? '—'} local epoch(s) · FedAvg
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <BarChart3 size={20} />
            </div>
            <div className="card-title">Held-out test AUROC</div>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Global federated GAT (proxy labels)</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>
            {formatAuroc(evalSum.test_auroc, evalSum.test_auroc_unavailable_reason)}
          </div>
          <div style={{ fontSize: '0.75rem', marginTop: '0.5rem', color: 'var(--text-muted)' }}>
            n={evalSum.test_n_samples ?? '—'}
            {evalSum.test_class_counts ? ` · classes ${JSON.stringify(evalSum.test_class_counts)}` : ''}
          </div>
          {baseline.valid_same_test_split && baseline.centralized_test_auroc != null && (
            <div style={{ fontSize: '0.75rem', marginTop: '0.35rem', color: 'var(--text-muted)' }}>
              Centralized GAT baseline (same test embeddings): {formatAuroc(baseline.centralized_test_auroc)}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-icon text-info">
              <Shield size={20} />
            </div>
            <div className="card-title">Global checkpoint</div>
          </div>
          <StatusBadge status={verify.ok ? 'AVAILABLE' : 'NOT_MEASURED'} />
          <div style={{ fontSize: '0.8rem', marginTop: '0.5rem', color: 'var(--text-muted)' }}>
            Hash: {report.parameter_hash_final || report.checkpoint_summary?.global_parameter_hash || '—'}
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
      </div>

      <button type="button" className="btn btn-secondary" onClick={load}>
        <RefreshCw size={16} style={{ marginRight: '0.35rem' }} />
        Refresh
      </button>
    </div>
  );
}
