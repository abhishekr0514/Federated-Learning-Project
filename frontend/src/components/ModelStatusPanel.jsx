import React, { useState, useEffect, useCallback } from 'react';
import {
  Cpu,
  Activity,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Server,
  Layers,
  Database,
  Shield,
  HelpCircle,
} from 'lucide-react';
import { apiGetHealth, apiGetEcgReportReadiness, ApiError } from '../services/api';
import { fetchPlatformCapabilities, invalidatePlatformCapabilitiesCache } from '../services/capabilityClient';

export function ModelStatusPanel({ onReadinessChange }) {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [capError, setCapError] = useState(null);

  const [health, setHealth] = useState(null);
  const [capabilityReport, setCapabilityReport] = useState(null);
  const [deepVerify, setDeepVerify] = useState(null);
  const [lastChecked, setLastChecked] = useState(null);

  const ecgReportReadiness = capabilityReport?.capability_details?.ecg_report_inference;
  const gatReadiness = capabilityReport?.capability_details?.multimodal_gat_fusion;

  const fetchReadiness = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    else setLoading(true);
    setError(null);
    setCapError(null);

    try {
      const [hRes, capRes] = await Promise.allSettled([
        apiGetHealth(),
        fetchPlatformCapabilities({ force: isManual }),
      ]);

      if (hRes.status === 'fulfilled') setHealth(hRes.value);
      else setError(hRes.reason?.message || 'Health check failed');

      if (capRes.status === 'fulfilled') {
        setCapabilityReport(capRes.value);
      } else {
        setCapError(
          capRes.reason instanceof ApiError
            ? `GET /api/v1/capabilities — ${capRes.reason.message}`
            : 'Capability registry unavailable',
        );
      }

      if (isManual) {
        try {
          const verified = await apiGetEcgReportReadiness({ verify: true });
          setDeepVerify(verified);
        } catch (verifyErr) {
          setDeepVerify({ ready: false, blockers: [verifyErr.message] });
        }
      }

      const ecgReady = Boolean(capRes.status === 'fulfilled' && capRes.value?.capability_details?.ecg_report_inference?.ready);
      const gatReady = Boolean(capRes.status === 'fulfilled' && capRes.value?.capability_details?.multimodal_gat_fusion?.ready);
      setLastChecked(new Date().toLocaleTimeString());

      if (onReadinessChange) {
        onReadinessChange({
          backendHealthy: hRes.status === 'fulfilled' && hRes.value?.status === 'healthy',
          ecgReady,
          gatReady,
        });
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [onReadinessChange]);

  useEffect(() => {
    fetchReadiness();
  }, [fetchReadiness]);

  const handleRefresh = () => {
    invalidatePlatformCapabilitiesCache();
    fetchReadiness(true);
  };

  return (
    <div className="card model-status-panel" style={{ marginBottom: '1.5rem', border: '1px solid #1e293b' }}>
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
          <div className="card-icon" style={{ background: 'rgba(6, 182, 212, 0.1)', color: '#06b6d4' }}>
            <Cpu size={20} />
          </div>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
              Live Neural Architecture & Model Readiness
            </h3>
            <p style={{ fontSize: '0.75rem', color: '#94a3b8', margin: 0 }}>
              Lightweight registry checks; use Refresh for optional encoder forward verification.
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {lastChecked && (
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
              Checked: {lastChecked}
            </span>
          )}
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={handleRefresh}
            disabled={loading || refreshing}
            style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <RefreshCw size={13} className={refreshing ? 'spin' : ''} />
            <span>{refreshing ? 'Querying...' : 'Refresh Readiness'}</span>
          </button>
        </div>
      </div>

      {capError && (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--warning)', margin: '1rem' }}>
          <AlertTriangle size={16} style={{ display: 'inline', marginRight: '6px' }} />
          {capError}
        </div>
      )}
      {error && (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--danger)', margin: '1rem' }}>
          <AlertTriangle size={16} style={{ display: 'inline', marginRight: '6px' }} />
          <strong>Health:</strong> {error}
        </div>
      )}

      {loading && !refreshing ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          <RefreshCw size={24} className="spin" style={{ margin: '0 auto 0.5rem' }} />
          <p style={{ fontSize: '0.875rem' }}>Loading capability registry (cached, lightweight)...</p>
        </div>
      ) : (
        <div style={{ padding: '1rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem' }}>
            <div style={{
              background: '#090d16',
              border: ecgReportReadiness?.ready ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <strong style={{ fontSize: '0.875rem', color: '#f1f5f9' }}>ECG + Clinical Report Fusion</strong>
                <span className={`badge ${ecgReportReadiness?.ready ? 'badge-success' : 'badge-danger'}`}>
                  {ecgReportReadiness?.status || (ecgReportReadiness?.ready ? 'AVAILABLE' : 'NOT READY')}
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{ecgReportReadiness?.reason}</p>
              {ecgReportReadiness?.blockers?.length > 0 && (
                <ul style={{ fontSize: '0.75rem', color: '#fca5a5', paddingLeft: '1rem' }}>
                  {ecgReportReadiness.blockers.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
              )}
            </div>

            <div style={{
              background: '#090d16',
              border: gatReadiness?.ready ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <strong style={{ fontSize: '0.875rem', color: '#f1f5f9' }}>Two-Modality GAT Fusion</strong>
                <span className={`badge ${gatReadiness?.ready ? 'badge-success' : 'badge-warning'}`}>
                  {gatReadiness?.status || 'UNKNOWN'}
                </span>
              </div>
              <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{gatReadiness?.reason}</p>
              {gatReadiness?.blockers?.length > 0 && (
                <ul style={{ fontSize: '0.75rem', color: '#fde68a', paddingLeft: '1rem' }}>
                  {gatReadiness.blockers.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          {deepVerify && (
            <p style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.75rem' }}>
              Deep encoder verify: {deepVerify.ready ? 'forward pass OK' : deepVerify.blockers?.join('; ')}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
