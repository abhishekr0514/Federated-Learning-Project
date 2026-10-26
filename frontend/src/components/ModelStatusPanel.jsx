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
import { apiGetHealth, apiGetEcgReportReadiness, apiGetGatFusionReadiness, ApiError } from '../services/api';

export function ModelStatusPanel({ onReadinessChange }) {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const [health, setHealth] = useState(null);
  const [ecgReportReadiness, setEcgReportReadiness] = useState(null);
  const [gatReadiness, setGatReadiness] = useState(null);
  const [lastChecked, setLastChecked] = useState(null);

  const fetchReadiness = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const [hRes, ecgRes, gatRes] = await Promise.allSettled([
        apiGetHealth(),
        apiGetEcgReportReadiness(),
        apiGetGatFusionReadiness(),
      ]);

      if (hRes.status === 'fulfilled') setHealth(hRes.value);
      else throw hRes.reason;

      const ecgData = ecgRes.status === 'fulfilled' ? ecgRes.value : null;
      const gatData = gatRes.status === 'fulfilled' ? gatRes.value : null;

      setEcgReportReadiness(ecgData);
      setGatReadiness(gatData);
      setLastChecked(new Date().toLocaleTimeString());

      if (onReadinessChange) {
        onReadinessChange({
          backendHealthy: hRes.value?.status === 'healthy',
          ecgReady: Boolean(ecgData?.ready),
          gatReady: Boolean(gatData?.ready),
        });
      }
    } catch (err) {
      console.error('Model readiness diagnostic check failed:', err);
      setError(err.message || 'Unable to query backend readiness endpoints');
      if (onReadinessChange) {
        onReadinessChange({
          backendHealthy: false,
          ecgReady: false,
          gatReady: false,
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
              Real-time verification against server memory weights and artifact contracts.
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
            onClick={() => fetchReadiness(true)}
            disabled={loading || refreshing}
            style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <RefreshCw size={13} className={refreshing ? 'spin' : ''} />
            <span>{refreshing ? 'Querying...' : 'Refresh Readiness'}</span>
          </button>
        </div>
      </div>

      {error ? (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--danger)', margin: '1rem' }}>
          <AlertTriangle size={16} className="text-danger" style={{ display: 'inline', marginRight: '6px' }} />
          <strong>Readiness Telemetry Error:</strong> {error}. Make sure the FastAPI backend is listening on port 8000.
        </div>
      ) : loading && !refreshing ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          <RefreshCw size={24} className="spin" style={{ margin: '0 auto 0.5rem' }} />
          <p style={{ fontSize: '0.875rem' }}>Querying neural model runtimes and checkpoints...</p>
        </div>
      ) : (
        <div style={{ padding: '1rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem' }}>
            {/* Model Card 1: Two-Modality ECG + Report Fusion */}
            <div style={{
              background: '#090d16',
              border: ecgReportReadiness?.ready ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Activity size={16} className={ecgReportReadiness?.ready ? 'text-success' : 'text-danger'} />
                    <strong style={{ fontSize: '0.875rem', color: '#f1f5f9' }}>
                      ECG + Clinical Report Fusion
                    </strong>
                  </div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', fontFamily: 'monospace' }}>
                    {ecgReportReadiness?.architecture_id || 'medshield.ecg_report_fusion.v1'}
                  </span>
                </div>

                <span
                  className={`badge ${ecgReportReadiness?.ready ? 'badge-success' : 'badge-danger'}`}
                  style={{ fontSize: '0.75rem', fontWeight: 600 }}
                >
                  {ecgReportReadiness?.ready ? 'OPERATIONAL' : 'NOT READY'}
                </span>
              </div>

              <div className="info-list" style={{ fontSize: '0.8125rem' }}>
                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>Model Checkpoint</span>
                  <span className="info-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {ecgReportReadiness?.checkpoint_present ? (
                      <><CheckCircle2 size={13} className="text-success" /> Present & Trained</>
                    ) : (
                      <><XCircle size={13} className="text-danger" /> Missing</>
                    )}
                  </span>
                </div>

                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>BioClinicalBERT Text Model</span>
                  <span className="info-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {ecgReportReadiness?.text_model_loaded ? (
                      <><CheckCircle2 size={13} className="text-success" /> Loaded (128d)</>
                    ) : (
                      <><XCircle size={13} className="text-warning" /> Pending Warm-up</>
                    )}
                  </span>
                </div>

                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>1D-CNN + BiLSTM Weights</span>
                  <span className="info-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {ecgReportReadiness?.model_loaded ? (
                      <><CheckCircle2 size={13} className="text-success" /> Active in CPU Memory</>
                    ) : (
                      <><XCircle size={13} className="text-warning" /> Lazy Load on First Call</>
                    )}
                  </span>
                </div>

                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>Normalization Stats</span>
                  <span className="info-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {ecgReportReadiness?.normalization_present ? (
                      <><CheckCircle2 size={13} className="text-success" /> Verified (12-Lead)</>
                    ) : (
                      <><XCircle size={13} className="text-danger" /> Missing</>
                    )}
                  </span>
                </div>
              </div>

              {ecgReportReadiness?.blockers && ecgReportReadiness.blockers.length > 0 && (
                <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '4px', fontSize: '0.75rem', color: '#fca5a5' }}>
                  <strong>Blockers:</strong>
                  <ul style={{ paddingLeft: '1rem', marginTop: '2px' }}>
                    {ecgReportReadiness.blockers.map((b, i) => (
                      <li key={i}>{b}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Model Card 2: Two-Modality GAT Fusion */}
            <div style={{
              background: '#090d16',
              border: gatReadiness?.ready ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Layers size={16} className={gatReadiness?.ready ? 'text-success' : 'text-warning'} />
                    <strong style={{ fontSize: '0.875rem', color: '#f1f5f9' }}>
                      Graph Attention Network (GAT Fusion)
                    </strong>
                  </div>
                  <span style={{ fontSize: '0.7rem', color: '#64748b', fontFamily: 'monospace' }}>
                    {gatReadiness?.architecture_id || 'medshield.diagnostic_net.gat_ecg_report'}
                  </span>
                </div>

                <span
                  className={`badge ${gatReadiness?.ready ? 'badge-success' : 'badge-warning'}`}
                  style={{ fontSize: '0.75rem', fontWeight: 600 }}
                >
                  {gatReadiness?.ready ? 'READY' : 'UNAVAILABLE (DISABLED)'}
                </span>
              </div>

              <div className="info-list" style={{ fontSize: '0.8125rem' }}>
                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>GAT Checkpoint</span>
                  <span className="info-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {gatReadiness?.checkpoint_present ? (
                      <><CheckCircle2 size={13} className="text-success" /> Present</>
                    ) : (
                      <><XCircle size={13} className="text-danger" /> Missing Artifact</>
                    )}
                  </span>
                </div>

                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>Inference Enabled Flag</span>
                  <span className="info-value">
                    <span className="badge" style={{ fontSize: '0.7rem' }}>
                      {gatReadiness?.inference_enabled ? 'true' : 'false (Safety Gate Active)'}
                    </span>
                  </span>
                </div>

                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>Pairing Gate Verdict</span>
                  <span className="info-value" style={{ color: gatReadiness?.pairing_gate_allowed ? '#10b981' : '#f59e0b' }}>
                    {gatReadiness?.pairing_gate_verdict || 'BLOCKED'}
                  </span>
                </div>

                <div className="info-item" style={{ padding: '0.35rem 0' }}>
                  <span className="info-label" style={{ color: '#94a3b8' }}>Embedding Contracts</span>
                  <span className="info-value" style={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                    ECG: 128d | Clinical: 128d
                  </span>
                </div>
              </div>

              {gatReadiness?.blockers && gatReadiness.blockers.length > 0 && (
                <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(245, 158, 11, 0.1)', borderRadius: '4px', fontSize: '0.75rem', color: '#fde68a' }}>
                  <strong>Safety Blockers (Genuine Server Response):</strong>
                  <ul style={{ paddingLeft: '1rem', marginTop: '2px' }}>
                    {gatReadiness.blockers.map((b, i) => (
                      <li key={i}>{b}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
