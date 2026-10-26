import React, { useState, useEffect, useCallback } from 'react';
import {
  Radio,
  Play,
  History,
  AlertCircle,
  Building2,
  RefreshCw,
  BarChart3,
} from 'lucide-react';
import {
  apiGetLatestTabularExperiment,
  apiGetFlOrchestrationStatus,
  apiStartFlCampaign,
  apiStopFlCampaign,
  apiGetFlRounds,
  ApiError,
} from '../services/api';
import { invalidatePlatformCapabilitiesCache } from '../services/capabilityClient';
import {
  unwrapApiData,
  isFlowerRunActive,
  flowerLifecycleLabel,
  connectionStatusBadge,
  demoStartBlockedReason,
} from '../utils/federatedDemo';

const DEMO_ROUNDS = 2;

function notifyFederatedChanged() {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('medshield:federate_changed'));
  }
}

export function FederatedLearningPanel() {
  const [loading, setLoading] = useState(true);
  const [statusError, setStatusError] = useState(null);
  const [accessDenied, setAccessDenied] = useState(false);
  const [latestExperiment, setLatestExperiment] = useState(null);
  const [orchestration, setOrchestration] = useState(null);
  const [roundHistory, setRoundHistory] = useState(null);

  const [actionLoading, setActionLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState(null);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [actionType, setActionType] = useState(null);

  const fetchFlTelemetry = useCallback(async ({ silent = false } = {}) => {
    if (!silent) setLoading(true);
    setStatusError(null);
    setAccessDenied(false);
    try {
      const [expRes, orchRes] = await Promise.allSettled([
        apiGetLatestTabularExperiment(),
        apiGetFlOrchestrationStatus(),
      ]);

      if (expRes.status === 'fulfilled') {
        setLatestExperiment(unwrapApiData(expRes.value));
      }

      if (orchRes.status === 'fulfilled') {
        setOrchestration(unwrapApiData(orchRes.value));
      } else {
        const reason = orchRes.reason;
        const denied = reason instanceof ApiError && reason.status === 403;
        setAccessDenied(denied);
        const msg =
          reason instanceof ApiError
            ? `${reason.message} (HTTP ${reason.status})`
            : reason?.message || 'Failed to load SuperLink orchestration status';
        setStatusError(
          denied
            ? `${msg} Sign in as the seeded Super Admin (admin@medshield.org) — not Clinician or Hospital Admin.`
            : msg,
        );
        setOrchestration(null);
      }
    } catch (err) {
      setStatusError(err.message || 'Federated learning service query failed');
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFlTelemetry();
  }, [fetchFlTelemetry]);

  useEffect(() => {
    const lifecycle = orchestration?.lifecycle;
    if (!isFlowerRunActive(lifecycle)) {
      return undefined;
    }
    const id = setInterval(() => fetchFlTelemetry({ silent: true }), 2500);
    return () => clearInterval(id);
  }, [orchestration?.lifecycle, fetchFlTelemetry]);

  const handleExecuteAction = async () => {
    setShowConfirmModal(false);
    setActionLoading(true);
    setActionMessage(null);

    try {
      if (actionType === 'start') {
        const res = unwrapApiData(await apiStartFlCampaign(DEMO_ROUNDS));
        setActionMessage({
          type: 'success',
          text: `SuperLink demo run started (${DEMO_ROUNDS} rounds). Run ID: ${res?.run_id || '—'}`,
        });
        invalidatePlatformCapabilitiesCache();
        notifyFederatedChanged();
        await fetchFlTelemetry();
      } else if (actionType === 'stop') {
        await apiStopFlCampaign();
        setActionMessage({ type: 'success', text: 'Stop requested for SuperLink processes.' });
        invalidatePlatformCapabilitiesCache();
        notifyFederatedChanged();
        await fetchFlTelemetry();
      } else if (actionType === 'rounds') {
        const res = await apiGetFlRounds();
        setRoundHistory(res);
        const rounds = res?.rounds || [];
        setActionMessage({
          type: 'success',
          text:
            rounds.length > 0
              ? `Loaded ${rounds.length} round record(s) from the current/last SuperLink run.`
              : 'No round history on server yet (start a demo run first).',
        });
      }
    } catch (err) {
      setActionMessage({
        type: 'error',
        status: err.status,
        code: err.code,
        message: err.message || 'Request failed',
      });
    } finally {
      setActionLoading(false);
    }
  };

  const demoNodes = accessDenied || statusError ? [] : orchestration?.demo_supernodes || [];
  const onlineCount = demoNodes.filter((n) => n.connection_status === 'online').length;
  const startBlocked =
    accessDenied || statusError
      ? statusError || 'Orchestration status unavailable.'
      : demoStartBlockedReason(orchestration);
  const lifecycle = orchestration?.lifecycle;
  const canStop = isFlowerRunActive(lifecycle);

  return (
    <div className="federated-learning-panel">
      <div className="card" style={{ marginBottom: '1.25rem', border: '1px solid #1e293b' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
            <div className="card-icon" style={{ background: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6' }}>
              <Radio size={20} />
            </div>
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                Flower SuperLink Demo (synthetic / non-clinical)
              </h3>
              <p style={{ fontSize: '0.75rem', color: '#94a3b8', margin: 0 }}>
                Real SuperLink + two local SuperNodes; not production multi-hospital FL.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => fetchFlTelemetry()} disabled={loading}>
              <RefreshCw size={13} className={loading ? 'spin' : ''} style={{ marginRight: '4px' }} />
              Refresh
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => {
                setActionType('rounds');
                setShowConfirmModal(true);
              }}
              disabled={actionLoading}
            >
              <History size={13} style={{ marginRight: '4px' }} />
              Query Historical Rounds
            </button>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => {
                setActionType('start');
                setShowConfirmModal(true);
              }}
              disabled={actionLoading || Boolean(startBlocked)}
              title={startBlocked || 'Start SuperLink demo'}
            >
              <Play size={13} style={{ marginRight: '4px' }} />
              Start SuperLink run
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => {
                setActionType('stop');
                setShowConfirmModal(true);
              }}
              disabled={actionLoading || !canStop}
            >
              Stop
            </button>
          </div>
        </div>
      </div>

      {accessDenied && (
        <div className="notice-banner" style={{ borderLeftColor: '#f97316', marginBottom: '1.25rem', background: '#1c1917' }}>
          <AlertCircle size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
          <strong>Access denied — SuperLink demo requires Super Admin.</strong>
          <p style={{ margin: '0.35rem 0 0', fontSize: '0.8125rem', color: '#e2e8f0' }}>{statusError}</p>
          <span style={{ display: 'block', fontSize: '0.75rem', color: '#94a3b8', marginTop: '6px' }}>
            Demo account: <code>admin@medshield.org</code> (permission <code>federated:monitor</code>). Re-run{' '}
            <code>python scripts/seed_postgres_dev.py --non-interactive</code> if that user lacks{' '}
            <code>is_super_admin</code>.
          </span>
        </div>
      )}

      {statusError && !accessDenied && (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--warning)', marginBottom: '1.25rem', background: '#090d16' }}>
          <AlertCircle size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
          <strong>Status unavailable:</strong> {statusError}
        </div>
      )}

      {orchestration && !orchestration.orchestration_enabled && (
        <div className="notice-banner" style={{ borderLeftColor: '#64748b', marginBottom: '1.25rem', background: '#0f172a' }}>
          <p style={{ margin: 0, fontSize: '0.8125rem', color: '#cbd5e1' }}>
            <strong>Demo not armed on API server.</strong> {orchestration.reason}
          </p>
          <p style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', color: '#94a3b8' }}>
            {orchestration.demo_local_setup_hint}
          </p>
        </div>
      )}

      {orchestration && (
        <div className="card" style={{ marginBottom: '1.25rem', border: '1px solid #1e293b' }}>
          <div style={{ padding: '1rem', fontSize: '0.8125rem', color: '#cbd5e1' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
              <strong>Lifecycle:</strong>
              <span className="badge badge-sm">{flowerLifecycleLabel(lifecycle)}</span>
              <span style={{ color: '#64748b' }}>· {orchestration.summary}</span>
            </div>
            <div style={{ color: '#94a3b8' }}>{orchestration.reason}</div>
            <div style={{ marginTop: '0.35rem' }}>
              flwr {orchestration.flower_version || '—'} · mode {orchestration.deployment_mode || '—'} · TLS required:{' '}
              {String(orchestration.tls_required)} · security ready: {String(orchestration.security_configured)}
            </div>
            {(lifecycle === 'RUNNING' || lifecycle === 'COMPLETED' || lifecycle === 'FAILED') && (
              <div style={{ marginTop: '0.35rem' }}>
                Round {orchestration.current_round ?? '—'}/{orchestration.total_rounds ?? '—'} · last round OK clients{' '}
                {orchestration.successful_clients_last_round ?? '—'} · failed{' '}
                {orchestration.failed_clients_last_round ?? '—'}
              </div>
            )}
            {orchestration.initial_parameter_hash && (
              <div style={{ fontFamily: 'monospace', fontSize: '0.75rem', marginTop: '0.35rem' }}>
                param hash init → global: {orchestration.initial_parameter_hash} → {orchestration.global_parameter_hash || '—'}
              </div>
            )}
            {orchestration.last_error && (
              <div style={{ color: '#fca5a5', marginTop: '0.35rem' }}>Error: {orchestration.last_error}</div>
            )}
            {Array.isArray(orchestration.processes) && orchestration.processes.length > 0 && (
              <ul style={{ fontSize: '0.75rem', marginTop: '0.35rem' }}>
                {orchestration.processes.map((p) => (
                  <li key={p.name}>
                    {p.name} pid={p.pid} {p.alive ? 'alive' : 'stopped'}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      {actionMessage && (
        <div
          className="notice-banner"
          style={{
            borderLeftColor: actionMessage.type === 'error' ? 'var(--warning)' : 'var(--success)',
            marginBottom: '1.25rem',
            background: '#090d16',
          }}
        >
          <p style={{ margin: 0, fontSize: '0.8125rem', color: '#e2e8f0' }}>{actionMessage.message || actionMessage.text}</p>
        </div>
      )}

      {showConfirmModal && (
        <div
          className="modal-backdrop"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
          }}
        >
          <div className="card" style={{ maxWidth: '440px', width: '90%', border: '1px solid #334155', background: '#0f172a' }}>
            <div className="card-header">
              <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                {actionType === 'start' && 'Start SuperLink demo run'}
                {actionType === 'stop' && 'Stop SuperLink demo'}
                {actionType === 'rounds' && 'Query round history'}
              </h3>
            </div>
            <div style={{ padding: '1rem', fontSize: '0.875rem', color: '#cbd5e1' }}>
              {actionType === 'start' && (
                <p style={{ margin: 0 }}>
                  Start a synthetic {DEMO_ROUNDS}-round FedAvg run on two local SuperNodes (non-clinical data only).
                </p>
              )}
              {actionType === 'stop' && <p style={{ margin: 0 }}>Terminate SuperLink and SuperNode processes started by the supervisor.</p>}
              {actionType === 'rounds' && <p style={{ margin: 0 }}>Fetch round records from the backend (current or last SuperLink run).</p>}
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', padding: '0.75rem 1rem', borderTop: '1px solid #1e293b' }}>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setShowConfirmModal(false)}>
                Cancel
              </button>
              <button type="button" className="btn btn-primary btn-sm" onClick={handleExecuteAction}>
                Proceed
              </button>
            </div>
          </div>
        </div>
      )}

      {!accessDenied && (
      <div className="grid grid-2" style={{ gap: '1.25rem' }}>
        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Building2 size={18} className="text-accent" />
              <h4 style={{ fontSize: '0.9375rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                SuperLink Demo Nodes
              </h4>
            </div>
            <span className="badge badge-sm" style={{ background: '#1e293b' }}>
              {orchestration
                ? `${onlineCount} online / ${demoNodes.length} configured`
                : '—'}
            </span>
          </div>
          <div style={{ padding: '0.5rem 0' }}>
            {demoNodes.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {demoNodes.map((n) => {
                  const badge = connectionStatusBadge(n.connection_status);
                  return (
                    <div
                      key={n.demo_id}
                      style={{
                        background: '#090d16',
                        border: '1px solid #1e293b',
                        borderRadius: 'var(--radius-sm)',
                        padding: '0.625rem 0.75rem',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <strong style={{ fontSize: '0.8125rem', color: '#f1f5f9' }}>{n.display_name}</strong>
                          <span style={{ fontSize: '0.7rem', color: '#64748b', marginLeft: '6px', fontFamily: 'monospace' }}>
                            [{n.code}]
                          </span>
                        </div>
                        <span className={`badge badge-${badge.tone === 'success' ? 'success' : 'secondary'}`} style={{ fontSize: '0.65rem' }}>
                          {badge.label}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '4px' }}>
                        {n.node_id != null && <>Flower node_id {String(n.node_id)} · </>}
                        {n.process_pid ? <>pid {n.process_pid} · </> : null}
                        process {n.process_alive ? 'running' : 'not running'}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ padding: '1.5rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.8125rem' }}>
                {orchestration
                  ? 'No demo node telemetry yet. Start a SuperLink run or refresh.'
                  : 'Waiting for orchestration status…'}
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart3 size={18} className="text-primary" />
              <h4 style={{ fontSize: '0.9375rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                Round history (SuperLink)
              </h4>
            </div>
          </div>
          <div style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: '#cbd5e1' }}>
            {(roundHistory?.rounds?.length > 0 || orchestration?.history?.length > 0) && (
              <ul style={{ margin: 0, paddingLeft: '1.1rem' }}>
                {(roundHistory?.rounds || orchestration?.history || []).map((r) => (
                  <li key={`round-${r.round}`}>
                    Round {r.round}: {r.successful_clients ?? r.successful_clients_last_round ?? '—'} ok,{' '}
                    {r.failed_clients ?? r.failed_clients_last_round ?? 0} failed
                    {r.global_parameter_hash ? ` · hash ${r.global_parameter_hash}` : ''}
                  </li>
                ))}
              </ul>
            )}
            {!roundHistory?.rounds?.length && !orchestration?.history?.length && (
              <p style={{ margin: 0, color: '#94a3b8' }}>
                Use &quot;Query Historical Rounds&quot; or complete a demo run to populate round records.
              </p>
            )}
          </div>
        </div>
      </div>
      )}

      <details style={{ marginTop: '1.25rem', fontSize: '0.75rem', color: '#64748b' }}>
        <summary style={{ cursor: 'pointer', color: '#94a3b8' }}>Legacy tabular FL experiment (separate research path)</summary>
        <div className="card" style={{ marginTop: '0.5rem' }}>
          {latestExperiment ? (
            <pre style={{ padding: '0.75rem', overflow: 'auto', fontSize: '0.7rem' }}>{JSON.stringify(latestExperiment, null, 2)}</pre>
          ) : (
            <p style={{ padding: '1rem', margin: 0 }}>No saved tabular FL experiment on this node (unrelated to SuperLink demo).</p>
          )}
        </div>
      </details>
    </div>
  );
}
