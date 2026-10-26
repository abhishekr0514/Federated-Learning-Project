import React, { useState, useEffect } from 'react';
import { AdminLayout } from '../../layouts/AdminLayout';
import { FeatureNotImplemented } from '../../components/ErrorStates';
import { GatFederatedResearchPanel } from '../../components/admin/GatFederatedResearchPanel';
import { TabularFederatedResearchPanel } from '../../components/admin/TabularFederatedResearchPanel';
import { ModelStatusPanel } from '../../components/ModelStatusPanel';
import { DiagnosticWorkspace } from '../../components/DiagnosticWorkspace';
import { FederatedDemoShell } from '../../components/FederatedDemoShell';
import { apiGetAccessibleNodes, apiRequest } from '../../services/api';
import { Server, Building2, ShieldCheck, Activity, Cpu, Radio, Sparkles, History, Users, Database, AlertCircle, RefreshCw } from 'lucide-react';

export function SuperAdminDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [hospitals, setHospitals] = useState([]);
  const [loadingHospitals, setLoadingHospitals] = useState(false);
  const [overviewMetrics, setOverviewMetrics] = useState(null);
  const [loadingOverview, setLoadingOverview] = useState(false);
  const [auditLogs, setAuditLogs] = useState([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [loadingAudit, setLoadingAudit] = useState(false);
  const [auditError, setAuditError] = useState(null);

  useEffect(() => {
    if (activeTab === 'overview') {
      setLoadingOverview(true);
      apiRequest('/api/v1/admin/overview')
        .then((res) => {
          setOverviewMetrics(res?.data || res);
        })
        .catch((err) => {
          console.warn('Failed to load admin overview:', err);
        })
        .finally(() => setLoadingOverview(false));
    }
    if (activeTab === 'hospitals') {
      setLoadingHospitals(true);
      apiRequest('/api/v1/admin/hospitals')
        .then((res) => {
          const data = res?.data || res;
          setHospitals(data?.hospitals || []);
        })
        .catch((err) => {
          console.warn('Failed to load hospitals:', err);
          // Fallback to accessible-nodes
          return apiGetAccessibleNodes().then((r) => setHospitals(r?.hospitals || []));
        })
        .finally(() => setLoadingHospitals(false));
    }
    if (activeTab === 'audit') {
      setLoadingAudit(true);
      setAuditError(null);
      apiRequest('/api/v1/admin/audit?limit=50&offset=0')
        .then((res) => {
          const data = res?.data || res;
          setAuditLogs(data?.logs || []);
          setAuditTotal(data?.total_count || 0);
        })
        .catch((err) => {
          console.warn('Failed to load audit logs:', err);
          setAuditError(err.message || 'Failed to load audit log entries.');
        })
        .finally(() => setLoadingAudit(false));
    }
  }, [activeTab]);

  return (
    <AdminLayout activeTab={activeTab} onSelectTab={setActiveTab}>
      <header className="page-header" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Super Admin System Control</h1>
          <p style={{ color: 'var(--text-muted)' }}>
            Global federated architecture management, institutional nodes, and cross-site audit monitoring.
          </p>
        </div>

        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={() => setActiveTab('inference')}
          style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          <Activity size={14} />
          Diagnostic Studio
        </button>
      </header>

      {/* OVERVIEW TAB */}
      {activeTab === 'overview' && (
        <div>
          {/* Live DB Metrics Summary */}
          {loadingOverview && (
            <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem', textAlign: 'center' }}>
              <RefreshCw size={20} className="animate-spin" style={{ display: 'inline', marginRight: '8px' }} />
              Loading system metrics...
            </div>
          )}
          {overviewMetrics && (
            <div className="grid" style={{ marginBottom: '1.5rem' }}>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon text-info"><Building2 size={20} /></div>
                  <div className="card-title">Hospital Nodes</div>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700, margin: '0.5rem 0' }}>
                  {overviewMetrics.total_hospitals ?? 0}
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Active nodes registered</div>
              </div>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon text-info"><Users size={20} /></div>
                  <div className="card-title">Active Users</div>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700, margin: '0.5rem 0' }}>
                  {overviewMetrics.active_users ?? 0}
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Non-deleted accounts</div>
              </div>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon text-info"><ShieldCheck size={20} /></div>
                  <div className="card-title">Login Events</div>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700, margin: '0.5rem 0' }}>
                  {overviewMetrics.successful_logins ?? 0}
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                  {overviewMetrics.failed_logins ?? 0} failed attempts
                </div>
              </div>
              <div className="card">
                <div className="card-header">
                  <div className="card-icon text-info"><History size={20} /></div>
                  <div className="card-title">Audit Events</div>
                </div>
                <div style={{ fontSize: '2rem', fontWeight: 700, margin: '0.5rem 0' }}>
                  {overviewMetrics.audit_events_count ?? 0}
                </div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Persisted audit entries</div>
              </div>
            </div>
          )}

          {/* Live Neural Model Readiness Diagnostics */}
          <ModelStatusPanel />

          <div className="grid">
            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <Server size={20} />
                </div>
                <div className="card-title">Backend Architecture</div>
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                <p>FastAPI Backend v0.1.0 operational with PostgreSQL, Alembic migrations, and JWT RBAC security.</p>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <Building2 size={20} />
                </div>
                <div className="card-title">Hospital Institutional Nodes</div>
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                <p>Decentralized client nodes operate inside hospital boundaries with privacy masking enabled.</p>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <ShieldCheck size={20} />
                </div>
                <div className="card-title">Security & Audit Log</div>
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                <p>Append-only audit ledger tracking security events and permission authorization requests.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DIAGNOSTIC STUDIO TAB */}
      {activeTab === 'inference' && (
        <DiagnosticWorkspace />
      )}

      {/* HOSPITALS TAB */}
      {activeTab === 'hospitals' && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Registered Institutional Hospital Nodes</div>
          </div>
          <div style={{ padding: '1rem' }}>
            {loadingHospitals ? (
              <p style={{ color: 'var(--text-muted)' }}>Loading accessible hospital nodes...</p>
            ) : hospitals.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {hospitals.map((h) => (
                  <div
                    key={h.id}
                    style={{
                      background: '#090d16',
                      border: '1px solid #1e293b',
                      borderRadius: 'var(--radius-sm)',
                      padding: '0.75rem 1rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <strong style={{ color: '#f8fafc' }}>{h.name}</strong>
                      <span style={{ fontSize: '0.75rem', color: '#64748b', marginLeft: '8px', fontFamily: 'monospace' }}>
                        Code: {h.code}
                      </span>
                      {h.member_count !== undefined && (
                        <span style={{ fontSize: '0.75rem', color: '#64748b', marginLeft: '8px' }}>
                          · {h.member_count} member{h.member_count !== 1 ? 's' : ''}
                        </span>
                      )}
                      <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '2px', fontFamily: 'monospace' }}>
                        UUID: {h.id}
                      </div>
                    </div>
                    <span className="badge badge-success">Active Node</span>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>
                No hospital nodes returned. Run{' '}
                <code style={{ fontSize: '0.8rem' }}>python scripts/seed_demo_synthetic.py --non-interactive</code>{' '}
                (adds [SYNTHETIC] DEMO-METRO and DEMO-RIVERSIDE) or{' '}
                <code style={{ fontSize: '0.8rem' }}>python scripts/seed_postgres_dev.py --non-interactive</code> for a single dev hospital.
              </p>
            )}
          </div>
        </div>
      )}

      {/* FEDERATED LEARNING TAB */}
      {activeTab === 'federated' && (
        <div>
          <FederatedDemoShell />

          <div style={{ marginTop: '1.5rem' }}>
            <header style={{ marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 600 }}>Federated GAT Research Demo (Read-Only)</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                Persisted results from Flower SuperLink runs training the fused ECG + clinical-report GAT head on
                frozen PTB-XL embeddings — not live multi-hospital deployment.
              </p>
            </header>
            <GatFederatedResearchPanel />
            <details style={{ marginTop: '1.5rem' }}>
              <summary style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                Legacy tabular cardio_train.csv experiment (archived)
              </summary>
              <div style={{ marginTop: '0.75rem' }}>
                <TabularFederatedResearchPanel />
              </div>
            </details>
          </div>
        </div>
      )}

      {/* MODELS TAB */}
      {activeTab === 'models' && (
        <div>
          <ModelStatusPanel />
        </div>
      )}

      {activeTab === 'metrics' && (
        <FeatureNotImplemented
          title="Global System Metrics & ROC-AUC"
          moduleName="Global Evaluation Metrics"
          description="System ROC-AUC curves, sensitivity, and cross-institution validation metrics are disabled to prevent metric fabrication."
        />
      )}

      {activeTab === 'security' && (
        <FeatureNotImplemented
          title="Real-Time Security Event Monitor"
          moduleName="Security Event Monitor"
          description="Live suspicious authentication tracking and IP anomaly alerts are unexposed."
        />
      )}

      {activeTab === 'audit' && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div className="card-header" style={{ justifyContent: 'space-between' }}>
              <div>
                <div className="card-title">Append-Only Audit Log Ledger</div>
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Live entries from audit_logs table — populated by patient portal access, document operations, and assignments.
                </div>
              </div>
              <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                Total: {auditTotal}
              </span>
            </div>
          </div>

          {loadingAudit && (
            <div className="card" style={{ textAlign: 'center', padding: '2.5rem' }}>
              <RefreshCw size={24} className="animate-spin" style={{ display: 'inline', marginRight: '8px' }} />
              Loading audit log entries...
            </div>
          )}

          {auditError && (
            <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
              <AlertCircle size={28} style={{ display: 'block', margin: '0 auto 1rem', color: '#ef4444' }} />
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>{auditError}</p>
            </div>
          )}

          {!loadingAudit && !auditError && auditLogs.length === 0 && (
            <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
              <History size={36} style={{ display: 'block', margin: '0 auto 1rem', color: 'var(--text-muted)' }} />
              <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>No Audit Events Recorded</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '0.5rem', maxWidth: '480px', margin: '0.5rem auto 0' }}>
                Audit entries are written when patients access the portal, documents are uploaded, or patient assignments are made.
                Run the demo seed and log in as a patient to generate the first audit entries.
              </p>
            </div>
          )}

          {!loadingAudit && !auditError && auditLogs.length > 0 && (
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <div className="table-responsive">
                <table className="table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-color, #334155)', textAlign: 'left' }}>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Event Type</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Resource</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Actor</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Outcome</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditLogs.map((log) => (
                      <tr key={log.id} style={{ borderBottom: '1px solid var(--border-color, #1e293b)' }}>
                        <td style={{ padding: '0.875rem 1rem', fontFamily: 'monospace', fontSize: '0.8125rem', color: '#38bdf8' }}>
                          {log.event_type}
                        </td>
                        <td style={{ padding: '0.875rem 1rem', fontSize: '0.8125rem' }}>
                          <div>{log.resource_type}</div>
                          {log.resource_id && (
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                              {log.resource_id.slice(0, 18)}…
                            </div>
                          )}
                        </td>
                        <td style={{ padding: '0.875rem 1rem', fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                          {log.actor_id ? log.actor_id.slice(0, 13) + '…' : 'system'}
                        </td>
                        <td style={{ padding: '0.875rem 1rem' }}>
                          <span className={`badge ${log.outcome === 'SUCCESS' ? 'badge-success' : log.outcome === 'DENIED' ? 'badge-danger' : 'badge-warning'}`}>
                            {log.outcome}
                          </span>
                        </td>
                        <td style={{ padding: '0.875rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          {log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </AdminLayout>
  );
}
