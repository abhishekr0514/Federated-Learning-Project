import React, { useState, useEffect, useCallback } from 'react';
import { HospitalAdminLayout } from '../../layouts/HospitalAdminLayout';
import { apiRequest, ApiError } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { DiagnosticWorkspace } from '../../components/DiagnosticWorkspace';
import { ModelStatusPanel } from '../../components/ModelStatusPanel';
import { FederatedLearningPanel } from '../../components/FederatedLearningPanel';
import {
  Building2,
  Users,
  UserCheck,
  Cpu,
  BarChart3,
  History,
  Activity,
  ShieldCheck,
  Bell,
  FileSpreadsheet,
  Radio,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Database,
  Server,
  Lock,
} from 'lucide-react';

/**
 * Canonical State Badge Component
 */
export function StatusBadge({ status }) {
  const norm = (status || '').toUpperCase();
  let badgeClass = 'badge badge-muted';
  let label = norm || 'UNKNOWN';

  switch (norm) {
    case 'AVAILABLE':
    case 'SUCCESS':
    case 'ACTIVE':
    case 'RESOLVED':
      badgeClass = 'badge badge-success';
      break;
    case 'NO_DATA':
      badgeClass = 'badge badge-muted';
      label = 'NO DATA';
      break;
    case 'NOT_ENABLED':
      badgeClass = 'badge badge-warning';
      label = 'NOT ENABLED';
      break;
    case 'NOT_MEASURED':
      badgeClass = 'badge badge-info';
      label = 'NOT MEASURED';
      break;
    case 'NOT_IMPLEMENTED':
      badgeClass = 'badge badge-purple';
      label = 'NOT IMPLEMENTED';
      break;
    case 'ERROR':
    case 'CRITICAL':
    case 'DENIED':
      badgeClass = 'badge badge-danger';
      break;
    case 'WARNING':
    case 'IN_REVIEW':
    case 'OPEN':
      badgeClass = 'badge badge-warning';
      break;
    case 'INFO':
      badgeClass = 'badge badge-info';
      break;
    default:
      badgeClass = 'badge badge-muted';
  }

  return <span className={badgeClass}>{label}</span>;
}

/**
 * Metric Card Component with Explicit Operational States
 */
function MetricCard({ title, icon: Icon, state, value, subtitle, iconColor = 'text-info' }) {
  const isAvailable = state === 'AVAILABLE';

  return (
    <div className="card">
      <div className="card-header" style={{ justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className={`card-icon ${iconColor}`}>
            <Icon size={20} />
          </div>
          <div className="card-title">{title}</div>
        </div>
        <StatusBadge status={state} />
      </div>
      <div style={{ marginTop: '0.5rem' }}>
        {isAvailable ? (
          <div style={{ fontSize: '1.75rem', fontWeight: '700', color: 'var(--text-primary)' }}>
            {value}
          </div>
        ) : (
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            {subtitle || 'No stored records available'}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Honest Empty State Container
 */
function EmptyState({ title, description, icon: Icon = HelpCircle, status = 'NO_DATA' }) {
  return (
    <div className="honest-state-box">
      <div className="honest-state-icon" style={{ background: 'rgba(156, 163, 175, 0.1)', color: 'var(--text-muted)' }}>
        <Icon size={32} />
      </div>
      <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', maxWidth: '500px', margin: '0 auto 1rem' }}>
        {description}
      </p>
      <StatusBadge status={status} />
    </div>
  );
}

/**
 * Honest Not Enabled Container
 */
function NotEnabledBanner({ title, description, badge = 'NOT_ENABLED' }) {
  return (
    <div className="notice-banner" style={{ borderLeftColor: 'var(--warning)', background: 'rgba(245, 158, 11, 0.08)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
        <strong style={{ color: 'var(--warning)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertTriangle size={16} />
          {title}
        </strong>
        <StatusBadge status={badge} />
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>{description}</p>
    </div>
  );
}

/**
 * Main Hospital Admin Command Center Dashboard
 */
export function HospitalAdminDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [accessibleNodes, setAccessibleNodes] = useState([]);
  const [selectedHospitalId, setSelectedHospitalId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Tab Data States
  const [overviewData, setOverviewData] = useState(null);
  const [healthData, setHealthData] = useState(null);
  const [cliniciansData, setCliniciansData] = useState({ clinicians: [], total_count: 0 });
  const [patientsData, setPatientsData] = useState({ patients: [], total_count: 0 });
  const [modelsData, setModelsData] = useState({ models: [] });
  const [metricsData, setMetricsData] = useState({ metrics: [] });
  const [federatedData, setFederatedData] = useState({ historical_clients: [] });
  const [privacyData, setPrivacyData] = useState(null);
  const [qualityData, setQualityData] = useState({ events: [] });
  const [auditData, setAuditData] = useState({ logs: [], total_count: 0 });
  const [alertsData, setAlertsData] = useState({ alerts: [] });

  // Pagination offsets
  const [pagination, setPagination] = useState({
    cliniciansOffset: 0,
    patientsOffset: 0,
    auditOffset: 0,
  });
  const pageSize = 20;

  // 1. Fetch Accessible Nodes for Multi-Hospital Selector
  useEffect(() => {
    async function loadAccessibleNodes() {
      try {
        const res = await apiRequest('/api/v1/hospital/accessible-nodes');
        if (res && res.hospitals && res.hospitals.length > 0) {
          setAccessibleNodes(res.hospitals);
          setSelectedHospitalId(res.hospitals[0].id);
        }
      } catch (err) {
        console.warn('Could not load accessible hospital nodes:', err);
      }
    }
    loadAccessibleNodes();
  }, []);

  // Construct Query String for Scoped Requests
  const getHospitalParam = useCallback(() => {
    return selectedHospitalId ? `?hospital_id=${selectedHospitalId}` : '';
  }, [selectedHospitalId]);

  // 2. Fetch Data for Active Tab
  const fetchActiveTabData = useCallback(async () => {
    setLoading(true);
    setError(null);
    const hospParam = getHospitalParam();
    const joinParam = (param) => (hospParam ? `${hospParam}&${param}` : `?${param}`);

    try {
      if (activeTab === 'overview') {
        const res = await apiRequest(`/api/v1/hospital/overview${hospParam}`);
        setOverviewData(res);
      } else if (activeTab === 'system_health') {
        const res = await apiRequest(`/api/v1/hospital/system-health${hospParam}`);
        setHealthData(res);
      } else if (activeTab === 'clinicians') {
        const res = await apiRequest(
          `/api/v1/hospital/clinicians${joinParam(`limit=${pageSize}&offset=${pagination.cliniciansOffset}`)}`
        );
        setCliniciansData(res);
      } else if (activeTab === 'patients') {
        const res = await apiRequest(
          `/api/v1/hospital/patients${joinParam(`limit=${pageSize}&offset=${pagination.patientsOffset}`)}`
        );
        setPatientsData(res);
      } else if (activeTab === 'models') {
        const res = await apiRequest(`/api/v1/hospital/models${hospParam}`);
        setModelsData(res);
      } else if (activeTab === 'metrics') {
        const res = await apiRequest(`/api/v1/hospital/metrics${hospParam}`);
        setMetricsData(res);
      } else if (activeTab === 'federated') {
        const res = await apiRequest(`/api/v1/hospital/federated${hospParam}`);
        setFederatedData(res);
      } else if (activeTab === 'privacy') {
        const res = await apiRequest(`/api/v1/hospital/privacy${hospParam}`);
        setPrivacyData(res);
      } else if (activeTab === 'data_quality') {
        const res = await apiRequest(`/api/v1/hospital/data-quality${hospParam}`);
        setQualityData(res);
      } else if (activeTab === 'audit') {
        const res = await apiRequest(
          `/api/v1/hospital/audit${joinParam(`limit=${pageSize}&offset=${pagination.auditOffset}`)}`
        );
        setAuditData(res);
      } else if (activeTab === 'alerts') {
        const res = await apiRequest(`/api/v1/hospital/alerts${hospParam}`);
        setAlertsData(res);
      }
    } catch (err) {
      setError(err.message || 'Failed to retrieve hospital telemetry');
    } finally {
      setLoading(false);
    }
  }, [activeTab, getHospitalParam, pagination, pageSize]);

  useEffect(() => {
    fetchActiveTabData();
  }, [fetchActiveTabData]);

  const activeHospitalObj = accessibleNodes.find((h) => h.id === selectedHospitalId);

  return (
    <HospitalAdminLayout activeTab={activeTab} onSelectTab={setActiveTab}>
      {/* Top Header & Hospital Selector */}
      <header
        className="page-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '1.5rem',
          flexWrap: 'wrap',
          gap: '1rem',
        }}
      >
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Hospital Command Center</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            Institutional Node Isolation & Compliance Dashboard. Strictly server-scoped telemetry.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {accessibleNodes.length > 1 && (
            <div className="hospital-badge">
              <Building2 size={16} />
              <select
                className="hospital-dropdown"
                value={selectedHospitalId}
                onChange={(e) => setSelectedHospitalId(e.target.value)}
              >
                {accessibleNodes.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.name} ({h.code})
                  </option>
                ))}
              </select>
            </div>
          )}

          {accessibleNodes.length === 1 && (
            <div className="hospital-badge">
              <Building2 size={16} />
              <span>
                {activeHospitalObj?.name || 'Local Hospital'} ({activeHospitalObj?.code || 'NODE'})
              </span>
            </div>
          )}

          <button
            className="btn btn-secondary btn-sm"
            onClick={fetchActiveTabData}
            disabled={loading}
            title="Refresh current tab"
          >
            <RefreshCw size={14} className={loading ? 'spinner' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </header>

      {/* Global Error Notice */}
      {error && (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--danger)', marginBottom: '1.5rem' }}>
          <strong style={{ color: 'var(--danger)' }}>Telemetry Fetch Error:</strong> {error}
        </div>
      )}

      {/* =========================================================================
          TAB: OVERVIEW
      ========================================================================= */}
      {activeTab === 'overview' && (
        <div>
          {/* Live Model Architecture Readiness Diagnostics */}
          <ModelStatusPanel />

          <NotEnabledBanner
            title="Training Subsystems Notice"
            description="Flower federated training and local model gradient updates are intentionally NOT enabled in this deployment. All metrics shown reflect historical offline baseline evaluations or zero-mock empty states."
          />

          <div className="grid">
            <MetricCard
              title="Authorized Clinicians"
              icon={UserCheck}
              state={overviewData?.clinicians?.status || 'NO_DATA'}
              value={overviewData?.clinicians?.count}
              subtitle={overviewData?.clinicians?.message}
            />

            <MetricCard
              title="Opaque Patient Hashes"
              icon={Users}
              state={overviewData?.patients?.status || 'NO_DATA'}
              value={overviewData?.patients?.count}
              subtitle={overviewData?.patients?.message}
            />

            <MetricCard
              title="Federated Clients"
              icon={Cpu}
              state={overviewData?.federated_clients?.status || 'NO_DATA'}
              value={overviewData?.federated_clients?.count}
              subtitle={overviewData?.federated_clients?.message}
            />

            <MetricCard
              title="Latest Local Metric"
              icon={BarChart3}
              state={overviewData?.latest_metric?.status || 'NO_DATA'}
              value={
                overviewData?.latest_metric?.metric_value !== undefined
                  ? `${overviewData.latest_metric.metric_name}: ${overviewData.latest_metric.metric_value}`
                  : null
              }
              subtitle="No local evaluation metric recorded"
            />
          </div>

          <div className="grid" style={{ marginTop: '1.25rem' }}>
            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <Activity size={20} />
                </div>
                <div className="card-title">Latest Model Version</div>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                {overviewData?.latest_model_version?.status === 'AVAILABLE' ? (
                  <div>
                    <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>
                      {overviewData.latest_model_version.version_tag}
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.8125rem', marginTop: '0.25rem' }}>
                      Target: {overviewData.latest_model_version.target_definition}
                    </div>
                  </div>
                ) : (
                  <StatusBadge status="NO_DATA" />
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <History size={20} />
                </div>
                <div className="card-title">Latest Node Activity</div>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                {overviewData?.latest_audit_activity?.status === 'AVAILABLE' ? (
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {new Date(overviewData.latest_audit_activity.timestamp).toLocaleString()}
                  </span>
                ) : (
                  <StatusBadge status="NO_DATA" />
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <Bell size={20} />
                </div>
                <div className="card-title">Latest Node Alert</div>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                {overviewData?.latest_alert?.status === 'AVAILABLE' ? (
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {new Date(overviewData.latest_alert.timestamp).toLocaleString()}
                  </span>
                ) : (
                  <StatusBadge status="NO_DATA" />
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =========================================================================
          TAB: DIAGNOSTIC STUDIO (INFERENCE)
      ========================================================================= */}
      {activeTab === 'inference' && (
        <DiagnosticWorkspace />
      )}

      {/* =========================================================================
          TAB: SYSTEM HEALTH
      ========================================================================= */}
      {activeTab === 'system_health' && (
        <div>
          <div className="grid">
            <MetricCard
              title="API Service Health"
              icon={Server}
              state={healthData?.api_health?.status || 'AVAILABLE'}
              value="AVAILABLE"
              subtitle="FastAPI v0.1.0 operational"
            />

            <MetricCard
              title="PostgreSQL / DB Latency"
              icon={Database}
              state={healthData?.database?.status || 'AVAILABLE'}
              value={healthData?.database?.latency_ms !== null ? `${healthData?.database?.latency_ms} ms` : 'N/A'}
              subtitle="Roundtrip SQL Ping"
            />

            <MetricCard
              title="Active User Sessions"
              icon={Users}
              state="AVAILABLE"
              value={healthData?.active_sessions_count || 0}
              subtitle="Authenticated active tokens"
            />
          </div>

          <div style={{ marginTop: '1.5rem' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>
              Infrastructure Hardware Telemetry
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8125rem', marginBottom: '1rem' }}>
              The following hardware metrics are unmeasured by current server configuration to prevent metric fabrication:
            </p>

            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Component</th>
                    <th>Measurement Type</th>
                    <th>Status</th>
                    <th>Reported Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>CPU Utilization</td>
                    <td>Host Hardware</td>
                    <td><StatusBadge status={healthData?.system_hardware?.cpu_usage || 'NOT_MEASURED'} /></td>
                    <td style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Unmeasured</td>
                  </tr>
                  <tr>
                    <td>Memory (RAM) Usage</td>
                    <td>Host Hardware</td>
                    <td><StatusBadge status={healthData?.system_hardware?.memory_usage || 'NOT_MEASURED'} /></td>
                    <td style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Unmeasured</td>
                  </tr>
                  <tr>
                    <td>Disk I/O Utilization</td>
                    <td>Storage Volume</td>
                    <td><StatusBadge status={healthData?.system_hardware?.disk_usage || 'NOT_MEASURED'} /></td>
                    <td style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Unmeasured</td>
                  </tr>
                  <tr>
                    <td>GPU Compute Telemetry</td>
                    <td>Accelerator</td>
                    <td><StatusBadge status={healthData?.system_hardware?.gpu_usage || 'NOT_MEASURED'} /></td>
                    <td style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Unmeasured</td>
                  </tr>
                  <tr>
                    <td>Worker Process Health</td>
                    <td>Background Celery/Worker</td>
                    <td><StatusBadge status={healthData?.system_hardware?.worker_health || 'NOT_IMPLEMENTED'} /></td>
                    <td style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Not Implemented</td>
                  </tr>
                  <tr>
                    <td>Inference Queue Depth</td>
                    <td>Queue Backlog</td>
                    <td><StatusBadge status={healthData?.system_hardware?.queue_depth || 'NOT_IMPLEMENTED'} /></td>
                    <td style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Not Implemented</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* =========================================================================
          TAB: CLINICIANS
      ========================================================================= */}
      {activeTab === 'clinicians' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: '600' }}>Hospital Clinician Directory</h2>
            <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
              Total: {cliniciansData.total_count || 0}
            </span>
          </div>

          {cliniciansData.clinicians && cliniciansData.clinicians.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Clinician ID</th>
                    <th>User ID</th>
                    <th>Department</th>
                    <th>Status</th>
                    <th>Created At</th>
                  </tr>
                </thead>
                <tbody>
                  {cliniciansData.clinicians.map((c) => (
                    <tr key={c.id}>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem' }}>{c.id}</td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem' }}>{c.user_id}</td>
                      <td>{c.department || 'General Clinical'}</td>
                      <td><StatusBadge status={c.user_status} /></td>
                      <td>{c.created_at ? new Date(c.created_at).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="pagination-bar">
                <span>
                  Showing {cliniciansData.clinicians.length} of {cliniciansData.total_count} clinicians
                </span>
                <div className="pagination-actions">
                  <button
                    className="btn btn-secondary btn-sm"
                    disabled={pagination.cliniciansOffset === 0}
                    onClick={() =>
                      setPagination((p) => ({ ...p, cliniciansOffset: Math.max(0, p.cliniciansOffset - pageSize) }))
                    }
                  >
                    <ChevronLeft size={14} /> Previous
                  </button>
                  <button
                    className="btn btn-secondary btn-sm"
                    disabled={pagination.cliniciansOffset + pageSize >= (cliniciansData.total_count || 0)}
                    onClick={() =>
                      setPagination((p) => ({ ...p, cliniciansOffset: p.cliniciansOffset + pageSize }))
                    }
                  >
                    Next <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <EmptyState
              title="No Clinician Records Available"
              description="There are currently zero clinician membership records registered for this hospital node."
              icon={UserCheck}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: PATIENTS
      ========================================================================= */}
      {activeTab === 'patients' && (
        <div>
          <div className="notice-banner" style={{ borderLeftColor: 'var(--accent)', marginBottom: '1rem' }}>
            <strong style={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Lock size={16} /> Privacy Boundary Strict Enforcement
            </strong>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
              Central database contains strictly opaque SHA-256 patient references. Zero patient names, MRNs, notes, or raw ECG signal arrays are accessible centrally.
            </p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: '600' }}>Opaque Patient Reference Directory</h2>
            <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
              Total: {patientsData.total_count || 0}
            </span>
          </div>

          {patientsData.patients && patientsData.patients.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Association ID</th>
                    <th>Patient UUID</th>
                    <th>Opaque Hash Reference</th>
                    <th>Registered At</th>
                  </tr>
                </thead>
                <tbody>
                  {patientsData.patients.map((p) => (
                    <tr key={p.id}>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem' }}>{p.id}</td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem' }}>{p.patient_id}</td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem', color: 'var(--accent)' }}>
                        {p.opaque_patient_id}
                      </td>
                      <td>{p.created_at ? new Date(p.created_at).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="pagination-bar">
                <span>
                  Showing {patientsData.patients.length} of {patientsData.total_count} references
                </span>
                <div className="pagination-actions">
                  <button
                    className="btn btn-secondary btn-sm"
                    disabled={pagination.patientsOffset === 0}
                    onClick={() =>
                      setPagination((p) => ({ ...p, patientsOffset: Math.max(0, p.patientsOffset - pageSize) }))
                    }
                  >
                    <ChevronLeft size={14} /> Previous
                  </button>
                  <button
                    className="btn btn-secondary btn-sm"
                    disabled={pagination.patientsOffset + pageSize >= (patientsData.total_count || 0)}
                    onClick={() =>
                      setPagination((p) => ({ ...p, patientsOffset: p.patientsOffset + pageSize }))
                    }
                  >
                    Next <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <EmptyState
              title="No Patient References Recorded"
              description="No patient records have been associated with this institutional hospital node yet."
              icon={Users}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: MODELS
      ========================================================================= */}
      {activeTab === 'models' && (
        <div>
          <NotEnabledBanner
            title="Local Weight Training Not Enabled"
            description="Local training routines and gradient checkpoint generation are not enabled. Registered models reflect global architecture configurations."
          />

          <h2 style={{ fontSize: '1.125rem', fontWeight: '600', marginTop: '1rem', marginBottom: '0.5rem' }}>
            Registered Neural Model Architectures
          </h2>

          {modelsData.models && modelsData.models.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Version Tag</th>
                    <th>Model Family</th>
                    <th>Task & Target Definition</th>
                    <th>Status</th>
                    <th>Created At</th>
                  </tr>
                </thead>
                <tbody>
                  {modelsData.models.map((m) => (
                    <tr key={m.id}>
                      <td style={{ fontWeight: '600', color: 'var(--primary)' }}>{m.version_tag}</td>
                      <td>{m.model_family}</td>
                      <td style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>{m.target_definition}</td>
                      <td><StatusBadge status={m.status} /></td>
                      <td>{m.created_at ? new Date(m.created_at).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No Model Versions Available"
              description="Zero model version records found in the registry."
              icon={Cpu}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: METRICS
      ========================================================================= */}
      {activeTab === 'metrics' && (
        <div>
          <div className="notice-banner" style={{ borderLeftColor: 'var(--info)', marginBottom: '1rem' }}>
            <strong style={{ color: 'var(--info)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart3 size={16} /> Scientific Integrity & Metric Provenance
            </strong>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
              Historical PTB-XL proxy experiment is NOT a production heart-disease diagnostic metric. Declared task targets are shown explicitly with every metric.
            </p>
          </div>

          <h2 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>
            Hospital Node Evaluation Metrics
          </h2>

          {metricsData.metrics && metricsData.metrics.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Metric Name</th>
                    <th>Reported Value</th>
                    <th>Model Version</th>
                    <th>Declared Task Target</th>
                    <th>Evaluated At</th>
                  </tr>
                </thead>
                <tbody>
                  {metricsData.metrics.map((m) => (
                    <tr key={m.id}>
                      <td style={{ fontWeight: '600' }}>{m.metric_name}</td>
                      <td style={{ fontSize: '1.125rem', fontWeight: '700', color: 'var(--success)' }}>
                        {m.metric_value}
                      </td>
                      <td>{m.model_version || 'Global Model'}</td>
                      <td style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>{m.target_definition}</td>
                      <td>{m.evaluated_at ? new Date(m.evaluated_at).toLocaleString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No Evaluation Results Available"
              description="There are currently zero stored evaluation metrics for this hospital node."
              icon={BarChart3}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: FEDERATED
      ========================================================================= */}
      {activeTab === 'federated' && (
        <div>
          {/* Live Federated Learning Telemetry & Orchestration */}
          <FederatedLearningPanel />

          <NotEnabledBanner
            title="Flower Federated Aggregation Disabled"
            description="Federated learning Flower aggregation engine is not active. Central server does not broadcast training rounds to clients."
            badge="NOT_ENABLED"
          />

          <h2 style={{ fontSize: '1.125rem', fontWeight: '600', marginTop: '1.25rem', marginBottom: '0.5rem' }}>
            Registered Hospital Federated Clients
          </h2>

          {federatedData.historical_clients && federatedData.historical_clients.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Client Identifier</th>
                    <th>Status</th>
                    <th>Classification</th>
                    <th>Registered At</th>
                  </tr>
                </thead>
                <tbody>
                  {federatedData.historical_clients.map((fc) => (
                    <tr key={fc.id}>
                      <td style={{ fontWeight: '600' }}>{fc.client_identifier}</td>
                      <td><StatusBadge status={fc.status} /></td>
                      <td><StatusBadge status="INFO" /> <span style={{ fontSize: '0.75rem', marginLeft: '4px' }}>HISTORICAL</span></td>
                      <td>{fc.registered_at ? new Date(fc.registered_at).toLocaleString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No Federated Clients Registered"
              description="Zero federated client nodes are registered for this hospital."
              icon={Radio}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: PRIVACY
      ========================================================================= */}
      {activeTab === 'privacy' && (
        <div>
          <div className="notice-banner" style={{ borderLeftColor: 'var(--success)', marginBottom: '1rem' }}>
            <strong style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldCheck size={16} /> Privacy Contract Guarantee
            </strong>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
              Zero raw notes, clinical text, PII entity spans, or replacement maps are serialized in responses.
            </p>
          </div>

          <div className="grid">
            <MetricCard
              title="Documents Sanitized"
              icon={ShieldCheck}
              state={privacyData?.status || 'NO_DATA'}
              value={privacyData?.metrics?.documents_processed}
              subtitle="Zero documents processed"
            />

            <MetricCard
              title="Entities Redacted"
              icon={ShieldCheck}
              state={privacyData?.status || 'NO_DATA'}
              value={privacyData?.metrics?.entities_detected}
              subtitle="Zero entity redactions"
            />

            <MetricCard
              title="Sanitization Status"
              icon={CheckCircle2}
              state={privacyData?.status || 'NO_DATA'}
              value={privacyData?.metrics?.sanitization_status?.toUpperCase()}
              subtitle="No privacy jobs recorded"
            />
          </div>

          {privacyData?.status !== 'AVAILABLE' && (
            <EmptyState
              title="No Privacy Masking Records Available"
              description="No clinical document privacy sanitization jobs have been executed for this node."
              icon={ShieldCheck}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: DATA QUALITY
      ========================================================================= */}
      {activeTab === 'data_quality' && (
        <div>
          <h2 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>
            Hospital Data Quality Events
          </h2>

          {qualityData.events && qualityData.events.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Severity</th>
                    <th>Status</th>
                    <th>Created At</th>
                    <th>Resolved At</th>
                  </tr>
                </thead>
                <tbody>
                  {qualityData.events.map((e) => (
                    <tr key={e.id}>
                      <td style={{ fontWeight: '500' }}>{e.category}</td>
                      <td><StatusBadge status={e.severity} /></td>
                      <td><StatusBadge status={e.status} /></td>
                      <td>{e.created_at ? new Date(e.created_at).toLocaleString() : 'N/A'}</td>
                      <td>{e.resolved_at ? new Date(e.resolved_at).toLocaleString() : 'Unresolved'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No Data Quality Events Recorded"
              description="Zero data quality anomalies or validation exceptions recorded for this hospital node."
              icon={FileSpreadsheet}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: AUDIT LOG
      ========================================================================= */}
      {activeTab === 'audit' && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: '600' }}>Hospital Node Immutable Audit Ledger</h2>
            <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
              Total: {auditData.total_count || 0}
            </span>
          </div>

          {auditData.logs && auditData.logs.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Event Type</th>
                    <th>Resource</th>
                    <th>Request ID</th>
                    <th>Outcome</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {auditData.logs.map((log) => (
                    <tr key={log.id}>
                      <td style={{ fontWeight: '600' }}>{log.event_type}</td>
                      <td>{log.resource_type}</td>
                      <td style={{ fontFamily: 'monospace', fontSize: '0.8125rem' }}>{log.request_id}</td>
                      <td><StatusBadge status={log.outcome} /></td>
                      <td>{log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="pagination-bar">
                <span>
                  Showing {auditData.logs.length} of {auditData.total_count} audit records
                </span>
                <div className="pagination-actions">
                  <button
                    className="btn btn-secondary btn-sm"
                    disabled={pagination.auditOffset === 0}
                    onClick={() =>
                      setPagination((p) => ({ ...p, auditOffset: Math.max(0, p.auditOffset - pageSize) }))
                    }
                  >
                    <ChevronLeft size={14} /> Previous
                  </button>
                  <button
                    className="btn btn-secondary btn-sm"
                    disabled={pagination.auditOffset + pageSize >= (auditData.total_count || 0)}
                    onClick={() =>
                      setPagination((p) => ({ ...p, auditOffset: p.auditOffset + pageSize }))
                    }
                  >
                    Next <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <EmptyState
              title="No Audit Records Stored"
              description="Zero audit log events recorded for this hospital node."
              icon={History}
              status="NO_DATA"
            />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB: ALERTS
      ========================================================================= */}
      {activeTab === 'alerts' && (
        <div>
          <NotEnabledBanner
            title="Automatic Alert Generation Not Enabled"
            description="Background telemetry heuristics and automatic anomaly alert generation are disabled in this environment."
            badge="NOT_ENABLED"
          />

          <h2 style={{ fontSize: '1.125rem', fontWeight: '600', marginTop: '1.25rem', marginBottom: '0.5rem' }}>
            Hospital Node Operational Alerts
          </h2>

          {alertsData.alerts && alertsData.alerts.length > 0 ? (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Title</th>
                    <th>Category</th>
                    <th>Severity</th>
                    <th>Status</th>
                    <th>Recorded At</th>
                  </tr>
                </thead>
                <tbody>
                  {alertsData.alerts.map((a) => (
                    <tr key={a.id}>
                      <td style={{ fontWeight: '600' }}>{a.title}</td>
                      <td>{a.category}</td>
                      <td><StatusBadge status={a.severity} /></td>
                      <td><StatusBadge status={a.status} /></td>
                      <td>{a.created_at ? new Date(a.created_at).toLocaleString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              title="No Operational Alerts"
              description="Zero active or historical alerts recorded for this hospital node."
              icon={Bell}
              status="NO_DATA"
            />
          )}
        </div>
      )}
    </HospitalAdminLayout>
  );
}
