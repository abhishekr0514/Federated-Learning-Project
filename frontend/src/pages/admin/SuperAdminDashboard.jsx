import React, { useState, useEffect } from 'react';
import { AdminLayout } from '../../layouts/AdminLayout';
import { FeatureNotImplemented } from '../../components/ErrorStates';
import { TabularFederatedResearchPanel } from '../../components/admin/TabularFederatedResearchPanel';
import { ModelStatusPanel } from '../../components/ModelStatusPanel';
import { DiagnosticWorkspace } from '../../components/DiagnosticWorkspace';
import { FederatedLearningPanel } from '../../components/FederatedLearningPanel';
import { apiGetAccessibleNodes } from '../../services/api';
import { Server, Building2, ShieldCheck, Activity, Cpu, Radio, Sparkles } from 'lucide-react';

export function SuperAdminDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [hospitals, setHospitals] = useState([]);
  const [loadingHospitals, setLoadingHospitals] = useState(false);

  useEffect(() => {
    if (activeTab === 'hospitals') {
      setLoadingHospitals(true);
      apiGetAccessibleNodes()
        .then((res) => {
          setHospitals(res.hospitals || []);
        })
        .catch((err) => {
          console.warn('Failed to load hospitals:', err);
        })
        .finally(() => setLoadingHospitals(false));
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
                      <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '2px', fontFamily: 'monospace' }}>
                        UUID: {h.id}
                      </div>
                    </div>
                    <span className="badge badge-success">Active Node</span>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>No hospital nodes returned.</p>
            )}
          </div>
        </div>
      )}

      {/* FEDERATED LEARNING TAB */}
      {activeTab === 'federated' && (
        <div>
          <FederatedLearningPanel />

          <div style={{ marginTop: '1.5rem' }}>
            <header style={{ marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 600 }}>Tabular Federated Research (Read-Only)</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                Saved cardio_train.csv simulation results — not live multi-hospital training.
              </p>
            </header>
            <TabularFederatedResearchPanel />
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
        <FeatureNotImplemented
          title="Append-Only Audit Log Ledger"
          moduleName="Audit Ledger"
          description="Global security event audit log viewer is pending implementation."
        />
      )}
    </AdminLayout>
  );
}
