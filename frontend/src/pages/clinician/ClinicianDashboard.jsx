import React, { useState, useEffect, useCallback } from 'react';
import { ClinicianLayout } from '../../layouts/ClinicianLayout';
import {
  apiRequest,
  ApiError,
  apiUploadClinicalDocument,
  apiProcessClinicalDocument,
} from '../../services/api';
import { DiagnosticWorkspace } from '../../components/DiagnosticWorkspace';
import { ModelStatusPanel } from '../../components/ModelStatusPanel';
import { FederatedLearningPanel } from '../../components/FederatedLearningPanel';
import {
  ClinicalEmptyState,
  ClinicalNotImplemented,
  ClinicalNotEnabled,
  ClinicalForbidden,
  ClinicalNotFound,
  ClinicalError,
  ClinicalLoading,
  ClinicalStatusBadge,
} from '../../components/ClinicalStates';
import {
  Stethoscope,
  FileText,
  Activity,
  Cpu,
  BarChart3,
  ShieldCheck,
  ChevronRight,
  ArrowLeft,
  ChevronDown,
  Info,
  Layers,
  Lock,
  RefreshCw,
  Search,
  CheckCircle,
  FileSpreadsheet,
  Upload,
  Play,
  Sparkles,
} from 'lucide-react';

export function ClinicianDashboard() {
  const [activeTab, setActiveTab] = useState('dashboard');

  // Loading & error states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Overview data
  const [overview, setOverview] = useState(null);

  // Patients data & pagination
  const [patients, setPatients] = useState([]);
  const [patientTotal, setPatientTotal] = useState(0);
  const [patientPage, setPatientPage] = useState(0);
  const [patientSearch, setPatientSearch] = useState('');
  const patientLimit = 20;

  const [eligiblePatients, setEligiblePatients] = useState([]);
  const [eligibleLoading, setEligibleLoading] = useState(false);
  const [assigningPatientId, setAssigningPatientId] = useState(null);
  const [assignError, setAssignError] = useState(null);

  // Selected patient for dossier drill-down
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [patientDossier, setPatientDossier] = useState(null);
  const [dossierLoading, setDossierLoading] = useState(false);
  const [dossierError, setDossierError] = useState(null);

  // Accordion state for dossier
  const [openSections, setOpenSections] = useState({
    whyPrediction: false,
    whatChange: false,
    causalInsights: false,
    modelInfo: false,
    techDetails: false,
  });

  const toggleSection = (key) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Documents data & pagination
  const [documents, setDocuments] = useState([]);
  const [documentTotal, setDocumentTotal] = useState(0);
  const [documentPage, setDocumentPage] = useState(0);
  const documentLimit = 20;

  // Selected document for inspection
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docDetailLoading, setDocDetailLoading] = useState(false);

  // Diagnosis data
  const [diagnosisData, setDiagnosisData] = useState(null);

  // Document upload modal state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadPatientId, setUploadPatientId] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [processingDocId, setProcessingDocId] = useState(null);

  const handleUploadDocument = async (e) => {
    e.preventDefault();
    if (!uploadFile) {
      setUploadError('Please select a medical document to upload.');
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const formData = new FormData();
      formData.append('file', uploadFile);
      if (uploadPatientId) {
        formData.append('patient_id', uploadPatientId);
      }

      const res = await apiUploadClinicalDocument(formData);
      setUploadSuccess(`Document ${uploadFile.name} uploaded successfully.`);
      setUploadFile(null);
      fetchDocuments();

      // Trigger automatic process job
      const newDocId = res?.document?.id || res?.data?.id || res?.id;
      if (newDocId) {
        setProcessingDocId(newDocId);
        try {
          await apiProcessClinicalDocument(newDocId);
          fetchDocuments();
        } catch (procErr) {
          console.warn('Post-upload processing job error:', procErr);
        } finally {
          setProcessingDocId(null);
        }
      }
    } catch (err) {
      setUploadError(err.message || 'Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  const handleProcessDocument = async (docId) => {
    setProcessingDocId(docId);
    try {
      await apiProcessClinicalDocument(docId);
      fetchDocuments();
    } catch (err) {
      console.error('Document processing failed:', err);
    } finally {
      setProcessingDocId(null);
    }
  };

  // Fetch overview telemetry
  const fetchOverview = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiRequest('/api/v1/clinician/overview');
      setOverview(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch assigned patients
  const fetchPatients = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const offset = patientPage * patientLimit;
      const data = await apiRequest(`/api/v1/clinician/patients?limit=${patientLimit}&offset=${offset}`);
      setPatients(data.patients || []);
      setPatientTotal(data.total_count || 0);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [patientPage]);

  const fetchEligiblePatients = useCallback(async () => {
    setEligibleLoading(true);
    setAssignError(null);
    try {
      const data = await apiRequest('/api/v1/clinician/patients/eligible?limit=50&offset=0');
      setEligiblePatients(data.patients || []);
    } catch (err) {
      setAssignError(err);
      setEligiblePatients([]);
    } finally {
      setEligibleLoading(false);
    }
  }, []);

  const handleAssignPatient = async (patientId) => {
    setAssigningPatientId(patientId);
    setAssignError(null);
    try {
      await apiRequest(`/api/v1/clinician/patients/${patientId}/assignment`, { method: 'POST' });
      await fetchPatients();
      await fetchEligiblePatients();
      if (activeTab === 'dashboard') {
        fetchOverview();
      }
    } catch (err) {
      setAssignError(err);
    } finally {
      setAssigningPatientId(null);
    }
  };

  // Fetch patient dossier detail
  const fetchPatientDossier = useCallback(async (patientId) => {
    setDossierLoading(true);
    setDossierError(null);
    try {
      const data = await apiRequest(`/api/v1/clinician/patients/${patientId}`);
      setPatientDossier(data);
    } catch (err) {
      setDossierError(err);
    } finally {
      setDossierLoading(false);
    }
  }, []);

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const offset = documentPage * documentLimit;
      const data = await apiRequest(`/api/v1/clinician/documents?limit=${documentLimit}&offset=${offset}`);
      setDocuments(data.documents || []);
      setDocumentTotal(data.total_count || 0);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [documentPage]);

  // Fetch document detail
  const fetchDocumentDetail = async (docId) => {
    setDocDetailLoading(true);
    try {
      const data = await apiRequest(`/api/v1/clinician/documents/${docId}`);
      setSelectedDoc(data);
    } catch (err) {
      setError(err);
    } finally {
      setDocDetailLoading(false);
    }
  };

  // Fetch diagnosis predictions
  const fetchDiagnosis = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiRequest('/api/v1/clinician/diagnosis');
      setDiagnosisData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Switch tabs and load relevant data
  useEffect(() => {
    if (activeTab === 'dashboard') {
      fetchOverview();
    } else if (activeTab === 'patients') {
      fetchPatients();
      if (!selectedPatientId) {
        fetchEligiblePatients();
      }
    } else if (activeTab === 'documents') {
      fetchDocuments();
    } else if (activeTab === 'diagnosis') {
      fetchDiagnosis();
    }
  }, [activeTab, fetchOverview, fetchPatients, fetchEligiblePatients, fetchDocuments, fetchDiagnosis, selectedPatientId]);

  // When patient drilldown is opened
  const handleSelectPatient = (patientId) => {
    setSelectedPatientId(patientId);
    fetchPatientDossier(patientId);
  };

  const handleBackToPatients = () => {
    setSelectedPatientId(null);
    setPatientDossier(null);
    setDossierError(null);
  };

  // Filter patients by search term
  const filteredPatients = patients.filter((p) => {
    if (!patientSearch) return true;
    return p.opaque_patient_id.toLowerCase().includes(patientSearch.toLowerCase());
  });

  return (
    <ClinicianLayout activeTab={activeTab} onSelectTab={setActiveTab}>
      <header className="page-header" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Clinician Diagnostic Workspace</h1>
            <p style={{ color: 'var(--text-muted)' }}>
              Assigned patient diagnostics, clinical metadata dossiers, and AI capability telemetry.
            </p>
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => {
              if (activeTab === 'dashboard') fetchOverview();
              else if (activeTab === 'patients') {
                if (selectedPatientId) fetchPatientDossier(selectedPatientId);
                else fetchPatients();
              } else if (activeTab === 'documents') fetchDocuments();
              else if (activeTab === 'diagnosis') fetchDiagnosis();
            }}
            disabled={loading || dossierLoading}
          >
            <RefreshCw size={14} className={loading || dossierLoading ? 'spinner' : ''} />
            Refresh
          </button>
        </div>
      </header>

      {/* Security & Scientific Invariants Banner */}
      <div
        style={{
          background: 'rgba(59, 130, 246, 0.05)',
          borderLeft: '4px solid var(--primary)',
          padding: '0.75rem 1rem',
          borderRadius: '4px',
          marginBottom: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontSize: '0.8125rem',
          color: 'var(--text-secondary)',
        }}
      >
        <ShieldCheck size={18} style={{ color: 'var(--primary)', flexShrink: 0 }} />
        <div>
          <strong>Zero Mock Clinical Data Contract Active:</strong> Clinician access is restricted exclusively to assigned patients.
          All missing or offline AI subsystems (GAT inference, DiCE counterfactuals, DoWhy causal DAGs) explicitly surface as canonical unavailable states.
          Zero PII, unmasked notes, or raw ECG waveform samples are serialized across boundaries.
        </div>
      </div>

      {/* Global Error Banner */}
      {error && (
        <div style={{ marginBottom: '1.5rem' }}>
          {error.status === 403 ? (
            <ClinicalForbidden title="Clinician Access Forbidden" description={error.message} />
          ) : error.status === 404 ? (
            <ClinicalNotFound title="Clinical Resource Not Found" description={error.message} />
          ) : (
            <ClinicalError title="Telemetry Request Failed" message={error.message} />
          )}
        </div>
      )}

      {/* =========================================================================
          TAB 1: CLINICIAN DASHBOARD / OVERVIEW
      ========================================================================= */}
      {activeTab === 'dashboard' && (
        <div>
          {/* Live Model Architecture Readiness Diagnostics */}
          <ModelStatusPanel />

          {/* Quick Launch Diagnostic Studio Action Banner */}
          <div className="card" style={{
            marginBottom: '1.5rem',
            background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%)',
            border: '1px solid rgba(6, 182, 212, 0.3)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sparkles size={18} className="text-accent" />
                <strong style={{ fontSize: '1rem', color: '#f8fafc' }}>
                  Interactive Multimodal Diagnostic Studio is Operational
                </strong>
              </div>
              <p style={{ margin: '4px 0 0', fontSize: '0.8125rem', color: '#94a3b8' }}>
                Analyze 12-lead ECG waveforms and clinical narratives using the verified PTB-XL multimodal neural network with real-time privacy masking.
              </p>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setActiveTab('inference')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Activity size={16} />
              Open Diagnostic Studio
            </button>
          </div>

          {loading && !overview ? (
            <ClinicalLoading message="Loading clinician profile and telemetry..." />
          ) : overview ? (
            <>
              {/* Profile Card & Counters */}
              <div className="grid" style={{ marginBottom: '1.5rem' }}>
                <div className="card">
                  <div className="card-header">
                    <div className="card-icon text-info">
                      <Stethoscope size={20} />
                    </div>
                    <div>
                      <div className="card-title">Clinician Profile</div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {overview.clinician.department || 'General Clinical'}
                      </span>
                    </div>
                  </div>
                  <div style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Hospital Node:</span>
                      <span style={{ fontWeight: '600' }}>
                        {overview.clinician.hospital_name || 'Hospital Node'} ({overview.clinician.hospital_code || 'LOCAL'})
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Account Status:</span>
                      <ClinicalStatusBadge status={overview.clinician.status} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Scope Boundary:</span>
                      <span style={{ color: 'var(--success)', fontWeight: '600' }}>Explicit Assignment Only</span>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-icon text-primary">
                      <Stethoscope size={20} />
                    </div>
                    <div className="card-title">Assigned Patients</div>
                  </div>
                  <div style={{ marginTop: '0.5rem' }}>
                    <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                      {overview.metrics.assigned_patients.count}
                    </div>
                    <div style={{ marginTop: '0.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Authorized patient records</span>
                      <ClinicalStatusBadge status={overview.metrics.assigned_patients.status} />
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-icon text-warning">
                      <FileText size={20} />
                    </div>
                    <div className="card-title">Clinical Documents</div>
                  </div>
                  <div style={{ marginTop: '0.5rem' }}>
                    <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                      {overview.metrics.clinical_documents.count}
                    </div>
                    <div style={{ marginTop: '0.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Sanitized report metadata</span>
                      <ClinicalStatusBadge status={overview.metrics.clinical_documents.status} />
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-icon text-success">
                      <Activity size={20} />
                    </div>
                    <div className="card-title">ECG Telemetry Records</div>
                  </div>
                  <div style={{ marginTop: '0.5rem' }}>
                    <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--text-primary)' }}>
                      {overview.metrics.ecg_records.count}
                    </div>
                    <div style={{ marginTop: '0.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Acquisition records</span>
                      <ClinicalStatusBadge status={overview.metrics.ecg_records.status} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Subsystem Capabilities Grid */}
              <div className="card" style={{ marginBottom: '1.5rem' }}>
                <div className="card-header">
                  <div className="card-icon text-purple">
                    <Cpu size={20} />
                  </div>
                  <div className="card-title">Diagnostic Subsystem Capability Matrix</div>
                </div>
                <div style={{ overflowX: 'auto', marginTop: '0.75rem' }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Subsystem Module</th>
                        <th>Target Modality</th>
                        <th>Operational State</th>
                        <th>Diagnostic Impact</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Multimodal GAT Fusion</strong></td>
                        <td>ECG (1D CNN) + Notes (BERT) + Tabular</td>
                        <td><ClinicalStatusBadge status={overview.capabilities.multimodal_gat_fusion || overview.capabilities.clinical_ai_inference} /></td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                          Runtime prediction suspended; no risk scores manufactured.
                        </td>
                      </tr>
                      <tr>
                        <td><strong>DiCE Counterfactuals</strong></td>
                        <td>Actionable Tabular Feature Adjustments</td>
                        <td><ClinicalStatusBadge status={overview.capabilities.dice_counterfactuals} /></td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                          Counterfactual explanations unavailable; zero fake perturbed instances.
                        </td>
                      </tr>
                      <tr>
                        <td><strong>DoWhy Causal AI DAG</strong></td>
                        <td>Causal Graph Estimators & Interventions</td>
                        <td><ClinicalStatusBadge status={overview.capabilities.dowhy_causal} /></td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                          Causal inference unavailable; zero treatment suggestions.
                        </td>
                      </tr>
                      <tr>
                        <td><strong>Flower Federated Client</strong></td>
                        <td>Cross-Hospital Decentralized Weights</td>
                        <td><ClinicalStatusBadge status={overview.capabilities.flower_federated} /></td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                          Node federated training loop currently idle.
                        </td>
                      </tr>
                      <tr>
                        <td><strong>Clinical Decision Support</strong></td>
                        <td>Automated Diagnostic Recommendations</td>
                        <td><ClinicalStatusBadge status={overview.capabilities.clinical_decision_support} /></td>
                        <td style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                          Disabled pending certified clinical inference validation.
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : null}
        </div>
      )}

      {/* =========================================================================
          TAB: DIAGNOSTIC STUDIO (MULTIMODAL INFERENCE)
      ========================================================================= */}
      {activeTab === 'inference' && (
        <DiagnosticWorkspace
          assignedPatients={patients}
          onPredictionComplete={() => fetchDiagnosis()}
        />
      )}

      {/* =========================================================================
          TAB: FEDERATED LEARNING PANEL
      ========================================================================= */}
      {activeTab === 'federated' && (
        <FederatedLearningPanel />
      )}

      {/* =========================================================================
          TAB 2: ASSIGNED PATIENTS (Directory & Dossier Drill-down)
      ========================================================================= */}
      {activeTab === 'patients' && (
        <div>
          {/* Patient Dossier View */}
          {selectedPatientId ? (
            <div>
              <button
                className="btn btn-secondary"
                onClick={handleBackToPatients}
                style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                <ArrowLeft size={16} /> Back to Assigned Patients
              </button>

              {dossierLoading ? (
                <ClinicalLoading message="Retrieving authorized patient clinical dossier..." />
              ) : dossierError ? (
                dossierError.status === 404 ? (
                  <ClinicalNotFound
                    title="Patient Dossier Not Accessible"
                    description="This patient reference is not assigned to your clinician account or belongs to another hospital node."
                  />
                ) : (
                  <ClinicalError title="Dossier Retrieval Error" message={dossierError.message} />
                )
              ) : patientDossier ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  {/* Patient Dossier Header */}
                  <div className="card" style={{ borderLeft: '4px solid var(--primary)' }}>
                    <div className="card-header" style={{ justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div className="card-icon text-primary">
                          <Stethoscope size={20} />
                        </div>
                        <div>
                          <div className="card-title">Patient Clinical Dossier</div>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            Opaque SHA-256 Identifier Scoped to Hospital Node
                          </span>
                        </div>
                      </div>
                      <ClinicalStatusBadge status="AVAILABLE" />
                    </div>
                    <div style={{ marginTop: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '1.5rem', fontSize: '0.875rem' }}>
                      <div>
                        <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>OPAQUE PATIENT ID</span>
                        <code style={{ background: 'var(--bg-secondary)', padding: '2px 6px', borderRadius: '4px' }}>
                          {patientDossier.patient.opaque_patient_id}
                        </code>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>HOSPITAL NODE</span>
                        <span style={{ fontWeight: '600' }}>
                          {patientDossier.patient.hospital_name || 'Hospital Node'} ({patientDossier.patient.hospital_code || 'LOCAL'})
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.75rem' }}>REGISTERED AT</span>
                        <span>{patientDossier.patient.created_at ? new Date(patientDossier.patient.created_at).toLocaleDateString() : 'N/A'}</span>
                      </div>
                    </div>
                  </div>

                  {/* Section 1: Clinical Observations */}
                  <div className="card">
                    <div className="card-header" style={{ justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <FileSpreadsheet size={18} className="text-info" />
                        <span style={{ fontWeight: '600' }}>Clinical Observations</span>
                      </div>
                      <ClinicalStatusBadge status={patientDossier.observations.status} />
                    </div>
                    <div style={{ marginTop: '0.75rem' }}>
                      {patientDossier.observations.count === 0 ? (
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                          No tabular clinical observations recorded for this patient.
                        </p>
                      ) : (
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Observation Type</th>
                              <th>Feature Schema</th>
                              <th>Acquisition Date</th>
                            </tr>
                          </thead>
                          <tbody>
                            {patientDossier.observations.items.map((obs) => (
                              <tr key={obs.id}>
                                <td><code>{obs.observation_type}</code></td>
                                <td>{obs.feature_schema_version}</td>
                                <td>{obs.created_at ? new Date(obs.created_at).toLocaleString() : 'N/A'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  </div>

                  {/* Section 2: Medical Documents (Metadata only, Zero raw text) */}
                  <div className="card">
                    <div className="card-header" style={{ justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <FileText size={18} className="text-warning" />
                        <span style={{ fontWeight: '600' }}>Medical Documents (Privacy Sanitized)</span>
                      </div>
                      <ClinicalStatusBadge status={patientDossier.documents.status} />
                    </div>
                    <div style={{ marginTop: '0.75rem' }}>
                      {patientDossier.documents.count === 0 ? (
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                          No clinical documents on file for this patient reference.
                        </p>
                      ) : (
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Document Type</th>
                              <th>Status</th>
                              <th>Versions</th>
                              <th>Masking Method</th>
                              <th>Entities Sanitized</th>
                              <th>Timestamp</th>
                            </tr>
                          </thead>
                          <tbody>
                            {patientDossier.documents.items.map((doc) => (
                              <tr key={doc.id}>
                                <td><code>{doc.document_type}</code></td>
                                <td><ClinicalStatusBadge status={doc.status} /></td>
                                <td>{doc.versions_count}</td>
                                <td><span className="badge badge-success">{doc.masking_status}</span></td>
                                <td>{doc.entities_detected_count}</td>
                                <td>{doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'N/A'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  </div>

                  {/* Section 3: 12-Lead ECG Records (Metadata only, Zero raw waveforms) */}
                  <div className="card">
                    <div className="card-header" style={{ justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Activity size={18} className="text-success" />
                        <span style={{ fontWeight: '600' }}>12-Lead ECG Telemetry (Acquisition Metadata)</span>
                      </div>
                      <ClinicalStatusBadge status={patientDossier.ecg_records.status} />
                    </div>
                    <div style={{ marginTop: '0.75rem' }}>
                      {patientDossier.ecg_records.count === 0 ? (
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                          No ECG records acquired for this patient reference.
                        </p>
                      ) : (
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Sampling Rate</th>
                              <th>Duration</th>
                              <th>Leads</th>
                              <th>Dataset Provenance</th>
                              <th>Acquisition Timestamp</th>
                            </tr>
                          </thead>
                          <tbody>
                            {patientDossier.ecg_records.items.map((ecg) => (
                              <tr key={ecg.id}>
                                <td>{ecg.sampling_rate_hz} Hz</td>
                                <td>{ecg.duration_seconds} s</td>
                                <td>{ecg.lead_count} Leads</td>
                                <td><span className="badge badge-info">{ecg.dataset_source}</span></td>
                                <td>{ecg.acquired_at ? new Date(ecg.acquired_at).toLocaleString() : 'N/A'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  </div>

                  {/* Section 4: AI Analysis Capability Banner */}
                  <div className="card" style={{ borderLeft: '4px solid #8b5cf6', background: 'rgba(139, 92, 246, 0.03)' }}>
                    <div className="card-header" style={{ justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div className="card-icon" style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#8b5cf6' }}>
                          <Cpu size={20} />
                        </div>
                        <div>
                          <div className="card-title">Clinical AI Analysis Capability</div>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            Graph Attention Network Multimodal Fusion Subsystem
                          </span>
                        </div>
                      </div>
                      <ClinicalStatusBadge status="NOT_IMPLEMENTED" />
                    </div>
                    <div style={{ marginTop: '0.75rem', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                      <p>
                        Clinical AI inference runtime is offline. The system strictly adheres to the Zero Mock Data contract:
                        no synthetic diagnoses, risk levels, or confidence scores are fabricated.
                      </p>
                    </div>
                  </div>

                  {/* Section 5: Predicted Risk & Model Confidence (Honest state) */}
                  <div className="card">
                    <div className="card-header" style={{ justifyContent: 'space-between' }}>
                      <div className="card-title">Predicted Risk & Model Confidence</div>
                      <ClinicalStatusBadge status={patientDossier.diagnosis.status} />
                    </div>
                    <div style={{ marginTop: '0.75rem' }}>
                      {patientDossier.diagnosis.prediction ? (
                        <div style={{ fontSize: '0.875rem' }}>
                          <div style={{ marginBottom: '0.5rem' }}>
                            <strong>Target:</strong> {patientDossier.diagnosis.prediction.target_definition}
                          </div>
                          <div style={{ marginBottom: '0.5rem' }}>
                            <strong>Model Version:</strong> {patientDossier.diagnosis.prediction.model_version_tag}
                          </div>
                          <div>
                            <strong>Provenance:</strong> {patientDossier.diagnosis.prediction.provenance}
                          </div>
                        </div>
                      ) : (
                        <div className="honest-state-box" style={{ padding: '1.5rem', margin: 0 }}>
                          <h4 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                            Prediction Unavailable
                          </h4>
                          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                            Clinical inference is not implemented. Individual prediction confidence is not computed.
                          </p>
                          <div style={{ marginTop: '0.5rem' }}>
                            <ClinicalStatusBadge status="NOT_IMPLEMENTED" />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Section 6: Why this prediction? (DiCE) Expandable */}
                  <div className="card">
                    <div
                      className="card-header"
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                      onClick={() => toggleSection('whyPrediction')}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {openSections.whyPrediction ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span style={{ fontWeight: '600' }}>Why this prediction? (Feature Attribution)</span>
                      </div>
                      <ClinicalStatusBadge status="NOT_IMPLEMENTED" />
                    </div>
                    {openSections.whyPrediction && (
                      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)' }}>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                          Explanation generation is not yet available. Feature attribution bars and SHAP weightings are not fabricated.
                        </p>
                        <ClinicalNotImplemented
                          title="DiCE Feature Attribution Offline"
                          moduleName="DiCE Explainability Engine"
                          description="Multimodal feature attribution requires active runtime model weights."
                        />
                      </div>
                    )}
                  </div>

                  {/* Section 7: What could change the prediction? (Counterfactuals) Expandable */}
                  <div className="card">
                    <div
                      className="card-header"
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                      onClick={() => toggleSection('whatChange')}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {openSections.whatChange ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span style={{ fontWeight: '600' }}>What could change the prediction? (Counterfactuals)</span>
                      </div>
                      <ClinicalStatusBadge status="NOT_IMPLEMENTED" />
                    </div>
                    {openSections.whatChange && (
                      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)' }}>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                          Counterfactual explanations are not yet available. Diverse perturbed clinical instances are not simulated.
                        </p>
                        <ClinicalNotImplemented
                          title="DiCE Counterfactuals Offline"
                          moduleName="DiCE Counterfactual Optimization"
                          description="Optimization engine for actionable feature boundaries requires active trained neural latent space."
                        />
                      </div>
                    )}
                  </div>

                  {/* Section 8: Causal insights (DoWhy) Expandable */}
                  <div className="card">
                    <div
                      className="card-header"
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                      onClick={() => toggleSection('causalInsights')}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {openSections.causalInsights ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span style={{ fontWeight: '600' }}>Causal Insights & Interventions (DoWhy)</span>
                      </div>
                      <ClinicalStatusBadge status="NOT_IMPLEMENTED" />
                    </div>
                    {openSections.causalInsights && (
                      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)' }}>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                          Causal inference is not yet available. Causal DAG intervention effects are not manufactured.
                        </p>
                        <ClinicalNotImplemented
                          title="DoWhy Causal DAG Offline"
                          moduleName="DoWhy Causal Inference"
                          description="Structural causal models and backdoor adjustment estimators are pending causal pipeline integration."
                        />
                      </div>
                    )}
                  </div>

                  {/* Section 9: Clinical Decision Support (Disabled) */}
                  <div className="card" style={{ opacity: 0.65, borderLeft: '4px solid var(--text-muted)' }}>
                    <div className="card-header" style={{ justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Lock size={18} className="text-muted" />
                        <span style={{ fontWeight: '600' }}>Clinical Decision Support (Disabled)</span>
                      </div>
                      <ClinicalStatusBadge status="NOT_IMPLEMENTED" />
                    </div>
                    <div style={{ marginTop: '0.5rem' }}>
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                        Clinical decision support will become available after the validated inference and causal-analysis pipeline is implemented.
                      </p>
                    </div>
                  </div>

                  {/* Section 10: Model Information Expandable */}
                  <div className="card">
                    <div
                      className="card-header"
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                      onClick={() => toggleSection('modelInfo')}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {openSections.modelInfo ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span style={{ fontWeight: '600' }}>Model Information</span>
                      </div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Architecture Metadata</span>
                    </div>
                    {openSections.modelInfo && (
                      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)', fontSize: '0.875rem' }}>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Architecture:</strong> MedShield Multimodal Graph Attention Network (GAT Fusion)
                        </div>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Modalities:</strong> 1D CNN-BiLSTM (ECG 500Hz), BioClinicalBERT (Sanitized Notes), MLP (Tabular Observations)
                        </div>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Scientific Task:</strong> PTB-XL Normal Sinus Rhythm vs Pathological Proxy Task
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                          <em>Notice: Individual prediction confidence is distinct from aggregate held-out validation accuracy. Zero model weights exposed centrally.</em>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Section 11: Technical Details Expandable (Zero secrets) */}
                  <div className="card">
                    <div
                      className="card-header"
                      style={{ cursor: 'pointer', userSelect: 'none' }}
                      onClick={() => toggleSection('techDetails')}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {openSections.techDetails ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                        <span style={{ fontWeight: '600' }}>Technical Details & Provenance</span>
                      </div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Telemetry Proof</span>
                    </div>
                    {openSections.techDetails && (
                      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-color)', fontSize: '0.875rem' }}>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Patient Opaque Hash:</strong> <code>{patientDossier.patient.opaque_patient_id}</code>
                        </div>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Hospital Boundary:</strong> <code>{patientDossier.patient.hospital_id}</code>
                        </div>
                        <div style={{ marginBottom: '0.5rem' }}>
                          <strong>Data Privacy:</strong> Zero PII transmitted; raw waveforms and unmasked text stored solely in local hospital enclave.
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
                          Zero secrets, tokens, or plaintext medical identifiers serialized.
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            /* Patient Directory Table */
            <div className="card">
              <div className="card-header" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div className="card-icon text-primary">
                    <Stethoscope size={20} />
                  </div>
                  <div>
                    <div className="card-title">Assigned Patients Directory</div>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Total Authorized Patients: {patientTotal}
                    </span>
                  </div>
                </div>

                {/* Search Bar */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ position: 'relative' }}>
                    <Search
                      size={14}
                      style={{ position: 'absolute', left: '10px', top: '10px', color: 'var(--text-muted)' }}
                    />
                    <input
                      type="text"
                      placeholder="Search opaque hash..."
                      value={patientSearch}
                      onChange={(e) => setPatientSearch(e.target.value)}
                      style={{
                        padding: '6px 12px 6px 32px',
                        borderRadius: '6px',
                        border: '1px solid var(--border-color)',
                        background: 'var(--bg-secondary)',
                        color: 'var(--text-primary)',
                        fontSize: '0.8125rem',
                        width: '220px',
                      }}
                    />
                  </div>
                </div>
              </div>

              {assignError && assignError.status !== 401 && assignError.status !== 403 ? (
                <ClinicalError
                  title="Assignment Request Failed"
                  message={assignError.message || 'Unable to update patient assignment.'}
                />
              ) : null}

              {loading ? (
                <ClinicalLoading message="Loading assigned patient directory..." />
              ) : filteredPatients.length === 0 ? (
                <>
                  <ClinicalEmptyState
                    title="No Assigned Patients Found"
                    description={
                      patientSearch
                        ? `No patients match the opaque hash query "${patientSearch}".`
                        : 'There are currently no patients assigned to your clinician account. Eligible hospital patients can be assigned below.'
                    }
                    icon={Stethoscope}
                  />
                  {!patientSearch ? (
                    <div className="card" style={{ marginTop: '1rem' }}>
                      <div className="card-header">
                        <div className="card-title">Eligible Hospital Patients</div>
                      </div>
                      {eligibleLoading ? (
                        <ClinicalLoading message="Loading eligible patients at your hospital node..." />
                      ) : eligiblePatients.length === 0 ? (
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', padding: '0.5rem 0' }}>
                          No unassigned patients are registered at your hospital node.
                        </p>
                      ) : (
                        <table className="data-table" style={{ marginTop: '0.5rem' }}>
                          <thead>
                            <tr>
                              <th>Opaque Patient ID</th>
                              <th>Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {eligiblePatients.map((p) => (
                              <tr key={p.patient_id}>
                                <td>
                                  <code style={{ background: 'var(--bg-secondary)', padding: '2px 6px', borderRadius: '4px' }}>
                                    {p.opaque_patient_id.length > 20
                                      ? `${p.opaque_patient_id.slice(0, 16)}...`
                                      : p.opaque_patient_id}
                                  </code>
                                </td>
                                <td>
                                  <button
                                    className="btn btn-secondary"
                                    style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                                    disabled={assigningPatientId === p.patient_id}
                                    onClick={() => handleAssignPatient(p.patient_id)}
                                  >
                                    {assigningPatientId === p.patient_id ? 'Assigning…' : 'Assign to Me'}
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  ) : null}
                </>
              ) : (
                <>
                  <div style={{ overflowX: 'auto', marginTop: '0.75rem' }}>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Opaque Patient ID</th>
                          <th>Hospital Node</th>
                          <th>Assigned Date</th>
                          <th>Documents</th>
                          <th>ECG Records</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredPatients.map((p) => (
                          <tr key={p.patient_id}>
                            <td>
                              <code style={{ background: 'var(--bg-secondary)', padding: '2px 6px', borderRadius: '4px' }}>
                                {p.opaque_patient_id.length > 20
                                  ? `${p.opaque_patient_id.slice(0, 16)}...`
                                  : p.opaque_patient_id}
                              </code>
                            </td>
                            <td>
                              <span className="badge badge-muted">{p.hospital_code || 'NODE'}</span>
                            </td>
                            <td>{p.assigned_at ? new Date(p.assigned_at).toLocaleDateString() : 'N/A'}</td>
                            <td>{p.documents_count}</td>
                            <td>{p.ecg_records_count}</td>
                            <td>
                              <button
                                className="btn btn-primary"
                                style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                                onClick={() => handleSelectPatient(p.patient_id)}
                              >
                                View Dossier <ChevronRight size={12} />
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination Controls */}
                  <div className="pagination-bar" style={{ marginTop: '1rem' }}>
                    <button
                      className="btn btn-secondary"
                      disabled={patientPage === 0}
                      onClick={() => setPatientPage((prev) => Math.max(0, prev - 1))}
                      style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                    >
                      Previous
                    </button>
                    <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                      Page {patientPage + 1} of {Math.max(1, Math.ceil(patientTotal / patientLimit))}
                    </span>
                    <button
                      className="btn btn-secondary"
                      disabled={(patientPage + 1) * patientLimit >= patientTotal}
                      onClick={() => setPatientPage((prev) => prev + 1)}
                      style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                    >
                      Next
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* =========================================================================
          TAB 3: CLINICAL REPORTS & DOCUMENTS
      ========================================================================= */}
      {activeTab === 'documents' && (
        <div className="card">
          <div className="card-header" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div className="card-icon text-warning">
                <FileText size={20} />
              </div>
              <div>
                <div className="card-title">Clinical Reports & Documents</div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Total Reports for Assigned Patients: {documentTotal}
                </span>
              </div>
            </div>

            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => {
                setShowUploadModal(true);
                setUploadError(null);
                setUploadSuccess(null);
              }}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Upload size={14} />
              Upload Medical Document
            </button>
          </div>

          {loading ? (
            <ClinicalLoading message="Loading clinical document directory..." />
          ) : documents.length === 0 ? (
            <ClinicalEmptyState
              title="No Clinical Documents On File"
              description="No sanitized medical documents have been registered for your assigned patients."
              icon={FileText}
            />
          ) : (
            <>
              <div style={{ overflowX: 'auto', marginTop: '0.75rem' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Document Type</th>
                      <th>Patient Reference</th>
                      <th>Status</th>
                      <th>Versions</th>
                      <th>Privacy Masking</th>
                      <th>Sanitized Entities</th>
                      <th>Registered At</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((d) => (
                      <tr key={d.id}>
                        <td><code>{d.document_type}</code></td>
                        <td>
                          <code style={{ background: 'var(--bg-secondary)', padding: '2px 6px', borderRadius: '4px' }}>
                            {d.opaque_patient_id.length > 16
                              ? `${d.opaque_patient_id.slice(0, 12)}...`
                              : d.opaque_patient_id}
                          </code>
                        </td>
                        <td><ClinicalStatusBadge status={d.status} /></td>
                        <td>{d.versions_count}</td>
                        <td><span className="badge badge-success">{d.masking_status}</span></td>
                        <td>{d.entities_detected_count}</td>
                        <td>{d.created_at ? new Date(d.created_at).toLocaleDateString() : 'N/A'}</td>
                        <td>
                          <div style={{ display: 'flex', gap: '4px' }}>
                            <button
                              className="btn btn-secondary"
                              style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                              onClick={() => fetchDocumentDetail(d.id)}
                            >
                              Metadata
                            </button>
                            <button
                              className="btn btn-primary"
                              style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                              disabled={processingDocId === d.id}
                              onClick={() => handleProcessDocument(d.id)}
                            >
                              {processingDocId === d.id ? 'Processing...' : 'Run Privacy Job'}
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination Controls */}
              <div className="pagination-bar" style={{ marginTop: '1rem' }}>
                <button
                  className="btn btn-secondary"
                  disabled={documentPage === 0}
                  onClick={() => setDocumentPage((prev) => Math.max(0, prev - 1))}
                  style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                >
                  Previous
                </button>
                <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                  Page {documentPage + 1} of {Math.max(1, Math.ceil(documentTotal / documentLimit))}
                </span>
                <button
                  className="btn btn-secondary"
                  disabled={(documentPage + 1) * documentLimit >= documentTotal}
                  onClick={() => setDocumentPage((prev) => prev + 1)}
                  style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                >
                  Next
                </button>
              </div>
            </>
          )}

          {/* Document Upload Modal */}
          {showUploadModal && (
            <div
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'rgba(0,0,0,0.7)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                zIndex: 1000,
              }}
            >
              <div
                className="card"
                style={{ width: '520px', maxWidth: '90vw', background: '#0f172a', border: '1px solid #334155' }}
              >
                <div className="card-header" style={{ justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Upload size={18} className="text-accent" />
                    <div className="card-title">Upload Clinical Document</div>
                  </div>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => {
                      setShowUploadModal(false);
                      setUploadFile(null);
                      setUploadError(null);
                      setUploadSuccess(null);
                    }}
                  >
                    Close
                  </button>
                </div>

                <form onSubmit={handleUploadDocument} style={{ padding: '1rem' }}>
                  {uploadError && (
                    <div className="notice-banner" style={{ borderLeftColor: 'var(--danger)', marginBottom: '1rem', fontSize: '0.75rem' }}>
                      <AlertCircle size={14} className="text-danger" style={{ display: 'inline', marginRight: '4px' }} />
                      {uploadError}
                    </div>
                  )}

                  {uploadSuccess && (
                    <div className="notice-banner" style={{ borderLeftColor: 'var(--success)', marginBottom: '1rem', fontSize: '0.75rem' }}>
                      <CheckCircle size={14} className="text-success" style={{ display: 'inline', marginRight: '4px' }} />
                      {uploadSuccess}
                    </div>
                  )}

                  <div className="form-group" style={{ marginBottom: '1rem' }}>
                    <label className="form-label">
                      Select Medical Document File (.pdf, .jpg, .jpeg, .png — max 10MB)
                    </label>
                    <input
                      type="file"
                      accept=".pdf,.jpg,.jpeg,.png"
                      className="form-input"
                      onChange={(e) => {
                        const file = e.target.files && e.target.files[0];
                        if (file) {
                          if (file.size > 10 * 1024 * 1024) {
                            setUploadError('File exceeds maximum size of 10 MB.');
                            setUploadFile(null);
                            return;
                          }
                          setUploadFile(file);
                          setUploadError(null);
                        }
                      }}
                      required
                    />
                    {uploadFile && (
                      <span style={{ fontSize: '0.75rem', color: '#38bdf8', marginTop: '4px', display: 'block' }}>
                        Selected: {uploadFile.name} ({(uploadFile.size / 1024).toFixed(1)} KB)
                      </span>
                    )}
                  </div>

                  <div className="form-group" style={{ marginBottom: '1.25rem' }}>
                    <label className="form-label">
                      Assign to Patient Reference (Optional)
                    </label>
                    <select
                      className="form-input"
                      value={uploadPatientId}
                      onChange={(e) => setUploadPatientId(e.target.value)}
                    >
                      <option value="">Unassigned Hospital Document</option>
                      {patients.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.opaque_patient_id}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => setShowUploadModal(false)}
                      disabled={uploading}
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="btn btn-primary btn-sm"
                      disabled={uploading || !uploadFile}
                    >
                      {uploading ? 'Uploading & Processing...' : 'Upload & Process'}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {/* Document Metadata Inspector Modal */}
          {selectedDoc && (
            <div
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'rgba(0,0,0,0.6)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                zIndex: 1000,
              }}
            >
              <div
                className="card"
                style={{ width: '600px', maxWidth: '90vw', maxHeight: '85vh', overflowY: 'auto' }}
              >
                <div className="card-header" style={{ justifyContent: 'space-between' }}>
                  <div className="card-title">Document Metadata & Verification</div>
                  <button className="btn btn-secondary" onClick={() => setSelectedDoc(null)}>
                    Close
                  </button>
                </div>
                <div style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                  <div style={{ marginBottom: '0.5rem' }}>
                    <strong>Document UUID:</strong> <code>{selectedDoc.id}</code>
                  </div>
                  <div style={{ marginBottom: '0.5rem' }}>
                    <strong>Patient Hash:</strong> <code>{selectedDoc.opaque_patient_id}</code>
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <strong>Type:</strong> <code>{selectedDoc.document_type}</code>
                  </div>

                  <h4 style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Version Checksums (SHA-256)</h4>
                  {selectedDoc.versions.map((v) => (
                    <div key={v.id} style={{ marginBottom: '0.25rem', fontSize: '0.8125rem' }}>
                      v{v.version_number}: <code>{v.checksum_sha256}</code>
                    </div>
                  ))}

                  <h4 style={{ fontWeight: '600', marginTop: '1rem', marginBottom: '0.5rem' }}>Processing Jobs</h4>
                  {selectedDoc.jobs.length === 0 ? (
                    <p style={{ color: 'var(--text-muted)' }}>No job history.</p>
                  ) : (
                    selectedDoc.jobs.map((j) => (
                      <div key={j.id} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                        <span>Job: {j.id.slice(0, 8)}</span>
                        <ClinicalStatusBadge status={j.status} />
                      </div>
                    ))
                  )}

                  <div
                    style={{
                      marginTop: '1.25rem',
                      padding: '0.5rem 0.75rem',
                      background: 'var(--bg-secondary)',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      color: 'var(--text-muted)',
                    }}
                  >
                    Privacy contract: Raw document text is retained exclusively in local hospital on-premise storage and is never exposed in central API endpoints.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* =========================================================================
          TAB 4: GAT DIAGNOSIS
      ========================================================================= */}
      {activeTab === 'diagnosis' && (
        <div>
          <ClinicalNotImplemented
            title="Multimodal GAT Diagnostic Inference Pipeline"
            moduleName="GAT Neural Fusion Classifier"
            description="Diagnostic inference is currently not enabled in this deployment. Evaluated predictions will render here once model weights are validated."
          />

          <div style={{ marginTop: '1.5rem' }}>
            <div className="card">
              <div className="card-header" style={{ justifyContent: 'space-between' }}>
                <div className="card-title">Assigned Diagnosis Records</div>
                <ClinicalStatusBadge status={diagnosisData?.status || 'NO_DATA'} />
              </div>
              <div style={{ marginTop: '0.75rem' }}>
                {diagnosisData && diagnosisData.predictions.length > 0 ? (
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Patient Hash</th>
                        <th>Model Version</th>
                        <th>Target Definition</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diagnosisData.predictions.map((p) => (
                        <tr key={p.id}>
                          <td><code>{p.opaque_patient_id}</code></td>
                          <td>{p.model_version_tag}</td>
                          <td>{p.target_definition}</td>
                          <td><ClinicalStatusBadge status={p.status} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <ClinicalEmptyState
                    title="No Diagnosis Predictions Recorded"
                    description="Zero predictions recorded. The system does not fabricate synthetic risk scores or diagnostic conclusions."
                    icon={Activity}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =========================================================================
          TAB 5: DiCE EXPLANATIONS
      ========================================================================= */}
      {activeTab === 'explanations' && (
        <ClinicalNotImplemented
          title="DiCE Counterfactual Explanations"
          moduleName="DiCE Explainability Engine"
          description="Diverse counterfactual explanation generation for clinical feature adjustment is offline pending ML pipeline completion. Zero fake feature importance weights are rendered."
        />
      )}

      {/* =========================================================================
          TAB 6: DoWhy CAUSAL AI
      ========================================================================= */}
      {activeTab === 'causal' && (
        <ClinicalNotImplemented
          title="DoWhy Causal Inference & Interventions"
          moduleName="DoWhy Causal Graph Engine"
          description="Causal DAG estimation, backdoor adjustment, and treatment intervention effect evaluation are offline pending causal module integration. Zero treatment suggestions are simulated."
        />
      )}
    </ClinicianLayout>
  );
}
