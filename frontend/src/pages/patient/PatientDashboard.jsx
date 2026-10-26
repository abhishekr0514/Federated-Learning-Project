import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { patientPortalNavBadge, patientMessagingNavBadge } from '../../utils/patientPortalNav';
import { PatientLayout } from '../../layouts/PatientLayout';
import { apiRequest, ApiError } from '../../services/api';
import {
  PatientStatusBadge,
  PatientSafetyDisclaimer,
  PatientLoadingState,
  PatientEmptyState,
  PatientNotEnabled,
  PatientNotImplemented,
  PatientForbidden,
  PatientNotFound,
  PatientError,
} from '../../components/PatientStates';
import {
  Heart,
  FileText,
  Activity,
  HelpCircle,
  Stethoscope,
  FileSpreadsheet,
  ShieldCheck,
  RefreshCw,
  Search,
  Calendar,
  Clock,
  ChevronRight,
  Info,
  CheckCircle,
  MessageSquare,
  Lock,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import {
  PatientAssessmentCharts,
  PatientAssessmentDetailGrid,
} from '../../components/PatientAssessmentCharts';
import '../../components/PatientAssessmentCharts.css';

export function PatientDashboard() {
  const [activeTab, setActiveTab] = useState('health');
  const { user } = useAuth();

  // Loading & error state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Data states
  const [overview, setOverview] = useState(null);
  const [recordsData, setRecordsData] = useState(null);
  const [reportsData, setReportsData] = useState(null);
  const [diagnosisData, setDiagnosisData] = useState(null);
  const [questionsData, setQuestionsData] = useState(null);
  const [questionDraft, setQuestionDraft] = useState('');
  const [questionSubmitting, setQuestionSubmitting] = useState(false);
  const [validationOpen, setValidationOpen] = useState(false);

  // UI filters & selections
  const [recordsSearch, setRecordsSearch] = useState('');
  const [selectedRecord, setSelectedRecord] = useState(null);

  // Fetch overview
  const fetchOverview = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiRequest('/api/v1/patient/overview');
      setOverview(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch records
  const fetchRecords = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiRequest('/api/v1/patient/records');
      setRecordsData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch reports
  const fetchReports = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiRequest('/api/v1/patient/reports');
      setReportsData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch diagnosis / results
  const fetchQuestions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiRequest('/api/v1/patient/questions');
      setQuestionsData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchDiagnosis = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiRequest('/api/v1/patient/diagnosis');
      setDiagnosisData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const portalNavBadges = useMemo(() => {
    if (!overview) return null;
    const rec = overview.summary?.records_count ?? 0;
    const rep = overview.summary?.reports_count ?? 0;
    return {
      records: rec > 0 ? String(rec) : 'No records',
      reports: rep > 0 ? String(rep) : 'No reports',
      assessment: patientPortalNavBadge(overview.assessment?.status),
      results: patientPortalNavBadge(overview.results?.status),
      messaging: patientMessagingNavBadge(overview.capabilities),
    };
  }, [overview]);

  useEffect(() => {
    fetchOverview();
  }, [fetchOverview]);

  // Tab switch effect
  useEffect(() => {
    if (activeTab === 'health') {
      fetchOverview();
    } else if (activeTab === 'records') {
      fetchRecords();
    } else if (activeTab === 'reports') {
      fetchReports();
    } else if (activeTab === 'assessment') {
      fetchOverview();
      fetchDiagnosis();
    } else if (activeTab === 'results') {
      fetchDiagnosis();
    } else if (activeTab === 'questions') {
      fetchQuestions();
    }
  }, [activeTab, fetchOverview, fetchRecords, fetchReports, fetchDiagnosis, fetchQuestions]);

  const layoutProps = { activeTab, onSelectTab: setActiveTab, portalNavBadges };

  // Handle errors
  if (error) {
    if (error instanceof ApiError && error.status === 403) {
      return (
        <PatientLayout {...layoutProps}>
          <PatientForbidden message="You do not have authorization to view this patient portal." />
        </PatientLayout>
      );
    }
    if (error instanceof ApiError && error.status === 404) {
      return (
        <PatientLayout {...layoutProps}>
          <PatientNotFound message="No patient profile found associated with your account." />
        </PatientLayout>
      );
    }
    return (
      <PatientLayout {...layoutProps}>
        <PatientError
          message={error.message || 'Unable to connect to the patient service.'}
          onRetry={() => {
            if (activeTab === 'health') fetchOverview();
            else if (activeTab === 'records') fetchRecords();
            else if (activeTab === 'reports') fetchReports();
            else fetchDiagnosis();
          }}
        />
      </PatientLayout>
    );
  }

  return (
    <PatientLayout {...layoutProps}>
      {/* Top Header */}
      <header className="page-header" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: '700', margin: 0 }}>Personal Health Portal</h1>
          <p style={{ color: 'var(--text-muted)', margin: '0.25rem 0 0' }}>
            Welcome back, <strong>{user?.full_name || user?.email || 'Patient'}</strong>. Access your authorized personal health information securely.
          </p>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => {
            if (activeTab === 'health') fetchOverview();
            else if (activeTab === 'records') fetchRecords();
            else if (activeTab === 'reports') fetchReports();
            else fetchDiagnosis();
          }}
          disabled={loading}
          style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          <span>Refresh</span>
        </button>
      </header>

      {/* Loading state indicator */}
      {loading && !overview && !recordsData && !reportsData && !diagnosisData && (
        <PatientLoadingState />
      )}

      {/* TAB 1: MY HEALTH OVERVIEW */}
      {activeTab === 'health' && overview && (
        <div>
          {/* Summary Metric Cards */}
          <div className="grid" style={{ marginBottom: '1.5rem' }}>
            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <FileText size={20} />
                </div>
                <div className="card-title">Health Records</div>
              </div>
              <div style={{ fontSize: '1.75rem', fontWeight: '700', margin: '0.5rem 0' }}>
                {overview.summary?.records_count ?? 0}
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                {overview.summary?.records_count === 0 ? 'No records recorded' : 'Records on file with your hospital'}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <Activity size={20} />
                </div>
                <div className="card-title">Diagnostic Reports</div>
              </div>
              <div style={{ fontSize: '1.75rem', fontWeight: '700', margin: '0.5rem 0' }}>
                {overview.summary?.reports_count ?? 0}
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                {overview.summary?.reports_count === 0 ? 'No reports on file' : 'Privacy-sanitized reports'}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <Calendar size={20} />
                </div>
                <div className="card-title">Latest Test Date</div>
              </div>
              <div style={{ fontSize: '1.125rem', fontWeight: '600', margin: '0.75rem 0', wordBreak: 'break-all' }}>
                {overview.summary?.latest_record_date
                  ? new Date(overview.summary.latest_record_date).toLocaleDateString()
                  : 'None Recorded'}
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                Most recent health activity
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-warning">
                  <Stethoscope size={20} />
                </div>
                <div className="card-title">AI Assessment</div>
              </div>
              <div style={{ margin: '0.75rem 0' }}>
                <PatientStatusBadge status={overview.assessment?.status || 'NOT_ENABLED'} />
              </div>
              <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                {overview.assessment?.message || 'Not currently available'}
              </div>
            </div>
          </div>

          {/* Health Status Box */}
          {overview.summary?.health_status === 'NO_DATA' ? (
            <PatientEmptyState
              title="My Health"
              message="No health information is currently available."
              hint="When your care team adds health records, encounter documents, or test results, they will appear here."
            />
          ) : (
            <div className="card" style={{ marginBottom: '1.5rem' }}>
              <div className="card-header">
                <div className="card-icon text-info">
                  <Heart size={20} />
                </div>
                <div className="card-title">My Health Summary</div>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', lineHeight: 1.6 }}>
                You have <strong>{overview.summary?.records_count}</strong> total records on file, including{' '}
                <strong>{overview.summary?.ecg_tests_count}</strong> cardiac recordings,{' '}
                <strong>{overview.summary?.vitals_tests_count}</strong> health observation readings, and{' '}
                <strong>{overview.summary?.reports_count}</strong> medical documents.
              </p>
              <div style={{ marginTop: '1rem', display: 'flex', gap: '0.75rem' }}>
                <button className="btn btn-primary btn-sm" onClick={() => setActiveTab('records')}>
                  View My Records
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => setActiveTab('reports')}>
                  View My Reports
                </button>
              </div>
            </div>
          )}

          {/* Privacy & Safety Information Cards */}
          <div className="grid grid-2">
            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <Lock size={18} />
                </div>
                <div className="card-title">Privacy Protection Active</div>
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                <p>
                  Your health records are safeguarded by hospital-level privacy protection. Raw personal identifying details
                  and doctor notes remain stored securely within your hospital node.
                </p>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="card-icon text-info">
                  <HelpCircle size={18} />
                </div>
                <div className="card-title">Questions for Your Doctor?</div>
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                <p>
                  Review suggested discussion topics and questions to bring to your next medical appointment.
                </p>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setActiveTab('questions')}
                  style={{ marginTop: '0.5rem' }}
                >
                  View Discussion Guide
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: MY RECORDS */}
      {activeTab === 'records' && recordsData && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <h2 style={{ fontSize: '1.125rem', fontWeight: 600, margin: 0 }}>My Health Records</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
                  Authorized clinical recordings and test entries on file for your care.
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ position: 'relative' }}>
                  <Search size={16} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                  <input
                    type="text"
                    placeholder="Search records..."
                    value={recordsSearch}
                    onChange={(e) => setRecordsSearch(e.target.value)}
                    style={{ paddingLeft: '32px', fontSize: '0.875rem' }}
                  />
                </div>
              </div>
            </div>
          </div>

          {recordsData.status === 'NO_DATA' || recordsData.total_count === 0 ? (
            <PatientEmptyState
              title="No Health Records"
              message="No records are currently available."
              hint="When test recordings or health documents are logged by your clinical care team, they will appear here."
            />
          ) : (
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <div className="table-responsive">
                <table className="table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-color, #334155)', textAlign: 'left' }}>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Record Type</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Description</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Date</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>Status</th>
                      <th style={{ padding: '0.75rem 1rem', fontSize: '0.8125rem', color: 'var(--text-muted)', textAlign: 'right' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recordsData.records
                      .filter((r) => {
                        if (!recordsSearch) return true;
                        const q = recordsSearch.toLowerCase();
                        return (
                          r.record_type?.toLowerCase().includes(q) ||
                          r.category?.toLowerCase().includes(q) ||
                          r.details?.toLowerCase().includes(q)
                        );
                      })
                      .map((record) => (
                        <tr key={record.record_id} style={{ borderBottom: '1px solid var(--border-color, #1e293b)' }}>
                          <td style={{ padding: '0.875rem 1rem', fontWeight: 600 }}>{record.record_type}</td>
                          <td style={{ padding: '0.875rem 1rem', color: 'var(--text-main)' }}>
                            <div>{record.category}</div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{record.details}</div>
                          </td>
                          <td style={{ padding: '0.875rem 1rem', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                            {record.date ? new Date(record.date).toLocaleDateString() : 'N/A'}
                          </td>
                          <td style={{ padding: '0.875rem 1rem' }}>
                            <span className="badge badge-success">{record.status}</span>
                          </td>
                          <td style={{ padding: '0.875rem 1rem', textAlign: 'right' }}>
                            <button
                              className="btn btn-secondary btn-sm"
                              onClick={() => setSelectedRecord(record)}
                            >
                              Details
                            </button>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Record Detail Modal */}
          {selectedRecord && (
            <div
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'rgba(0,0,0,0.7)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 999,
                padding: '1rem',
              }}
            >
              <div className="card" style={{ maxWidth: '500px', width: '100%' }}>
                <div className="card-header" style={{ marginBottom: '1rem' }}>
                  <div className="card-title">Record Details</div>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => setSelectedRecord(null)}
                  >
                    Close
                  </button>
                </div>
                <div style={{ fontSize: '0.875rem', lineHeight: 1.8 }}>
                  <div><strong>Type:</strong> {selectedRecord.record_type}</div>
                  <div><strong>Category:</strong> {selectedRecord.category}</div>
                  <div><strong>Status:</strong> {selectedRecord.status}</div>
                  <div><strong>Date Recorded:</strong> {selectedRecord.date ? new Date(selectedRecord.date).toLocaleString() : 'N/A'}</div>
                  <div style={{ marginTop: '0.5rem' }}><strong>Description:</strong> {selectedRecord.details}</div>
                </div>
                <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-color, #334155)', paddingTop: '1rem' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    <ShieldCheck size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
                    Raw clinical notes and raw electrical signals are protected and retained within your hospital repository.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 3: MY REPORTS */}
      {activeTab === 'reports' && reportsData && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, margin: 0 }}>My Diagnostic Reports</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
              Finalized clinical reports and diagnostic documentation on file with your care team.
            </p>
          </div>

          {reportsData.status === 'NO_DATA' || reportsData.total_count === 0 ? (
            <PatientEmptyState
              title="No Diagnostic Reports"
              message="No reports are currently available."
              hint="When your physician or hospital clinic uploads encounter reports, they will be listed here."
            />
          ) : (
            <div className="grid">
              {reportsData.reports.map((report) => (
                <div className="card" key={report.report_id}>
                  <div className="card-header">
                    <div className="card-icon text-info">
                      <FileText size={20} />
                    </div>
                    <div className="card-title">{report.report_title}</div>
                  </div>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', margin: '0.5rem 0 1rem' }}>
                    {report.summary}
                  </p>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8125rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>
                      Date: {report.date ? new Date(report.date).toLocaleDateString() : 'N/A'}
                    </span>
                    <span className="badge badge-success">{report.status}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 4: MY ASSESSMENT */}
      {activeTab === 'assessment' && overview && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, margin: 0 }}>AI-Assisted Assessment</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
              Research/decision-support dashboard using stored results only (not live inference on this screen).
            </p>
            <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <PatientStatusBadge status={overview.assessment?.status || 'NOT_ENABLED'} />
              {diagnosisData?.inference_capability && (
                <span className="badge badge-secondary">
                  New inference capability: {diagnosisData.inference_capability}
                </span>
              )}
            </div>
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
              {overview.assessment?.message}
            </p>
          </div>

          {diagnosisData?.prediction ? (
            <>
              <div className="card" style={{ marginBottom: '1.5rem' }}>
                <div className="card-title" style={{ marginBottom: '0.75rem' }}>Latest stored assessment</div>
                <PatientAssessmentDetailGrid
                  prediction={diagnosisData.prediction}
                  explainer={diagnosisData.model_confidence_explainer}
                />
              </div>
              <PatientAssessmentCharts charts={diagnosisData.charts} history={diagnosisData.history} />
            </>
          ) : (
            <PatientNotEnabled
              title="No stored assessment yet"
              message={
                overview.assessment?.status === 'NOT_ENABLED'
                  ? overview.assessment?.message
                  : 'No completed assessment is stored for your account. Your clinicians may run authorized evaluations in Diagnostic Studio.'
              }
              disclaimer={overview.assessment?.safety_wording}
            />
          )}

          <div className="card" style={{ marginTop: '1.5rem' }}>
            <div className="card-header">
              <div className="card-icon text-info"><Info size={18} /></div>
              <div className="card-title">Capability vs stored results</div>
            </div>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Stored results mean a past authorized evaluation was saved. They do not by themselves mean new AI inference
              is enabled (`INFERENCE_ENABLED` and fusion flags are evaluated separately on the server).
            </p>
          </div>
        </div>
      )}

      {/* TAB 5: MY RESULTS */}
      {activeTab === 'results' && diagnosisData && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, margin: 0 }}>My Diagnostic Results</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
              Verified AI-assisted diagnostic evaluation results.
            </p>
          </div>

          {diagnosisData.prediction ? (
            <div className="card" style={{ marginBottom: '1.5rem' }}>
              <div className="card-header">
                <div className="card-icon text-success">
                  <CheckCircle size={20} />
                </div>
                <div className="card-title">Stored research-proxy result (not clinically validated)</div>
              </div>
              <PatientAssessmentDetailGrid prediction={diagnosisData.prediction} />
              <dl className="patient-assessment-detail-grid" style={{ marginTop: '0.75rem' }}>
                <div className="patient-assessment-detail-grid__item">
                  <dt>Provenance</dt>
                  <dd>{diagnosisData.prediction.provenance}</dd>
                </div>
                <div className="patient-assessment-detail-grid__item">
                  <dt>Status</dt>
                  <dd><span className="badge badge-success">{diagnosisData.prediction.status}</span></dd>
                </div>
              </dl>

              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setValidationOpen((v) => !v)}
                style={{ marginBottom: '0.75rem' }}
              >
                {validationOpen ? 'Hide' : 'What does “not validated” mean?'}
              </button>
              {validationOpen && (
                <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.7, marginBottom: '1rem' }}>
                  <p><strong>{diagnosisData.validation_notice}</strong></p>
                  <p>{diagnosisData.model_confidence_explainer}</p>
                  <p><strong>NORM:</strong> {diagnosisData.proxy_class_explainer?.NORM}</p>
                  <p><strong>MI/STTC:</strong> {diagnosisData.proxy_class_explainer?.['MI/STTC']}</p>
                  <p>
                    Clinical validation would require independent cohorts, external validation, calibration, subgroup
                    evaluation, and formal clinical safety review. This platform does not claim regulatory approval or
                    proven diagnostic accuracy for this proxy output.
                  </p>
                </div>
              )}

              <div style={{ marginTop: '1.25rem' }}>
                <PatientAssessmentCharts charts={diagnosisData.charts} history={diagnosisData.history} />
              </div>
              <div style={{ marginTop: '1.25rem' }}>
                <PatientSafetyDisclaimer wording={diagnosisData.safety_wording} />
              </div>
            </div>
          ) : (
            <div>
              <PatientEmptyState
                title="No Assessment Results Available"
                message="No diagnostic assessment results are currently available."
                hint="Absence of an AI assessment result does not imply absence of disease."
              />
              <PatientSafetyDisclaimer wording={diagnosisData.safety_wording} />
            </div>
          )}
        </div>
      )}

      {/* TAB 6: QUESTIONS FOR MY CLINICIAN */}
      {activeTab === 'questions' && questionsData && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, margin: 0 }}>Questions for My Clinician</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
              Submit a non-urgent question to your authorized care team. Messages are not monitored continuously.
            </p>
            <p style={{ color: '#f87171', fontSize: '0.8125rem', marginTop: '0.5rem' }}>
              {questionsData.emergency_notice}
            </p>
          </div>

          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', fontSize: '0.875rem', marginBottom: '0.35rem' }}>Your question</label>
            <textarea
              className="form-input"
              rows={4}
              value={questionDraft}
              onChange={(e) => setQuestionDraft(e.target.value)}
              placeholder="Describe your non-emergency question..."
              maxLength={2000}
            />
            <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {(questionsData.suggested_templates || []).map((t) => (
                <button
                  key={t}
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => setQuestionDraft(t)}
                >
                  Use template
                </button>
              ))}
            </div>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              style={{ marginTop: '0.75rem' }}
              disabled={questionSubmitting || !questionDraft.trim()}
              onClick={async () => {
                try {
                  setQuestionSubmitting(true);
                  await apiRequest('/api/v1/patient/questions', {
                    method: 'POST',
                    body: JSON.stringify({ question_text: questionDraft.trim() }),
                  });
                  setQuestionDraft('');
                  await fetchQuestions();
                } catch (err) {
                  setError(err);
                } finally {
                  setQuestionSubmitting(false);
                }
              }}
            >
              {questionSubmitting ? 'Submitting…' : 'Submit question'}
            </button>
          </div>

          {questionsData.questions?.length ? (
            <div className="card">
              <div className="card-title" style={{ marginBottom: '0.75rem' }}>Previous submissions</div>
              {questionsData.questions.map((q) => (
                <div key={q.question_id} style={{ borderTop: '1px solid var(--border-color,#334155)', padding: '0.75rem 0' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {q.submitted_at ? new Date(q.submitted_at).toLocaleString() : ''} · {q.status}
                  </div>
                  <p style={{ fontSize: '0.875rem', margin: '0.35rem 0' }}>{q.question_text}</p>
                  {q.response_text && (
                    <p style={{ fontSize: '0.875rem', color: '#86efac' }}>
                      <strong>Clinician response:</strong> {q.response_text}
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <PatientEmptyState title="No questions submitted yet" message="Your submitted questions will appear here." />
          )}
        </div>
      )}
    </PatientLayout>
  );
}
