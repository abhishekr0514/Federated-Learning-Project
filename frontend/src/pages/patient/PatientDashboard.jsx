import React, { useState, useEffect, useCallback } from 'react';
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

  // Tab switch effect
  useEffect(() => {
    if (activeTab === 'health') {
      fetchOverview();
    } else if (activeTab === 'records') {
      fetchRecords();
    } else if (activeTab === 'reports') {
      fetchReports();
    } else if (activeTab === 'results' || activeTab === 'assessment') {
      fetchDiagnosis();
    }
  }, [activeTab, fetchOverview, fetchRecords, fetchReports, fetchDiagnosis]);

  // Handle errors
  if (error) {
    if (error instanceof ApiError && error.status === 403) {
      return (
        <PatientLayout activeTab={activeTab} onSelectTab={setActiveTab}>
          <PatientForbidden message="You do not have authorization to view this patient portal." />
        </PatientLayout>
      );
    }
    if (error instanceof ApiError && error.status === 404) {
      return (
        <PatientLayout activeTab={activeTab} onSelectTab={setActiveTab}>
          <PatientNotFound message="No patient profile found associated with your account." />
        </PatientLayout>
      );
    }
    return (
      <PatientLayout activeTab={activeTab} onSelectTab={setActiveTab}>
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
    <PatientLayout activeTab={activeTab} onSelectTab={setActiveTab}>
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
      {activeTab === 'assessment' && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, margin: 0 }}>AI-Assisted Assessment</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
              Information about future automated clinical assessment capabilities.
            </p>
          </div>

          <PatientNotEnabled
            title="AI-Assisted Assessment is Not Currently Available"
            message="Automated clinical diagnosis models are currently undergoing scientific research and validation and are not active for clinical decision making."
            disclaimer="AI-assisted assessment is not currently available. Absence of an AI-generated result does not indicate absence of disease. Please discuss your health questions with your qualified healthcare professional."
          />

          <div className="card" style={{ marginTop: '1.5rem' }}>
            <div className="card-header">
              <div className="card-icon text-info">
                <Info size={18} />
              </div>
              <div className="card-title">What this means for you</div>
            </div>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              <p>
                Our AI-assisted clinical decision tools are strictly supplemental and remain offline until rigorous
                validation is complete. Your doctors continue to assess your health using established medical standards.
              </p>
              <p style={{ margin: '0.5rem 0 0' }}>
                We never display artificial or placeholder health scores. If you have questions about your symptoms or tests,
                please speak directly with your doctor.
              </p>
            </div>
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
                <div className="card-title">Verified Assessment Result</div>
              </div>
              <div style={{ margin: '1rem 0', fontSize: '0.9375rem', lineHeight: 1.8 }}>
                <div><strong>Assessment Target:</strong> {diagnosisData.prediction.assessment_target}</div>
                <div><strong>Evaluation Date:</strong> {diagnosisData.prediction.assessment_date ? new Date(diagnosisData.prediction.assessment_date).toLocaleDateString() : 'N/A'}</div>
                {diagnosisData.prediction.model_confidence && (
                  <div><strong>Model Confidence:</strong> {diagnosisData.prediction.model_confidence}</div>
                )}
                <div><strong>Status:</strong> <span className="badge badge-success">{diagnosisData.prediction.status}</span></div>
              </div>

              {/* Mandatory Medical Safety Disclaimer */}
              <PatientSafetyDisclaimer wording={diagnosisData.safety_wording} />
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
      {activeTab === 'questions' && (
        <div>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, margin: 0 }}>Questions for My Clinician</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem', margin: '0.25rem 0 0' }}>
              Suggested discussion topics and questions to bring to your next healthcare appointment.
            </p>
          </div>

          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div className="card-header">
              <div className="card-icon text-info">
                <MessageSquare size={20} />
              </div>
              <div className="card-title">Suggested Appointment Questions</div>
            </div>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.8 }}>
              <p>Consider asking your doctor or cardiologist the following questions during your examination:</p>
              <ul style={{ paddingLeft: '1.25rem', margin: '0.5rem 0' }}>
                <li>What do my recent cardiac recordings or test reports indicate about my heart health?</li>
                <li>Are there any specific lifestyle, dietary, or exercise recommendations based on my readings?</li>
                <li>Do my recorded observations suggest that any medications should be adjusted?</li>
                <li>When would you recommend scheduling my next routine follow-up appointment?</li>
                <li>What symptoms should prompt me to contact your office or seek immediate care?</li>
              </ul>
            </div>
          </div>

          <PatientNotImplemented
            title="Online Clinician Messaging"
            message="Direct online messaging with your healthcare team is still being developed. For medical advice or urgent matters, please contact your hospital clinic directly."
          />
        </div>
      )}
    </PatientLayout>
  );
}
