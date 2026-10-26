import React, { useState, useEffect, useCallback, useId } from 'react';
import {
  Activity,
  FileText,
  Upload,
  ShieldCheck,
  ShieldAlert,
  Play,
  RotateCcw,
  CheckCircle,
  AlertCircle,
  Clock,
  Layers,
  Info,
  Key,
  FileCheck,
  FileWarning,
  Sparkles,
  ExternalLink,
  Lock,
} from 'lucide-react';
import { EcgWaveformCanvas } from './EcgWaveformCanvas';
import {
  apiEvaluateEcgReport,
  apiValidatePrivacyMasking,
  apiGetEcgReportReadiness,
  apiGetGatFusionReadiness,
  generateUUID,
  ApiError,
} from '../services/api';
import {
  SAMPLE_ECG_NORMAL,
  SAMPLE_ECG_PATHOLOGICAL,
  SAMPLE_ECG_INVALID,
  SAMPLE_REPORTS,
} from '../utils/sampleData';

export function DiagnosticWorkspace({ assignedPatients = [], onPredictionComplete }) {
  // Workflow mode: 'ecg_report' | 'gat_fusion'
  const [workflow, setWorkflow] = useState('ecg_report');

  // Readiness states
  const [ecgReady, setEcgReady] = useState(false);
  const [gatReady, setGatReady] = useState(false);
  const [gatBlockers, setGatBlockers] = useState([]);
  const [checkingReadiness, setCheckingReadiness] = useState(true);

  // ECG Input State
  const [ecgLeads, setEcgLeads] = useState(SAMPLE_ECG_NORMAL.leads);
  const [ecgSource, setEcgSource] = useState('sample:normal'); // 'sample:normal' | 'sample:mi' | 'upload'
  const [ecgFileName, setEcgFileName] = useState('Benchmark: Normal Sinus Rhythm (12x1000)');
  const [ecgFileSize, setEcgFileSize] = useState('~48 KB');
  const [ecgValidationStatus, setEcgValidationStatus] = useState({ valid: true, message: 'Valid 12-lead signal (12 leads x 1000 samples)' });
  const [dragActive, setDragActive] = useState(false);

  // Clinical Report Input State
  const [reportText, setReportText] = useState(SAMPLE_REPORTS[0].text);
  const [reportFileName, setReportFileName] = useState(null);

  // Privacy Masking Verification State
  // 'IDLE' | 'CHECKING' | 'VALIDATED' | 'BLOCKED'
  const [privacyState, setPrivacyState] = useState('IDLE');
  const [privacyDetails, setPrivacyDetails] = useState(null);
  const [privacyError, setPrivacyError] = useState(null);

  // Optional Patient & Idempotency Key
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [idempotencyKey, setIdempotencyKey] = useState(generateUUID());

  // Execution & Prediction State
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [submitError, setSubmitError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  // Fetch live readiness for both workflows
  const checkReadiness = useCallback(async () => {
    setCheckingReadiness(true);
    try {
      const [ecgRes, gatRes] = await Promise.allSettled([
        apiGetEcgReportReadiness(),
        apiGetGatFusionReadiness(),
      ]);

      if (ecgRes.status === 'fulfilled') {
        setEcgReady(Boolean(ecgRes.value?.ready));
      }
      if (gatRes.status === 'fulfilled') {
        setGatReady(Boolean(gatRes.value?.ready));
        setGatBlockers(gatRes.value?.blockers || []);
      }
    } catch (err) {
      console.warn('Failed to fetch initial workflow readiness:', err);
    } finally {
      setCheckingReadiness(false);
    }
  }, []);

  useEffect(() => {
    checkReadiness();
  }, [checkReadiness]);

  // Elapsed timer while submitting
  useEffect(() => {
    let timer;
    if (isSubmitting) {
      setElapsedSeconds(0);
      timer = setInterval(() => {
        setElapsedSeconds((s) => s + 1);
      }, 1000);
    } else {
      clearInterval(timer);
    }
    return () => clearInterval(timer);
  }, [isSubmitting]);

  // Validate parsed ECG data
  const validateParsedEcg = (arr) => {
    if (!Array.isArray(arr)) {
      return { valid: false, message: 'File does not contain a JSON array.' };
    }
    if (arr.length !== 12) {
      return { valid: false, message: `ECG must have exactly 12 leads, found ${arr.length}.` };
    }
    for (let i = 0; i < arr.length; i++) {
      if (!Array.isArray(arr[i])) {
        return { valid: false, message: `Lead ${i + 1} is not a valid numeric array.` };
      }
      if (arr[i].length !== 1000) {
        return { valid: false, message: `Lead ${i + 1} must have 1000 samples, found ${arr[i].length}.` };
      }
      for (let j = 0; j < arr[i].length; j++) {
        const val = Number(arr[i][j]);
        if (!Number.isFinite(val)) {
          return { valid: false, message: `Non-finite sample (NaN or Inf) detected at lead ${i + 1}, index ${j}.` };
        }
      }
    }
    return { valid: true, message: 'Valid 12-lead signal (12 leads x 1000 samples at 100Hz).' };
  };

  // Handle uploaded ECG file (.json or .csv)
  const processEcgFile = async (file) => {
    if (!file) return;
    const name = file.name;
    const sizeKb = (file.size / 1024).toFixed(1) + ' KB';
    setEcgFileName(name);
    setEcgFileSize(sizeKb);
    setEcgSource('upload');
    setSubmitError(null);

    try {
      const text = await file.text();
      let parsed = null;

      if (name.endsWith('.json')) {
        const json = JSON.parse(text);
        parsed = json.ecg_leads || json.leads || json;
      } else if (name.endsWith('.csv') || name.endsWith('.txt')) {
        // Parse CSV: rows can be 12 rows of 1000 or 1000 rows of 12
        const lines = text.trim().split(/\r?\n/).map((l) => l.split(/[,;\t]/).map(Number));
        if (lines.length === 12 && lines[0].length === 1000) {
          parsed = lines;
        } else if (lines.length === 1000 && lines[0].length === 12) {
          // Transpose 1000 rows x 12 cols to 12 x 1000
          parsed = Array.from({ length: 12 }, (_, col) => lines.map((row) => row[col]));
        } else {
          parsed = lines;
        }
      } else {
        setEcgValidationStatus({ valid: false, message: 'Unsupported file extension. Please upload a .json or .csv file.' });
        setEcgLeads(null);
        return;
      }

      const val = validateParsedEcg(parsed);
      setEcgValidationStatus(val);
      if (val.valid) {
        setEcgLeads(parsed);
      } else {
        setEcgLeads(null);
      }
    } catch (err) {
      setEcgValidationStatus({ valid: false, message: `Failed to parse file: ${err.message}` });
      setEcgLeads(null);
    }
  };

  const handleEcgDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processEcgFile(e.dataTransfer.files[0]);
    }
  };

  // Quick load sample ECG
  const handleLoadSampleEcg = (sample) => {
    setEcgLeads(sample.leads);
    setEcgFileName(sample.name);
    setEcgFileSize('~48 KB (In-memory benchmark)');
    setEcgSource(sample.id);
    const val = validateParsedEcg(sample.leads);
    setEcgValidationStatus(val);
    setSubmitError(null);
  };

  // Quick load sample Clinical Report
  const handleLoadSampleReport = (sample) => {
    setReportText(sample.text);
    setReportFileName(null);
    setPrivacyState('IDLE');
    setPrivacyDetails(null);
    setPrivacyError(null);
    setSubmitError(null);
  };

  // Clear ECG
  const handleClearEcg = () => {
    setEcgLeads(null);
    setEcgFileName(null);
    setEcgFileSize(null);
    setEcgValidationStatus({ valid: false, message: 'No ECG signal loaded.' });
  };

  // Validate Privacy Masking via backend
  const handleValidatePrivacy = async () => {
    if (!reportText || !reportText.trim()) {
      setPrivacyError('Please enter clinical report text before requesting privacy validation.');
      return;
    }

    setPrivacyState('CHECKING');
    setPrivacyError(null);
    setPrivacyDetails(null);

    try {
      const res = await apiValidatePrivacyMasking(reportText);
      setPrivacyDetails(res);
      if (res.is_valid && res.status === 'VALIDATED') {
        setPrivacyState('VALIDATED');
      } else {
        setPrivacyState('BLOCKED');
        setPrivacyError(res.validation_message || 'Fail-closed privacy check rejected the document.');
      }
    } catch (err) {
      console.error('Privacy masking check error:', err);
      setPrivacyState('BLOCKED');
      setPrivacyError(err.message || 'Privacy verification service unavailable.');
    }
  };

  // When report text changes, invalidate previous validation
  const handleReportTextChange = (e) => {
    setReportText(e.target.value);
    if (privacyState !== 'IDLE') {
      setPrivacyState('IDLE');
      setPrivacyDetails(null);
      setPrivacyError(null);
    }
    setSubmitError(null);
  };

  // Submit model evaluation
  const handleSubmitPrediction = async () => {
    if (!ecgValidationStatus.valid || !ecgLeads) {
      setSubmitError('Please supply a valid 12-lead ECG waveform array before submission.');
      return;
    }
    if (!reportText || !reportText.trim()) {
      setSubmitError('Please supply a clinical report narrative text before submission.');
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);
    setPredictionResult(null);

    try {
      const payload = {
        ecg_leads: ecgLeads,
        report_text: reportText.trim(),
        patient_id: selectedPatientId ? selectedPatientId : undefined,
        idempotency_key: idempotencyKey.trim() || undefined,
      };

      const result = await apiEvaluateEcgReport(payload, selectedPatientId || null);
      setPredictionResult(result);
      setPrivacyState('VALIDATED');

      if (onPredictionComplete) {
        onPredictionComplete(result);
      }
    } catch (err) {
      console.error('Evaluation request failed:', err);
      setSubmitError(err.message || 'Model prediction evaluation failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Safe form reset
  const handleResetForm = () => {
    handleLoadSampleEcg(SAMPLE_ECG_NORMAL);
    setReportText(SAMPLE_REPORTS[0].text);
    setReportFileName(null);
    setPrivacyState('IDLE');
    setPrivacyDetails(null);
    setPrivacyError(null);
    setSelectedPatientId('');
    setIdempotencyKey(generateUUID());
    setSubmitError(null);
    setPredictionResult(null);
  };

  const isFormSubmittable =
    ecgValidationStatus.valid &&
    Boolean(ecgLeads) &&
    Boolean(reportText && reportText.trim()) &&
    privacyState !== 'BLOCKED' &&
    !isSubmitting;

  return (
    <div className="diagnostic-workspace">
      {/* Workflow Architecture Switcher */}
      <div className="card" style={{ marginBottom: '1.25rem', border: '1px solid #1e293b' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={18} className="text-accent" />
              <h2 style={{ fontSize: '1.125rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                Diagnostic Multimodal Inference Studio
              </h2>
            </div>
            <p style={{ fontSize: '0.8125rem', color: '#94a3b8', margin: '4px 0 0' }}>
              Real-time inference pipeline executing verified neural weights on local hospital nodes.
            </p>
          </div>

          {/* Workflow Tabs */}
          <div style={{ display: 'flex', gap: '0.5rem', background: '#090d16', padding: '4px', borderRadius: 'var(--radius-md)' }}>
            <button
              type="button"
              className={`btn btn-sm ${workflow === 'ecg_report' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setWorkflow('ecg_report')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Activity size={14} />
              <span>Two-Modality ECG + Report</span>
              <span className={`badge ${ecgReady ? 'badge-success' : 'badge-warning'}`} style={{ fontSize: '0.65rem', padding: '1px 5px' }}>
                {ecgReady ? 'Ready' : 'Checking'}
              </span>
            </button>

            <button
              type="button"
              className={`btn btn-sm ${workflow === 'gat_fusion' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setWorkflow('gat_fusion')}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <Layers size={14} />
              <span>GAT Multimodal Fusion</span>
              <span className="badge badge-danger" style={{ fontSize: '0.65rem', padding: '1px 5px' }}>
                Unavailable
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* When GAT Workflow is selected, display honest server state */}
      {workflow === 'gat_fusion' && (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--warning)', marginBottom: '1.5rem', background: '#1c1917' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
            <AlertCircle size={20} className="text-warning" style={{ marginTop: '2px', flexShrink: 0 }} />
            <div>
              <strong style={{ fontSize: '0.9375rem', color: '#fef08a' }}>
                Two-Modality GAT Fusion Pipeline is Disabled (Honest Server State)
              </strong>
              <p style={{ margin: '4px 0 8px', fontSize: '0.8125rem', color: '#e7e5e4' }}>
                The backend service reports that Graph Attention Network (GAT) fusion checkpoints are not deployed in this environment, and <code>FUSION_INFERENCE_ENABLED</code> is set to <code>false</code>.
              </p>
              <div style={{ background: '#0c0a09', padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.75rem' }}>
                <span style={{ color: '#a8a29e' }}>Verified Server Blockers:</span>
                <ul style={{ margin: '4px 0 0', paddingLeft: '1.25rem', color: '#fca5a5' }}>
                  {gatBlockers.length > 0 ? (
                    gatBlockers.map((b, i) => <li key={i}>{b}</li>)
                  ) : (
                    <>
                      <li>ECG-report encoder checkpoint file missing</li>
                      <li>FUSION_MODEL_DIR does not exist</li>
                      <li>FUSION_INFERENCE_ENABLED is false</li>
                    </>
                  )}
                </ul>
              </div>
              <p style={{ margin: '8px 0 0', fontSize: '0.75rem', color: '#a8a29e' }}>
                Switch back to the operational <strong>Two-Modality ECG + Report</strong> workflow above to execute live predictions.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Two-Column Diagnostic Grid */}
      <div className="grid grid-2" style={{ gap: '1.25rem', marginBottom: '1.25rem' }}>
        {/* LEFT COLUMN: 12-Lead ECG Modality Input */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div className="card-icon" style={{ background: 'rgba(34, 211, 238, 0.1)', color: '#22d3ee' }}>
                <Activity size={18} />
              </div>
              <div>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                  12-Lead ECG Signal Input
                </h3>
                <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Shape requirement: exactly (12, 1000) mV</span>
              </div>
            </div>

            {/* Remove / Reset file button */}
            {ecgLeads && (
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={handleClearEcg}
                title="Remove current ECG file"
                style={{ fontSize: '0.75rem' }}
              >
                Clear
              </button>
            )}
          </div>

          {/* Drag & Drop File Zone */}
          <div
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={() => setDragActive(true)}
            onDragLeave={() => setDragActive(false)}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleEcgDrop}
            style={{
              padding: '1rem',
              border: '2px dashed #334155',
              borderRadius: 'var(--radius-md)',
              background: dragActive ? 'rgba(6, 182, 212, 0.05)' : '#090d16',
              textAlign: 'center',
              cursor: 'pointer',
              marginBottom: '0.75rem'
            }}
            onClick={() => document.getElementById('ecg-file-upload-input')?.click()}
          >
            <input
              id="ecg-file-upload-input"
              type="file"
              accept=".json,.csv,.txt"
              style={{ display: 'none' }}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  processEcgFile(e.target.files[0]);
                }
              }}
            />
            <Upload size={24} style={{ margin: '0 auto 0.5rem', color: '#38bdf8' }} />
            <p style={{ fontSize: '0.8125rem', color: '#f1f5f9', margin: 0 }}>
              <strong>Drop your 12-lead ECG waveform file here</strong>, or browse
            </p>
            <p style={{ fontSize: '0.7rem', color: '#64748b', margin: '2px 0 0' }}>
              Accepts .json (array shape 12x1000) or .csv (12 columns or 12 rows of 1000 points)
            </p>
          </div>

          {/* Quick-load Sample Buttons */}
          <div style={{ marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.7rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>
              Or load verified PTB-XL benchmark signals:
            </span>
            <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
              <button
                type="button"
                className={`btn btn-sm ${ecgSource === 'sample:normal' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => handleLoadSampleEcg(SAMPLE_ECG_NORMAL)}
                style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
              >
                Normal Sinus Rhythm
              </button>
              <button
                type="button"
                className={`btn btn-sm ${ecgSource === 'sample:mi' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => handleLoadSampleEcg(SAMPLE_ECG_PATHOLOGICAL)}
                style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
              >
                Anteroseptal MI / STTC
              </button>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => handleLoadSampleEcg(SAMPLE_ECG_INVALID)}
                style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', borderColor: '#ef4444' }}
                title="Test validation with 8 leads instead of 12"
              >
                Malformed (8 Leads Test)
              </button>
            </div>
          </div>

          {/* File validation status bar */}
          <div style={{
            padding: '0.5rem 0.75rem',
            borderRadius: 'var(--radius-sm)',
            background: ecgValidationStatus.valid ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${ecgValidationStatus.valid ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            marginBottom: '0.75rem',
            fontSize: '0.75rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              {ecgValidationStatus.valid ? (
                <FileCheck size={14} className="text-success" />
              ) : (
                <FileWarning size={14} className="text-danger" />
              )}
              <span style={{ color: ecgValidationStatus.valid ? '#86efac' : '#fca5a5' }}>
                {ecgValidationStatus.message}
              </span>
            </div>
            {ecgFileName && (
              <span style={{ color: '#94a3b8', fontSize: '0.7rem' }}>
                {ecgFileSize}
              </span>
            )}
          </div>

          {/* Interactive Waveform Canvas Preview */}
          <div style={{ flex: 1 }}>
            <EcgWaveformCanvas leads={ecgLeads} title={ecgFileName || 'Loaded ECG'} />
          </div>
        </div>

        {/* RIGHT COLUMN: Clinical Report Modality & Privacy Masker */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div className="card-icon" style={{ background: 'rgba(13, 148, 136, 0.1)', color: '#0d9488' }}>
                <FileText size={18} />
              </div>
              <div>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, color: '#f8fafc', margin: 0 }}>
                  Clinical Report Observation
                </h3>
                <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>Tokenized via BioClinicalBERT with Privacy Masking</span>
              </div>
            </div>

            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => {
                setReportText('');
                setPrivacyState('IDLE');
                setPrivacyDetails(null);
              }}
              style={{ fontSize: '0.75rem' }}
            >
              Clear Text
            </button>
          </div>

          {/* Sample Narratives Selector */}
          <div style={{ marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.7rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>
              Pre-load clinical report narrative template:
            </span>
            <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
              {SAMPLE_REPORTS.map((s, idx) => (
                <button
                  key={s.id}
                  type="button"
                  className="btn btn-sm btn-secondary"
                  onClick={() => handleLoadSampleReport(s)}
                  style={{
                    fontSize: '0.75rem',
                    padding: '0.2rem 0.5rem',
                    borderColor: idx === 2 ? '#f59e0b' : undefined,
                  }}
                  title={s.title}
                >
                  {idx === 0 ? 'Normal Report' : idx === 1 ? 'Acute MI Report' : 'PII Stress Test'}
                </button>
              ))}
            </div>
          </div>

          {/* Clinical Narrative Textarea */}
          <div style={{ position: 'relative', flex: 1, minHeight: '180px', marginBottom: '0.75rem' }}>
            <textarea
              className="form-input"
              value={reportText}
              onChange={handleReportTextChange}
              placeholder="Enter patient clinical findings, ECG machine observations, or doctor notes..."
              style={{
                width: '100%',
                height: '100%',
                minHeight: '180px',
                resize: 'vertical',
                fontFamily: 'Inter, system-ui, sans-serif',
                fontSize: '0.8125rem',
                lineHeight: 1.6,
                background: '#090d16',
                border: '1px solid #1e293b',
                color: '#f8fafc',
                padding: '0.75rem',
              }}
            />
            <div style={{ position: 'absolute', bottom: '8px', right: '12px', fontSize: '0.7rem', color: '#64748b', pointerEvents: 'none' }}>
              {reportText.length} chars • {reportText.trim().split(/\s+/).filter(Boolean).length} words
            </div>
          </div>

          {/* Privacy Boundary & Sanitization Section */}
          <div style={{
            background: '#090d16',
            border: privacyState === 'VALIDATED'
              ? '1px solid rgba(16, 185, 129, 0.4)'
              : privacyState === 'BLOCKED'
              ? '1px solid rgba(239, 68, 68, 0.4)'
              : '1px solid #1e293b',
            borderRadius: 'var(--radius-md)',
            padding: '0.75rem',
            marginBottom: '0.75rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Lock size={14} className="text-accent" />
                <strong style={{ fontSize: '0.8125rem', color: '#f8fafc' }}>
                  Hospital Privacy Masking Gate
                </strong>
              </div>

              {/* Status Badge */}
              {privacyState === 'IDLE' && (
                <span className="badge" style={{ fontSize: '0.7rem', background: '#1e293b', color: '#94a3b8' }}>
                  Awaiting Validation
                </span>
              )}
              {privacyState === 'CHECKING' && (
                <span className="badge badge-warning" style={{ fontSize: '0.7rem' }}>
                  Validating Masking...
                </span>
              )}
              {privacyState === 'VALIDATED' && (
                <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>
                  <ShieldCheck size={12} style={{ display: 'inline', marginRight: '3px' }} />
                  Validated by Privacy Gate
                </span>
              )}
              {privacyState === 'BLOCKED' && (
                <span className="badge badge-danger" style={{ fontSize: '0.7rem' }}>
                  <ShieldAlert size={12} style={{ display: 'inline', marginRight: '3px' }} />
                  Masking Blocked
                </span>
              )}
            </div>

            <p style={{ fontSize: '0.75rem', color: '#94a3b8', margin: '0 0 0.5rem' }}>
              Direct identifiers (names, MRNs, phone numbers, emails, dates of birth) are sanitized before neural encoding. Zero raw text or PHI ever leaves local hospital memory.
            </p>

            {/* Validation Breakdown Telemetry */}
            {privacyDetails && (
              <div style={{
                background: '#040711',
                borderRadius: 'var(--radius-sm)',
                padding: '0.5rem',
                fontSize: '0.75rem',
                marginBottom: '0.5rem'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#86efac', fontWeight: 600 }}>
                  <span>Status: {privacyDetails.status}</span>
                  <span>Direct Identifiers Sanitized: {privacyDetails.entities_detected_count}</span>
                </div>
                {privacyDetails.entity_categories && Object.keys(privacyDetails.entity_categories).length > 0 && (
                  <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap', marginTop: '4px' }}>
                    {Object.entries(privacyDetails.entity_categories).map(([cat, count]) => (
                      <span key={cat} className="badge" style={{ fontSize: '0.65rem', background: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee' }}>
                        {cat}: {count}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}

            {privacyError && (
              <div style={{
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.5rem',
                fontSize: '0.75rem',
                color: '#fca5a5',
                marginBottom: '0.5rem'
              }}>
                <strong>Masking Validation Warning:</strong> {privacyError}
              </div>
            )}

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleValidatePrivacy}
              disabled={privacyState === 'CHECKING' || !reportText.trim()}
              style={{ width: '100%', justifyContent: 'center', fontSize: '0.75rem' }}
            >
              <ShieldCheck size={13} style={{ marginRight: '4px' }} />
              Verify Privacy Boundary & Sanitize
            </button>
          </div>
        </div>
      </div>

      {/* Execution Control & Context Bar */}
      <div className="card" style={{ marginBottom: '1.25rem', border: '1px solid #1e293b' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          {/* Metadata Controls: Patient assignment & Idempotency Key */}
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
            {/* Patient dropdown */}
            <div>
              <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>
                Associate Authorized Patient (Optional)
              </label>
              <select
                className="form-input"
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
                style={{ fontSize: '0.8125rem', padding: '0.35rem 0.6rem', minWidth: '220px' }}
              >
                <option value="">Research Proxy (No Patient Linked)</option>
                {assignedPatients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.opaque_patient_id} ({p.gender || 'Patient'})
                  </option>
                ))}
              </select>
            </div>

            {/* Idempotency token */}
            <div>
              <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '2px' }}>
                Idempotency Token
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <input
                  type="text"
                  className="form-input"
                  value={idempotencyKey}
                  onChange={(e) => setIdempotencyKey(e.target.value)}
                  placeholder="UUID or safe retry token"
                  style={{ fontSize: '0.75rem', padding: '0.35rem 0.6rem', width: '220px', fontFamily: 'monospace' }}
                />
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => setIdempotencyKey(generateUUID())}
                  title="Generate new idempotency key"
                  style={{ padding: '0.35rem 0.5rem' }}
                >
                  <RotateCcw size={12} />
                </button>
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleResetForm}
              disabled={isSubmitting}
            >
              <RotateCcw size={14} style={{ marginRight: '6px' }} />
              Reset Form
            </button>

            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSubmitPrediction}
              disabled={!isFormSubmittable}
              style={{
                minWidth: '200px',
                justifyContent: 'center',
                boxShadow: isFormSubmittable ? '0 0 16px rgba(6, 182, 212, 0.35)' : 'none'
              }}
            >
              {isSubmitting ? (
                <>
                  <Activity size={16} className="spin" style={{ marginRight: '6px' }} />
                  Running Neural Pass ({elapsedSeconds}s)...
                </>
              ) : (
                <>
                  <Play size={16} style={{ marginRight: '6px' }} />
                  Run Model Inference
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Submission Error Banner */}
      {submitError && (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--danger)', marginBottom: '1.25rem' }}>
          <AlertCircle size={18} className="text-danger" style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }} />
          <strong>Inference Execution Error:</strong> {submitError}
        </div>
      )}

      {/* REAL PREDICTION RESULT CARD */}
      {predictionResult && (
        <div className="card prediction-result-card" style={{
          border: predictionResult.predicted_class_index === 1
            ? '1px solid rgba(245, 158, 11, 0.5)'
            : '1px solid rgba(16, 185, 129, 0.5)',
          background: '#090d16',
          boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
          animation: 'fadeIn 0.3s ease-in-out'
        }}>
          <div className="card-header" style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderBottom: '1px solid #1e293b',
            paddingBottom: '0.75rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
              <div className="card-icon" style={{
                background: predictionResult.predicted_class_index === 1
                  ? 'rgba(245, 158, 11, 0.15)'
                  : 'rgba(16, 185, 129, 0.15)',
                color: predictionResult.predicted_class_index === 1 ? '#f59e0b' : '#10b981'
              }}>
                <Activity size={22} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.125rem', fontWeight: 700, color: '#f8fafc', margin: 0 }}>
                  Model Prediction Evaluation (Genuine Inference Output)
                </h3>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  Execution ID: {predictionResult.prediction_id}
                </span>
              </div>
            </div>

            <span className={`badge ${predictionResult.predicted_class_index === 1 ? 'badge-warning' : 'badge-success'}`} style={{
              fontSize: '0.875rem',
              fontWeight: 700,
              padding: '0.35rem 0.85rem'
            }}>
              Proxy Target: {predictionResult.predicted_class_label}
            </span>
          </div>

          <div style={{ padding: '1.25rem 0' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
              {/* Predicted Class */}
              <div style={{ background: '#0f172a', borderRadius: 'var(--radius-md)', padding: '1rem', border: '1px solid #1e293b' }}>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Predicted Proxy Class
                </span>
                <div style={{
                  fontSize: '1.5rem',
                  fontWeight: 700,
                  marginTop: '4px',
                  color: predictionResult.predicted_class_index === 1 ? '#f59e0b' : '#10b981'
                }}>
                  {predictionResult.predicted_class_label}
                  <span style={{ fontSize: '0.8125rem', color: '#64748b', fontWeight: 400, marginLeft: '6px' }}>
                    (Index {predictionResult.predicted_class_index})
                  </span>
                </div>
                <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
                  {predictionResult.predicted_class_index === 0 ? 'Normal Sinus Rhythm Proxy' : 'Myocardial Infarction / STTC Proxy'}
                </span>
              </div>

              {/* Model Sigmoid Probability */}
              <div style={{ background: '#0f172a', borderRadius: 'var(--radius-md)', padding: '1rem', border: '1px solid #1e293b' }}>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Sigmoid Class-1 Probability
                </span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', color: '#38bdf8' }}>
                  {(predictionResult.probability * 100).toFixed(2)}%
                </div>
                <div style={{
                  height: '6px',
                  background: '#1e293b',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginTop: '8px'
                }}>
                  <div style={{
                    width: `${Math.min(100, Math.max(0, predictionResult.probability * 100))}%`,
                    height: '100%',
                    background: predictionResult.probability >= 0.5 ? '#f59e0b' : '#10b981'
                  }} />
                </div>
                <span style={{ fontSize: '0.6875rem', color: '#64748b', display: 'block', marginTop: '4px' }}>
                  Threshold: 0.50 (NORM &lt; 0.50 &le; MI/STTC)
                </span>
              </div>

              {/* Architecture ID */}
              <div style={{ background: '#0f172a', borderRadius: 'var(--radius-md)', padding: '1rem', border: '1px solid #1e293b' }}>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Architecture & Version
                </span>
                <div style={{ fontSize: '0.875rem', fontWeight: 600, color: '#f8fafc', marginTop: '6px', fontFamily: 'monospace' }}>
                  {predictionResult.architecture_id}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px', fontFamily: 'monospace' }}>
                  tag: {predictionResult.model_version_tag}
                </div>
                <div style={{ marginTop: '6px' }}>
                  <span className="badge badge-success" style={{ fontSize: '0.65rem' }}>
                    Privacy: {predictionResult.privacy_validation_status || 'VALIDATED'}
                  </span>
                </div>
              </div>

              {/* Provenance & Timestamp */}
              <div style={{ background: '#0f172a', borderRadius: 'var(--radius-md)', padding: '1rem', border: '1px solid #1e293b' }}>
                <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Evaluation Provenance
                </span>
                <div style={{ fontSize: '0.75rem', color: '#e2e8f0', marginTop: '4px', fontFamily: 'monospace', wordBreak: 'break-all' }}>
                  {predictionResult.provenance}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '6px' }}>
                  <Clock size={11} style={{ display: 'inline', marginRight: '3px' }} />
                  {predictionResult.evaluated_at || new Date().toISOString()}
                </div>
              </div>
            </div>

            {/* Target Definition from Model Contract */}
            <div style={{ background: '#040711', padding: '0.75rem', borderRadius: 'var(--radius-sm)', marginBottom: '1rem', fontSize: '0.8125rem' }}>
              <span style={{ color: '#94a3b8', fontWeight: 600 }}>Target Definition:</span>{' '}
              <span style={{ color: '#e2e8f0' }}>{predictionResult.target_definition}</span>
            </div>

            {/* Strict Medical Safety Disclaimer */}
            <div className="notice-banner" style={{ borderLeftColor: 'var(--warning)', margin: 0, fontSize: '0.78125rem' }}>
              <strong style={{ color: '#fef08a', display: 'block', marginBottom: '2px' }}>
                Medical Research Safety Notice & Limitations:
              </strong>
              <p style={{ margin: 0, color: '#d6d3d1' }}>
                {predictionResult.disclaimer}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
