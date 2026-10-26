import React, { useCallback, useEffect, useState } from 'react';
import { AlertCircle, BarChart3, Play, RefreshCw } from 'lucide-react';
import {
  apiGetCausalReadiness,
  apiListCausalCohorts,
  apiRunCausalAnalysis,
  ApiError,
} from '../services/api';
import { ClinicalStatusBadge } from './ClinicalStates';

const DEFAULT_OBS_CONF = ['age_years'];

export function CausalAnalysisPanel() {
  const [readiness, setReadiness] = useState(null);
  const [cohorts, setCohorts] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [treatment, setTreatment] = useState('sex');
  const [outcome, setOutcome] = useState('research_proxy_label');
  const [confoundersText, setConfoundersText] = useState(DEFAULT_OBS_CONF.join(', '));

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [r, c] = await Promise.all([apiGetCausalReadiness(), apiListCausalCohorts()]);
      setReadiness(r);
      setCohorts(c);
      const cohort = c?.cohorts?.[0];
      if (r?.causal_effect_estimation_allowed && r?.metadata_variables?.length) {
        setTreatment('sex');
        setOutcome('research_proxy_label');
        setConfoundersText(DEFAULT_OBS_CONF.join(', '));
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to load causal analysis readiness.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    const confounders = confoundersText
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    const cohortId = cohorts?.cohorts?.[0]?.cohort_id || 'gat_ecg_report_ptbxl_research_v1';
    try {
      const res = await apiRunCausalAnalysis({
        cohort_id: cohortId,
        treatment,
        outcome,
        confounders,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Causal analysis request failed.');
    } finally {
      setRunning(false);
    }
  };

  const summary = readiness?.summary || 'UNKNOWN';
  const cohortMeta = cohorts?.cohorts?.[0];
  const canRunDoWhy = Boolean(readiness?.causal_effect_estimation_allowed && readiness?.causal_pipeline_available);
  const descriptiveOnly = readiness?.cohort_available && !readiness?.causal_estimand_identifiable;
  const metaVars = readiness?.metadata_variables || [];

  return (
    <div className="card">
      <div className="card-header" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BarChart3 size={20} className="text-accent" />
          <h2 style={{ margin: 0, fontSize: '1.125rem' }}>DoWhy GAT-Grounded Cohort Analysis</h2>
          <ClinicalStatusBadge status={summary} />
        </div>
        <button type="button" className="btn btn-secondary btn-sm" onClick={load} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
        </button>
      </div>

      <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
        Research-only cohort built from frozen ECG+report embeddings and trained two-modality GAT outputs.
        GAT predictions are descriptive only. DoWhy pipeline availability, estimand identifiability, and validated
        causal claims are separate; refutation success does not validate causation.
      </p>

      {readiness && (
        <ul style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
          <li>Pipeline available: {String(readiness.causal_pipeline_available)}</li>
          <li>Estimand identifiable (platform review): {String(readiness.causal_estimand_identifiable)}</li>
          <li>Claim validated: {String(readiness.causal_claim_validated)}</li>
          <li>Effect estimation API: {readiness.causal_effect_estimation_allowed ? 'enabled' : 'disabled (default)'}</li>
        </ul>
      )}

      {readiness?.reason && (
        <div className="notice-banner" style={{ marginTop: '0.75rem', fontSize: '0.8125rem' }}>
          {readiness.reason}
        </div>
      )}

      {descriptiveOnly && (
        <div className="notice-banner" style={{ marginTop: '0.75rem', borderLeftColor: 'var(--warning)' }}>
          {readiness?.causal_design_review?.rationale
            || 'Descriptive GAT cohort available; automated causal effect claims are not approved for this design.'}
        </div>
      )}

      {error && (
        <div className="notice-banner" style={{ borderLeftColor: 'var(--danger)', marginTop: '0.75rem' }}>
          <AlertCircle size={16} style={{ display: 'inline', marginRight: 6 }} />
          {error}
        </div>
      )}

      {cohortMeta && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.75rem' }}>
          <div>
            Cohort: {cohortMeta.cohort_id} · mode: {cohortMeta.cohort_mode || readiness?.cohort_mode}
            {cohortMeta.sample_count != null && <> · n={cohortMeta.sample_count}</>}
          </div>
          {readiness?.provenance?.gat_model_version && (
            <div>GAT model: {readiness.provenance.gat_model_version}</div>
          )}
          {cohortMeta.descriptive_variables && (
            <div>Descriptive fields: {cohortMeta.descriptive_variables.join(', ')}</div>
          )}
          {cohortMeta.descriptive_statistics && (
            <pre style={{ fontSize: '0.65rem', overflow: 'auto', marginTop: '0.35rem' }}>
              {JSON.stringify(cohortMeta.descriptive_statistics, null, 2)}
            </pre>
          )}
        </div>
      )}

      {canRunDoWhy && (
        <>
          <div className="grid grid-2" style={{ gap: '1rem', marginTop: '1rem' }}>
            <div>
              <label className="form-label">Treatment / exposure (sourced metadata)</label>
              <select className="form-input" value={treatment} onChange={(e) => setTreatment(e.target.value)}>
                {metaVars.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="form-label">Outcome</label>
              <select className="form-input" value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                {metaVars.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ marginTop: '0.75rem' }}>
            <label className="form-label">Confounders (comma-separated)</label>
            <input
              className="form-input"
              value={confoundersText}
              onChange={(e) => setConfoundersText(e.target.value)}
            />
          </div>

          <button
            type="button"
            className="btn btn-primary"
            style={{ marginTop: '1rem' }}
            disabled={running}
            onClick={handleRun}
          >
            {running ? 'Running DoWhy pipeline…' : (
              <>
                <Play size={14} style={{ marginRight: 6 }} /> Run model → identify → estimate → refute
              </>
            )}
          </button>
        </>
      )}

      {result && (
        <div style={{ marginTop: '1.25rem', padding: '1rem', background: '#0f172a', borderRadius: 8 }}>
          <div style={{ marginBottom: '0.5rem' }}>
            <strong>Status:</strong> {result.status}
          </div>
          {result.causal_question && (
            <div style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>{result.causal_question}</div>
          )}
          {result.causal_claim_validated === false && (
            <div style={{ color: '#fcd34d' }}>Causal claim validated: false (refutation ≠ causation)</div>
          )}
          {result.effect_estimate != null && (
            <div><strong>Effect estimate:</strong> {result.effect_estimate.toFixed(6)}</div>
          )}
          {result.estimation_method && (
            <div><strong>Method:</strong> {result.estimation_method}</div>
          )}
          {result.sample_size != null && (
            <div><strong>Sample size:</strong> {result.sample_size}</div>
          )}
          {result.message && <div style={{ color: '#fca5a5' }}>{result.message}</div>}
          {result.refutation && (
            <pre style={{ fontSize: '0.7rem', overflow: 'auto', marginTop: '0.5rem' }}>
              {JSON.stringify(result.refutation, null, 2)}
            </pre>
          )}
          <div className="notice-banner" style={{ marginTop: '0.75rem', fontSize: '0.75rem' }}>
            {result.disclaimer}
          </div>
        </div>
      )}

      <div className="notice-banner" style={{ marginTop: '1rem', fontSize: '0.75rem' }}>
        {readiness?.disclaimer || 'Research-only; not for clinical treatment decisions.'}
      </div>
    </div>
  );
}
