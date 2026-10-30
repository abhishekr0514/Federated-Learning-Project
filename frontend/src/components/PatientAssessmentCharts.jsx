import React, { useMemo, useState, useCallback } from 'react';
import {
  buildAssessmentHistoryRows,
  buildPlottableConfidenceSeries,
  CHART_VIEW_HEIGHT,
  CHART_VIEW_WIDTH,
  computeConfidencePlotCoordinates,
  filterSeriesByRange,
  formatConfidencePercent,
  MARGIN,
  normalizeConfidenceForPlot,
} from '../utils/patientAssessmentChartMath';
import './PatientAssessmentCharts.css';

function formatDateLabel(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function AssessmentHistorySection({ history, classAxisLabel }) {
  const rows = useMemo(() => buildAssessmentHistoryRows(history), [history]);

  if (!rows.length) {
    return (
      <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', margin: 0 }} role="status">
        No stored proxy-class history is available.
      </p>
    );
  }

  return (
    <>
      <p id="class-chart-desc" className="patient-assessment-charts__section" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: '0 0 0.75rem' }}>
        {classAxisLabel}. NORM = 0, MI/STTC = 1 in the PTB-XL research task — not a clinical diagnosis.
      </p>

      <div className="patient-assessment-history-wrap patient-assessment-history-wrap--desktop">
        <table className="patient-assessment-history-table" aria-describedby="class-chart-desc">
          <thead>
            <tr>
              <th scope="col">Date / time</th>
              <th scope="col">Proxy class</th>
              <th scope="col">Model confidence</th>
              <th scope="col">Model version</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.predictionId}>
                <td data-label="Date / time">{row.dateTimeLabel}</td>
                <td data-label="Proxy class">{row.proxyClass}</td>
                <td data-label="Model confidence">{row.confidenceLabel}</td>
                <td data-label="Model version">{row.modelVersion}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="patient-assessment-history-cards" aria-describedby="class-chart-desc">
        {rows.map((row) => (
          <article key={row.predictionId} className="patient-assessment-history-card">
            <dl>
              <div>
                <dt>Date / time</dt>
                <dd>{row.dateTimeLabel}</dd>
              </div>
              <div>
                <dt>Proxy class</dt>
                <dd>{row.proxyClass}</dd>
              </div>
              <div>
                <dt>Model confidence</dt>
                <dd>{row.confidenceLabel}</dd>
              </div>
              <div>
                <dt>Model version</dt>
                <dd>{row.modelVersion}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </>
  );
}

export function PatientAssessmentCharts({ charts, history }) {
  const [range, setRange] = useState('all');
  const [activePointId, setActivePointId] = useState(null);

  const versionByPredictionId = useMemo(() => {
    const map = {};
    for (const h of history || []) {
      if (h.prediction_id) map[h.prediction_id] = h;
    }
    return map;
  }, [history]);

  const timelinePoints = charts?.timeline_points || [];

  const plottableSeries = useMemo(
    () => buildPlottableConfidenceSeries(timelinePoints, versionByPredictionId),
    [timelinePoints, versionByPredictionId]
  );

  const filtered = useMemo(
    () => filterSeriesByRange(plottableSeries, range),
    [plottableSeries, range]
  );

  const plot = useMemo(() => computeConfidencePlotCoordinates(filtered), [filtered]);

  const activePoint = plot.plotPoints.find((p) => p.predictionId === activePointId) || null;

  const summaryText = useMemo(() => {
    if (filtered.length === 0) {
      return plottableSeries.length
        ? 'No stored assessments match the selected period.'
        : 'No valid confidence values are available to plot.';
    }
    if (filtered.length === 1) {
      return 'One assessment recorded; more history is needed to show a trend.';
    }
    return `${filtered.length} stored research-proxy assessments in the selected period. The line connects stored events for visibility only—not clinical progression.`;
  }, [filtered, plottableSeries.length]);

  const handlePointFocus = useCallback((id) => setActivePointId(id), []);

  if (!timelinePoints.length) {
    return (
      <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }} role="status">
        Charts appear when at least one assessment result is stored for your account.
      </p>
    );
  }

  if (plottableSeries.length === 0) {
    return (
      <div className="card patient-assessment-charts__section" role="status">
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', margin: 0 }}>
          Assessment records exist, but none have a plottable confidence value or valid date.
        </p>
      </div>
    );
  }

  const plotH = CHART_VIEW_HEIGHT - MARGIN.top - MARGIN.bottom;
  const plotW = CHART_VIEW_WIDTH - MARGIN.left - MARGIN.right;

  return (
    <div className="patient-assessment-charts">
      <div className="patient-assessment-charts__section">
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Period{' '}
            <select className="form-input" value={range} onChange={(e) => setRange(e.target.value)}>
              <option value="all">All stored</option>
              <option value="90d">Last 90 days</option>
              <option value="365d">Last 12 months</option>
            </select>
          </label>
        </div>
        <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', margin: '0.75rem 0 0' }}>{summaryText}</p>
      </div>

      <section className="card patient-assessment-charts__section" aria-labelledby="confidence-chart-title">
        <h3 id="confidence-chart-title" style={{ fontSize: '0.9375rem', margin: '0 0 0.5rem' }}>
          Model confidence over time
        </h3>
        <p id="confidence-chart-desc" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
          {charts?.confidence_axis_label || 'Model confidence'} — not disease probability.
        </p>

        {filtered.length === 0 ? (
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: '0.75rem' }} role="status">
            {summaryText}
          </p>
        ) : (
          <div className="patient-assessment-charts__chart-shell">
            <svg
              role="img"
              aria-labelledby="confidence-chart-title confidence-chart-desc"
              viewBox={`0 0 ${CHART_VIEW_WIDTH} ${CHART_VIEW_HEIGHT}`}
              preserveAspectRatio="xMidYMid meet"
            >
              <rect
                x={MARGIN.left}
                y={MARGIN.top}
                width={plotW}
                height={plotH}
                fill="rgba(15, 23, 42, 0.35)"
                stroke="#334155"
                strokeWidth="1"
              />
              {plot.yTicks.map((tick) => (
                <g key={tick.label}>
                  <line
                    x1={MARGIN.left}
                    y1={tick.y}
                    x2={MARGIN.left + plotW}
                    y2={tick.y}
                    stroke="#475569"
                    strokeDasharray="4 4"
                    opacity="0.6"
                  />
                  <text x={MARGIN.left - 8} y={tick.y + 4} textAnchor="end" fill="#94a3b8" fontSize="10">
                    {tick.label}
                  </text>
                </g>
              ))}
              <text
                x={12}
                y={MARGIN.top + plotH / 2}
                fill="#94a3b8"
                fontSize="10"
                transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
                textAnchor="middle"
              >
                Confidence
              </text>
              <line
                x1={MARGIN.left}
                y1={MARGIN.top + plotH}
                x2={MARGIN.left + plotW}
                y2={MARGIN.top + plotH}
                stroke="#64748b"
                strokeWidth="1"
              />
              {plot.xTickLabels.map((tick, idx) => (
                <text
                  key={`${tick.label}-${idx}`}
                  x={tick.x}
                  y={MARGIN.top + plotH + 16}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="10"
                >
                  {tick.label}
                </text>
              ))}
              {plot.polyline && (
                <polyline fill="none" stroke="#38bdf8" strokeWidth="2" points={plot.polyline} opacity="0.85" />
              )}
              {plot.plotPoints.map((p) => {
                const isActive = activePointId === p.predictionId;
                const label = `${formatDateLabel(p.dateIso)}: ${p.confidenceDisplay} model confidence${
                  p.modelVersion ? `, model ${p.modelVersion}` : ''
                }`;
                return (
                  <circle
                    key={p.predictionId}
                    cx={p.x}
                    cy={p.y}
                    r={isActive ? 7 : 5}
                    fill="#0ea5e9"
                    stroke={isActive ? '#f8fafc' : 'none'}
                    strokeWidth={2}
                    tabIndex={0}
                    role="button"
                    aria-label={label}
                    onMouseEnter={() => handlePointFocus(p.predictionId)}
                    onMouseLeave={() => setActivePointId(null)}
                    onFocus={() => handlePointFocus(p.predictionId)}
                    onBlur={() => setActivePointId(null)}
                  />
                );
              })}
            </svg>

            {activePoint && (
              <div className="patient-assessment-charts__tooltip" role="status">
                <div className="patient-assessment-charts__tooltip-row">
                  <strong>Date:</strong> {formatDateLabel(activePoint.dateIso)}
                </div>
                <div className="patient-assessment-charts__tooltip-row">
                  <strong>Model confidence:</strong> {activePoint.confidenceDisplay}
                </div>
                {activePoint.modelVersion && (
                  <div className="patient-assessment-charts__tooltip-row">
                    <strong>Model version:</strong> {activePoint.modelVersion}
                  </div>
                )}
                {activePoint.predictedProxyClass && (
                  <div className="patient-assessment-charts__tooltip-row">
                    <strong>Proxy class:</strong> {activePoint.predictedProxyClass}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </section>

      <section className="card patient-assessment-charts__section" aria-labelledby="proxy-history-title">
        <h3 id="proxy-history-title" style={{ fontSize: '0.9375rem', margin: '0 0 0.5rem' }}>
          Assessment history (research labels)
        </h3>
        <AssessmentHistorySection
          history={history}
          classAxisLabel={charts?.class_axis_label || 'Research class (NORM vs MI/STTC)'}
        />
      </section>

      <section className="card patient-assessment-charts__section" aria-labelledby="activity-summary-title">
        <h3 id="activity-summary-title" style={{ fontSize: '0.9375rem', margin: '0 0 0.5rem' }}>
          Activity summary
        </h3>
        <p style={{ fontSize: '0.875rem', margin: 0 }}>
          Plottable assessments in view: <strong>{filtered.length}</strong> of{' '}
          <strong>{plottableSeries.length}</strong> with valid confidence (
          {charts?.assessment_count ?? timelinePoints.length} total timeline entries).
        </p>
      </section>
    </div>
  );
}

/** Latest assessment detail block for parent pages (responsive grid). */
export function PatientAssessmentDetailGrid({ prediction, explainer }) {
  if (!prediction) return null;
  return (
    <dl className="patient-assessment-detail-grid">
      <div className="patient-assessment-detail-grid__item">
        <dt>Date</dt>
        <dd>{prediction.assessment_date ? new Date(prediction.assessment_date).toLocaleString() : 'N/A'}</dd>
      </div>
      <div className="patient-assessment-detail-grid__item">
        <dt>Assessment target</dt>
        <dd>{prediction.assessment_target}</dd>
      </div>
      <div className="patient-assessment-detail-grid__item">
        <dt>Proxy class</dt>
        <dd>{prediction.predicted_proxy_class}</dd>
      </div>
      <div className="patient-assessment-detail-grid__item">
        <dt>Model confidence</dt>
        <dd>
          {prediction.model_confidence ||
            formatConfidencePercent(normalizeConfidenceForPlot(prediction.model_confidence_raw)) ||
            'N/A'}
        </dd>
      </div>
      <div className="patient-assessment-detail-grid__item" style={{ gridColumn: '1 / -1' }}>
        <dt>Model version</dt>
        <dd>{prediction.model_version_tag}</dd>
      </div>
      {explainer && (
        <div className="patient-assessment-detail-grid__item" style={{ gridColumn: '1 / -1' }}>
          <dt>Note</dt>
          <dd style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>{explainer}</dd>
        </div>
      )}
    </dl>
  );
}
