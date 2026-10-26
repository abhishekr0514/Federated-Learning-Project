/** Pure helpers for patient assessment confidence timeline SVG (no synthetic data). */

export const CHART_VIEW_WIDTH = 520;
export const CHART_VIEW_HEIGHT = 200;
export const MARGIN = { top: 16, right: 20, bottom: 40, left: 52 };

/**
 * Normalize API confidence to 0–1 for plotting. Display formatting uses the same normalized value.
 * Accepts fractional 0–1 or percentage 0–100; rejects invalid/null.
 */
export function normalizeConfidenceForPlot(raw) {
  if (raw === null || raw === undefined) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  if (n >= 0 && n <= 1) return n;
  if (n > 1 && n <= 100) return n / 100;
  return null;
}

export function formatConfidencePercent(normalized) {
  if (normalized === null) return 'n/a';
  return `${(normalized * 100).toFixed(1)}%`;
}

export function parseChartTimestamp(iso) {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

/**
 * @param {Array<{ prediction_id: string, date?: string, model_confidence_raw?: number|null }>} timelinePoints
 * @param {Record<string, { model_version_tag?: string }>} versionByPredictionId
 */
export function buildPlottableConfidenceSeries(timelinePoints, versionByPredictionId = {}) {
  const series = [];
  for (const p of timelinePoints || []) {
    const normalized = normalizeConfidenceForPlot(p.model_confidence_raw);
    const ts = parseChartTimestamp(p.date);
    if (normalized === null) continue;
    series.push({
      predictionId: p.prediction_id,
      dateIso: p.date,
      timestamp: ts,
      confidenceNormalized: normalized,
      confidenceDisplay: formatConfidencePercent(normalized),
      modelVersion: versionByPredictionId[p.prediction_id]?.model_version_tag || null,
      predictedProxyClass: p.predicted_proxy_class ?? null,
    });
  }
  series.sort((a, b) => {
    if (a.timestamp != null && b.timestamp != null) return a.timestamp - b.timestamp;
    if (a.timestamp != null) return -1;
    if (b.timestamp != null) return 1;
    return 0;
  });
  return series;
}

export function filterSeriesByRange(series, range) {
  if (range === 'all' || series.length <= 1) return series;
  const days = range === '90d' ? 90 : 365;
  const cutoff = Date.now() - days * 86400000;
  return series.filter((p) => p.timestamp != null && p.timestamp >= cutoff);
}

/** Map confidence series to SVG coordinates in viewBox space. Y domain fixed [0,1]. */
export function computeConfidencePlotCoordinates(series) {
  const plotW = CHART_VIEW_WIDTH - MARGIN.left - MARGIN.right;
  const plotH = CHART_VIEW_HEIGHT - MARGIN.top - MARGIN.bottom;
  const n = series.length;

  if (n === 0) {
    return { plotPoints: [], polyline: '', yTicks: defaultYTicks(), xTickLabels: [] };
  }

  const times = series.map((p) => p.timestamp).filter((t) => t != null);
  const minT = times.length ? Math.min(...times) : 0;
  const maxT = times.length ? Math.max(...times) : 1;
  const timeSpan = maxT - minT;

  const plotPoints = series.map((p, i) => {
    let x;
    if (n === 1) {
      x = MARGIN.left + plotW / 2;
    } else if (p.timestamp != null && timeSpan > 0) {
      x = MARGIN.left + ((p.timestamp - minT) / timeSpan) * plotW;
    } else {
      x = MARGIN.left + (i / (n - 1)) * plotW;
    }
    const y = MARGIN.top + plotH * (1 - p.confidenceNormalized);
    return { ...p, x, y };
  });

  const polyline =
    plotPoints.length >= 2 ? plotPoints.map((p) => `${p.x},${p.y}`).join(' ') : '';

  const xTickLabels = buildXTicks(plotPoints, minT, maxT, plotW);

  return {
    plotPoints,
    polyline,
    yTicks: defaultYTicks(),
    xTickLabels,
    plotW,
    plotH,
  };
}

function defaultYTicks() {
  return [0, 0.25, 0.5, 0.75, 1].map((v) => ({
    value: v,
    label: `${Math.round(v * 100)}%`,
    y: MARGIN.top + (CHART_VIEW_HEIGHT - MARGIN.top - MARGIN.bottom) * (1 - v),
  }));
}

function buildXTicks(plotPoints, minT, maxT, plotW) {
  if (plotPoints.length === 0) return [];
  if (plotPoints.length === 1) {
    const p = plotPoints[0];
    return [{ x: p.x, label: formatShortDate(p.dateIso) }];
  }
  const labels = [];
  const first = plotPoints[0];
  const last = plotPoints[plotPoints.length - 1];
  labels.push({ x: first.x, label: formatShortDate(first.dateIso) });
  if (plotPoints.length > 2) {
    const mid = plotPoints[Math.floor(plotPoints.length / 2)];
    labels.push({ x: mid.x, label: formatShortDate(mid.dateIso) });
  }
  if (last.predictionId !== first.predictionId) {
    labels.push({ x: last.x, label: formatShortDate(last.dateIso) });
  }
  return labels;
}

export function formatShortDate(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return String(iso);
  }
}

/** Normalized rows for assessment history table/cards (newest first). */
export function buildAssessmentHistoryRows(history) {
  const rows = [];
  for (const h of history || []) {
    if (!h?.prediction_id) continue;
    const normalized = normalizeConfidenceForPlot(h.model_confidence_raw);
    rows.push({
      predictionId: h.prediction_id,
      dateTimeLabel: h.assessment_date ? formatAssessmentDateTime(h.assessment_date) : '—',
      dateTimeIso: h.assessment_date || null,
      proxyClass: h.predicted_proxy_class || 'Unknown',
      confidenceLabel:
        h.model_confidence || (normalized !== null ? formatConfidencePercent(normalized) : 'n/a'),
      modelVersion: h.model_version_tag || '—',
      architectureConfig: h.architecture_config || null,
      targetDefinition: h.assessment_target || null,
    });
  }
  rows.sort((a, b) => {
    const ta = parseChartTimestamp(a.dateTimeIso);
    const tb = parseChartTimestamp(b.dateTimeIso);
    if (ta != null && tb != null) return tb - ta;
    return 0;
  });
  return rows;
}

export function formatAssessmentDateTime(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return String(iso);
  }
}
