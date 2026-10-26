/**
 * Patient assessment confidence chart math: scaling and SVG coordinates.
 */

import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildAssessmentHistoryRows,
  buildPlottableConfidenceSeries,
  computeConfidencePlotCoordinates,
  normalizeConfidenceForPlot,
  formatConfidencePercent,
  CHART_VIEW_HEIGHT,
  MARGIN,
} from '../src/utils/patientAssessmentChartMath.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

console.log('--- patient_assessment_chart.test.js ---');

assert.strictEqual(normalizeConfidenceForPlot(0.62), 0.62);
assert.strictEqual(normalizeConfidenceForPlot(62), 0.62);
assert.strictEqual(normalizeConfidenceForPlot(null), null);
assert.strictEqual(normalizeConfidenceForPlot(150), null);
assert.strictEqual(formatConfidencePercent(0.625), '62.5%');

const empty = buildPlottableConfidenceSeries([
  { prediction_id: 'a', date: '2026-01-01T00:00:00Z', model_confidence_raw: null },
]);
assert.strictEqual(empty.length, 0);

const one = buildPlottableConfidenceSeries([
  {
    prediction_id: 'p1',
    date: '2026-06-01T12:00:00Z',
    model_confidence_raw: 0.75,
  },
]);
assert.strictEqual(one.length, 1);
const onePlot = computeConfidencePlotCoordinates(one);
assert.strictEqual(onePlot.plotPoints.length, 1);
assert.strictEqual(onePlot.polyline, '');
const plotH = CHART_VIEW_HEIGHT - MARGIN.top - MARGIN.bottom;
const expectedY = MARGIN.top + plotH * (1 - 0.75);
assert.ok(Math.abs(onePlot.plotPoints[0].y - expectedY) < 0.01);
assert.ok(onePlot.plotPoints[0].y < MARGIN.top + plotH * 0.3, '75% confidence should plot in upper third');

const multi = buildPlottableConfidenceSeries([
  { prediction_id: 'a', date: '2026-01-01T00:00:00Z', model_confidence_raw: 0.4 },
  { prediction_id: 'b', date: '2026-03-01T00:00:00Z', model_confidence_raw: 40 },
  { prediction_id: 'c', date: '2026-05-01T00:00:00Z', model_confidence_raw: 0.9 },
]);
assert.strictEqual(multi.length, 3);
const multiPlot = computeConfidencePlotCoordinates(multi);
assert.strictEqual(multiPlot.plotPoints.length, 3);
assert.ok(multiPlot.polyline.includes(','));
assert.ok(multiPlot.plotPoints[0].x < multiPlot.plotPoints[2].x);

const flat = buildPlottableConfidenceSeries([
  { prediction_id: 'a', date: '2026-01-01T00:00:00Z', model_confidence_raw: 0.5 },
  { prediction_id: 'b', date: '2026-02-01T00:00:00Z', model_confidence_raw: 0.5 },
]);
const flatPlot = computeConfidencePlotCoordinates(flat);
assert.ok(Math.abs(flatPlot.plotPoints[0].y - flatPlot.plotPoints[1].y) < 0.01);
assert.ok(flatPlot.polyline.length > 0);

const longVersion = 'medshield.diagnostic_net.gat_ecg_report@1.0.0-extra-build-metadata-that-wraps';
const historyRows = buildAssessmentHistoryRows([
  {
    prediction_id: 'p1',
    assessment_date: '2026-01-15T10:00:00Z',
    predicted_proxy_class: 'NORM',
    model_confidence_raw: 0.42,
    model_version_tag: longVersion,
  },
  {
    prediction_id: 'p2',
    assessment_date: '2026-03-20T14:30:00Z',
    predicted_proxy_class: 'MI/STTC',
    model_confidence: '61.0%',
    model_version_tag: 'gat-v2',
  },
]);
assert.strictEqual(historyRows.length, 2);
assert.strictEqual(historyRows[0].predictionId, 'p2');
assert.strictEqual(historyRows[0].modelVersion, 'gat-v2');
assert.ok(historyRows[1].modelVersion.includes('gat_ecg_report'));

const cssPath = path.join(__dirname, '../src/components/PatientAssessmentCharts.css');
const css = fs.readFileSync(cssPath, 'utf8');
assert.ok(css.includes('patient-assessment-charts'));
assert.ok(css.includes('overflow-wrap: anywhere'));
assert.ok(css.includes('min-width: 0'));
assert.ok(css.includes('@media (max-width: 639px)'));
assert.ok(css.includes('@media (min-width: 640px)'));
assert.ok(!css.includes('position: absolute'), 'layout CSS must not use absolute positioning for sections');

console.log('patient_assessment_chart.test.js: all assertions passed');
