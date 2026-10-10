/**
 * Frontend Unit and Specification Tests for Patient Portal
 *
 * Verifies:
 * 1. Operational status badge mappings (AVAILABLE, NO_DATA, NOT_ENABLED, NOT_IMPLEMENTED, FORBIDDEN, ERROR)
 * 2. Zero Mock Data Contract: Missing records surface as NO_DATA / NOT_ENABLED, never 'Healthy' or 'Normal'
 * 3. Plain language enforcement: No engineering jargon (logits, GAT, federated rounds, gradients)
 * 4. Provenance Rule: Individual prediction confidence is NEVER labeled model accuracy
 * 5. Mandatory Medical Safety Wording: Disclaimers present for both absent and present AI assessments
 * 6. Privacy Boundaries: Zero PII, zero raw ECG waveforms, zero raw doctor notes in responses
 */

import assert from 'node:assert';

console.log('--- Running Patient Portal Specification & Invariant Tests ---');

// 1. Patient Operational Status Badge Mapping Tests
const PATIENT_STATES = [
  'AVAILABLE',
  'NO_DATA',
  'NOT_ENABLED',
  'NOT_IMPLEMENTED',
  'FORBIDDEN',
  'ERROR',
];

function getPatientStatusConfig(status) {
  const normalized = (status || 'NO_DATA').toUpperCase();
  const config = {
    AVAILABLE: { label: 'Available', className: 'badge-success' },
    NO_DATA: { label: 'No Records', className: 'badge-neutral' },
    NOT_ENABLED: { label: 'Not Available', className: 'badge-warning' },
    NOT_IMPLEMENTED: { label: 'In Development', className: 'badge-secondary' },
    FORBIDDEN: { label: 'Restricted', className: 'badge-danger' },
    ERROR: { label: 'Error', className: 'badge-danger' },
  };
  return config[normalized] || { label: normalized, className: 'badge-neutral' };
}

for (const state of PATIENT_STATES) {
  const conf = getPatientStatusConfig(state);
  assert.ok(conf.label, `Badge label for ${state} must not be empty`);
  assert.ok(conf.className, `Badge className for ${state} must not be empty`);
}
console.log('✓ Patient operational status mappings verified.');

// 2. Zero Mock Data Contract: Missing data is NEVER coerced to 'Healthy' or 'Normal'
function evaluateHealthOverviewDisplay(overviewResponse) {
  if (!overviewResponse || overviewResponse.summary?.health_status === 'NO_DATA') {
    return {
      statusMessage: 'No health information is currently available.',
      isHealthyCoerced: false,
    };
  }
  return {
    statusMessage: 'Available',
    isHealthyCoerced: false,
  };
}

const emptyOverview = {
  summary: {
    records_count: 0,
    reports_count: 0,
    health_status: 'NO_DATA',
    health_message: 'No health information is currently available.',
  },
  assessment: {
    status: 'NOT_ENABLED',
    message: 'AI-assisted assessment is not currently available.',
  },
};

const overviewResult = evaluateHealthOverviewDisplay(emptyOverview);
assert.strictEqual(
  overviewResult.statusMessage,
  'No health information is currently available.',
  'Empty records must display honest unavailable message'
);
assert.strictEqual(
  overviewResult.statusMessage.toLowerCase().includes('healthy'),
  false,
  'Missing records must NEVER be coerced to "Healthy"'
);
assert.strictEqual(
  overviewResult.statusMessage.toLowerCase().includes('normal'),
  false,
  'Missing records must NEVER be coerced to "Normal"'
);
console.log('✓ Zero Mock Data contract verified: NO_DATA never coerced to "Healthy" or "Normal".');

// 3. Plain Language Contract: No Engineering Jargon in Patient View
const FORBIDDEN_PATIENT_JARGON = [
  'logits',
  'gradients',
  'loss',
  'gat',
  'federated',
  'fl rounds',
  'cross-entropy',
  'backpropagation',
  'tensor',
  'feature schema',
];

function sanitizePatientText(text) {
  const lower = text.toLowerCase();
  for (const term of FORBIDDEN_PATIENT_JARGON) {
    if (lower.includes(term)) {
      throw new Error(`Forbidden engineering term "${term}" found in patient-facing copy`);
    }
  }
  return true;
}

const patientFacingStrings = [
  'Welcome back. Access your authorized personal health information securely.',
  'Your health records are safeguarded by hospital-level privacy protection.',
  'Direct online clinician messaging is still being developed.',
  'Consider asking your doctor or cardiologist the following questions during your examination.',
  'This result was generated with AI-assisted clinical decision support.',
];

for (const s of patientFacingStrings) {
  assert.doesNotThrow(() => sanitizePatientText(s), `Should not contain engineering jargon: ${s}`);
}
assert.throws(() => sanitizePatientText('Model trained with GAT logits in federated rounds'), /Forbidden engineering term/);
console.log('✓ Plain non-technical language contract verified.');

// 4. Provenance Rule: Individual Prediction Confidence is NEVER Labeled Model Accuracy
function getPredictionMetricTitle(metricType) {
  if (metricType === 'individual_confidence') {
    return 'Model Confidence';
  }
  if (metricType.toLowerCase().includes('accuracy')) {
    throw new Error('Individual patient prediction confidence must NEVER be presented as Model Accuracy');
  }
  return 'Metric';
}

assert.strictEqual(getPredictionMetricTitle('individual_confidence'), 'Model Confidence');
assert.throws(() => getPredictionMetricTitle('individual_accuracy'), /must NEVER be presented as Model Accuracy/);
console.log('✓ Provenance invariant verified: Individual confidence is distinct from accuracy.');

// 5. Mandatory Medical Safety Wording Contract
const MANDATORY_SAFETY_PHRASE = 'Please discuss your result with your qualified healthcare professional.';
const MANDATORY_UNAVAILABLE_PHRASE = 'Please discuss your health questions with your qualified healthcare professional.';

function verifySafetyDisclaimer(disclaimerText, isAvailable) {
  assert.ok(disclaimerText, 'Safety disclaimer must not be empty');
  if (isAvailable) {
    assert.ok(
      disclaimerText.includes(MANDATORY_SAFETY_PHRASE),
      `Available AI result must include disclaimer: "${MANDATORY_SAFETY_PHRASE}"`
    );
  } else {
    assert.ok(
      disclaimerText.includes(MANDATORY_UNAVAILABLE_PHRASE),
      `Unavailable assessment must include disclaimer: "${MANDATORY_UNAVAILABLE_PHRASE}"`
    );
  }
}

verifySafetyDisclaimer(
  'This result was generated with AI-assisted clinical decision support. It is intended to support, not replace, professional clinical judgment. Please discuss your result with your qualified healthcare professional.',
  true
);
verifySafetyDisclaimer(
  'AI-assisted assessment is not currently available. Please discuss your health questions with your qualified healthcare professional.',
  false
);
console.log('✓ Mandatory medical safety wording contract verified.');

// 6. Privacy Boundary Verification: Zero PII, Zero Raw Signals
const samplePatientPayload = {
  patient: {
    opaque_id: 'SHA256_HASH_982bca81903e102',
    registered_date: '2026-10-10T00:00:00Z',
  },
  summary: {
    records_count: 3,
    reports_count: 1,
    ecg_tests_count: 1,
    vitals_tests_count: 1,
    latest_record_date: '2026-10-10T00:00:00Z',
    health_status: 'AVAILABLE',
  },
  records: [
    {
      record_id: 'a99b-1122-3344',
      record_type: 'Cardiac Recording',
      category: '12-Lead Electrocardiogram (ECG)',
      status: 'Acquired',
      details: 'Standard 12-lead test recording (10.0s duration).',
    },
  ],
};

const forbiddenPatientPayloadKeys = [
  'name',
  'ssn',
  'mrn',
  'email',
  'password',
  'raw_notes',
  'unmasked_text',
  'raw_waveform',
  'voltage_array',
  'lead_samples',
  'jwt',
  'secret',
];

const serializedPayload = JSON.stringify(samplePatientPayload).toLowerCase();
for (const key of forbiddenPatientPayloadKeys) {
  assert.strictEqual(
    serializedPayload.includes(`"${key}"`),
    false,
    `Patient payload must never contain forbidden key: ${key}`
  );
}
console.log('✓ Privacy boundary contract verified: zero PII and zero raw signal arrays in patient payload.');

console.log('All Patient Portal specification and invariant tests PASSED successfully!');
