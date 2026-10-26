/**
 * Diagnostic Studio patient dropdown: UUID binding vs opaque display labels.
 */

import assert from 'node:assert';
import {
  DEMO_SYNTHETIC_OPAQUE_PREFIX,
  formatPatientOptionLabel,
  isDemoSyntheticOpaque,
  partitionPatientOptions,
  resolvePatientUuid,
} from '../src/utils/patientReferenceDisplay.js';

console.log('--- patient_reference_display.test.js ---');

const demoUuid = 'a1b2c3d4-0005-4000-8000-000000000001';
const demoRow = {
  patient_id: demoUuid,
  opaque_patient_id: 'DEMO-SYNTH-PATIENT-0001',
  is_demo_synthetic: true,
  display_label: 'DEMO-SYNTH-PATIENT-0001 · DEMO-METRO',
  hospital_code: 'DEMO-METRO',
};

assert.strictEqual(resolvePatientUuid(demoRow), demoUuid);
assert.strictEqual(formatPatientOptionLabel(demoRow), demoRow.display_label);
assert.ok(isDemoSyntheticOpaque('DEMO-SYNTH-PATIENT-0001'));
assert.ok(DEMO_SYNTHETIC_OPAQUE_PREFIX.startsWith('DEMO-SYNTH-'));

const hashRow = {
  patient_id: '40004d41-e306-459d-abb6-c76dcec54add',
  opaque_patient_id:
    'c89a71239bf049b1a520fa981e7d34107298319f030f81d86d5268c07e594d21',
  is_demo_synthetic: false,
};

const parts = partitionPatientOptions([hashRow, demoRow]);
assert.strictEqual(parts.demo.length, 1);
assert.strictEqual(parts.other.length, 1);
assert.strictEqual(resolvePatientUuid(parts.demo[0]), demoUuid);

console.log('patient_reference_display.test.js: all assertions passed');
