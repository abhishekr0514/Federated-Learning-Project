/** Opaque patient list labels for Diagnostic Studio (no PII). */

export const DEMO_SYNTHETIC_OPAQUE_PREFIX = 'DEMO-SYNTH-';

export function isDemoSyntheticOpaque(opaquePatientId) {
  return typeof opaquePatientId === 'string' && opaquePatientId.startsWith(DEMO_SYNTHETIC_OPAQUE_PREFIX);
}

export function resolvePatientUuid(patientRef) {
  if (!patientRef) return '';
  return patientRef.patient_id || patientRef.id || '';
}

export function formatPatientOptionLabel(patientRef) {
  if (!patientRef) return '';
  if (patientRef.display_label) return patientRef.display_label;
  const opaque = patientRef.opaque_patient_id || '';
  if (isDemoSyntheticOpaque(opaque)) {
    return patientRef.hospital_code ? `${opaque} · ${patientRef.hospital_code}` : opaque;
  }
  if (opaque.length > 36) {
    return `${opaque.slice(0, 12)}…${opaque.slice(-8)}`;
  }
  return opaque || resolvePatientUuid(patientRef);
}

export function partitionPatientOptions(patients) {
  const demo = [];
  const other = [];
  for (const p of patients || []) {
    const opaque = p.opaque_patient_id || '';
    if (p.is_demo_synthetic || isDemoSyntheticOpaque(opaque)) {
      demo.push(p);
    } else {
      other.push(p);
    }
  }
  return { demo, other };
}
