/**
 * Clinical and ECG Sample Data for MedShield-FL Diagnostic Studio
 * Provides synthetic, mathematically generated 12-lead ECG waveforms (12 x 1000 samples)
 * and representative clinical report narratives for testing and demonstrations.
 * All patient names and identifiers are strictly fictional.
 */

/**
 * Generate a synthetic 12-lead ECG signal of shape (12, 1000)
 * Simulates 10 seconds of 100Hz ECG with P, QRS, and T complexes.
 */
function generateSyntheticEcg(isPathological = false) {
  const numLeads = 12;
  const numSamples = 1000;
  const leads = [];

  // Heart rate: ~72 bpm -> ~1.2 beats/sec -> beat every ~83 samples at 100Hz
  const beatInterval = 83;

  for (let l = 0; l < numLeads; l++) {
    const lead = new Float32Array(numSamples);
    // Baseline wander
    const wanderFreq = 0.05 + l * 0.01;
    for (let t = 0; t < numSamples; t++) {
      lead[t] = 0.05 * Math.sin((2 * Math.PI * wanderFreq * t) / 100);
    }

    // Lead amplitude scaling
    const scale = l < 6 ? (l === 1 ? 1.0 : l === 0 ? 0.7 : 0.5) : (l >= 6 && l <= 9 ? 1.2 : 0.9);

    // Render beats
    for (let beat = 20; beat < numSamples - 60; beat += beatInterval) {
      // P wave (small positive deflection)
      const pCenter = beat - 16;
      for (let dt = -6; dt <= 6; dt++) {
        const idx = pCenter + dt;
        if (idx >= 0 && idx < numSamples) {
          lead[idx] += 0.12 * Math.exp(-Math.pow(dt / 3, 2)) * scale;
        }
      }

      // Q wave (small negative dip)
      const qIdx = beat - 4;
      if (qIdx >= 0 && qIdx < numSamples) {
        lead[qIdx] -= 0.15 * scale;
      }

      // R wave (sharp tall peak)
      const rIdx = beat;
      if (rIdx >= 0 && rIdx < numSamples) {
        lead[rIdx] += 1.25 * scale;
      }
      if (rIdx - 1 >= 0) lead[rIdx - 1] += 0.6 * scale;
      if (rIdx + 1 < numSamples) lead[rIdx + 1] += 0.5 * scale;

      // S wave (negative dip)
      const sIdx = beat + 4;
      if (sIdx >= 0 && sIdx < numSamples) {
        lead[sIdx] -= 0.3 * scale;
      }

      // T wave
      const tCenter = beat + 24;
      // In pathological STEMI mode for chest leads V2-V4 (indices 7, 8, 9), elevate ST segment
      const stElevation = isPathological && (l >= 6 && l <= 9) ? 0.35 : 0.0;
      for (let dt = -10; dt <= 14; dt++) {
        const idx = tCenter + dt;
        if (idx >= 0 && idx < numSamples) {
          const tWave = 0.25 * Math.exp(-Math.pow(dt / 5, 2)) * scale;
          lead[idx] += tWave + stElevation;
        }
      }
      // Add ST segment elevation
      if (isPathological && l >= 6 && l <= 9) {
        for (let dt = 5; dt <= 16; dt++) {
          const idx = beat + dt;
          if (idx >= 0 && idx < numSamples) {
            lead[idx] += 0.3 * scale;
          }
        }
      }
    }

    // Small measurement noise
    for (let t = 0; t < numSamples; t++) {
      lead[t] += (Math.random() - 0.5) * 0.02;
    }

    leads.push(Array.from(lead).map((v) => Number(v.toFixed(4))));
  }

  return leads;
}

export const SAMPLE_ECG_NORMAL = {
  id: 'ecg_norm_ptbxl_001',
  name: 'PTB-XL Benchmark: Normal Sinus Rhythm (Class 0)',
  leads: generateSyntheticEcg(false),
  leadNames: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
  description: '12-lead ECG (1000 samples at 100Hz). Regular rhythm, PR interval 160ms, QRS 84ms, normal axis, no ST displacement.',
};

export const SAMPLE_ECG_PATHOLOGICAL = {
  id: 'ecg_mi_ptbxl_002',
  name: 'PTB-XL Benchmark: Anteroseptal MI / STTC (Class 1)',
  leads: generateSyntheticEcg(true),
  leadNames: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
  description: '12-lead ECG (1000 samples at 100Hz). Acute ST-segment elevation in leads V2-V4 (>2mm) consistent with acute myocardial ischemia/infarction proxy.',
};

export const SAMPLE_ECG_INVALID = {
  id: 'ecg_invalid_shape_003',
  name: 'Malformed ECG: 8 Leads Only (Quality Check Failure Test)',
  leads: generateSyntheticEcg(false).slice(0, 8),
  leadNames: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2'],
  description: 'ECG array with only 8 leads instead of required 12 leads. Demonstrates fail-closed schema validation.',
};

export const SAMPLE_REPORTS = [
  {
    id: 'report_norm',
    title: 'Normal Clinical Cardiology Follow-up',
    text: 'Normal sinus rhythm with a heart rate of 72 bpm. Normal PR interval and QRS duration. Axis is normal. No significant ST-segment or T-wave abnormalities identified. Comparison with prior electrocardiogram shows no acute changes.',
  },
  {
    id: 'report_mi',
    title: 'Acute Coronary Syndrome Clinical Observation',
    text: 'Sinus tachycardia at 96 bpm with 2.5 mm ST-segment elevation in anteroseptal leads V2, V3, and V4. Reciprocal ST depression in inferior leads. Findings are concerning for acute anterior myocardial infarction. Immediate invasive cardiac evaluation requested.',
  },
  {
    id: 'report_pii_stress',
    title: 'Report with Direct Identifiers (Privacy Masking Test)',
    text: 'Patient Name: Robert Henderson, MRN: 48921-X, DOB: 04/18/1962. Address: 442 Pine Crest Ave, Phone: (555) 234-5678, Email: rhenderson@metrohealth.org. Patient presents with acute substernal chest discomfort radiating to left shoulder. History of hypertension and hyperlipidemia. Troponin I elevated at 1.8 ng/mL. Electrocardiogram demonstrates ST-segment elevation in leads V2-V4.',
  },
];
