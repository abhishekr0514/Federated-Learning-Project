-- Dev-only repair: align dr_metro demo clinician + assignments with DEMO-METRO
-- (Run after seed_postgres_dev + seed_demo_synthetic if Diagnostic Studio shows no demo patients.)
-- Does NOT delete patients or diagnosis_predictions.

UPDATE clinician_profiles cp
SET hospital_id = h_demo.id,
    department = '[SYNTHETIC] Cardiology & Electrophysiology',
    license_number_hash = 'DEMO-LIC-METRO-001'
FROM users u, hospitals h_demo
WHERE cp.user_id = u.id
  AND u.email = 'dr_metro@metro.org'
  AND h_demo.code = 'DEMO-METRO'
  AND cp.hospital_id IS DISTINCT FROM h_demo.id;

UPDATE patient_clinician_assignments pca
SET hospital_id = h_demo.id
FROM clinician_profiles cp, users u, hospitals h_demo, patients p
WHERE pca.clinician_id = cp.id
  AND cp.user_id = u.id
  AND u.email = 'dr_metro@metro.org'
  AND h_demo.code = 'DEMO-METRO'
  AND p.id = pca.patient_id
  AND p.opaque_patient_id LIKE 'DEMO-SYNTH-%'
  AND pca.hospital_id IS DISTINCT FROM h_demo.id;

INSERT INTO hospital_memberships (id, user_id, hospital_id, role_in_hospital, created_at, updated_at)
SELECT gen_random_uuid(), u.id, h_demo.id, 'CLINICIAN', NOW(), NOW()
FROM users u
CROSS JOIN hospitals h_demo
WHERE u.email = 'dr_metro@metro.org'
  AND h_demo.code = 'DEMO-METRO'
  AND NOT EXISTS (
    SELECT 1 FROM hospital_memberships hm
    WHERE hm.user_id = u.id AND hm.hospital_id = h_demo.id
  );
