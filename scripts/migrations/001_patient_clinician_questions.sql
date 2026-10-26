-- Patient question workflow (run once on PostgreSQL when upgrading)
CREATE TABLE IF NOT EXISTS patient_clinician_questions (
    id UUID PRIMARY KEY,
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    hospital_id UUID NOT NULL REFERENCES hospitals(id) ON DELETE RESTRICT,
    author_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assigned_clinician_id UUID REFERENCES clinician_profiles(id) ON DELETE SET NULL,
    question_text TEXT NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'submitted',
    response_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    responded_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_patient_clinician_questions_patient_id
    ON patient_clinician_questions(patient_id);
CREATE INDEX IF NOT EXISTS ix_patient_clinician_questions_hospital_id
    ON patient_clinician_questions(hospital_id);

-- When this script is run as postgres, grant the app role (see POSTGRES_USER in .env).
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.patient_clinician_questions TO medshield_admin;
