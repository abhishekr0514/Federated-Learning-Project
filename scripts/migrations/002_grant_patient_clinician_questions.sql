-- Grant application role access when the table was created by a superuser (manual 001 run).
-- Safe to re-run. Replace medshield_admin if POSTGRES_USER differs in your .env.
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.patient_clinician_questions TO medshield_admin;
