-- Manual apply if Alembic is unavailable (PostgreSQL).
-- Run as DB owner or postgres, then GRANT to medshield_admin if needed.

CREATE TABLE IF NOT EXISTS privacy_masking_entity_counts (
    id UUID PRIMARY KEY,
    masking_result_id UUID NOT NULL REFERENCES privacy_masking_results(id) ON DELETE CASCADE,
    entity_category VARCHAR(50) NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT uq_masking_entity_category UNIQUE (masking_result_id, entity_category)
);

CREATE INDEX IF NOT EXISTS ix_privacy_masking_entity_counts_masking_result_id
    ON privacy_masking_entity_counts (masking_result_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON privacy_masking_entity_counts TO medshield_admin;
