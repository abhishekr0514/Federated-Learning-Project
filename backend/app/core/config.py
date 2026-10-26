"""Core Application Configuration Settings."""

from pathlib import Path
from typing import List, Union

from typing_extensions import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Application Settings
    PROJECT_NAME: str = "MedShield-FL"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    SECRET_KEY: str = Field(
        default="DEV_ONLY_INSECURE_SECRET_KEY_CHANGE_IN_PRODUCTION"
    )

    # JWT Security
    JWT_SECRET: str = Field(
        default="DEV_ONLY_INSECURE_JWT_SECRET_CHANGE_IN_PRODUCTION"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database Settings (PostgreSQL)
    POSTGRES_USER: str = "medshield_admin"
    POSTGRES_PASSWORD: str = "medshield_dev_password"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "medshield_fl_db"
    # Assembled from POSTGRES_* after validation (see build_database_url). Not loaded from env.
    DATABASE_URL: str = Field(default="", validate_default=False)

    @model_validator(mode="after")
    def build_database_url(self) -> Self:
        """Build async SQLAlchemy URL from POSTGRES_* (handles special characters safely)."""
        url = URL.create(
            drivername="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            database=self.POSTGRES_DB,
        )
        object.__setattr__(self, "DATABASE_URL", url.render_as_string(hide_password=False))
        return self

    # CORS Settings
    CORS_ORIGINS: Union[str, List[str]] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"]
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    # ML artifact paths (optional; used for capability diagnostics only)
    MODEL_DIR: str = Field(default="models")
    INFERENCE_ENABLED: bool = Field(default=False)
    ECG_REPORT_MODEL_DIR: str = Field(default="models/ecg_report_fusion")
    ECG_REPORT_INFERENCE_ENABLED: bool = Field(default=True)

    # Three-modality GAT fusion settings
    FUSION_MODEL_DIR: str = Field(default="models/gat_ecg_report")
    FUSION_INFERENCE_ENABLED: bool = Field(default=False)

    # Observational causal analysis (DoWhy, GAT-grounded PTB-XL cohort)
    CAUSAL_ANALYSIS_ENABLED: bool = Field(default=True)
    CAUSAL_GAT_EMBEDDING_CACHE_PATH: str = Field(
        default="",
        description="Optional path to embedding_cache_train.pt; auto-discovered when empty.",
    )
    CAUSAL_PTBNXL_ROOT: str = Field(
        default="",
        description="Optional PTB-XL root for metadata alignment; defaults to data/raw/ptb-xl.",
    )
    CAUSAL_COHORT_PSEUDONYM_SALT: str = Field(default="medshield-fl-research-cohort-v1")
    CAUSAL_ANALYSIS_MAX_ROWS: int = Field(default=8000, ge=500, le=70000)
    CAUSAL_ANALYSIS_MIN_SAMPLE_SIZE: int = Field(default=500, ge=100)
    CAUSAL_EFFECT_ESTIMATION_ALLOWED: bool = Field(
        default=False,
        description=(
            "When false (default), POST /causal/analyze is blocked regardless of DoWhy availability. "
            "Does not imply a scientifically validated causal design when true."
        ),
    )

    # Federated Learning Settings
    FLOWER_SERVER_ADDRESS: str = "127.0.0.1:8080"
    FLOWER_NUM_ROUNDS: int = Field(default=2, ge=1, le=100)
    FLOWER_ORCHESTRATION_ENABLED: bool = Field(default=False)
    FLOWER_TRAINING_ENABLED: bool = Field(default=False)
    FLOWER_DEPLOYMENT_MODE: str = Field(
        default="superlink",
        description="superlink = real SuperLink+SuperNode processes; legacy_inprocess = deprecated thread smoke only",
    )
    FLOWER_APP_DIR: str = "fl/flower_deploy"
    FLOWER_TLS_CERT_DIR: str = "deployment/flower/dev"
    FLOWER_SUPERLINK_CONTROL_ADDRESS: str = "127.0.0.1:9093"
    FLOWER_FLEET_API_ADDRESS: str = "127.0.0.1:9092"
    FLOWER_REQUIRE_TLS: bool = Field(default=True)
    FLOWER_MIN_AVAILABLE_CLIENTS: int = Field(default=2, ge=1, le=32)

    # Differential Privacy Settings
    DP_EPSILON: float = 1.0
    DP_DELTA: float = 1e-5
    DP_CLIP_NORM: float = 1.0


settings = Settings()
