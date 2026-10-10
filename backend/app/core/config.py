"""Core Application Configuration Settings."""

import os
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://medshield_admin:medshield_dev_password@localhost:5432/medshield_fl_db"
    )

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

    # Federated Learning Settings
    FLOWER_SERVER_ADDRESS: str = "127.0.0.1:8080"
    FLOWER_NUM_ROUNDS: int = 10

    # Differential Privacy Settings
    DP_EPSILON: float = 1.0
    DP_DELTA: float = 1e-5
    DP_CLIP_NORM: float = 1.0


settings = Settings()
