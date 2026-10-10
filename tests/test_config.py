"""Test suite for application configuration loading."""

from backend.app.core.config import Settings, settings


def test_settings_initialization():
    """Verify application settings load with default fallback configuration."""
    assert settings.PROJECT_NAME == "MedShield-FL"
    assert settings.VERSION == "0.1.0"
    assert isinstance(settings.CORS_ORIGINS, list)
    assert len(settings.CORS_ORIGINS) > 0


def test_cors_origins_parsing():
    """Verify CORS origins string is correctly parsed into a list."""
    custom_settings = Settings(CORS_ORIGINS="http://localhost:3000, http://example.com")
    assert custom_settings.CORS_ORIGINS == ["http://localhost:3000", "http://example.com"]
