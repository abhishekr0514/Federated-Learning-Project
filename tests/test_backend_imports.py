"""Test suite verifying backend module importability."""

def test_backend_imports():
    """Verify core backend packages and modules import cleanly without errors."""
    import backend.app.main as main_module
    import backend.app.core.config as config_module
    import backend.app.core.request_context as request_context_module
    import backend.app.core.logging as logging_module
    import backend.app.core.errors as errors_module
    import backend.app.core.security as security_module
    import backend.app.db.session as session_module
    import backend.app.db.base as base_module
    import backend.app.repositories.base as repo_base
    import backend.app.services.base as service_base
    import backend.app.api.v1.router as v1_router

    assert main_module.app is not None
    assert config_module.settings is not None
    assert request_context_module.RequestIDMiddleware is not None
    assert logging_module.logger is not None
    assert errors_module.BaseAppException is not None
    assert security_module.UserRole is not None
    assert session_module.async_engine is not None
    assert base_module.Base is not None
    assert repo_base.BaseRepository is not None
    assert service_base.BaseService is not None
    assert v1_router.api_v1_router is not None
