"""Regression: require_permission must not raise UnboundLocalError on UserRole."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.app.core.security import SystemPermission, UserRole
from backend.app.api.dependencies import require_permission


@pytest.mark.anyio
async def test_require_permission_grants_clinician_clinical_read():
    checker = require_permission(SystemPermission.CLINICAL_READ)
    clinician_user = SimpleNamespace(
        is_super_admin=False,
        roles=[SimpleNamespace(name=UserRole.CLINICIAN.value)],
    )
    result = await checker(current_user=clinician_user)
    assert result is clinician_user


@pytest.mark.anyio
async def test_require_permission_super_admin_bypass():
    checker = require_permission(SystemPermission.GLOBAL_ADMIN)
    admin_user = SimpleNamespace(
        is_super_admin=True,
        roles=[SimpleNamespace(name=UserRole.SUPER_ADMIN.value)],
    )
    result = await checker(current_user=admin_user)
    assert result is admin_user


@pytest.mark.anyio
async def test_require_permission_federated_monitor_via_super_admin_role():
    checker = require_permission(SystemPermission.FEDERATED_MONITOR)
    admin_user = SimpleNamespace(
        is_super_admin=False,
        roles=[SimpleNamespace(name=UserRole.SUPER_ADMIN.value)],
    )
    result = await checker(current_user=admin_user)
    assert result is admin_user


@pytest.mark.anyio
async def test_require_permission_federated_monitor_denied_for_clinician():
    checker = require_permission(SystemPermission.FEDERATED_MONITOR)
    clinician_user = SimpleNamespace(
        is_super_admin=False,
        roles=[SimpleNamespace(name=UserRole.CLINICIAN.value)],
    )
    with pytest.raises(Exception, match="federated:monitor"):
        await checker(current_user=clinician_user)
