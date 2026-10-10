"""Authentication and Session Management Service."""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.errors import AuthenticationAppError, SecurityError
from backend.app.core.logging import get_logger
from backend.app.core.request_context import get_request_id
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.models.audit import SecurityEvent
from backend.app.models.identity import User, UserSession

logger = get_logger("medshield.auth")


def hash_token_secret(token: str) -> str:
    """Hash raw token secret using SHA-256 for secure DB storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthService:
    """Service handling user authentication, session lifecycle, and security audit events."""

    @staticmethod
    async def record_security_event(
        db: AsyncSession,
        event_name: str,
        severity: str = "INFO",
        actor_id: Optional[uuid.UUID] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist security event to DB security_events table."""
        req_id = get_request_id() or "system"
        safe_details_str = str(details) if details else None
        event = SecurityEvent(
            event_name=event_name,
            severity=severity,
            request_id=req_id,
            actor_id=actor_id,
            details_json=safe_details_str,
        )
        db.add(event)
        await db.commit()
        logger.info(f"Security event recorded: {event_name} [severity={severity}]", extra={"actor_id": str(actor_id)})

    @staticmethod
    async def authenticate_user(
        db: AsyncSession, email: str, password: str, ip_address: Optional[str] = None
    ) -> User:
        """Verify user credentials with timing-safe error responses (generic error on failure)."""
        stmt = select(User).where(User.email == email, User.deleted_at.is_(None)).options(selectinload(User.roles))
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or not verify_password(password, user.hashed_password):
            await AuthService.record_security_event(
                db, event_name="LOGIN_FAILED", severity="WARNING", details={"email_domain": email.split("@")[-1] if "@" in email else "invalid"}
            )
            raise AuthenticationAppError("Invalid email or password")

        if not user.is_active:
            await AuthService.record_security_event(
                db, event_name="ACCOUNT_DISABLED", severity="WARNING", actor_id=user.id
            )
            raise AuthenticationAppError("User account is disabled")

        return user

    @staticmethod
    async def create_user_session(
        db: AsyncSession,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate JWT access and refresh tokens and create persistent UserSession record."""
        access_token, access_jti = create_access_token(user_id=str(user.id))
        refresh_token, refresh_jti, refresh_expires_at = create_refresh_token(user_id=str(user.id))

        token_hash = hash_token_secret(refresh_token)

        session = UserSession(
            user_id=user.id,
            token_jti=refresh_jti,
            refresh_token_hash=token_hash,
            is_revoked=False,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=refresh_expires_at,
        )
        db.add(session)
        await db.commit()

        await AuthService.record_security_event(
            db, event_name="LOGIN_SUCCESS", severity="INFO", actor_id=user.id
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 900,  # 15 minutes in seconds
        }

    @staticmethod
    async def refresh_user_tokens(
        db: AsyncSession,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate refresh token, enforce rotation, detect reuse, and issue new token pair."""
        payload = decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise AuthenticationAppError("Invalid or expired refresh token")

        jti = payload.get("jti")
        user_id_str = payload.get("sub")
        if not jti or not user_id_str:
            raise AuthenticationAppError("Malformed refresh token")

        stmt = select(UserSession).where(UserSession.token_jti == jti)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            raise AuthenticationAppError("Refresh session not found")

        # REUSE DETECTION: If session was already revoked, invalidate ALL sessions for this user!
        if session.is_revoked:
            await AuthService.record_security_event(
                db,
                event_name="REFRESH_REUSE_DETECTED",
                severity="CRITICAL",
                actor_id=session.user_id,
                details={"jti": jti},
            )
            # Revoke all sessions for this user to mitigate session compromise
            stmt_all = select(UserSession).where(UserSession.user_id == session.user_id)
            res_all = await db.execute(stmt_all)
            for s in res_all.scalars().all():
                s.is_revoked = True
            await db.commit()
            raise SecurityError("Refresh token reuse detected. Session invalidated.")

        # Check expiration
        now = datetime.now(timezone.utc)
        if session.expires_at < now:
            session.is_revoked = True
            await db.commit()
            raise AuthenticationAppError("Refresh session expired")

        # Fetch active user
        stmt_user = select(User).where(User.id == session.user_id, User.deleted_at.is_(None))
        res_user = await db.execute(stmt_user)
        user = res_user.scalar_one_or_none()
        if not user or not user.is_active:
            session.is_revoked = True
            await db.commit()
            raise AuthenticationAppError("User account disabled or deleted")

        # Rotate token: revoke old session and issue new pair
        session.is_revoked = True
        
        access_token, _ = create_access_token(user_id=str(user.id))
        new_refresh_token, new_refresh_jti, new_refresh_expires_at = create_refresh_token(user_id=str(user.id))
        new_token_hash = hash_token_secret(new_refresh_token)

        new_session = UserSession(
            user_id=user.id,
            token_jti=new_refresh_jti,
            refresh_token_hash=new_token_hash,
            is_revoked=False,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=new_refresh_expires_at,
        )
        db.add(new_session)
        await db.commit()

        await AuthService.record_security_event(
            db, event_name="REFRESH_SUCCESS", severity="INFO", actor_id=user.id
        )

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": 900,
        }

    @staticmethod
    async def logout_user(db: AsyncSession, refresh_token: str) -> None:
        """Revoke refresh session on logout."""
        payload = decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return

        jti = payload.get("jti")
        if not jti:
            return

        stmt = select(UserSession).where(UserSession.token_jti == jti)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if session and not session.is_revoked:
            session.is_revoked = True
            await db.commit()
            await AuthService.record_security_event(
                db, event_name="LOGOUT", severity="INFO", actor_id=session.user_id
            )
