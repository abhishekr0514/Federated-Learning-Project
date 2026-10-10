"""Authentication and Session Management API v1 Endpoints."""

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies import get_current_user, get_db
from backend.app.models.identity import User
from backend.app.schemas.auth import LoginRequest, RefreshTokenRequest, TokenResponse, UserMeResponse
from backend.app.schemas.common import GenericSuccessEnvelope
from backend.app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=GenericSuccessEnvelope[TokenResponse],
    status_code=status.HTTP_200_OK,
    summary="Authenticate User and Issue Tokens",
)
async def login(
    credentials: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> GenericSuccessEnvelope[TokenResponse]:
    """Authenticate user credentials and issue signed JWT access and refresh tokens."""
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    user = await AuthService.authenticate_user(
        db, email=credentials.email, password=credentials.password, ip_address=client_ip
    )
    tokens = await AuthService.create_user_session(
        db, user=user, ip_address=client_ip, user_agent=user_agent
    )

    return GenericSuccessEnvelope(data=TokenResponse(**tokens))


@router.post(
    "/refresh",
    response_model=GenericSuccessEnvelope[TokenResponse],
    status_code=status.HTTP_200_OK,
    summary="Rotate Refresh Token and Issue New Access Token",
)
async def refresh_token(
    body: RefreshTokenRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> GenericSuccessEnvelope[TokenResponse]:
    """Rotate JWT refresh token, check session validity and token reuse, and issue new token pair."""
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    tokens = await AuthService.refresh_user_tokens(
        db, refresh_token=body.refresh_token, ip_address=client_ip, user_agent=user_agent
    )

    return GenericSuccessEnvelope(data=TokenResponse(**tokens))


@router.post(
    "/logout",
    response_model=GenericSuccessEnvelope[dict],
    status_code=status.HTTP_200_OK,
    summary="Revoke User Refresh Session",
)
async def logout(
    body: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
) -> GenericSuccessEnvelope[dict]:
    """Invalidate active refresh session on user logout."""
    await AuthService.logout_user(db, refresh_token=body.refresh_token)
    return GenericSuccessEnvelope(data={"message": "Successfully logged out"})


@router.get(
    "/me",
    response_model=GenericSuccessEnvelope[UserMeResponse],
    status_code=status.HTTP_200_OK,
    summary="Retrieve Current Authenticated User Account Details",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> GenericSuccessEnvelope[UserMeResponse]:
    """Return safe account information for the currently authenticated user."""
    role_names = [r.name for r in current_user.roles]
    user_data = UserMeResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_super_admin=current_user.is_super_admin,
        roles=role_names,
    )
    return GenericSuccessEnvelope(data=user_data)
