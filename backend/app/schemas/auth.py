"""Pydantic Schemas for Authentication Requests and User Responses."""

import uuid
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Credentials payload for user authentication."""

    email: EmailStr = Field(..., description="User account email address")
    password: str = Field(..., min_length=1, description="Account password")


class RefreshTokenRequest(BaseModel):
    """Refresh token payload for token rotation."""

    refresh_token: str = Field(..., description="Active JWT refresh token")


class TokenResponse(BaseModel):
    """JWT Access and Refresh token pair response."""

    access_token: str = Field(..., description="Short-lived JWT access token")
    refresh_token: str = Field(..., description="Rotated JWT refresh token")
    token_type: str = Field(default="bearer", description="Token authentication type")
    expires_in: int = Field(default=900, description="Access token expiration window in seconds")


class UserMeResponse(BaseModel):
    """Safe authenticated user account profile."""

    id: uuid.UUID = Field(..., description="User unique identifier")
    email: EmailStr = Field(..., description="User account email address")
    full_name: Optional[str] = Field(None, description="Full name of system actor")
    is_active: bool = Field(..., description="Account active status flag")
    is_super_admin: bool = Field(..., description="Super Admin status flag")
    roles: List[str] = Field(default_factory=list, description="Assigned system roles")
