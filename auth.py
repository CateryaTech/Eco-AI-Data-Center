"""
security/auth.py
================
JWT-based authentication & Role-Based Access Control (RBAC)
for Eco AI Data Center.

Supports two auth providers:
  1. Built-in JWT (default — no external deps beyond PyJWT)
  2. Auth0 (optional — requires python-jose + Auth0 tenant)

Roles:
  - viewer      : read-only dashboard access
  - analyst     : viewer + run evaluations / simulations
  - engineer    : analyst + data upload + monitor controls
  - admin       : full access including user management
  - compliance  : dedicated compliance scanning access
  - api_client  : programmatic API access (no UI)

Eco AI Data Center — CateryaTech
Author  : Ary HH
Email   : cateryatech@proton.me
GitHub  : https://github.com/cateryatech/Eco-AI-Data-Center
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger("eco_ai.security.auth")

# ---------------------------------------------------------------------------
# Optional deps — graceful degradation
# ---------------------------------------------------------------------------

try:
    import jwt as pyjwt            # PyJWT
    PYJWT_AVAILABLE = True
except ImportError:
    pyjwt = None
    PYJWT_AVAILABLE = False
    logger.warning("[Auth] PyJWT not installed. JWT tokens will use fallback HMAC-signed tokens.")

try:
    from jose import jwt as jose_jwt, JWTError  # type: ignore
    JOSE_AVAILABLE = True
except ImportError:
    jose_jwt = None
    JWTError = Exception
    JOSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

class Role(str, Enum):
    VIEWER     = "viewer"
    ANALYST    = "analyst"
    ENGINEER   = "engineer"
    ADMIN      = "admin"
    COMPLIANCE = "compliance"
    API_CLIENT = "api_client"


# Permission registry: each permission maps to minimum required role(s)
PERMISSIONS: Dict[str, Set[Role]] = {
    # Dashboard
    "dashboard:read":           {Role.VIEWER, Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE, Role.API_CLIENT},
    "dashboard:refresh_cos":    {Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.API_CLIENT},

    # Data
    "data:read":                {Role.VIEWER, Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE, Role.API_CLIENT},
    "data:upload":              {Role.ENGINEER, Role.ADMIN},
    "data:delete":              {Role.ADMIN},
    "data:export":              {Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.API_CLIENT},

    # Evaluations
    "evaluation:run":           {Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.API_CLIENT},
    "evaluation:read":          {Role.VIEWER, Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE, Role.API_CLIENT},

    # Monitor
    "monitor:read":             {Role.VIEWER, Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE},
    "monitor:control":          {Role.ENGINEER, Role.ADMIN},
    "monitor:alerts":           {Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE},

    # Quantum
    "quantum:simulate":         {Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.API_CLIENT},
    "quantum:read":             {Role.VIEWER, Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE},

    # Compliance
    "compliance:scan":          {Role.COMPLIANCE, Role.ADMIN},
    "compliance:read":          {Role.VIEWER, Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE},
    "compliance:export":        {Role.COMPLIANCE, Role.ADMIN},

    # Security / Audit
    "audit:read":               {Role.COMPLIANCE, Role.ADMIN},
    "audit:export":             {Role.COMPLIANCE, Role.ADMIN},
    "security:admin":           {Role.ADMIN},

    # API
    "api:read":                 {Role.API_CLIENT, Role.ADMIN},
    "api:write":                {Role.API_CLIENT, Role.ADMIN},

    # Reports
    "reports:create":           {Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE},
    "reports:share":            {Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE},
    "reports:read":             {Role.VIEWER, Role.ANALYST, Role.ENGINEER, Role.ADMIN, Role.COMPLIANCE},
}


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

@dataclass
class User:
    """Authenticated user entity."""
    user_id:    str
    username:   str
    email:      str
    role:       Role
    full_name:  str = ""
    department: str = ""
    active:     bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_login: Optional[str] = None
    mfa_enabled: bool = False

    def has_permission(self, permission: str) -> bool:
        """Check if this user's role has the given permission."""
        allowed_roles = PERMISSIONS.get(permission, set())
        return self.role in allowed_roles

    def to_dict(self) -> dict:
        return {
            "user_id":    self.user_id,
            "username":   self.username,
            "email":      self.email,
            "role":       self.role.value,
            "full_name":  self.full_name,
            "department": self.department,
            "active":     self.active,
        }


# ---------------------------------------------------------------------------
# Demo user store (replace with DB in production)
# ---------------------------------------------------------------------------

_DEMO_PASSWORD_HASH = hashlib.sha256(b"demo_pass_eco_ai_2025").hexdigest()

DEMO_USERS: Dict[str, dict] = {
    "admin": {
        "user_id": "usr-001",
        "username": "admin",
        "email": "admin@cateryatech.com",
        "password_hash": hashlib.sha256(b"Admin@EcoAI2025!").hexdigest(),
        "role": Role.ADMIN,
        "full_name": "System Administrator",
        "department": "IT Security",
    },
    "analyst": {
        "user_id": "usr-002",
        "username": "analyst",
        "email": "analyst@cateryatech.com",
        "password_hash": hashlib.sha256(b"Analyst@EcoAI2025!").hexdigest(),
        "role": Role.ANALYST,
        "full_name": "Data Analyst",
        "department": "Data Science",
    },
    "compliance_officer": {
        "user_id": "usr-003",
        "username": "compliance_officer",
        "email": "compliance@cateryatech.com",
        "password_hash": hashlib.sha256(b"Comply@EcoAI2025!").hexdigest(),
        "role": Role.COMPLIANCE,
        "full_name": "Compliance Officer",
        "department": "Risk & Compliance",
    },
    "engineer": {
        "user_id": "usr-004",
        "username": "engineer",
        "email": "engineer@cateryatech.com",
        "password_hash": hashlib.sha256(b"Engineer@EcoAI2025!").hexdigest(),
        "role": Role.ENGINEER,
        "full_name": "Infrastructure Engineer",
        "department": "Operations",
    },
    "viewer": {
        "user_id": "usr-005",
        "username": "viewer",
        "email": "viewer@cateryatech.com",
        "password_hash": hashlib.sha256(b"Viewer@EcoAI2025!").hexdigest(),
        "role": Role.VIEWER,
        "full_name": "Dashboard Viewer",
        "department": "Management",
    },
    "api_service": {
        "user_id": "usr-006",
        "username": "api_service",
        "email": "api@cateryatech.com",
        "password_hash": hashlib.sha256(b"ApiService@EcoAI2025!").hexdigest(),
        "role": Role.API_CLIENT,
        "full_name": "API Service Account",
        "department": "Integration",
    },
}


# ---------------------------------------------------------------------------
# JWT Token Manager
# ---------------------------------------------------------------------------

class JWTTokenManager:
    """
    Manages JWT token issuance, validation, and refresh.

    Supports:
      - PyJWT (preferred)
      - HMAC-signed fallback (no external deps)
      - Auth0 JWKS validation (when configured)

    Parameters
    ----------
    secret_key      : JWT signing secret (auto-generated if not provided)
    algorithm       : JWT algorithm (default HS256)
    access_ttl      : access token TTL in minutes (default 60)
    refresh_ttl     : refresh token TTL in hours (default 24)
    auth0_domain    : Auth0 domain (optional)
    auth0_audience  : Auth0 API audience (optional)
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_ttl: int = 60,
        refresh_ttl: int = 24,
        auth0_domain: Optional[str] = None,
        auth0_audience: Optional[str] = None,
    ):
        self.secret_key = secret_key or os.getenv(
            "JWT_SECRET_KEY",
            secrets.token_urlsafe(64),
        )
        self.algorithm = algorithm
        self.access_ttl = access_ttl
        self.refresh_ttl = refresh_ttl
        self.auth0_domain = auth0_domain or os.getenv("AUTH0_DOMAIN")
        self.auth0_audience = auth0_audience or os.getenv("AUTH0_AUDIENCE")
        self._revoked_tokens: Set[str] = set()

        logger.info(
            "[Auth] JWT Manager initialised | algo=%s | access_ttl=%dm | auth0=%s",
            algorithm, access_ttl,
            "enabled" if self.auth0_domain else "disabled",
        )

    # ------------------------------------------------------------------
    # Token issuance
    # ------------------------------------------------------------------

    def create_access_token(self, user: User) -> str:
        """Issue a signed access token for the given user."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub":        user.user_id,
            "username":   user.username,
            "email":      user.email,
            "role":       user.role.value,
            "department": user.department,
            "type":       "access",
            "iat":        int(now.timestamp()),
            "exp":        int((now + timedelta(minutes=self.access_ttl)).timestamp()),
            "jti":        secrets.token_urlsafe(16),
            "iss":        "eco-ai-datacenter",
            "aud":        "eco-ai-api",
        }
        return self._sign(payload)

    def create_refresh_token(self, user: User) -> str:
        """Issue a refresh token (longer TTL, fewer claims)."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub":  user.user_id,
            "type": "refresh",
            "iat":  int(now.timestamp()),
            "exp":  int((now + timedelta(hours=self.refresh_ttl)).timestamp()),
            "jti":  secrets.token_urlsafe(16),
            "iss":  "eco-ai-datacenter",
        }
        return self._sign(payload)

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify and decode a JWT token.
        Returns payload dict if valid, None if invalid/expired.
        """
        if token in self._revoked_tokens:
            logger.warning("[Auth] Token is revoked.")
            return None

        if self.auth0_domain and JOSE_AVAILABLE:
            return self._verify_auth0(token)

        return self._verify_local(token)

    def revoke_token(self, token: str) -> None:
        """Add token to revocation list (in-memory; use Redis in production)."""
        self._revoked_tokens.add(token)
        logger.info("[Auth] Token revoked.")

    def get_user_from_token(self, token: str) -> Optional[User]:
        """Decode token and return User object if valid."""
        payload = self.verify_token(token)
        if not payload:
            return None

        user_id = payload.get("sub")
        username = payload.get("username")
        if not username:
            # Try to find user by sub
            for udata in DEMO_USERS.values():
                if udata["user_id"] == user_id:
                    username = udata["username"]
                    break

        udata = DEMO_USERS.get(username or "")
        if not udata:
            return None

        return User(
            user_id=udata["user_id"],
            username=udata["username"],
            email=udata["email"],
            role=udata["role"],
            full_name=udata.get("full_name", ""),
            department=udata.get("department", ""),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sign(self, payload: dict) -> str:
        if PYJWT_AVAILABLE:
            return pyjwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return self._fallback_sign(payload)

    def _verify_local(self, token: str) -> Optional[dict]:
        if PYJWT_AVAILABLE:
            try:
                return pyjwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                    audience="eco-ai-api",
                    options={"verify_exp": True},
                )
            except Exception as exc:
                logger.debug("[Auth] JWT decode failed: %s", exc)
                return None
        return self._fallback_verify(token)

    def _verify_auth0(self, token: str) -> Optional[dict]:
        """Validate token against Auth0 JWKS endpoint."""
        try:
            from jose import jwk, JWTError as JE  # type: ignore
            import urllib.request, json as _json
            jwks_url = f"https://{self.auth0_domain}/.well-known/jwks.json"
            with urllib.request.urlopen(jwks_url, timeout=5) as resp:
                jwks = _json.loads(resp.read())
            header = jose_jwt.get_unverified_header(token)
            rsa_key = {}
            for key in jwks.get("keys", []):
                if key.get("kid") == header.get("kid"):
                    rsa_key = {
                        "kty": key["kty"], "kid": key["kid"],
                        "use": key["use"], "n": key["n"], "e": key["e"],
                    }
                    break
            if not rsa_key:
                return None
            payload = jose_jwt.decode(
                token, rsa_key,
                algorithms=["RS256"],
                audience=self.auth0_audience,
                issuer=f"https://{self.auth0_domain}/",
            )
            return payload
        except Exception as exc:
            logger.debug("[Auth] Auth0 verify failed: %s", exc)
            return None

    def _fallback_sign(self, payload: dict) -> str:
        """HMAC-SHA256 fallback when PyJWT not available."""
        import base64
        import json
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
        body_bytes = json.dumps(payload, separators=(",", ":")).encode()
        body = base64.urlsafe_b64encode(body_bytes).rstrip(b"=").decode()
        msg = f"{header}.{body}".encode()
        sig_bytes = hmac.new(self.secret_key.encode(), msg, hashlib.sha256).digest()
        sig = base64.urlsafe_b64encode(sig_bytes).rstrip(b"=").decode()
        return f"{header}.{body}.{sig}"

    def _fallback_verify(self, token: str) -> Optional[dict]:
        """Verify HMAC-signed fallback token."""
        import base64
        import json
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header_body = f"{parts[0]}.{parts[1]}".encode()
            sig_bytes = hmac.new(self.secret_key.encode(), header_body, hashlib.sha256).digest()
            expected_sig = base64.urlsafe_b64encode(sig_bytes).rstrip(b"=").decode()
            if not hmac.compare_digest(parts[2], expected_sig):
                return None
            padding = 4 - len(parts[1]) % 4
            body = base64.urlsafe_b64decode(parts[1] + "=" * padding)
            payload = json.loads(body)
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Auth service
# ---------------------------------------------------------------------------

class AuthService:
    """
    High-level authentication service.
    Handles login, logout, and permission checks.
    """

    def __init__(self, token_manager: Optional[JWTTokenManager] = None):
        self.token_mgr = token_manager or JWTTokenManager()

    def login(self, username: str, password: str) -> Optional[dict]:
        """
        Authenticate user with username + password.
        Returns {access_token, refresh_token, user} or None on failure.
        """
        udata = DEMO_USERS.get(username)
        if not udata:
            logger.warning("[Auth] Login failed — unknown user: %s", username)
            return None

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if not hmac.compare_digest(udata["password_hash"], password_hash):
            logger.warning("[Auth] Login failed — wrong password for: %s", username)
            return None

        user = User(
            user_id=udata["user_id"],
            username=udata["username"],
            email=udata["email"],
            role=udata["role"],
            full_name=udata.get("full_name", ""),
            department=udata.get("department", ""),
        )
        udata["last_login"] = datetime.now(timezone.utc).isoformat()

        access_token = self.token_mgr.create_access_token(user)
        refresh_token = self.token_mgr.create_refresh_token(user)

        logger.info("[Auth] Login successful: %s (%s)", username, user.role.value)
        return {
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "token_type":    "bearer",
            "expires_in":    self.token_mgr.access_ttl * 60,
            "user":          user.to_dict(),
        }

    def logout(self, access_token: str) -> None:
        self.token_mgr.revoke_token(access_token)
        logger.info("[Auth] Token revoked (logout).")

    def get_current_user(self, token: str) -> Optional[User]:
        return self.token_mgr.get_user_from_token(token)

    def check_permission(self, user: User, permission: str) -> bool:
        has_it = user.has_permission(permission)
        if not has_it:
            logger.warning(
                "[Auth] PERMISSION DENIED: user=%s role=%s permission=%s",
                user.username, user.role.value, permission,
            )
        return has_it

    def require_permission(self, user: Optional[User], permission: str) -> User:
        """Raise PermissionError if user lacks the given permission."""
        if user is None:
            raise PermissionError("Authentication required.")
        if not self.check_permission(user, permission):
            raise PermissionError(
                f"User '{user.username}' (role: {user.role.value}) "
                f"does not have permission: '{permission}'"
            )
        return user


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
