"""Verify Supabase Auth access tokens.

Supabase issues two families of access token:

  * **HS256**, signed with the project's shared `SUPABASE_JWT_SECRET`. This is
    the long-standing default and needs no network call.
  * **Asymmetric** (RS256 / ES256), verified against the project's public JWKS.
    Newer projects and projects that have rotated to asymmetric keys use this.

Both are supported. The shared secret is preferred when configured because it
is offline; otherwise the JWKS is fetched once and cached, with a short retry
window so a key rotation heals without a redeploy.

We verify signature, expiry and audience, and we trust exactly three claims:
`sub` (the user id), `email`, and `role`. Everything else about the user —
which organizations they belong to, what they may do there — is read from our
own tables under RLS. A JWT never carries authorization in this system, only
identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient

from lumen_api.errors import Misconfigured, Unauthorized
from lumen_api.settings import get_settings

_JWKS_CACHE_SECONDS = 600
_jwks_client: PyJWKClient | None = None
_jwks_fetched_at: float = 0.0


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str | None
    role: str


def _jwks() -> PyJWKClient:
    global _jwks_client, _jwks_fetched_at

    now = time.monotonic()
    if _jwks_client is not None and now - _jwks_fetched_at < _JWKS_CACHE_SECONDS:
        return _jwks_client

    settings = get_settings()
    url = f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    try:
        # PyJWKClient fetches lazily; probe once so a misconfigured URL fails
        # here with a clear message rather than inside token verification.
        httpx.get(url, timeout=5.0).raise_for_status()
    except httpx.HTTPError as exc:
        raise Misconfigured(
            f"Cannot reach Supabase JWKS at {url}. Set SUPABASE_JWT_SECRET, or check SUPABASE_URL."
        ) from exc

    _jwks_client = PyJWKClient(url, cache_keys=True)
    _jwks_fetched_at = now
    return _jwks_client


def verify_access_token(token: str) -> AuthenticatedUser:
    """Return the identity a valid Supabase access token asserts.

    Raises `Unauthorized` for anything a caller could have caused, and
    `Misconfigured` for anything only the operator can fix.
    """
    settings = get_settings()
    secret = settings.supabase_jwt_secret.get_secret_value().strip()

    options = {"require": ["exp", "sub"], "verify_aud": True}

    try:
        if secret:
            claims: dict[str, Any] = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="authenticated",
                options=options,
            )
        else:
            signing_key = _jwks().get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience="authenticated",
                options=options,
            )
    except jwt.ExpiredSignatureError as exc:
        raise Unauthorized("Session expired — sign in again") from exc
    except jwt.InvalidAudienceError as exc:
        raise Unauthorized("Token was not issued for this application") from exc
    except jwt.InvalidTokenError as exc:
        raise Unauthorized("Invalid access token") from exc

    subject = claims.get("sub")
    if not subject:
        raise Unauthorized("Access token carries no subject")

    return AuthenticatedUser(
        id=str(subject),
        email=claims.get("email"),
        role=str(claims.get("role") or "authenticated"),
    )
