"""Request-scoped identity.

The web app signs in against Supabase Auth and sends the resulting access token
as `Authorization: Bearer <jwt>`. These dependencies turn that into (a) a user
id we can open an RLS-scoped transaction with, and (b) the organization and
role that user holds — read from our tables, never from the token.

`current_org_id` and `require_role` keep the signatures they had before Supabase
replaced the in-house session store, so every router downstream is unchanged.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy import text

from lumen_api.auth.supabase_jwt import AuthenticatedUser, verify_access_token
from lumen_api.db.session import user_session
from lumen_api.errors import Forbidden, Unauthorized

ROLE_RANK = {"viewer": 0, "member": 1, "admin": 2, "owner": 3}


@dataclass(frozen=True)
class Identity:
    user_id: uuid.UUID
    email: str
    display_name: str
    avatar_url: str | None
    org_id: uuid.UUID
    org_name: str
    org_slug: str
    plan_code: str
    role: str


def _bearer(authorization: str | None, request: Request) -> str:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    # The SSR layer may forward the token in a cookie it controls; accept it so
    # server-rendered routes work without duplicating the header everywhere.
    cookie = request.cookies.get("sb-access-token")
    if cookie:
        return cookie
    raise Unauthorized("No access token")


async def current_auth_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> AuthenticatedUser:
    return verify_access_token(_bearer(authorization, request))


async def current_identity(
    auth_user: Annotated[AuthenticatedUser, Depends(current_auth_user)],
) -> Identity:
    """Resolve the caller's active organization and role.

    Runs `public.current_identity()` inside the caller's own RLS context, so a
    token for a user with no membership yields nothing rather than another
    tenant's row.
    """
    user_id = uuid.UUID(auth_user.id)

    async with user_session(user_id) as db:
        row = (await db.execute(text("SELECT * FROM public.current_identity()"))).mappings().first()

    if row is None:
        raise Forbidden(
            "This account has no organization yet. Sign out and back in, or ask an "
            "administrator for an invitation."
        )

    return Identity(
        user_id=row["user_id"],
        email=row["email"],
        display_name=row["display_name"],
        avatar_url=row["avatar_url"],
        org_id=row["org_id"],
        org_name=row["org_name"],
        org_slug=row["org_slug"],
        plan_code=row["plan_code"],
        role=str(row["role"]),
    )


async def current_user_id(
    auth_user: Annotated[AuthenticatedUser, Depends(current_auth_user)],
) -> uuid.UUID:
    return uuid.UUID(auth_user.id)


async def current_org_id(
    identity: Annotated[Identity, Depends(current_identity)],
) -> uuid.UUID:
    return identity.org_id


def require_role(*allowed: str) -> Callable[..., object]:
    """Gate a route on organization role.

    Ranked, not exact: `require_role("member")` admits admins and owners too,
    which is what every caller has actually meant.
    """
    floor = min(ROLE_RANK[role] for role in allowed)

    async def _check(identity: Annotated[Identity, Depends(current_identity)]) -> Identity:
        if ROLE_RANK.get(identity.role, -1) < floor:
            raise Forbidden(
                f"Requires at least the '{min(allowed, key=lambda r: ROLE_RANK[r])}' role"
            )
        return identity

    return _check
