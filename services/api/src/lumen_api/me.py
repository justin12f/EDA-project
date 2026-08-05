"""Who is asking.

Every route already resolves this through `current_identity` to scope its own
queries; this endpoint just hands the same record to the browser, which needs
it for exactly one thing — rendering a workspace name instead of a placeholder.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from lumen_api.auth.dependencies import Identity, current_identity

router = APIRouter(prefix="/v1", tags=["identity"])


@router.get("/me")
async def me(identity: Annotated[Identity, Depends(current_identity)]) -> dict[str, object]:
    return {
        "user_id": str(identity.user_id),
        "email": identity.email,
        "display_name": identity.display_name,
        "avatar_url": identity.avatar_url,
        "org_id": str(identity.org_id),
        "org_name": identity.org_name,
        "org_slug": identity.org_slug,
        "plan_code": identity.plan_code,
        "role": identity.role,
    }
