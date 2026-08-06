"""The trust API (ADR-0009 §4): the first surface in this product an
external system can call without ever being a signed-in browser user.

Every request that reaches an endpoint here is logged before the endpoint
decides what to return — `_enforce_rate_limit` counts what `_record` has
already written for earlier requests, and a request rejected *for* being
over its rate limit is itself recorded via the same call. The audit trail
this ADR calls "a security review surface, not optional" cannot have a gap
shaped like "requests that got refused."

A key that fails authentication entirely (missing, invalid, revoked, wrong
scope) is not logged here: there is no org to attribute the attempt to yet,
and `api_key_audit_log.org_id` is not nullable on purpose — this table is
"what has a legitimate key been asked to do," not a general access log.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Header
from sqlalchemy import text

from lumen_api.api_keys import ApiKeyAuth, authenticate
from lumen_api.certification import certification_for
from lumen_api.db.session import user_session
from lumen_api.errors import NotFound, TooManyRequests, Unauthorized

router = APIRouter(prefix="/v1/public", tags=["trust-api"])

# Generous enough for a BI dashboard's refresh cycle, tight enough that a
# leaked key cannot be used to scrape a workspace's glossary or certification
# state wholesale.
_RATE_LIMIT_PER_MINUTE = 60


async def _authorize(authorization: str | None, scope: str, endpoint: str) -> ApiKeyAuth:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise Unauthorized("Missing 'Authorization: Bearer lum_...' header")
    auth = await authenticate(authorization[7:].strip(), scope)
    if auth is None:
        raise Unauthorized("Invalid, revoked, or wrong-scope API key")
    await _enforce_rate_limit(auth, endpoint)
    return auth


async def _enforce_rate_limit(auth: ApiKeyAuth, endpoint: str) -> None:
    async with user_session(auth.user_id) as db:
        count = (
            await db.execute(
                text(
                    "select count(*) from public.api_key_audit_log "
                    "where api_key_id = :key and requested_at > now() - interval '1 minute'"
                ),
                {"key": auth.key_id},
            )
        ).scalar_one()
    if count >= _RATE_LIMIT_PER_MINUTE:
        await _record(auth, endpoint, 429)
        raise TooManyRequests(f"More than {_RATE_LIMIT_PER_MINUTE} requests in the last minute")


async def _record(auth: ApiKeyAuth, endpoint: str, status_code: int) -> None:
    async with user_session(auth.user_id) as db:
        await db.execute(
            text(
                "insert into public.api_key_audit_log (api_key_id, org_id, endpoint, status_code) "
                "values (:key, :org, :endpoint, :status)"
            ),
            {"key": auth.key_id, "org": auth.org_id, "endpoint": endpoint, "status": status_code},
        )


@router.get("/glossary/{name}")
async def get_glossary_entry(
    name: str, authorization: Annotated[str | None, Header()] = None
) -> dict[str, Any]:
    endpoint = f"GET /v1/public/glossary/{name}"
    auth = await _authorize(authorization, "read:glossary", endpoint)

    async with user_session(auth.user_id) as db:
        entity = (
            await db.execute(
                text(
                    "select id, name, entity_type, reconciliation_rule, created_at "
                    "from public.canonical_entities where name = :name and status = 'approved'"
                ),
                {"name": name},
            )
        ).mappings().first()
        if entity is None:
            await _record(auth, endpoint, 404)
            raise NotFound(f"No approved glossary entry named '{name}'")

        members = (
            await db.execute(
                text(
                    "select m.source_id, s.name as source_name, m.column_name "
                    "from public.canonical_entity_members m "
                    "join public.data_sources s on s.id = m.source_id "
                    "where m.entity_id = :entity"
                ),
                {"entity": entity["id"]},
            )
        ).mappings().all()

    await _record(auth, endpoint, 200)
    return {
        "name": entity["name"],
        "entity_type": entity["entity_type"],
        "reconciliation_rule": entity["reconciliation_rule"],
        "members": [
            {
                "source_id": str(member["source_id"]),
                "source_name": member["source_name"],
                "column": member["column_name"],
            }
            for member in members
        ],
        "defined_at": entity["created_at"].isoformat(),
    }


@router.get("/certification/{source_id}")
async def get_public_certification(
    source_id: uuid.UUID, authorization: Annotated[str | None, Header()] = None
) -> dict[str, Any]:
    endpoint = f"GET /v1/public/certification/{source_id}"
    auth = await _authorize(authorization, "read:certification", endpoint)

    result = await certification_for(auth.org_id, auth.user_id, source_id)
    await _record(auth, endpoint, 200)
    # The narrower surface the ADR actually specifies: certified, last check,
    # open-drift count — not the internal reasoning (a pending review, which
    # columns) an external caller has no proposal UI to act on anyway.
    return {
        "source_id": str(source_id),
        "certified": result.certified,
        "last_checked_at": result.last_checked_at,
        "open_drift_count": result.open_drift_count,
    }
