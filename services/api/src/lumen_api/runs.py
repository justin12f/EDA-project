"""Runs: start the agent on a prompt, watch it work over Server-Sent Events.

One HTTP request per turn. The browser opens `POST /v1/runs`, gets back an
event stream, and the connection ends when the agent is done — there is no
separate "poll for status" endpoint because the whole point of streaming is
that the browser never has to ask.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy import text
from sse_starlette.sse import EventSourceResponse

from lumen_api.agents.runner import stream_run
from lumen_api.auth.dependencies import Identity, current_identity
from lumen_api.db.session import user_session

router = APIRouter(prefix="/v1", tags=["runs"])


class RunRequest(BaseModel):
    prompt: str
    source_id: uuid.UUID | None = None
    thread_id: uuid.UUID | None = None
    backend: str = "polars"


@router.post("/runs")
async def create_run(
    body: RunRequest,
    request: Request,
    identity: Annotated[Identity, Depends(current_identity)],
) -> EventSourceResponse:
    async def events():
        async for item in stream_run(
            identity,
            body.prompt,
            source_id=body.source_id,
            thread_id=body.thread_id,
            backend=body.backend,
        ):
            if await request.is_disconnected():
                break
            yield {"event": item["type"], "data": json.dumps(item["payload"], default=str)}

    return EventSourceResponse(
        events(),
        # No-buffering headers: several proxies (and dev servers) hold the
        # response open until a chunk hits a size threshold unless told not to,
        # which turns "streaming" into "arrives all at once when the run ends".
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        # Plain `\n`, not the library's default `\r\n`: the browser client parses
        # frames by hand (native EventSource cannot POST), and one separator
        # convention on both ends is simpler than negotiating it.
        sep="\n",
    )


@router.get("/threads/{thread_id}/events")
async def thread_events(
    thread_id: uuid.UUID,
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    """Replay a thread's transcript from persisted events, oldest first.

    What a reopened chat renders from — the live SSE stream only exists while a
    request is open, so this is the only source of history once the tab closes.
    """
    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select e.run_id, e.seq, e.type, e.payload, e.created_at "
                    "from public.agent_events e "
                    "join public.runs r on r.id = e.run_id "
                    "where r.thread_id = :thread "
                    "order by r.started_at, e.seq"
                ),
                {"thread": thread_id},
            )
        ).mappings().all()
        # The source this thread was about, if any — so reopening it can
        # reselect the same source rather than dropping back to general chat.
        source_id = (
            await db.execute(
                text(
                    "select source_id from public.runs "
                    "where thread_id = :thread and source_id is not null "
                    "order by started_at limit 1"
                ),
                {"thread": thread_id},
            )
        ).scalar_one_or_none()
    return {"events": [dict(row) for row in rows], "source_id": source_id}


@router.get("/threads")
async def list_threads(
    identity: Annotated[Identity, Depends(current_identity)],
) -> dict[str, Any]:
    """One row per thread, most recently active first — a conversation list."""
    async with user_session(identity.user_id) as db:
        rows = (
            await db.execute(
                text(
                    "select thread_id, max(started_at) as last_active, "
                    "       count(*) as run_count, "
                    "       (array_agg(source_id order by started_at))[1] as source_id "
                    "from public.runs group by thread_id order by last_active desc"
                )
            )
        ).mappings().all()
    return {"threads": [dict(row) for row in rows]}
