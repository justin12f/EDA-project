"""Run an agent for one request and stream what it does.

The loop in the engine knows nothing about tenants, databases or HTTP — it takes
a provider, a registry and a sink. This module supplies all three for a signed-in
person and turns the sink into two things at once:

* rows in `agent_events`, so a run can be reopened tomorrow, and
* items on an in-memory queue, so the browser sees each step as it happens.

Both, not either. Persisting only would mean polling; streaming only would mean a
run is gone the moment the tab closes.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import text

from lumen.agents.loop import AgentLoop
from lumen.llm.base import TokenUsage
from lumen.llm.registry import ModelTiers, get_provider
from lumen_api.agents.registry import build_tool_registry
from lumen_api.auth.dependencies import Identity
from lumen_api.context.store import ContextStore
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings

# The queue is closed by pushing this. `None` would be ambiguous with a dropped
# event, and a sentinel object survives being put on a typed queue.
_DONE = object()

SYSTEM_PROMPT = """\
You are Lumen's data analyst. You work on one workspace's data, through tools.

How to work:
1. Call `recall_context` first. Someone may have audited this source already, and
   this person may have rejected a proposal you are about to repeat.
2. `read_source` to load it, then `profile_source` to see nulls, types and
   duplicates. Never describe data you have not profiled.
3. Propose changes with `propose_cleaning_pipeline`. Cite the numbers you saw —
   "email is 12.4% null" — never "there appear to be some nulls".
4. Use `list_predictors`, `compare_predictors` and `forecast_series` when the
   question is about what happens next.
5. `remember_decision` when you learn something worth having next time.

Rules:
- You propose; a person accepts. You never apply a pipeline yourself.
- A tool that fails returns an error you can read. Try another route rather than
  giving up, but do not repeat the same failing call.
- When you are done, answer in plain prose. No JSON, no tool syntax.
"""


def _provider():
    settings = get_settings()
    return get_provider(
        "reasoning",
        anthropic_key=(
            settings.anthropic_api_key.get_secret_value() if settings.has_anthropic else None
        ),
        groq_key=(settings.groq_api_key.get_secret_value() if settings.has_groq else None),
        tiers=ModelTiers(
            reasoning=settings.model_reasoning,
            specialist=settings.model_specialist,
            fast=settings.model_fast,
        ),
        mode=settings.llm_mode,
        bridge_inbox=settings.llm_bridge_inbox,
    )


class StreamingSink:
    """Writes every event to Postgres and pushes it onto a queue.

    `append_agent_event` allocates `seq` in the database, so ordering survives two
    processes writing to the same run — which is exactly what happens the moment
    the worker starts applying an accepted proposal alongside a live chat.
    """

    def __init__(self, org_id: uuid.UUID, user_id: uuid.UUID, run_id: uuid.UUID) -> None:
        self._org_id = org_id
        self._user_id = user_id
        self._run_id = run_id
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.tokens = 0

    async def emit(self, type_: str, payload: dict[str, Any]) -> None:
        async with user_session(self._user_id) as db:
            row = (
                await db.execute(
                    text(
                        "select seq, created_at from public.append_agent_event("
                        "  :org, :run, cast(:type as public.event_type), cast(:payload as jsonb))"
                    ),
                    {
                        "org": self._org_id,
                        "run": self._run_id,
                        "type": type_,
                        "payload": json.dumps(payload, default=str),
                    },
                )
            ).mappings().first()

        await self.queue.put(
            {"seq": row["seq"] if row else 0, "type": type_, "payload": payload}
        )

    async def record_usage(self, usage: TokenUsage) -> None:
        self.tokens += usage.total


async def stream_run(
    identity: Identity,
    prompt: str,
    *,
    source_id: uuid.UUID | None = None,
    thread_id: uuid.UUID | None = None,
    backend: str = "polars",
) -> AsyncIterator[dict[str, Any]]:
    """Start a run and yield its events until the loop stops.

    The loop runs as a task rather than inline so that events reach the browser
    while it is still working. Awaiting the loop and *then* draining the queue
    would replay a finished run at full speed, which is not streaming — it is a
    slideshow after the fact.
    """
    settings = get_settings()
    thread = thread_id or uuid.uuid4()

    async with user_session(identity.user_id) as db:
        run_id = (
            await db.execute(
                text(
                    "insert into public.runs (org_id, source_id, thread_id, kind, status, "
                    "                         backend, created_by) "
                    "values (:org, :source, :thread, 'chat', 'running', :backend, :user) "
                    "returning id"
                ),
                {
                    "org": identity.org_id,
                    "source": source_id,
                    "thread": thread,
                    "backend": backend,
                    "user": identity.user_id,
                },
            )
        ).scalar_one()

    yield {"type": "run_started", "payload": {"run_id": str(run_id), "thread_id": str(thread)}}

    sink = StreamingSink(identity.org_id, identity.user_id, run_id)
    # `agent_events.type` has no "user_message" value — the enum's "message" is
    # shared with the assistant's own turns (emitted by the engine's loop), and
    # a `role` in the payload is what tells a replayed thread the two apart.
    await sink.emit("message", {"role": "user", "text": prompt})
    registry = build_tool_registry(
        identity.org_id, identity.user_id, backend, run_id=run_id, thread_id=thread
    )
    loop = AgentLoop(
        _provider(),
        registry,
        agent_name="analyst",
        sink=sink,
        max_iterations=settings.agent_max_iterations,
        deadline_seconds=settings.agent_deadline_seconds,
        max_total_tokens=settings.agent_max_total_tokens,
    )

    briefing = await _briefing(identity, source_id)
    task = asyncio.create_task(loop.run(SYSTEM_PROMPT + briefing, prompt))
    task.add_done_callback(lambda _: sink.queue.put_nowait(_DONE))

    while True:
        item = await sink.queue.get()
        if item is _DONE:
            break
        yield item

    error: str | None = None
    try:
        result = await task
        final, stop_reason = result.final_text, result.stop_reason
    except Exception as exc:  # noqa: BLE001 — the browser gets the message, not a dead socket
        final, stop_reason, error = None, "failed", f"{type(exc).__name__}: {exc}"

    async with user_session(identity.user_id) as db:
        await db.execute(
            text(
                "update public.runs set status = cast(:status as public.run_status), "
                "       finished_at = now(), error = :error where id = :id"
            ),
            {
                "status": "failed" if error else "succeeded",
                "error": error,
                "id": run_id,
            },
        )

    done_payload = {
        "run_id": str(run_id),
        "text": final,
        "stop_reason": stop_reason,
        "tokens": sink.tokens,
        "error": error,
    }
    # Persisted so a reopened thread can show how the run ended, and yielded
    # directly — not through the queue — because the queue's consumer already
    # broke out on the sentinel above.
    await sink.emit("done", done_payload)
    yield {"type": "done", "payload": done_payload}


async def _briefing(identity: Identity, source_id: uuid.UUID | None) -> str:
    """What this workspace and this person already know, appended to the prompt.

    Empty string when there is nothing yet, rather than a header with nothing
    under it — a heading over silence reads as "there is no history", which is a
    stronger claim than "this is a new workspace".
    """
    if source_id is None:
        return ""
    try:
        text_ = await ContextStore(identity.org_id, identity.user_id).briefing(source_id)
    except Exception:  # noqa: BLE001 — memory is an aid; losing it must not lose the run
        return ""
    return f"\n\n## What is already known\n\n{text_}\n" if text_ else ""
