"""The agent's memory: what it learned, and what this person decided.

Two scopes, chosen by what the row *is* rather than by who wrote it:

    ORG   objective facts about shared data — a profile, a schema, an applied
          pipeline. Measured once, useful to every member. Scoping these per
          user would mean paying for the same computation N times and letting
          each teammate rediscover the same 3.2% null rate.

    USER  facts about one person's judgment — what they asked, what they
          rejected and why, the columns they always care about. Private to
          their author even inside the same organization, because a colleague's
          rejected proposal is not evidence about your intent, and feeding it to
          the agent as if it were makes the agent worse for both of you.

Isolation is the database's job (see 20260804000004_personal_context.sql). This
module cannot leak one person's history into another's context even if it tries:
every read runs inside that user's RLS session.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy import text

from lumen.embeddings import EmbeddingProvider, get_embedding_provider
from lumen_api.db.session import user_session
from lumen_api.settings import get_settings


class Scope(StrEnum):
    ORG = "org"
    USER = "user"


class Kind(StrEnum):
    PROFILE = "profile"
    SCHEMA = "schema"
    RATIONALE = "rationale"
    DECISION = "decision"
    NOTE = "note"


# Which scope each kind belongs in. Encoded here rather than left to call sites,
# because "is this fact about the data or about the person" is exactly the
# judgment that gets made inconsistently when it is made twenty times.
DEFAULT_SCOPE: dict[Kind, Scope] = {
    Kind.PROFILE: Scope.ORG,
    Kind.SCHEMA: Scope.ORG,
    Kind.RATIONALE: Scope.ORG,
    Kind.DECISION: Scope.USER,
    Kind.NOTE: Scope.USER,
}


@dataclass(frozen=True)
class ContextEntry:
    kind: Kind
    title: str
    content: str
    scope: Scope | None = None
    source_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None
    rid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_scope(self) -> Scope:
        return self.scope or DEFAULT_SCOPE[self.kind]


@dataclass(frozen=True)
class ContextMatch:
    id: uuid.UUID
    kind: Kind
    scope: Scope
    title: str
    content: str
    similarity: float
    metadata: dict[str, Any]
    source_id: uuid.UUID | None
    is_mine: bool


class ContextStore:
    """Reads and writes `data_contexts` for one user in one organization."""

    def __init__(
        self,
        org_id: uuid.UUID,
        user_id: uuid.UUID,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._org_id = org_id
        self._user_id = user_id
        self._embedder = embedder or _default_embedder()

    # ── writing ─────────────────────────────────────────────────────────────

    async def remember(self, entry: ContextEntry) -> uuid.UUID:
        scope = entry.resolved_scope()
        embedding = self._embed(entry.content)

        async with user_session(self._user_id) as db:
            row = (
                await db.execute(
                    text(
                        "insert into public.data_contexts "
                        "(org_id, user_id, source_id, run_id, rid, kind, scope, title, "
                        " content, embedding, metadata) "
                        "values (:org, :user, :source, :run, :rid, :kind, :scope, :title, "
                        " :content, cast(:embedding as extensions.vector), cast(:meta as jsonb)) "
                        "returning id"
                    ),
                    {
                        "org": self._org_id,
                        # An org-scoped row still records who produced it — useful
                        # for attribution — but the policy does not gate on it.
                        "user": self._user_id,
                        "source": entry.source_id,
                        "run": entry.run_id,
                        "rid": entry.rid,
                        "kind": str(entry.kind),
                        "scope": str(scope),
                        "title": entry.title[:200],
                        "content": entry.content,
                        "embedding": embedding,
                        "meta": json.dumps(entry.metadata),
                    },
                )
            ).scalar_one()
        return row

    async def remember_many(self, entries: list[ContextEntry]) -> list[uuid.UUID]:
        """Embed in one batch — the model call dominates, and it vectorises."""
        if not entries:
            return []

        vectors = self._embed_many([e.content for e in entries])
        written: list[uuid.UUID] = []

        async with user_session(self._user_id) as db:
            for entry, embedding in zip(entries, vectors, strict=True):
                scope = entry.resolved_scope()
                row = (
                    await db.execute(
                        text(
                            "insert into public.data_contexts "
                            "(org_id, user_id, source_id, run_id, rid, kind, scope, title, "
                            " content, embedding, metadata) "
                            "values (:org, :user, :source, :run, :rid, :kind, :scope, :title, "
                            " :content, cast(:embedding as extensions.vector), cast(:meta as jsonb)) "
                            "returning id"
                        ),
                        {
                            "org": self._org_id,
                            "user": self._user_id,
                            "source": entry.source_id,
                            "run": entry.run_id,
                            "rid": entry.rid,
                            "kind": str(entry.kind),
                            "scope": str(scope),
                            "title": entry.title[:200],
                            "content": entry.content,
                            "embedding": embedding,
                            "meta": json.dumps(entry.metadata),
                        },
                    )
                ).scalar_one()
                written.append(row)
        return written

    # ── reading ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float = 0.25,
        source_id: uuid.UUID | None = None,
        kinds: list[Kind] | None = None,
        scope: Scope | None = None,
    ) -> list[ContextMatch]:
        embedding = self._embed(query)
        if embedding is None:
            return []

        async with user_session(self._user_id) as db:
            rows = (
                await db.execute(
                    text(
                        "select id, source_id, kind, scope, user_id, title, content, "
                        "       metadata, similarity "
                        "from public.match_data_contexts("
                        "  cast(:embedding as extensions.vector), :org, :limit, :min_sim, "
                        "  :source, cast(:kinds as public.context_kind[]), "
                        "  cast(:scope as public.context_scope))"
                    ),
                    {
                        "embedding": embedding,
                        "org": self._org_id,
                        "limit": limit,
                        "min_sim": min_similarity,
                        "source": source_id,
                        "kinds": [str(k) for k in kinds] if kinds else None,
                        "scope": str(scope) if scope else None,
                    },
                )
            ).mappings().all()

        return [
            ContextMatch(
                id=row["id"],
                kind=Kind(str(row["kind"])),
                scope=Scope(str(row["scope"])),
                title=row["title"],
                content=row["content"],
                similarity=float(row["similarity"]),
                metadata=dict(row["metadata"] or {}),
                source_id=row["source_id"],
                is_mine=row["user_id"] == self._user_id,
            )
            for row in rows
        ]

    async def briefing(self, source_id: uuid.UUID, max_personal: int = 10) -> str:
        """What the agent should know before it starts, as prompt-ready markdown.

        Exact recall, not similarity: when you know the key, an approximate
        search is a worse way to answer the question.
        """
        async with user_session(self._user_id) as db:
            rows = (
                await db.execute(
                    text(
                        "select scope, kind, title, content, created_at "
                        "from public.personal_source_context(:org, :source, :max)"
                    ),
                    {"org": self._org_id, "source": source_id, "max": max_personal},
                )
            ).mappings().all()

        if not rows:
            return ""

        shared = [r for r in rows if str(r["scope"]) == "org"]
        personal = [r for r in rows if str(r["scope"]) == "user"]

        parts: list[str] = []
        if shared:
            parts.append("## What is already known about this source\n")
            parts.extend(f"- **{r['title']}** — {r['content']}" for r in shared)
        if personal:
            parts.append("\n## What you previously decided here\n")
            parts.extend(
                f"- {r['created_at']:%Y-%m-%d} · **{r['title']}** — {r['content']}"
                for r in personal
            )
        return "\n".join(parts)

    # ── embedding ───────────────────────────────────────────────────────────

    def _embed(self, content: str) -> str | None:
        vectors = self._embed_many([content])
        return vectors[0] if vectors else None

    def _embed_many(self, contents: list[str]) -> list[str | None]:
        """Return pgvector literals, or None when no embedding could be produced.

        A null embedding is deliberate: the row is still written, still readable
        by exact lookup, and a later backfill can embed it. Refusing to record
        what the agent learned because a model hiccuped would lose the more
        valuable half.

        But the two reasons to fail are not the same, and conflating them cost
        real debugging time here. `EMBEDDING_PROVIDER=none` is a choice and stays
        silent. Anything else — a missing package, a corrupt cache — is a
        misconfiguration that presents as "semantic search finds nothing", which
        looks like a product defect. Those are logged, loudly, once.
        """
        if self._embedder.dimensions == 0:
            return [None] * len(contents)
        try:
            vectors = self._embedder.embed(contents)
        except Exception as exc:  # noqa: BLE001 — see docstring
            _warn_embeddings_unavailable(exc)
            return [None] * len(contents)
        return ["[" + ",".join(f"{value:.7f}" for value in vector) + "]" for vector in vectors]


_warned = False


def _warn_embeddings_unavailable(exc: Exception) -> None:
    """Once per process. A warning repeated per row is a warning nobody reads."""
    global _warned
    if _warned:
        return
    _warned = True
    logging.getLogger("lumen").error(
        "Embeddings unavailable — context is being stored WITHOUT vectors, so "
        "semantic recall will return nothing until this is fixed and the rows are "
        "backfilled. Cause: %s: %s",
        type(exc).__name__,
        exc,
    )


def _default_embedder() -> EmbeddingProvider:
    settings = get_settings()
    return get_embedding_provider(
        settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        cache_dir=settings.embedding_cache_dir,
    )
