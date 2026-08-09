"""Enqueueing arq jobs from the API process.

The worker (`lumen_worker.main.WorkerSettings`) owns its own long-lived
pool; the API is short-lived, many-processes, and needs its own. Cached per
event loop for the same reason `tenant_db.get_tenant_engine()` is: a pool
belongs to the loop that opened it, and pytest-asyncio gives every test a
fresh one.
"""

from __future__ import annotations

import asyncio

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from lumen_api.settings import get_settings

_pools: dict[int, ArqRedis] = {}


async def _pool() -> ArqRedis:
    try:
        key = id(asyncio.get_running_loop())
    except RuntimeError:
        key = 0

    pool = _pools.get(key)
    if pool is None:
        pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
        _pools[key] = pool
    return pool


async def dispose_queue_pools() -> None:
    for pool in list(_pools.values()):
        await pool.aclose()
    _pools.clear()


async def enqueue_job(function: str, *args: object) -> None:
    await (await _pool()).enqueue_job(function, *args)
