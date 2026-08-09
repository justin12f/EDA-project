"""The worker process: `arq lumen_worker.main.WorkerSettings`.

The first thing in this codebase to actually run `services/worker` — every
proposal accept still applies synchronously in the API request (see
`proposals.py`'s own note on why), and this is the caller that finally needed
not to block on it: a scheduled tick has no request to answer, so there is
nothing for it to block.
"""

from __future__ import annotations

from arq import cron
from arq.connections import RedisSettings

from lumen_api.settings import get_settings
from lumen_worker.glossary import propose_entity_mapping
from lumen_worker.ingest import design_schema_job, import_tables, ingest_to_staging
from lumen_worker.sentinel import diagnose_drift, dispatch_due_schedules, process_schedule


async def shutdown(ctx: dict) -> None:
    from lumen_api.db.session import dispose_engines

    await dispose_engines()


class WorkerSettings:
    functions = [
        process_schedule,
        diagnose_drift,
        propose_entity_mapping,
        ingest_to_staging,
        design_schema_job,
        import_tables,
    ]
    cron_jobs = [
        # Ticks every 15 minutes and enqueues whatever `data_source_schedules`
        # are due — see sentinel.py's module docstring for why this dispatch
        # step exists instead of a per-org cron() entry.
        cron(dispatch_due_schedules, minute=set(range(0, 60, 15)), run_at_startup=True),
    ]
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    # A tick that hangs on one slow source must not starve the rest of that
    # org's, or another org's, schedule.
    job_timeout = 300
    max_jobs = 10
