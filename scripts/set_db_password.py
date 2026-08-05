#!/usr/bin/env python
"""Put the Supabase database password into .env without it passing through a shell
history, a chat window, or a terminal echo.

    uv run --directory services/api python ../../scripts/set_db_password.py

It prompts with getpass, percent-encodes the value so a password containing `@`,
`/`, `#` or `:` does not corrupt the DSN, writes DATABASE_URL, and then opens one
connection to prove it works. Nothing is printed but the outcome.
"""

from __future__ import annotations

import getpass
import re
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV = REPO_ROOT / ".env"
PROJECT_REF_RE = re.compile(r"https://([a-z0-9]+)\.supabase\.co")


def project_ref() -> str:
    text = ENV.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("SUPABASE_URL="):
            match = PROJECT_REF_RE.search(line)
            if match:
                return match.group(1)
    raise SystemExit("SUPABASE_URL is not set in .env — fill in the Supabase block first.")


def main() -> int:
    if not ENV.exists():
        raise SystemExit(".env not found. Copy .env.example to .env first.")

    ref = project_ref()
    print(f"Supabase project: {ref}")
    print("Dashboard → Settings → Database → Reset password, then paste it here.")
    print("Nothing is echoed, logged, or stored anywhere but .env.\n")

    password = getpass.getpass("Database password: ").strip()
    if not password:
        raise SystemExit("No password entered; nothing changed.")

    dsn = (
        f"postgresql+asyncpg://postgres:{quote(password, safe='')}"
        f"@db.{ref}.supabase.co:5432/postgres"
    )

    lines = ENV.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("DATABASE_URL="):
            lines[index] = f"DATABASE_URL={dsn}"
            break
    else:
        lines.append(f"DATABASE_URL={dsn}")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("DATABASE_URL written.\n")

    return check()


def check() -> int:
    import asyncio

    try:
        from sqlalchemy import text

        from lumen_api.db.session import service_session
    except ImportError:
        print("Written, but the API package is not importable from here — run:")
        print("  uv run --directory services/api pytest tests -m integration -q")
        return 0

    async def probe() -> int:
        try:
            async with service_session() as db:
                tables = (
                    await db.execute(
                        text(
                            "select count(*) from pg_class c "
                            "join pg_namespace n on n.oid = c.relnamespace "
                            "where n.nspname = 'public' and c.relkind = 'r'"
                        )
                    )
                ).scalar_one()
            print(f"Connected. {tables} tables in public.")
            print("\nNow run: uv run --directory services/api pytest tests -m integration -q")
            return 0
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            print(f"Could not connect: {name}")
            if "Password" in name:
                print("  The password is wrong. Reset it in the dashboard and run this again.")
            else:
                print(
                    "  Not an authentication failure. If it timed out, check\n"
                    "  Settings → Database → Network bans — repeated failed attempts\n"
                    "  get the IP temporarily blocked."
                )
            return 1

    return asyncio.run(probe())


if __name__ == "__main__":
    sys.exit(main())
