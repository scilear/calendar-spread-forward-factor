"""
Standalone PostgreSQL connection module for the CSFF scanner.

Reads connection parameters from environment variables.  When no host is
set (or when CSFF_PGHOST is empty / absent), falls back to a local Unix
socket at /var/run/postgresql — the default for local hal development.

Environment variables (all optional):
    CSFF_PGHOST       — TCP hostname / IP (default: unset → Unix socket)
    CSFF_PGPORT       — TCP port (default: 5432)
    CSFF_PGDATABASE   — database name (default: earningsvol)
    CSFF_PGUSER       — user (default: fabien)
    CSFF_PGPASSWORD   — password (default: unset)

Usage:
    from db import csff_conn
    conn = csff_conn(pg_host="10.0.0.1", pg_db="earningsvol")
    # or rely on env vars:
    conn = csff_conn()

The keyword arguments (pg_host, pg_port, pg_db, pg_user, pg_password)
take precedence over environment variables.
"""

from __future__ import annotations

import os

import psycopg2


def csff_conn(
    pg_host: str | None = None,
    pg_port: int | None = None,
    pg_db: str | None = None,
    pg_user: str | None = None,
    pg_password: str | None = None,
    **kwargs,
):
    host = pg_host or os.environ.get("CSFF_PGHOST")
    port = pg_port or int(os.environ.get("CSFF_PGPORT", 5432))
    dbname = pg_db or os.environ.get("CSFF_PGDATABASE", "earningsvol")
    user = pg_user or os.environ.get("CSFF_PGUSER", "fabien")
    password = pg_password or os.environ.get("CSFF_PGPASSWORD")

    if not host:
        host = "/var/run/postgresql"

    conn_kwargs = dict(host=host, port=port, dbname=dbname, user=user)
    if password:
        conn_kwargs["password"] = password
    # Explicit search_path so unqualified table names (option_chain, ohlcv,
    # earnings_calendar) resolve to the dolt schema on remote TCP connections.
    # The default search_path "$user", public does not include dolt.
    conn_kwargs.setdefault("options", "-c search_path=dolt,public")
    conn_kwargs.update(kwargs)

    return psycopg2.connect(**conn_kwargs)
