"""Tenant identifier naming — pure, so it runs in the default suite.

Provisioning and isolation need a live instance and live in
test_tenant_isolation.py behind the integration marker.
"""

from __future__ import annotations

import uuid

from lumen_api.tenant_db import (
    tenant_raw_schema_name,
    tenant_role_name,
    tenant_schema_name,
)

ORG = uuid.UUID("f7930655-ed4d-40f0-9a8d-21cd99bf468a")


def test_the_schema_name_strips_dashes():
    assert tenant_schema_name(ORG) == "tenant_f7930655ed4d40f09a8d21cd99bf468a"


def test_the_raw_schema_is_the_schema_plus_a_suffix():
    assert tenant_raw_schema_name(ORG) == tenant_schema_name(ORG) + "_raw"


def test_the_role_is_the_schema_plus_a_suffix():
    assert tenant_role_name(ORG) == tenant_schema_name(ORG) + "_role"


def test_every_identifier_fits_the_postgres_limit():
    """63 bytes. The role name is the longest of the three, so if it fits,
    they all do — but assert each one rather than reasoning about it."""
    for name in (tenant_schema_name(ORG), tenant_raw_schema_name(ORG), tenant_role_name(ORG)):
        assert len(name.encode("utf-8")) <= 63


def test_names_are_deterministic():
    assert tenant_schema_name(ORG) == tenant_schema_name(ORG)


def test_different_orgs_get_different_names():
    assert tenant_schema_name(ORG) != tenant_schema_name(uuid.uuid4())
