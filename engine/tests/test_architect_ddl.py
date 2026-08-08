"""Identifier sanitisation — the only place in the system where untrusted
text becomes SQL, so the only place that needs injection review."""

from __future__ import annotations

import pytest

from lumen.architect.ddl import sanitize_identifier
from lumen.architect.spec import SpecError


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("orders", "orders"),
        ("Orders", "orders"),
        ("Customer ID", "customer_id"),
        ("customer-id", "customer_id"),
        ("customer..id", "customer_id"),
        ("  spaced  ", "spaced"),
        ("2024_revenue", "col_2024_revenue"),
        ("café", "caf"),
    ],
)
def test_sanitisation_cases(raw, expected):
    assert sanitize_identifier(raw) == expected


def test_a_reserved_word_is_suffixed_not_quoted_away():
    """Quoting would work, but a column a customer has to write as "select"
    in every query is a papercut we can spend one underscore to avoid."""
    assert sanitize_identifier("select") == "select_col"
    assert sanitize_identifier("ORDER") == "order_col"


def test_collisions_get_a_numeric_suffix():
    taken: set[str] = set()
    first = sanitize_identifier("Customer ID", taken=taken)
    taken.add(first)
    second = sanitize_identifier("customer.id", taken=taken)
    taken.add(second)
    third = sanitize_identifier("CUSTOMER-ID", taken=taken)
    assert [first, second, third] == ["customer_id", "customer_id_2", "customer_id_3"]


def test_a_long_name_is_truncated_to_the_postgres_limit():
    result = sanitize_identifier("a" * 200)
    assert len(result.encode("utf-8")) == 63


def test_a_long_name_built_from_a_multibyte_source_still_fits_the_limit():
    """Sanitisation always collapses non-ASCII content to underscores before
    truncation ever runs, so `collapsed` is pure ASCII by the time
    `_truncate_bytes` sees it and byte length equals character length there.
    The real risk this guards is upstream: a naive `raw[:63]` on the
    *original* multibyte string could split a character in half. Truncating
    the already-ASCII `collapsed` string sidesteps that entirely, which this
    proves by using a long, realistic multibyte-sourced name and asserting
    the result both fits and is still a valid identifier."""
    import re

    raw = ("a" + "ñ") * 50  # alternating valid/invalid -> a long collapsed name
    result = sanitize_identifier(raw)
    assert len(result.encode("utf-8")) <= 63
    assert re.fullmatch(r"[a-z_][a-z0-9_]*", result)


def test_a_name_with_nothing_usable_raises():
    with pytest.raises(SpecError, match="cannot be sanitised"):
        sanitize_identifier("!!!")


def test_an_empty_name_raises():
    with pytest.raises(SpecError, match="cannot be sanitised"):
        sanitize_identifier("")


def test_injection_attempts_are_neutralised():
    assert sanitize_identifier('x"; drop table users; --') == "x_drop_table_users"


def test_every_result_matches_the_validation_pattern():
    import re

    for raw in ["orders", "Customer ID", "2024", "select", "a" * 200, "ñoño"]:
        assert re.fullmatch(r"[a-z_][a-z0-9_]*", sanitize_identifier(raw))
