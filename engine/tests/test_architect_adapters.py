"""Source adapters. The contract is format-agnostic; v1 ships files and two
live databases, and adding a format is a new adapter and nothing else."""

from __future__ import annotations

import polars as pl
import pytest

from lumen.architect.adapters.file import FileAdapter
from lumen.architect.spec import SqlType


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text("id,amount,note\n1,10.5,hello\n2,20.0,world\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def parquet_path(tmp_path):
    path = tmp_path / "orders.parquet"
    pl.DataFrame({"id": [1, 2], "amount": [10.5, 20.0]}).write_parquet(path)
    return str(path)


async def test_discover_returns_one_table_named_after_the_file(csv_path):
    structure = await FileAdapter(csv_path).discover()
    assert len(structure.tables) == 1
    assert structure.tables[0].name == "orders"


async def test_a_file_declares_no_keys(csv_path):
    """Nothing in a CSV asserts a primary key or a relationship. Saying so
    explicitly is what lets the diagram distinguish a read constraint from
    an inferred one."""
    table = (await FileAdapter(csv_path).discover()).tables[0]
    assert table.primary_key is None
    assert table.foreign_keys == ()


async def test_discovery_is_marked_undeclared(csv_path):
    assert (await FileAdapter(csv_path).discover()).declared is False


async def test_columns_are_typed_from_the_frame(csv_path):
    table = (await FileAdapter(csv_path).discover()).tables[0]
    types = {c.name: c.sql_type for c in table.columns}
    assert types["id"] is SqlType.BIGINT
    assert types["amount"] is SqlType.DOUBLE
    assert types["note"] is SqlType.TEXT


async def test_parquet_works_through_the_same_adapter(parquet_path):
    structure = await FileAdapter(parquet_path).discover()
    assert {c.name for c in structure.tables[0].columns} == {"id", "amount"}


async def test_read_returns_a_frame(csv_path):
    frame = await FileAdapter(csv_path).read("orders")
    materialised = frame.collect() if hasattr(frame, "collect") else frame
    assert materialised.height == 2


async def test_read_honours_a_limit(csv_path):
    frame = await FileAdapter(csv_path).read("orders", limit=1)
    materialised = frame.collect() if hasattr(frame, "collect") else frame
    assert materialised.height == 1


async def test_an_unsupported_extension_names_what_is_supported(tmp_path):
    path = tmp_path / "data.docx"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="csv"):
        await FileAdapter(str(path)).discover()
