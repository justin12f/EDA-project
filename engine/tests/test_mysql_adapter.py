"""MySQLAdapter's pure parts, against recorded information_schema rows.

No live MySQL exists in this environment. Testing the mapping and assembly
directly is honest; a skipped integration test that has never run would
look like coverage without being any.
"""

from __future__ import annotations

from lumen.architect.adapters.mysql import _build_structure, _map_mysql_type
from lumen.architect.spec import SqlType


def test_tinyint_one_is_boolean():
    """MySQL has no bool. tinyint(1) is the convention every ORM emits, and
    reading it as an integer would show a customer 0/1 where they wrote
    true/false."""
    assert _map_mysql_type("tinyint", "tinyint(1)", None, None, None)[0] is SqlType.BOOLEAN


def test_a_wider_tinyint_stays_an_integer():
    assert _map_mysql_type("tinyint", "tinyint(4)", None, None, None)[0] is SqlType.INTEGER


def test_int_and_bigint():
    assert _map_mysql_type("int", "int(11)", None, None, None)[0] is SqlType.INTEGER
    assert _map_mysql_type("bigint", "bigint(20)", None, None, None)[0] is SqlType.BIGINT


def test_varchar_carries_its_length():
    assert _map_mysql_type("varchar", "varchar(120)", 120, None, None) == (SqlType.VARCHAR, "120")


def test_decimal_carries_precision_and_scale():
    assert _map_mysql_type("decimal", "decimal(12,2)", None, 12, 2) == (SqlType.NUMERIC, "12,2")


def test_datetime_is_naive_and_timestamp_is_aware():
    """MySQL's timestamp converts to UTC on store; datetime does not. They
    are genuinely different types and collapsing them loses the zone."""
    assert _map_mysql_type("datetime", "datetime", None, None, None)[0] is SqlType.TIMESTAMP
    assert _map_mysql_type("timestamp", "timestamp", None, None, None)[0] is SqlType.TIMESTAMPTZ


def test_text_variants_all_map_to_text():
    for name in ("text", "mediumtext", "longtext", "tinytext"):
        assert _map_mysql_type(name, name, None, None, None)[0] is SqlType.TEXT


def test_json_maps_to_jsonb():
    assert _map_mysql_type("json", "json", None, None, None)[0] is SqlType.JSONB


def test_an_unknown_type_falls_back_to_text():
    assert _map_mysql_type("geometry", "geometry", None, None, None)[0] is SqlType.TEXT


def test_structure_assembly_reads_keys_from_key_column_usage():
    """MySQL puts REFERENCED_TABLE_NAME directly on KEY_COLUMN_USAGE, so
    unlike Postgres there is no referential_constraints join."""
    columns = [
        {"TABLE_NAME": "customers", "COLUMN_NAME": "id", "DATA_TYPE": "varchar",
         "COLUMN_TYPE": "varchar(36)", "IS_NULLABLE": "NO",
         "CHARACTER_MAXIMUM_LENGTH": 36, "NUMERIC_PRECISION": None, "NUMERIC_SCALE": None},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "id", "DATA_TYPE": "bigint",
         "COLUMN_TYPE": "bigint(20)", "IS_NULLABLE": "NO",
         "CHARACTER_MAXIMUM_LENGTH": None, "NUMERIC_PRECISION": 20, "NUMERIC_SCALE": 0},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "customer_id", "DATA_TYPE": "varchar",
         "COLUMN_TYPE": "varchar(36)", "IS_NULLABLE": "YES",
         "CHARACTER_MAXIMUM_LENGTH": 36, "NUMERIC_PRECISION": None, "NUMERIC_SCALE": None},
    ]
    keys = [
        {"TABLE_NAME": "customers", "COLUMN_NAME": "id", "CONSTRAINT_NAME": "PRIMARY",
         "REFERENCED_TABLE_NAME": None, "REFERENCED_COLUMN_NAME": None},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "id", "CONSTRAINT_NAME": "PRIMARY",
         "REFERENCED_TABLE_NAME": None, "REFERENCED_COLUMN_NAME": None},
        {"TABLE_NAME": "orders", "COLUMN_NAME": "customer_id", "CONSTRAINT_NAME": "fk_cust",
         "REFERENCED_TABLE_NAME": "customers", "REFERENCED_COLUMN_NAME": "id"},
    ]

    structure = _build_structure(columns, keys)
    assert structure.declared is True

    customers = next(t for t in structure.tables if t.name == "customers")
    orders = next(t for t in structure.tables if t.name == "orders")
    assert customers.primary_key == ("id",)
    assert orders.primary_key == ("id",)
    assert orders.foreign_keys == (("customer_id", "customers", "id"),)
    assert next(c for c in orders.columns if c.name == "customer_id").nullable is True
