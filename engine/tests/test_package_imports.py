"""The engine must be importable as one package and must not shadow the stdlib."""
import statistics as stdlib_statistics


def test_stdlib_statistics_is_not_shadowed():
    assert stdlib_statistics.mean([1, 2, 3]) == 2
    assert "lumen" not in (stdlib_statistics.__file__ or "")


def test_master_factory_imports():
    from lumen.agents.master_factory import AgentMasterFactory

    master = AgentMasterFactory("polars")
    assert master.backend == "polars"


def test_domain_layers_resolve():
    from lumen.agents.master_factory import AgentMasterFactory

    master = AgentMasterFactory("pandas")
    assert master.readers().backend == "pandas"
    assert master.cleaning().backend == "pandas"
    assert master.analyzers().backend == "pandas"
