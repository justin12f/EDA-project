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


def test_every_backend_layer_resolves_without_pyspark():
    """The API and worker install lumen-engine without the spark extra."""
    import importlib.util

    if importlib.util.find_spec("pyspark") is not None:
        import pytest

        pytest.skip("pyspark installed; this guards the no-pyspark install")

    from lumen.agents.master_factory import AgentMasterFactory

    for backend in ("pandas", "polars"):
        master = AgentMasterFactory(backend)
        assert master.readers().backend == backend
        assert master.cleaning().backend == backend
        assert master.analyzers().backend == backend
        assert master.statistics().backend == backend
        assert master.encoders().backend == backend


def test_asking_for_spark_without_pyspark_gives_an_actionable_error():
    import importlib.util

    if importlib.util.find_spec("pyspark") is not None:
        import pytest

        pytest.skip("pyspark installed")

    import pytest

    from lumen.agents.master_factory import AgentMasterFactory

    with pytest.raises((ImportError, ValueError)) as excinfo:
        AgentMasterFactory("spark").readers().read("engine/tests/fixtures/data/shopping_trends.csv")
    assert "spark" in str(excinfo.value).lower()
