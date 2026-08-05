"""Generic registry-based factory for statistics backends.

Design
------
RegistryFactory is a **class-level** registry that maps a (calculator_name, backend)
pair to a concrete implementation class.  Every domain factory inherits from this
base without adding any methods — the inheritance itself scopes the registry to that
domain so registrations in `DescriptiveStatisticsFactory` do not pollute
`InferentialStatisticsFactory`.

Usage
-----
    # 1.  Inherit (one subclass per domain, no body needed)
    class DescriptiveStatisticsFactory(RegistryFactory): pass

    # 2.  Register (called once at module load inside each factory.py)
    DescriptiveStatisticsFactory.register("mean_calculator", "polars", MeanCalculatorPolars)
    DescriptiveStatisticsFactory.register("mean_calculator", "spark",  MeanCalculatorSpark)
    DescriptiveStatisticsFactory.register("mean_calculator", "pandas", MeanCalculatorPandas)

    # 3.  Create (called at runtime, typically from a DI container)
    calc = DescriptiveStatisticsFactory.create("mean_calculator", "polars")
"""
from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar

K = TypeVar("K")  # calculator / service name (usually str)
V = TypeVar("V")  # base type of the product (ABC)


class RegistryFactory(Generic[K, V]):
    """Two-key class-level registry factory.

    Attributes
    ----------
    _registry:
        Mapping of ``(name, backend)`` → implementation class.
        Each subclass gets its own mapping via ``__init_subclass__``.
    """

    _registry: ClassVar[dict[tuple[Any, str], type]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give every concrete subclass its own isolated registry dict."""
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: K, backend: str, implementation: type[V]) -> None:
        """Register an implementation class under ``(name, backend)``.

        Parameters
        ----------
        name:
            Logical calculator identifier (e.g. ``"mean_calculator"``).
        backend:
            Runtime target: ``"polars"``, ``"spark"``, or ``"pandas"``.
        implementation:
            Concrete class that will be instantiated via :meth:`create`.

        Raises
        ------
        ValueError
            If the same ``(name, backend)`` pair is registered twice, which
            almost always indicates a copy-paste error in a factory module.
        """
        key = (name, backend)
        if key in cls._registry:
            raise ValueError(
                f"[{cls.__name__}] Duplicate registration for "
                f"name='{name}', backend='{backend}'. "
                f"Already registered as '{cls._registry[key].__name__}'."
            )
        cls._registry[key] = implementation

    # ------------------------------------------------------------------
    # Creation API
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: K, backend: str, **kwargs: Any) -> V:
        """Instantiate the registered implementation.

        Parameters
        ----------
        name:
            Logical calculator identifier.
        backend:
            One of ``"polars"``, ``"spark"``, ``"pandas"``.
        **kwargs:
            Forwarded as keyword arguments to the implementation's
            ``__init__``.

        Returns
        -------
        V
            A freshly constructed implementation instance.

        Raises
        ------
        KeyError
            If ``(name, backend)`` has not been registered.
        """
        key = (name, backend)
        if key not in cls._registry:
            registered = sorted(cls._registry.keys())
            raise KeyError(
                f"[{cls.__name__}] No implementation registered for "
                f"name='{name}', backend='{backend}'. "
                f"Registered keys: {registered}"
            )
        return cls._registry[key](**kwargs)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @classmethod
    def registered_names(cls) -> list[K]:
        """Return a sorted, deduplicated list of all registered calculator names."""
        return sorted({name for name, _ in cls._registry})

    @classmethod
    def registered_backends(cls, name: K) -> list[str]:
        """Return all backends available for a given calculator name."""
        return sorted(backend for (n, backend) in cls._registry if n == name)

    @classmethod
    def is_registered(cls, name: K, backend: str) -> bool:
        """Return ``True`` if ``(name, backend)`` is registered."""
        return (name, backend) in cls._registry
