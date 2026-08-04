"""Unified statistics factory registry.

Keys are `{domain}.{calculator}`, so `StatisticsRegistry.create("descriptive.
central_tendency_calculator", "polars")` reaches one implementation out of
twelve domains × three backends.

**Discovery is tolerant on purpose.** A domain whose factory has not been written
yet, or one whose backend needs an optional dependency that is not installed,
is skipped and recorded — it does not take the other eleven domains down with
it. The previous hard-coded version failed the entire registry on the first
missing module, which meant a missing `graphs.factory` disabled descriptive
statistics for every caller.

Anything skipped is visible: `unavailable_domains()` reports what was not loaded
and why, and `/v1/config` surfaces it.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from lumen.core.abstract_factory import RegistryFactory
from lumen.core.backend import BACKENDS


class StatisticsRegistry(RegistryFactory[str, Any]):
    """Keys: `{domain}.{calculator}`"""


DOMAIN_FACTORIES: dict[str, type] = {}
_UNAVAILABLE: dict[str, str] = {}
_LOADED = False


def _domain_names() -> list[str]:
    """Every package under `lumen.statistics` that is a domain, not a helper."""
    import lumen.statistics as package

    return sorted(
        module.name
        for module in pkgutil.iter_modules(package.__path__)
        if module.ispkg and module.name != "core"
    )


def _register_domain(domain: str) -> None:
    module = importlib.import_module(f"lumen.statistics.{domain}.factory")

    factory = next(
        (
            candidate
            for name, candidate in vars(module).items()
            if isinstance(candidate, type)
            and issubclass(candidate, RegistryFactory)
            and candidate is not RegistryFactory
            and name.endswith("StatisticsFactory")
        ),
        None,
    )
    if factory is None:
        raise AttributeError(f"{module.__name__} defines no *StatisticsFactory")

    DOMAIN_FACTORIES[domain] = factory
    for key in factory.registered_keys():
        for backend in BACKENDS:
            if factory.is_registered(key, backend):
                StatisticsRegistry.register(
                    f"{domain}.{key}", backend, factory.get_class(key, backend)
                )


def _register_all(force: bool = False) -> None:
    """Load every domain that can be loaded. Idempotent."""
    global _LOADED
    if _LOADED and not force:
        return

    DOMAIN_FACTORIES.clear()
    _UNAVAILABLE.clear()

    for domain in _domain_names():
        try:
            _register_domain(domain)
        except Exception as exc:  # noqa: BLE001 — one bad domain must not sink the rest
            _UNAVAILABLE[domain] = f"{type(exc).__name__}: {exc}"

    _LOADED = True


def unavailable_domains() -> dict[str, str]:
    """Domains that failed to load, mapped to why. Empty when everything loaded."""
    _register_all()
    return dict(_UNAVAILABLE)


def available_domains() -> list[str]:
    _register_all()
    return sorted(DOMAIN_FACTORIES)


# Populate on import so callers can use the registry without a setup step.
_register_all()
