"""Reusable (key, backend) registry factory pattern."""

from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar

from lumen.core.backend import Backend, validate_backend

K = TypeVar("K")
V = TypeVar("V")


class RegistryFactory(Generic[K, V]):
    """Base factory mapping ``(key, backend)`` to a registered implementation."""

    _registry: ClassVar[dict[tuple[Any, str], type[V]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_registry" not in cls.__dict__:
            cls._registry = {}

    @classmethod
    def register(cls, key: K, backend: Backend | str, implementation: type[V]) -> None:
        validate_backend(str(backend))
        cls._registry[(key, str(backend))] = implementation

    @classmethod
    def get_class(cls, key: K, backend: Backend | str) -> type[V]:
        validate_backend(str(backend))
        impl = cls._registry.get((key, str(backend)))
        if impl is None:
            available = sorted(
                {f"{k} ({b})" for (k, b) in cls._registry if k == key}
            )
            raise ValueError(
                f"No registration for key={key!r}, backend={backend!r}. "
                f"Available for this key: {available or 'none'}"
            )
        return impl

    @classmethod
    def create(cls, key: K, backend: Backend | str, *args: Any, **kwargs: Any) -> V:
        impl_cls = cls.get_class(key, backend)
        return impl_cls(*args, **kwargs)

    @classmethod
    def is_registered(cls, key: K, backend: Backend | str) -> bool:
        return (key, str(backend)) in cls._registry

    @classmethod
    def registered_keys(cls, backend: Backend | str | None = None) -> list[Any]:
        if backend is None:
            return sorted({k for (k, _) in cls._registry})
        return sorted(k for (k, b) in cls._registry if b == str(backend))
