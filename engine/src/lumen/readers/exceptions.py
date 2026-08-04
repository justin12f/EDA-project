"""Custom exceptions for the readers module."""

from __future__ import annotations


class ReaderError(Exception):
    """Base exception for reader operations.

    Raised when a read operation fails due to file I/O, parsing,
    or format incompatibility.
    """


class BackendNotSupportedError(ReaderError, ImportError):
    """Raised when a requested backend is not registered.

    Also inherits from ``ImportError`` so callers doing the standard
    "missing optional dependency" check (``except ImportError``) catch it
    directly — this is the common case when the backend is a real engine
    (e.g. ``spark``) that simply is not installed.

    Args:
        backend: The backend name that was requested.
        available: List of available backend names.
        reason: Optional override for the default message — used to give
            an actionable hint (e.g. how to install the missing package)
            instead of the generic "not supported" wording.
    """

    def __init__(
        self,
        backend: str,
        available: list[str],
        reason: str | None = None,
    ) -> None:
        self.backend = backend
        self.available = available
        message = reason or (
            f"Backend '{backend}' is not supported. "
            f"Available backends: {available}"
        )
        super().__init__(message)


class FileFormatError(ReaderError):
    """Raised when the file format does not match the expected extension.

    Args:
        file_path: Path to the file.
        expected_format: Expected file format/extension.
    """

    def __init__(self, file_path: str, expected_format: str) -> None:
        self.file_path = file_path
        self.expected_format = expected_format
        super().__init__(
            f"File '{file_path}' does not match expected format '{expected_format}'."
        )
