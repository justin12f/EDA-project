"""Custom exceptions for the readers module."""

from __future__ import annotations


class ReaderError(Exception):
    """Base exception for reader operations.

    Raised when a read operation fails due to file I/O, parsing,
    or format incompatibility.
    """


class BackendNotSupportedError(ReaderError):
    """Raised when a requested backend is not registered.

    Args:
        backend: The backend name that was requested.
        available: List of available backend names.
    """

    def __init__(self, backend: str, available: list[str]) -> None:
        self.backend = backend
        self.available = available
        super().__init__(
            f"Backend '{backend}' is not supported. "
            f"Available backends: {available}"
        )


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
