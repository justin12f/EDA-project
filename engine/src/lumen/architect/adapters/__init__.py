"""Source adapters — one interface, any format."""

from lumen.architect.adapters.base import (
    DiscoveredColumn,
    DiscoveredStructure,
    DiscoveredTable,
    SourceAdapter,
)
from lumen.architect.adapters.file import FileAdapter
from lumen.architect.adapters.postgres import PostgresAdapter

__all__ = [
    "DiscoveredColumn",
    "DiscoveredStructure",
    "DiscoveredTable",
    "FileAdapter",
    "PostgresAdapter",
    "SourceAdapter",
]
