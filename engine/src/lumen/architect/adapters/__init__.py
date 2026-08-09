"""Source adapters — one interface, any format."""

from lumen.architect.adapters.base import (
    DiscoveredColumn,
    DiscoveredStructure,
    DiscoveredTable,
    SourceAdapter,
)
from lumen.architect.adapters.file import FileAdapter

__all__ = [
    "DiscoveredColumn",
    "DiscoveredStructure",
    "DiscoveredTable",
    "FileAdapter",
    "SourceAdapter",
]
