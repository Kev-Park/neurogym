"""Transports — pluggable byte-pipe implementations.

Construct via `.server(...)` / `.client(...)` classmethods. A new transport
medium is a class with `send` / `recv` / `close` (`Transport` is a Protocol).
"""

from .base import Transport
from .filesystem import FilesystemTransport
from .socket import SocketTransport

__all__ = ["Transport", "SocketTransport", "FilesystemTransport"]
