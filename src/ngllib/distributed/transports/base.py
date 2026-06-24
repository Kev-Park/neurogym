"""Transport Protocol — dumb byte pipe. Serialization lives one layer up."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Minimal byte-pipe interface — `send(bytes)`, `recv() -> bytes`, `close()`.

    Concrete implementations are expected to expose `server(...)` / `client(...)`
    classmethod factories so the role is unambiguous at the call site.
    """

    def send(self, message: bytes) -> None:
        """Send `message` to the peer. Blocks until the bytes are queued."""
        ...

    def recv(self) -> bytes:
        """Block until a message arrives from the peer; return its payload bytes.

        Raises `ngllib.errors.ConnectionLost` if the peer disconnects mid-stream,
        `ngllib.errors.TransportError` for other transport-level failures.
        """
        ...

    def close(self) -> None:
        """Release transport resources. Must be idempotent."""
        ...
