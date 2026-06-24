"""TCP `SocketTransport` — 4-byte big-endian length-prefix framing.

Server bind happens at construction; accept() and client connect() are
deferred to first send/recv (with retry until timeout), so server/client
startup order doesn't race.
"""

from __future__ import annotations

import socket as _socket
import struct
import time

from ...errors import ConnectionLost, TransportError

_HEADER_FMT = "!I"
_HEADER_LEN = struct.calcsize(_HEADER_FMT)


class SocketTransport:
    """TCP-backed `Transport`. Use `.server(...)` / `.client(...)` to construct."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        role: str,
        timeout: float = 60.0,
        connect_retry_interval: float = 0.05,
        backlog: int = 1,
    ):
        if role not in ("server", "client"):
            raise ValueError(f"role must be 'server' or 'client'; got {role!r}")
        self.host = host
        self.port = port
        self.role = role
        self.timeout = timeout
        self.connect_retry_interval = connect_retry_interval

        self._listener: _socket.socket | None = None
        self._conn: _socket.socket | None = None

        if role == "server":
            self._listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            self._listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            try:
                self._listener.bind((host, port))
                self._listener.listen(backlog)
            except OSError as e:
                self._listener.close()
                self._listener = None
                raise TransportError(
                    f"could not bind {host}:{port}: {e}"
                ) from e
            self._listener.settimeout(timeout)

    # -- factories ------------------------------------------------------------

    @classmethod
    def server(
        cls,
        host: str = "0.0.0.0",
        port: int = 5555,
        *,
        timeout: float = 60.0,
        backlog: int = 1,
    ) -> "SocketTransport":
        """Construct a server-side transport bound on `host:port`."""
        return cls(host=host, port=port, role="server", timeout=timeout, backlog=backlog)

    @classmethod
    def client(
        cls,
        host: str = "localhost",
        port: int = 5555,
        *,
        timeout: float = 60.0,
        connect_retry_interval: float = 0.05,
    ) -> "SocketTransport":
        """Construct a client-side transport that will connect on first send/recv."""
        return cls(
            host=host,
            port=port,
            role="client",
            timeout=timeout,
            connect_retry_interval=connect_retry_interval,
        )

    # -- connection management (deferred) -------------------------------------

    def _ensure_conn(self) -> None:
        if self._conn is not None:
            return
        if self.role == "server":
            try:
                conn, _addr = self._listener.accept()
            except _socket.timeout:
                raise TransportError(
                    f"no client connected within {self.timeout}s"
                ) from None
            except OSError as e:
                raise TransportError(f"accept failed: {e}") from e
            conn.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            self._conn = conn
        else:  # client
            deadline = time.monotonic() + self.timeout
            last_err: Exception | None = None
            while True:
                sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                try:
                    sock.connect((self.host, self.port))
                    self._conn = sock
                    return
                except (ConnectionRefusedError, OSError) as e:
                    last_err = e
                    sock.close()
                    if time.monotonic() > deadline:
                        raise TransportError(
                            f"could not connect to {self.host}:{self.port} "
                            f"within {self.timeout}s: {last_err}"
                        ) from last_err
                    time.sleep(self.connect_retry_interval)

    # -- Transport interface --------------------------------------------------

    def send(self, message: bytes) -> None:
        self._ensure_conn()
        header = struct.pack(_HEADER_FMT, len(message))
        try:
            self._conn.sendall(header + message)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise ConnectionLost(f"connection lost during send: {e}") from e

    def recv(self) -> bytes:
        self._ensure_conn()
        try:
            header = self._recv_exactly(_HEADER_LEN)
            (length,) = struct.unpack(_HEADER_FMT, header)
            return self._recv_exactly(length)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise ConnectionLost(f"connection lost during recv: {e}") from e

    def _recv_exactly(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionLost("peer closed connection mid-message")
            buf.extend(chunk)
        return bytes(buf)

    def close(self) -> None:
        for sock in (self._conn, self._listener):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._conn = None
        self._listener = None
