"""File-swap IPC over a shared directory pair (useful when TCP is awkward).

Mechanism: single-slot rename swap, inode-preserving. Each direction uses
two fixed filenames in the write_dir — `_pending.bin` (writer scratch) and
`_ready.bin` (atomic-rename target the reader watches). After the first
cycle one inode per direction is reused for every message. The reader
watches `_ready` only, so a half-written `_pending` is invisible.
"""

from __future__ import annotations

import os
import time

from ...errors import TransportError

_PENDING = "_pending.bin"
_READY = "_ready.bin"


class FilesystemTransport:
    """File-swap `Transport`. Use `.server(...)` / `.client(...)` to construct."""

    def __init__(
        self,
        *,
        read_dir: str,
        write_dir: str,
        timeout: float = 60.0,
        poll_interval: float = 0.01,
        cleanup_on_init: bool = False,
    ):
        self.read_dir = read_dir
        self.write_dir = write_dir
        self.timeout = timeout
        self.poll_interval = poll_interval

        os.makedirs(self.read_dir, exist_ok=True)
        os.makedirs(self.write_dir, exist_ok=True)

        # Fixed paths used for every message — only 2 files per direction, ever.
        self._write_pending = os.path.join(self.write_dir, _PENDING)
        self._write_ready = os.path.join(self.write_dir, _READY)
        self._read_pending = os.path.join(self.read_dir, _PENDING)
        self._read_ready = os.path.join(self.read_dir, _READY)

        if cleanup_on_init:
            for path in (
                self._write_pending,
                self._write_ready,
                self._read_pending,
                self._read_ready,
            ):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # -- factories ------------------------------------------------------------

    @classmethod
    def server(
        cls,
        action_dir: str,
        obs_dir: str,
        *,
        timeout: float = 60.0,
        poll_interval: float = 0.01,
        cleanup_on_init: bool = False,
    ) -> "FilesystemTransport":
        """Server reads actions, writes observations."""
        return cls(
            read_dir=action_dir,
            write_dir=obs_dir,
            timeout=timeout,
            poll_interval=poll_interval,
            cleanup_on_init=cleanup_on_init,
        )

    @classmethod
    def client(
        cls,
        action_dir: str,
        obs_dir: str,
        *,
        timeout: float = 60.0,
        poll_interval: float = 0.01,
        cleanup_on_init: bool = False,
    ) -> "FilesystemTransport":
        """Client writes actions, reads observations."""
        return cls(
            read_dir=obs_dir,
            write_dir=action_dir,
            timeout=timeout,
            poll_interval=poll_interval,
            cleanup_on_init=cleanup_on_init,
        )

    # -- Transport interface --------------------------------------------------

    def send(self, message: bytes) -> None:
        try:
            with open(self._write_pending, "wb") as f:
                f.write(message)
            os.replace(self._write_pending, self._write_ready)
        except OSError as e:
            raise TransportError(f"filesystem send failed: {e}") from e

    def recv(self) -> bytes:
        # Try-and-handle (not exists-then-open) to avoid TOCTOU and the Windows
        # sharing-violation race where `open` briefly fails on a just-renamed file.
        deadline = time.monotonic() + self.timeout
        data: bytes | None = None
        while data is None:
            try:
                with open(self._read_ready, "rb") as f:
                    data = f.read()
            except (FileNotFoundError, PermissionError):
                pass
            except OSError as e:
                raise TransportError(
                    f"filesystem recv failed reading {self._read_ready}: {e}"
                ) from e
            if data is None:
                if time.monotonic() > deadline:
                    raise TransportError(
                        f"timeout waiting {self.timeout}s for {self._read_ready}"
                    )
                time.sleep(self.poll_interval)

        # Rename _ready -> _pending so the peer's next send reuses the inode.
        # On Windows the rename may briefly hit a sharing violation; retry and
        # fall back to remove (costs one fresh inode next cycle).
        rename_deadline = time.monotonic() + 1.0
        while True:
            try:
                os.replace(self._read_ready, self._read_pending)
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if time.monotonic() > rename_deadline:
                    try: os.remove(self._read_ready)
                    except OSError: pass
                    break
                time.sleep(self.poll_interval)
            except OSError:
                try: os.remove(self._read_ready)
                except OSError: pass
                break
        return data

    def close(self) -> None:
        # Slot files survive close() so the peer can keep using the directory pair.
        pass
