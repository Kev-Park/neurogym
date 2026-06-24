"""`RemoteEnv` — client-side Gym proxy over a Transport.

Constructor handshakes to learn the server's spaces. Server-side exceptions
are re-raised class-faithful on the client.
"""

from __future__ import annotations

import importlib
import io
import logging
import pickle
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from .. import errors
from .transports import Transport

logger = logging.getLogger(__name__)


class RemoteEnv(gym.Env):
    """Gym-compliant proxy for an `Environment` running on the other end of a Transport."""

    metadata = {"render_modes": []}

    def __init__(self, transport: Transport):
        super().__init__()
        self.transport = transport
        self._closed = False

        # Handshake — learn the server's spaces and adopt them as our own.
        self._send({"cmd": "handshake"})
        response = self._recv()
        self.observation_space = response["observation_space"]
        self.action_space = response["action_space"]

    # =========================================================================
    # Gymnasium public API
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._send({"cmd": "reset", "seed": seed, "options": options})
        response = self._recv()
        return _decode_image_from_wire(response["obs"]), response["info"]

    def step(self, action):
        self._send({"cmd": "step", "action": action})
        response = self._recv()
        return (
            _decode_image_from_wire(response["obs"]),
            response["reward"],
            response["terminated"],
            response["truncated"],
            response["info"],
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._send({"cmd": "close"})
            self._recv()  # wait for ack
        except Exception as e:
            logger.debug("close ack failed (server may have already exited): %s", e)
        try:
            self.transport.close()
        except Exception as e:
            logger.debug("transport.close raised: %s", e)

    # =========================================================================
    # Internal wire helpers
    # =========================================================================

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            self.transport.send(pickle.dumps(payload))
        except (errors.ConnectionLost, errors.TransportError):
            raise
        except Exception as e:
            raise errors.ProtocolError(f"failed to send {payload.get('cmd')!r}: {e}") from e

    def _recv(self) -> dict[str, Any]:
        raw = self.transport.recv()
        try:
            msg = pickle.loads(raw)
        except Exception as e:
            raise errors.ProtocolError(f"could not unpickle server response: {e}") from e
        if isinstance(msg, dict) and "error" in msg:
            err = msg["error"]
            exc_cls = _resolve_exception_class(err.get("type", "NgllibError"), err.get("module", ""))
            server_tb = err.get("traceback", "")
            msg_text = err.get("message", "<no message>")
            raise exc_cls(
                f"server-side error: {msg_text}"
                + (f"\n--- server traceback ---\n{server_tb}" if server_tb else "")
            )
        return msg


def _resolve_exception_class(type_name: str, module_name: str) -> type[BaseException]:
    """ngllib.errors first, then the original module, then NgllibError as fallback."""
    cls = getattr(errors, type_name, None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls
    if module_name:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, type_name, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                return cls
        except Exception:
            pass
    return errors.NgllibError


def _decode_image_from_wire(obs: Any) -> Any:
    """Decode `obs["image"]` from a `{"__wire_image__", "data"}` marker back to numpy."""
    if not isinstance(obs, dict) or "image" not in obs:
        return obs
    image = obs["image"]
    if isinstance(image, dict) and image.get("__wire_image__") in ("jpeg", "png"):
        pil = Image.open(io.BytesIO(image["data"])).convert("RGB")
        return {**obs, "image": np.asarray(pil)}
    return obs
