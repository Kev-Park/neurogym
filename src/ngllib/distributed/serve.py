"""Server runner — dispatches Gym-shaped messages from a Transport to a local `Environment`.

Wire protocol (pickled per-message, framed by the transport):

    client -> server                             server -> client
    {"cmd": "handshake"}                         {"observation_space", "action_space"}
    {"cmd": "reset", "seed", "options"}          {"obs", "info"}
    {"cmd": "step",  "action"}                   {"obs", "reward", "terminated", "truncated", "info"}
    {"cmd": "close"}                             {"ok": True}

Error envelope (server -> client):
    {"error": {"type", "module", "message", "traceback"}}
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import sys
import traceback
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from PIL import Image

from .. import errors
from .transports import Transport

logger = logging.getLogger(__name__)

WireImageFormat = Literal["jpeg", "png", "raw"]


# ============================================================================
# Programmatic entry point
# ============================================================================


def serve(
    env: gym.Env,
    transport: Transport,
    *,
    wire_image_format: WireImageFormat = "jpeg",
) -> None:
    """Run the dispatch loop until the client closes or disconnects.

    JPEG-encodes obs["image"] by default (~20-80x bandwidth savings vs raw uint8).
    Always closes env and transport on exit.
    """
    if wire_image_format not in ("jpeg", "png", "raw"):
        raise ValueError(
            f"wire_image_format must be 'jpeg' | 'png' | 'raw'; got {wire_image_format!r}"
        )

    logger.info("ngllib.distributed.serve loop starting (wire_image_format=%s)", wire_image_format)
    try:
        while True:
            try:
                raw = transport.recv()
            except errors.ConnectionLost:
                logger.info("client disconnected")
                break
            except errors.TransportError as e:
                logger.warning("transport error during recv: %s", e)
                break

            try:
                msg = pickle.loads(raw)
            except Exception as e:
                _send_error(transport, errors.ProtocolError(f"could not unpickle message: {e}"))
                continue

            cmd = msg.get("cmd") if isinstance(msg, dict) else None

            try:
                if cmd == "handshake":
                    response = _handle_handshake(env)
                elif cmd == "reset":
                    response = _handle_reset(env, msg, wire_image_format)
                elif cmd == "step":
                    response = _handle_step(env, msg, wire_image_format)
                elif cmd == "close":
                    _send_response(transport, {"ok": True})
                    logger.info("client requested close")
                    break
                else:
                    raise errors.ProtocolError(f"unknown cmd: {cmd!r}")
            except Exception as e:
                # Surface env-side errors to the client; don't kill the server.
                _send_error(transport, e)
                continue

            _send_response(transport, response)
    finally:
        try:
            env.close()
        except Exception as e:
            logger.warning("env.close raised during shutdown: %s", e)
        try:
            transport.close()
        except Exception as e:
            logger.warning("transport.close raised during shutdown: %s", e)
        logger.info("ngllib.distributed.serve loop exited")


# ============================================================================
# Per-command handlers
# ============================================================================


def _handle_handshake(env: gym.Env) -> dict[str, Any]:
    return {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }


def _handle_reset(env: gym.Env, msg: dict[str, Any], wire_image_format: str) -> dict[str, Any]:
    obs, info = env.reset(seed=msg.get("seed"), options=msg.get("options"))
    if wire_image_format != "raw":
        obs = _encode_image_for_wire(obs, wire_image_format)
    return {"obs": obs, "info": info}


def _handle_step(env: gym.Env, msg: dict[str, Any], wire_image_format: str) -> dict[str, Any]:
    obs, reward, terminated, truncated, info = env.step(msg["action"])
    if wire_image_format != "raw":
        obs = _encode_image_for_wire(obs, wire_image_format)
    return {
        "obs": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


# ============================================================================
# Wire helpers
# ============================================================================


def _send_response(transport: Transport, payload: dict[str, Any]) -> None:
    try:
        transport.send(pickle.dumps(payload))
    except Exception as e:
        logger.error("failed to send response: %s", e)
        raise


def _send_error(transport: Transport, exc: BaseException) -> None:
    payload = {
        "error": {
            "type": exc.__class__.__name__,
            "module": exc.__class__.__module__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    }
    try:
        transport.send(pickle.dumps(payload))
    except Exception as e:
        logger.error("failed to send error envelope (%s): %s", exc, e)


def _encode_image_for_wire(obs: Any, fmt: str) -> Any:
    """Encode obs["image"] uint8 ndarray, wrapped in a `{"__wire_image__", "data"}` marker."""
    if not isinstance(obs, dict) or "image" not in obs:
        return obs
    image = obs["image"]
    if not isinstance(image, np.ndarray):
        return obs
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    if fmt == "jpeg":
        pil.save(buf, format="JPEG", quality=85)
    elif fmt == "png":
        pil.save(buf, format="PNG")
    else:
        return obs  # raw — no encoding
    return {**obs, "image": {"__wire_image__": fmt, "data": buf.getvalue()}}


# ============================================================================
# CLI
# ============================================================================


def _parse_size(s: str | None) -> tuple[int, int] | None:
    if s is None:
        return None
    try:
        w, h = s.lower().split("x")
        return (int(w), int(h))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"size must be 'WIDTHxHEIGHT' (e.g. 1800x900); got {s!r}"
        ) from e


def _add_env_kwargs(parser: argparse.ArgumentParser) -> None:
    """Add a subset of `Environment` constructor kwargs to a subparser."""
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--renderer", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--orientation", choices=["quaternion", "euler"], default="quaternion")
    parser.add_argument("--window-size", default="1800x900", help="WIDTHxHEIGHT (default 1800x900)")
    parser.add_argument("--image-size", default=None, help="WIDTHxHEIGHT; default = natural pane size")
    parser.add_argument("--screenshot-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--no-right-pane", dest="right_pane", action="store_false", default=True)
    parser.add_argument("--left-pane", dest="left_pane", action="store_true", default=False)
    parser.add_argument("--draw-mouse", action="store_true", default=False)
    parser.add_argument("--browser-restart-every", type=int, default=90,
                        help="Restart browser every N episodes; 0 to disable.")
    parser.add_argument("--retry-on-reset", type=int, default=3)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--wire-image-format", choices=["jpeg", "png", "raw"], default="jpeg")
    parser.add_argument("--verbose", action="store_true")


def _build_env(args: argparse.Namespace):
    from ..environment import Environment
    return Environment(
        headless=args.headless,
        renderer=args.renderer,
        orientation=args.orientation,
        window_size=_parse_size(args.window_size),
        image_size=_parse_size(args.image_size),
        screenshot_format=args.screenshot_format,
        right_pane=args.right_pane,
        left_pane=args.left_pane,
        draw_mouse=args.draw_mouse,
        browser_restart_every=args.browser_restart_every or None,
        retry_on_reset=args.retry_on_reset,
        config_path=args.config_path,
        verbose=args.verbose,
    )


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="python -m ngllib.distributed.serve",
        description="Run an ngllib.Environment server.",
    )
    sub = parser.add_subparsers(dest="transport", required=True)

    s = sub.add_parser("socket", help="TCP socket transport")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=5555)
    s.add_argument("--timeout", type=float, default=600.0)
    _add_env_kwargs(s)

    f = sub.add_parser("filesystem", help="Filesystem (file-swap) transport")
    f.add_argument("--action-dir", required=True)
    f.add_argument("--obs-dir", required=True)
    f.add_argument("--timeout", type=float, default=600.0)
    f.add_argument("--poll-interval", type=float, default=0.01)
    f.add_argument("--cleanup-on-init", action="store_true")
    _add_env_kwargs(f)

    args = parser.parse_args(argv)
    env = _build_env(args)

    if args.transport == "socket":
        from .transports import SocketTransport
        transport = SocketTransport.server(host=args.host, port=args.port, timeout=args.timeout)
        logger.info("serving on tcp://%s:%d", args.host, args.port)
    elif args.transport == "filesystem":
        from .transports import FilesystemTransport
        transport = FilesystemTransport.server(
            action_dir=args.action_dir,
            obs_dir=args.obs_dir,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            cleanup_on_init=args.cleanup_on_init,
        )
        logger.info("serving via filesystem: action=%s, obs=%s", args.action_dir, args.obs_dir)
    else:
        parser.error(f"unknown transport: {args.transport}")
        return 2

    try:
        serve(env, transport, wire_image_format=args.wire_image_format)
    except KeyboardInterrupt:
        logger.info("interrupted by user")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
