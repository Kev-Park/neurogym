"""CAVE credentials, shared by both renderers.

One secret serves the whole stack, but the two backends consume it differently:
Chrome needs it seeded into `localStorage`, while CloudVolume reads it only from
a FILE -- `cave_credentials()` looks at `$CLOUD_VOLUME_DIR/secrets/cave-secret.json`
and the legacy `chunkedgraph-secret.json`, with no env var for the token itself.
The simulator's fetch workers are spawned processes, so a file is the only way
to reach them.

So a token configured inline (`config.json`'s `cave_token`) is materialised into
a private temporary secrets directory and `CLOUD_VOLUME_DIR` is pointed at it.
Any secrets the user already has (google-secret.json, aws, ...) are linked into
that directory first, so relocating it cannot cost them other credentials.

No token is needed for public data: `ensure_cave_secret` is only called when a
source actually asks for one.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path

from .errors import RendererError

CAVE_SECRET_PATH = "~/.cloudvolume/secrets/cave-secret.json"
_MATERIALIZED: Path | None = None


def cave_token_from_config(config: dict) -> str | None:
    """`cave_token` (inline) or `cave_secret` (path) from a config, if set."""
    token = (config.get("cave_token") or "").strip()
    if token:
        return token
    path = (config.get("cave_secret") or "").strip()
    return read_cave_token(path) if path else None


def read_cave_token(path: str | None = None) -> str:
    """The token CloudVolume would read, from `path` or its default location."""
    f = Path(os.path.expanduser(path or CAVE_SECRET_PATH))
    if not f.is_file():
        raise RendererError(
            f"no CAVE token at {f}: set `cave_token` in your config, point "
            "`cave_secret` at a secret file, or mint one with "
            "caveclient's auth.setup_token()")
    data = json.loads(f.read_text())
    token = data.get("token") or data.get("middle_auth_token")
    if not token:
        raise RendererError(f"{f} has no 'token' field")
    return token


def cloudvolume_secret_exists() -> bool:
    """Does CloudVolume already have a CAVE secret where it looks?"""
    root = Path(os.environ.get("CLOUD_VOLUME_DIR", os.path.expanduser("~/.cloudvolume")))
    return any((root / "secrets" / name).is_file()
               for name in ("cave-secret.json", "chunkedgraph-secret.json"))


def ensure_cave_secret(token: str) -> Path:
    """Make `token` visible to CloudVolume, including in spawned workers.

    Writes a private secrets directory and points `CLOUD_VOLUME_DIR` at it, so
    every process this one spawns inherits it. Idempotent per process; the
    directory is removed at exit.
    """
    global _MATERIALIZED
    if _MATERIALIZED is not None:
        return _MATERIALIZED

    tmp = Path(tempfile.mkdtemp(prefix="ngllib-secrets-"))
    secrets = tmp / "secrets"
    secrets.mkdir(mode=0o700)
    # Carry over whatever else the user has, so redirecting CLOUD_VOLUME_DIR
    # cannot lose them google-secret.json or aws credentials.
    existing = Path(os.environ.get("CLOUD_VOLUME_DIR",
                                   os.path.expanduser("~/.cloudvolume"))) / "secrets"
    if existing.is_dir():
        for f in existing.iterdir():
            if f.is_file() and f.name not in ("cave-secret.json", "chunkedgraph-secret.json"):
                shutil.copy2(f, secrets / f.name)
                os.chmod(secrets / f.name, 0o600)

    out = secrets / "cave-secret.json"
    out.write_text(json.dumps({"token": token}))
    os.chmod(out, 0o600)
    os.environ["CLOUD_VOLUME_DIR"] = str(tmp)
    _MATERIALIZED = tmp
    atexit.register(lambda: shutil.rmtree(tmp, ignore_errors=True))
    return tmp
