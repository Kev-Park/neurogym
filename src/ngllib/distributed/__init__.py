"""Distributed-RL extension — used when env and agent live in separate processes."""

from . import transports
from .remote import RemoteEnv

# `serve` is intentionally NOT eagerly imported here — doing so makes runpy emit
# a RuntimeWarning when invoked as `python -m ngllib.distributed.serve`.
# Programmatic users: `from ngllib.distributed.serve import serve`.

__all__ = ["RemoteEnv", "transports"]
