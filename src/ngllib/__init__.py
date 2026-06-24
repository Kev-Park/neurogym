"""ngllib — Gymnasium-compliant RL environment driving Neuroglancer via Playwright."""

from . import distributed
from .distributed.remote import RemoteEnv
from .environment import Environment, RewardFactory, TerminationFactory
from .errors import (
    BrowserError,
    ConnectionLost,
    HandshakeFailed,
    NgllibError,
    ProtocolError,
    ProviderError,
    TransportError,
)
from .providers import NglState, StateProvider

__all__ = [
    "Environment",
    "RemoteEnv",
    "StateProvider",
    "NglState",
    "RewardFactory",
    "TerminationFactory",
    "NgllibError",
    "BrowserError",
    "ProviderError",
    "ProtocolError",
    "TransportError",
    "ConnectionLost",
    "HandshakeFailed",
    "distributed",
]

import gymnasium as _gym

_ENV_ID = "Neuroglancer-v0"
if _ENV_ID not in _gym.envs.registry:
    _gym.register(
        id=_ENV_ID,
        entry_point="ngllib:Environment",
        max_episode_steps=300,
    )
