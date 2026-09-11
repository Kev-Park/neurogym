"""ngllib — Gymnasium environment over Neuroglancer, with pluggable renderers."""

from . import distributed
from .chrome import ChromeRenderer
from .dataset import DatasetSpec
from .distributed.remote import RemoteEnv
from .environment import Environment, RewardFactory, TerminationFactory
from .errors import (
    BrowserError,
    ConnectionLost,
    HandshakeFailed,
    NgllibError,
    ProtocolError,
    ProviderError,
    RendererError,
    TransportError,
)
from .providers import NglState, StateProvider
from .renderer import PaneLayout, Renderer
from .simulator import SimulatorRenderer

__all__ = [
    "Environment",
    "Renderer",
    "ChromeRenderer",
    "SimulatorRenderer",
    "PaneLayout",
    "DatasetSpec",
    "RemoteEnv",
    "StateProvider",
    "NglState",
    "RewardFactory",
    "TerminationFactory",
    "NgllibError",
    "RendererError",
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
