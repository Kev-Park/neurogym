"""Typed exception hierarchy for ngllib.

Consumers can `except ngllib.errors.BrowserError` (or any specific subclass) to
handle failures precisely. In distributed mode, `serve` ships an error envelope
across the wire and `RemoteEnv` re-raises the same class locally, so the same
`except` clauses work for both local and remote envs.
"""


class NgllibError(Exception):
    """Base class for all ngllib-raised exceptions."""


class BrowserError(NgllibError):
    """Chromium failed to launch, hung, or crashed."""


class ProviderError(NgllibError):
    """A reset_state_provider, reward_factory, or termination_factory raised."""


class ProtocolError(NgllibError):
    """A malformed message was received on the wire."""


class TransportError(NgllibError):
    """The transport layer (socket / filesystem) failed."""


class ConnectionLost(TransportError):
    """The peer disconnected mid-stream."""


class HandshakeFailed(TransportError):
    """The connect-time handshake (spaces exchange) failed or was malformed."""
