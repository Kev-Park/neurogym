"""Optional JSONL event instrumentation (storm-precursor forensics).

Zero behavioural effect. Records are written only when NGLLIB_EVENT_LOG is set
(a path template that may contain `{pid}` and `{host}`); the environment logs
resets and slow steps, a renderer logs whatever it wants attributed (glitches,
warm-context readiness). One stream per environment, so a reset event and the
glitches that preceded it can be reconstructed in order.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any


class EventLog:
    def __init__(self, path_template: str | None = None):
        self._path = (path_template if path_template is not None
                      else os.environ.get("NGLLIB_EVENT_LOG"))
        self._fh = None
        self._host: str | None = None
        self.episode = 0

    @property
    def enabled(self) -> bool:
        return bool(self._path)

    def emit(self, evt: str, **fields: Any) -> None:
        if not self._path:
            return
        try:
            if self._fh is None:
                self._host = socket.gethostname()
                self._fh = open(
                    self._path.format(pid=os.getpid(), host=self._host), "a", buffering=1)
            rec = {
                "evt": evt, "ts": time.time(), "mono": time.monotonic(),
                "pid": os.getpid(), "host": self._host,
                "episode": self.episode, **fields,
            }
            self._fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # instrumentation must never break the env
