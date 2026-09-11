"""Gate 2: `import ngllib` must work with no GL, no CloudVolume and no
Playwright driver present, and constructing either renderer must not pull
them in either. Heavy imports happen when a renderer is opened."""

from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCKED = ("moderngl", "cloudvolume", "playwright")

_SCRIPT = textwrap.dedent(f"""
    import sys

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {BLOCKED!r}:
                raise ImportError(f"blocked import of {{name}} at import time")
            return None

    sys.meta_path.insert(0, _Block())
    import ngllib
    from ngllib import ChromeRenderer, Environment, SimulatorRenderer
    ChromeRenderer()
    sim = SimulatorRenderer(left_pane=True)
    env = Environment(backend=sim, orientation="euler")
    assert env.observation_space["image"].shape == (450, 900, 3)
    print("IMPORT-OK")
""")


def test_import_and_construct_without_heavy_deps():
    out = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "IMPORT-OK" in out.stdout
