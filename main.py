"""Minimal demo of the new Gymnasium-compliant ngllib.Environment API.

Drives the env through 10 steps with a hand-rolled `edit_state` action that pans
the camera and rotates slightly each step, and saves the resulting screenshots
under `screenshots/`.
"""

import os

import numpy as np
from PIL import Image

from ngllib import Environment


def main() -> None:
    # No factories supplied — defaults give reward=0, terminated=False (TimeLimit
    # would handle truncation in real use). Euler orientation matches the demo
    # action shape below.
    env = Environment(
        headless=True,
        orientation="euler",
        verbose=False,
        # Self-healing knobs are on by default; disable here to keep the demo simple.
        browser_restart_every=None,
    )

    obs, info = env.reset(seed=0)
    print(f"Reset OK. position={obs['position'].round(1).tolist()}, image shape={obs['image'].shape}")

    os.makedirs("screenshots", exist_ok=True)

    # An `edit_state` action that pans +10 in X and rotates a touch each step.
    action = {
        "action_type": 3,  # 0=left, 1=right, 2=double, 3=edit_state
        "mouse_xy": np.array([100, 100], dtype=np.float32),
        "modifiers": np.array([0, 0, 0], dtype=np.int8),
        "delta_pos": np.array([10, 0, 0], dtype=np.float32),
        "delta_xs_scale": np.array([0], dtype=np.float32),
        "delta_orient": np.array([0.2, 0, 0], dtype=np.float32),
        "delta_proj_scale": np.array([2000], dtype=np.float32),
    }

    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(action)
        Image.fromarray(obs["image"]).save(f"screenshots/step_{i:03d}.png")
        print(
            f"Step {i}: pos={obs['position'].round(1).tolist()}, "
            f"reward={reward}, terminated={terminated}"
        )

    env.close()
    print("Done. Screenshots saved to ./screenshots/")


if __name__ == "__main__":
    main()
