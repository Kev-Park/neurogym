"""
Server side of cross-node latency test.
Run this on the current node BEFORE submitting client.sh via sbatch.

  python server_latency.py
"""

import os
import sys
import time
import shutil

from ngllib import Environment
from ngllib.utils.Communication import FilesystemProtocol, NGLServer

# ── Config (must match client_latency.py) ───────────────────────────────
NUM_WARMUP = 2
NUM_ROUNDS = 10
IPC_DIR = "test_ipc_cross"
READY_SIGNAL = os.path.join(IPC_DIR, "server_ready")

TOTAL_STEPS = NUM_WARMUP + NUM_ROUNDS


def custom_reward(state, action, prev_state):
    return 0.0, False


if __name__ == "__main__":
    # Clean up previous run
    if os.path.exists(IPC_DIR):
        shutil.rmtree(IPC_DIR)
    os.makedirs(IPC_DIR, exist_ok=True)

    print("[SERVER] Starting Chrome ...")
    env = Environment(
        headless=True, config_path="config.json",
        verbose=False, reward_function=custom_reward,
    )
    env.start_session(euler_angles=True, fast=True)
    time.sleep(2)  # let initial page load finish

    protocol = FilesystemProtocol(
        action_file_path=os.path.join(IPC_DIR, "actions"),
        observation_file_path=os.path.join(IPC_DIR, "observations"),
        timeout=50_000_000,  # very high — client might be queued in sbatch
    )
    server = NGLServer(protocol, env)

    # Signal to the client that we are ready
    with open(READY_SIGNAL, "w") as f:
        f.write("ready")
    print(f"[SERVER] Ready. Wrote signal: {READY_SIGNAL}")
    print(f"[SERVER] Waiting for {TOTAL_STEPS} actions from client ...")

    for i in range(TOTAL_STEPS):
        server.process_actions()
        print(f"[SERVER] Processed step {i+1}/{TOTAL_STEPS}")

    env.end_session()
    print("[SERVER] Done.")
