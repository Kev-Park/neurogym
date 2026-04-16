"""
Server side of cross-node NAIVE latency test (no Communication protocol).
Uses direct pickle read/write with no atomic rename — exposes read/write conflicts.

Run on the current node BEFORE submitting client_naive.sh via sbatch.

  python server_latency_naive.py
"""

import os
import sys
import time
import shutil
import pickle

from ngllib import Environment

# ── Config (must match client_latency_naive.py) ────────────────────────
NUM_WARMUP = 2
NUM_ROUNDS = 10
IPC_DIR = "test_ipc_naive"
READY_SIGNAL = os.path.join(IPC_DIR, "server_ready")
ACTION_FILE = os.path.join(IPC_DIR, "action.pkl")
OBSERVATION_FILE = os.path.join(IPC_DIR, "observation.pkl")

TOTAL_STEPS = NUM_WARMUP + NUM_ROUNDS
POLL_TIMEOUT = 600  # seconds


def custom_reward(state, action, prev_state):
    return 0.0, False


def read_action(timeout=POLL_TIMEOUT):
    """Poll for action file, read it, delete it. No atomic rename."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(ACTION_FILE):
            try:
                with open(ACTION_FILE, "rb") as f:
                    action = pickle.load(f)
                os.remove(ACTION_FILE)
                return action
            except Exception:
                # Read/write conflict: file exists but is partially written.
                # Retry after a brief pause.
                time.sleep(0.001)
                continue
    raise TimeoutError("No action received within timeout.")


def write_observation(observation):
    """Write observation directly to file. No atomic rename."""
    with open(OBSERVATION_FILE, "wb") as f:
        pickle.dump(observation, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if os.path.exists(IPC_DIR):
        shutil.rmtree(IPC_DIR)
    os.makedirs(IPC_DIR, exist_ok=True)

    print("[SERVER-NAIVE] Starting Chrome ...")
    env = Environment(
        headless=True, config_path="config.json",
        verbose=False, reward_function=custom_reward,
    )
    env.start_session(euler_angles=True, fast=True)
    time.sleep(2)

    # Signal ready
    with open(READY_SIGNAL, "w") as f:
        f.write("ready")
    print(f"[SERVER-NAIVE] Ready. Waiting for {TOTAL_STEPS} actions ...")

    conflict_count = 0

    for i in range(TOTAL_STEPS):
        action = read_action()
        result = env.step(action)
        write_observation(result)
        print(f"[SERVER-NAIVE] Processed step {i+1}/{TOTAL_STEPS}")

    env.end_session()
    print("[SERVER-NAIVE] Done.")
