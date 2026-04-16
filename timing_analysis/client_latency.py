"""
Client side of cross-node latency test (with Communication protocol).
Submitted to a compute node via sbatch (client.sh).
Waits for the server to signal ready, then sends actions and measures latency.

Uses FilesystemProtocol's write_actions directly, but replaces the built-in
busy-poll read_observations with a sleep-based version to survive cross-node
GPFS latency (the original has no sleep between retries).
"""

import os
import sys
import time
import json
import pickle
import numpy as np

from ngllib.utils.Communication import FilesystemProtocol

# ── Config (must match server_latency.py) ───────────────────────────────
NUM_WARMUP = 2
NUM_ROUNDS = 10
IPC_DIR = "test_ipc_cross"
READY_SIGNAL = os.path.join(IPC_DIR, "server_ready")
RESULTS_FILE = os.path.join(IPC_DIR, "results.json")
CLIENT_ID = 1  # matches NGLServer._id (starts at 1)

ACTION = [
    0, 0, 0,       # left, right, double click
    0, 0,           # mouse x, y
    0, 0, 0,        # Shift, Ctrl, Alt
    1,              # json_change = True
    10, 0, 0,       # delta position x, y, z
    0,              # delta crossSectionScale
    0.1, 0, 0,      # delta euler angles
    500,            # delta projectionScale
]

POLL_TIMEOUT = 600  # seconds


def wait_for_server(timeout=POLL_TIMEOUT):
    """Poll for server ready signal."""
    print(f"[CLIENT] Waiting for server ready signal: {READY_SIGNAL}")
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(READY_SIGNAL):
            print("[CLIENT] Server is ready.")
            return True
        time.sleep(0.5)
    print("[CLIENT] ERROR: Server did not become ready within timeout.")
    return False


def read_observations_with_sleep(protocol, client_id, timeout=POLL_TIMEOUT):
    """
    Same logic as FilesystemProtocol.read_observations, but with time.sleep()
    between retries so we don't exhaust retries before the server responds
    over cross-node GPFS.
    """
    observation_file = os.path.join(protocol.observation_path, f"{client_id}_0")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(observation_file, "rb") as f:
                observations = pickle.load(f)
            os.rename(observation_file, os.path.join(protocol.observation_path, f"{client_id}_1"))
            return observations
        except Exception:
            time.sleep(0.001)
    raise TimeoutError(f"No observations received within {timeout}s.")


def send_actions(protocol, client_id, actions):
    """Write action via protocol, then read observation with sleep-based poll."""
    protocol.write_actions(actions, client_id)
    return read_observations_with_sleep(protocol, client_id)


if __name__ == "__main__":
    print(f"[CLIENT] Running on node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"[CLIENT] Config: {NUM_WARMUP} warmup + {NUM_ROUNDS} timed rounds\n")

    if not wait_for_server():
        sys.exit(1)

    protocol = FilesystemProtocol(
        action_file_path=os.path.join(IPC_DIR, "actions"),
        observation_file_path=os.path.join(IPC_DIR, "observations"),
        timeout=50_000_000,  # only used for write_actions; we override read
    )
    # Clear any leftover observation files (same as NGLClient.__init__)
    protocol.clear_observations(CLIENT_ID)

    # ── Warmup ──
    for i in range(NUM_WARMUP):
        send_actions(protocol, CLIENT_ID, ACTION)
        print(f"[CLIENT] warmup {i+1}/{NUM_WARMUP}")

    # ── Timed rounds ──
    latencies = []
    for i in range(NUM_ROUNDS):
        t0 = time.perf_counter()
        send_actions(protocol, CLIENT_ID, ACTION)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        print(f"[CLIENT] round {i+1}/{NUM_ROUNDS}: {latencies[-1]*1000:.1f} ms")

    # ── Report ──
    latencies_ms = np.array(latencies) * 1000
    print(f"\n{'=' * 50}")
    print(f"  Cross-Node Communication IPC")
    print(f"{'=' * 50}")
    print(f"  Rounds:  {len(latencies_ms)}")
    print(f"  Mean:    {np.mean(latencies_ms):8.2f} ms")
    print(f"  Std:     {np.std(latencies_ms):8.2f} ms")
    print(f"  Median:  {np.median(latencies_ms):8.2f} ms")
    print(f"  Min:     {np.min(latencies_ms):8.2f} ms")
    print(f"  Max:     {np.max(latencies_ms):8.2f} ms")
    print(f"{'=' * 50}")

    results = {
        "latencies_ms": latencies_ms.tolist(),
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[CLIENT] Results saved to {RESULTS_FILE}")
    print("[CLIENT] Done.")
