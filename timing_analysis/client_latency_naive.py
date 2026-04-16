"""
Client side of cross-node NAIVE latency test (no Communication protocol).
Uses direct pickle read/write with no atomic rename — exposes read/write conflicts.

Submitted to a compute node via sbatch (client_naive.sh).
"""

import os
import sys
import time
import json
import pickle
import numpy as np

# ── Config (must match server_latency_naive.py) ────────────────────────
NUM_WARMUP = 2
NUM_ROUNDS = 10
IPC_DIR = "test_ipc_naive"
READY_SIGNAL = os.path.join(IPC_DIR, "server_ready")
ACTION_FILE = os.path.join(IPC_DIR, "action.pkl")
OBSERVATION_FILE = os.path.join(IPC_DIR, "observation.pkl")
RESULTS_FILE = os.path.join(IPC_DIR, "results.json")

POLL_TIMEOUT = 600  # seconds

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


def wait_for_server(timeout=POLL_TIMEOUT):
    """Poll for server ready signal."""
    print(f"[CLIENT-NAIVE] Waiting for server ready signal: {READY_SIGNAL}")
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(READY_SIGNAL):
            print("[CLIENT-NAIVE] Server is ready.")
            return True
        time.sleep(0.5)
    print("[CLIENT-NAIVE] ERROR: Server did not become ready within timeout.")
    return False


def write_action(action):
    """Write action directly to file. No atomic rename."""
    with open(ACTION_FILE, "wb") as f:
        pickle.dump(action, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_observation(timeout=POLL_TIMEOUT):
    """Poll for observation file, read it, delete it. No atomic rename."""
    start = time.time()
    conflict_count = 0
    while time.time() - start < timeout:
        if os.path.exists(OBSERVATION_FILE):
            try:
                with open(OBSERVATION_FILE, "rb") as f:
                    observation = pickle.load(f)
                os.remove(OBSERVATION_FILE)
                if conflict_count > 0:
                    print(f"    [read/write conflicts before success: {conflict_count}]")
                return observation
            except Exception:
                # Read/write conflict: file exists but is partially written.
                conflict_count += 1
                time.sleep(0.001)
                continue
    raise TimeoutError("No observation received within timeout.")


if __name__ == "__main__":
    print(f"[CLIENT-NAIVE] Running on node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"[CLIENT-NAIVE] Config: {NUM_WARMUP} warmup + {NUM_ROUNDS} timed rounds\n")

    if not wait_for_server():
        sys.exit(1)

    # ── Warmup ──
    for i in range(NUM_WARMUP):
        write_action(ACTION)
        read_observation()
        print(f"[CLIENT-NAIVE] warmup {i+1}/{NUM_WARMUP}")

    # ── Timed rounds ──
    latencies = []
    total_conflicts = 0
    for i in range(NUM_ROUNDS):
        t0 = time.perf_counter()
        write_action(ACTION)
        obs = read_observation()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        print(f"[CLIENT-NAIVE] round {i+1}/{NUM_ROUNDS}: {latencies[-1]*1000:.1f} ms")

    # ── Report ──
    latencies_ms = np.array(latencies) * 1000
    print(f"\n{'=' * 50}")
    print(f"  Cross-Node Naive File IPC (no Communication)")
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
    print(f"\n[CLIENT-NAIVE] Results saved to {RESULTS_FILE}")
    print("[CLIENT-NAIVE] Done.")
