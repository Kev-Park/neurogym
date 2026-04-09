"""
Benchmark: measure latency of direct env.step() vs Communication-based IPC.

Test A (direct):  single-process, call env.step() directly.
Test B (IPC):     two processes — server (Chrome + env.step) and client
                  (send actions / read observations) via FilesystemProtocol.

Both run NUM_WARMUP warm-up steps (discarded) then NUM_ROUNDS timed steps.
"""

import os
import time
import shutil
import multiprocessing
import numpy as np

from ngllib import Environment
from ngllib.utils.Communication import FilesystemProtocol, NGLClient, NGLServer

# ── Config ──────────────────────────────────────────────────────────────
NUM_WARMUP = 2
NUM_ROUNDS = 10
IPC_DIR = "test_ipc"
SETTLE_TIME = 2  # seconds to let Chrome render after each URL change

# A small JSON-change action (no clicks)
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


def custom_reward(state, action, prev_state):
    return 0.0, False


# ── Test A: Direct ──────────────────────────────────────────────────────
def test_direct():
    """Measure per-step latency calling env.step() directly."""
    env = Environment(
        headless=True, config_path="config.json",
        verbose=False, reward_function=custom_reward,
    )
    env.start_session(euler_angles=True, fast=True)
    time.sleep(SETTLE_TIME)

    for i in range(NUM_WARMUP):
        env.step(ACTION)
        time.sleep(SETTLE_TIME)
        print(f"  warmup {i+1}/{NUM_WARMUP}")

    latencies = []
    for i in range(NUM_ROUNDS):
        t0 = time.perf_counter()
        env.step(ACTION)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        time.sleep(SETTLE_TIME)
        print(f"  round {i+1}/{NUM_ROUNDS}: {latencies[-1]*1000:.1f} ms")

    env.end_session()
    return np.array(latencies)


# ── Test B: With Communication (multiprocessing) ───────────────────────
def _server_worker(ipc_dir, num_steps, ready_event):
    """Runs in a child process with its own Chrome instance."""
    def _reward(state, action, prev_state):
        return 0.0, False

    env = Environment(
        headless=True, config_path="config.json",
        verbose=False, reward_function=_reward,
    )
    env.start_session(euler_angles=True, fast=True)
    time.sleep(SETTLE_TIME)  # let initial page load finish

    protocol = FilesystemProtocol(
        action_file_path=os.path.join(ipc_dir, "actions"),
        observation_file_path=os.path.join(ipc_dir, "observations"),
        timeout=5_000_000,
    )
    server = NGLServer(protocol, env)

    # Tell the client we are ready to receive actions
    ready_event.set()

    for _ in range(num_steps):
        server.process_actions()

    env.end_session()


def test_with_communication():
    """Measure per-step latency going through FilesystemProtocol IPC."""
    if os.path.exists(IPC_DIR):
        shutil.rmtree(IPC_DIR)

    total_steps = NUM_WARMUP + NUM_ROUNDS
    ready_event = multiprocessing.Event()

    # Start server in a separate process (its own Chrome, no GIL contention)
    server_proc = multiprocessing.Process(
        target=_server_worker,
        args=(IPC_DIR, total_steps, ready_event),
    )
    server_proc.start()

    # Block until the server has started Chrome and is polling for actions
    print("  waiting for server process to start Chrome ...")
    ready_event.wait(timeout=60)

    # Client side (main process, no Chrome needed)
    protocol = FilesystemProtocol(
        action_file_path=os.path.join(IPC_DIR, "actions"),
        observation_file_path=os.path.join(IPC_DIR, "observations"),
        timeout=5_000_000,
    )
    client = NGLClient(protocol)

    for i in range(NUM_WARMUP):
        client.send_actions(ACTION)
        print(f"  warmup {i+1}/{NUM_WARMUP}")

    latencies = []
    for i in range(NUM_ROUNDS):
        t0 = time.perf_counter()
        client.send_actions(ACTION)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        print(f"  round {i+1}/{NUM_ROUNDS}: {latencies[-1]*1000:.1f} ms")

    server_proc.join(timeout=120)

    if server_proc.exitcode != 0:
        print(f"[WARNING] Server process exited with code {server_proc.exitcode}")

    if os.path.exists(IPC_DIR):
        shutil.rmtree(IPC_DIR)

    return np.array(latencies)


# ── Report ──────────────────────────────────────────────────────────────
def print_stats(name, latencies_ms):
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Rounds:  {len(latencies_ms)}")
    print(f"  Mean:    {np.mean(latencies_ms):8.2f} ms")
    print(f"  Std:     {np.std(latencies_ms):8.2f} ms")
    print(f"  Median:  {np.median(latencies_ms):8.2f} ms")
    print(f"  Min:     {np.min(latencies_ms):8.2f} ms")
    print(f"  Max:     {np.max(latencies_ms):8.2f} ms")


if __name__ == "__main__":
    print(f"Config: {NUM_WARMUP} warmup + {NUM_ROUNDS} timed rounds")
    print(f"Settle time: {SETTLE_TIME}s between steps\n")

    print("[1/2] Running direct env.step() test ...")
    direct_ms = test_direct() * 1000

    print("\n[2/2] Running Communication IPC test ...")
    ipc_ms = test_with_communication() * 1000

    print_stats("Direct env.step()", direct_ms)
    print_stats("Communication IPC", ipc_ms)

    overhead = np.mean(ipc_ms) - np.mean(direct_ms)
    print(f"\n{'=' * 50}")
    print(f"  IPC overhead (mean): {overhead:+.2f} ms")
    print(f"  Ratio:               {np.mean(ipc_ms) / np.mean(direct_ms):.2f}x")
    print(f"{'=' * 50}")
