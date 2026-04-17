"""
Demo: Two-process IPC communication.
  Process A (environment): runs Chrome + Neuroglancer, waits for actions via IPC
  Process B (controller):  reads states via IPC, sends actions back
"""
import multiprocessing
import time
import os
import shutil

IPC_DIR = "ipc/agent_0"


def environment_process():
    """Environment side: start Chrome, run IPC loop."""
    from ngllib import Environment

    def reward_fn(state, action, prev_state):
        dz = state[0][0][2] - prev_state[0][0][2]
        return dz, False

    env = Environment(
        headless=True,
        config_path="config.json",
        reward_function=reward_fn,
        ipc_dir=IPC_DIR
    )
    env.start_session(euler_angles=True, fast=True)
    print("[ENV] Session started. Running IPC loop...")
    env.run_ipc_loop(max_steps=5)
    env.end_session()
    print("[ENV] Done.")


def controller_process():
    """Controller side: read states, send actions (no Chrome needed)."""
    from ngllib import IPCChannel

    ch = IPCChannel(IPC_DIR)
    os.makedirs("demo_ipc_screenshots", exist_ok=True)

    actions = [
        # Zoom in
        [0,0,0, 0,0, 0,0,0, 1, 0,0,0, 0, 0,0,0, -2000],
        # Move z
        [0,0,0, 0,0, 0,0,0, 1, 0,0,50, 0, 0,0,0, 0],
        # Rotate
        [0,0,0, 0,0, 0,0,0, 1, 0,0,0, 0, 0.3,0,0, 0],
        # Pan x
        [0,0,0, 0,0, 0,0,0, 1, 200,0,0, 0, 0,0,0, 0],
        # Pan y
        [0,0,0, 0,0, 0,0,0, 1, 0,200,0, 0, 0,0,0, 0],
    ]

    for i, action in enumerate(actions):
        # Wait for state from environment
        print(f"[CTRL] Waiting for state {i}...")
        while not ch.check_signal("state_ready"):
            time.sleep(0.01)

        pos_state, image = ch.read_state()
        print(f"[CTRL] Got state {i}: pos={[round(p,1) for p in pos_state[0]]}")
        image.save(f"demo_ipc_screenshots/step_{i}.png")

        # Send action
        ch.write_action(action)
        ch.signal("action_ready")
        print(f"[CTRL] Sent action {i}")

    # Read final state
    while not ch.check_signal("state_ready"):
        time.sleep(0.01)
    pos_state, image = ch.read_state()
    image.save("demo_ipc_screenshots/step_final.png")
    print(f"[CTRL] Final state: pos={[round(p,1) for p in pos_state[0]]}")
    print(f"[CTRL] Done! Screenshots in demo_ipc_screenshots/")


if __name__ == "__main__":
    # Clean up old IPC files
    if os.path.exists(IPC_DIR):
        shutil.rmtree(IPC_DIR)

    env_proc = multiprocessing.Process(target=environment_process)
    ctrl_proc = multiprocessing.Process(target=controller_process)

    env_proc.start()
    ctrl_proc.start()

    env_proc.join(timeout=180)
    ctrl_proc.join(timeout=180)

    print(f"\nBoth processes finished. env={env_proc.exitcode}, ctrl={ctrl_proc.exitcode}")
