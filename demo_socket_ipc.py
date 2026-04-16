"""
Demo: Socket-based IPC between two processes.

This test does NOT require Chrome/Neuroglancer — it uses a mock environment
to verify that SocketProtocol correctly shuttles actions and observations
between an NGLServer and NGLClient running in separate processes.

Usage:
    python demo_socket_ipc.py

You can also pass --host and --port if you want to test across machines:
    # On machine A (environment):
    python demo_socket_ipc.py --role server --host 0.0.0.0 --port 5555

    # On machine B (agent):
    python demo_socket_ipc.py --role client --host <machine-A-ip> --port 5555
"""

import multiprocessing
import argparse
import time
import sys

PORT = 5556  # Use a non-default port to avoid collisions


# ---------------------------------------------------------------------------
# Mock environment (no Chrome needed)
# ---------------------------------------------------------------------------

class MockEnvironment:
    """Mimics ngllib.Environment for testing purposes."""

    def __init__(self):
        self.position = [100.0, 200.0, 300.0]
        self.step_count = 0
        # These must exist before start_session because NGLServer reads them
        # to send the initial observation.
        self.prev_state = (self.position[:], None)
        self.prev_json = {"position": self.position[:]}

    def start_session(self, **options):
        print("[MockEnv] Session started with options:", options)
        # Refresh prev_state/prev_json so the initial observation reflects
        # the starting state.
        self.prev_state = (self.position[:], None)
        self.prev_json = {"position": self.position[:]}

    def step(self, actions):
        """Apply actions and return (state, reward, done, info)."""
        self.step_count += 1

        # Simulate position delta from the action vector
        # actions[9:12] are delta_position_x/y/z in the standard action layout
        if len(actions) >= 12:
            self.position[0] += actions[9]
            self.position[1] += actions[10]
            self.position[2] += actions[11]

        state = (self.position[:], None)  # (pos_state, image=None for mock)
        reward = actions[11] if len(actions) >= 12 else 0  # reward = dz
        done = self.step_count >= 5
        info = {"step": self.step_count}

        self.prev_state = state
        self.prev_json = {"position": self.position[:]}

        print(f"[MockEnv] Step {self.step_count}: pos={self.position}, reward={reward}, done={done}")
        return state, reward, done, info


# ---------------------------------------------------------------------------
# Server process (environment side)
# ---------------------------------------------------------------------------

def run_server(host, port, num_steps):
    from ngllib.utils.Communication import SocketProtocol, NGLServer

    protocol = SocketProtocol(host=host, port=port, is_server=True, timeout=30)
    env = MockEnvironment()
    server = NGLServer(protocol=protocol, environment=env)
    server.start_session(euler_angles=True, fast=True)

    print(f"[Server] Processing {num_steps} actions...")
    for i in range(num_steps):
        server.process_actions()
        print(f"[Server] Completed step {i + 1}/{num_steps}")

    protocol.close()
    print("[Server] Done.")


# ---------------------------------------------------------------------------
# Client process (agent side)
# ---------------------------------------------------------------------------

def run_client(host, port, num_steps):
    from ngllib.utils.Communication import SocketProtocol, NGLClient

    # Small delay to let the server start listening
    time.sleep(0.5)

    protocol = SocketProtocol(host=host, port=port, is_server=False, timeout=30)
    client = NGLClient(protocol=protocol)

    # Receive the initial observation BEFORE any actions are sent.
    # This is how an RL policy would work in practice: look at the state,
    # then decide on an action.
    initial_obs = client.get_initial_observation()
    state, reward, done, info = initial_obs
    print(f"[Client] Initial observation: pos={state[0]}")

    actions = [
        # Move in +z
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 50, 0, 0, 0, 0, 0],
        # Move in +x
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 100, 0, 0, 0, 0, 0, 0, 0],
        # Move in +y
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 200, 0, 0, 0, 0, 0, 0],
        # Move in -z
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -25, 0, 0, 0, 0, 0],
        # No movement
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for i in range(num_steps):
        # In a real RL loop, `action` would come from policy(state).
        action = actions[i % len(actions)]
        print(f"[Client] Sending action {i + 1}: delta_pos=({action[9]}, {action[10]}, {action[11]})")
        observation = client.send_actions(action)
        state, reward, done, info = observation
        print(f"[Client] Got observation {i + 1}: pos={state[0]}, reward={reward}, done={done}")

    protocol.close()
    print("[Client] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: Socket-based IPC")
    parser.add_argument("--role", choices=["server", "client", "both"], default="both",
                        help="Run server, client, or both in separate processes (default: both)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    if args.role == "both":
        server_proc = multiprocessing.Process(target=run_server, args=(args.host, args.port, args.steps))
        client_proc = multiprocessing.Process(target=run_client, args=(args.host, args.port, args.steps))

        server_proc.start()
        client_proc.start()

        server_proc.join(timeout=60)
        client_proc.join(timeout=60)

        print(f"\nBoth processes finished. server={server_proc.exitcode}, client={client_proc.exitcode}")
        sys.exit(0 if server_proc.exitcode == 0 and client_proc.exitcode == 0 else 1)

    elif args.role == "server":
        run_server(args.host, args.port, args.steps)

    elif args.role == "client":
        run_client(args.host, args.port, args.steps)
