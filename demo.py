"""
Demo: Simulates an agent navigating a fly brain neuron in Neuroglancer.
Shows different action types: zoom in, pan through slices, rotate view.
"""
import os, time
from ngllib import Environment

os.makedirs("demo_screenshots", exist_ok=True)

def custom_reward(state, action, prev_state):
    # Reward z-position change (moving deeper into the brain)
    dz = state[0][0][2] - prev_state[0][0][2]
    return dz, False

env = Environment(headless=True, config_path="config.json", verbose=False, reward_function=custom_reward)
env.start_session(euler_angles=True, resize=False, add_mouse=False, fast=True)

# Save initial state
state, json_state = env.prepare_state()
state[1].save("demo_screenshots/00_initial.png")
print(f"Initial: pos={[round(p,1) for p in state[0][0]]}, scale={round(state[0][3],1)}")

# --- Phase 1: Zoom in (decrease projectionScale) ---
print("\n--- Phase 1: Zooming in ---")
for i in range(5):
    action = [
        0, 0, 0,        # no click
        0, 0,            # mouse pos (unused)
        0, 0, 0,         # no modifier keys
        1,               # json_change = True
        0, 0, 0,         # no position change
        0,               # no crossSection change
        0, 0, 0,         # no rotation
        -2000            # zoom IN (decrease projectionScale)
    ]
    state, reward, done, json_state = env.step(action)
    state[1].save(f"demo_screenshots/01_zoom_{i}.png")
    print(f"  Zoom step {i}: scale={round(state[0][3],1)}")

# --- Phase 2: Pan through z-slices (go deeper into brain) ---
print("\n--- Phase 2: Moving through z-slices ---")
for i in range(5):
    action = [
        0, 0, 0,
        0, 0,
        0, 0, 0,
        1,
        0, 0, 50,        # move z by +50 (go deeper)
        0,
        0, 0, 0,
        0
    ]
    state, reward, done, json_state = env.step(action)
    state[1].save(f"demo_screenshots/02_zslice_{i}.png")
    print(f"  Z-slice step {i}: z={round(state[0][0][2],1)}, reward={round(reward,1)}")

# --- Phase 3: Rotate the 3D view ---
print("\n--- Phase 3: Rotating 3D view ---")
for i in range(5):
    action = [
        0, 0, 0,
        0, 0,
        0, 0, 0,
        1,
        0, 0, 0,
        0,
        0.3, 0, 0,       # rotate yaw by 0.3 radians (~17 degrees)
        0
    ]
    state, reward, done, json_state = env.step(action)
    state[1].save(f"demo_screenshots/03_rotate_{i}.png")
    euler = state[0][2]
    print(f"  Rotate step {i}: yaw={round(euler[0],2)}, pitch={round(euler[1],2)}, roll={round(euler[2],2)}")

# --- Phase 4: Pan in x-y plane ---
print("\n--- Phase 4: Panning x-y ---")
for i in range(5):
    action = [
        0, 0, 0,
        0, 0,
        0, 0, 0,
        1,
        200, 200, 0,      # move x+200, y+200
        0,
        0, 0, 0,
        0
    ]
    state, reward, done, json_state = env.step(action)
    state[1].save(f"demo_screenshots/04_pan_{i}.png")
    print(f"  Pan step {i}: pos={[round(p,1) for p in state[0][0]]}")

env.end_session()
print(f"\nDone! {len(os.listdir('demo_screenshots'))} screenshots saved to demo_screenshots/")
