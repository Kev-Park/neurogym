import os
from ngllib import Environment

# Initialize options for environmental interaction, including state return types
options = {
        'euler_angles': True,
        'resize': False,
        'add_mouse': False,
        'fast': True,
        'image_path': None
}

def custom_reward(state, action, prev_state):
    return 1, False

env = Environment(headless=True, config_path="config.json", verbose=False, reward_function=custom_reward)

env.start_session(**options)

os.makedirs("screenshots", exist_ok=True)

for i in range(10):

    # action_vector should reflect a model output; here it is hardcoded for demonstration purposes
    action_vector = [
        0, 0, 0,  # left, right, double click booleans
        100, 100,  # x, y
        0, 0, 0,  # no modifier keys
        1,  # json_change = 1, so apply delta to JSON state
        10, 0, 0,  # position change
        0,  # cross-section scaling
        0.2, 0, 0,  # orientation change in Euler angles
        2000  # projection scaling
        ]

    state, reward, done, json_state = env.step(action_vector)
    pos_state, image = state

    # Save screenshot so you can see what the agent sees
    image.save(f"screenshots/step_{i:03d}.png")
    print(f"Step {i}: pos={[round(p,1) for p in pos_state[0]]}, reward={reward}")

env.end_session()
print(f"Done! Screenshots saved to screenshots")