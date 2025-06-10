import re
from math import atan2, asin
from math import sin, cos, pi
from .Values import Values

def parse_action(action: str):
    json_keywords = ["Outside render:", "Drag", "Wheel", "Keyboard"]
    if any(keyword in action for keyword in json_keywords):
        # any of these operations act directly on the JSON state, the agent will learn to operate directly on it
        return None, True
    
    parsed_result = {
            'x': None,
            'y': None,
            'click_type': None,
            'keys_pressed': None,
            'json_change': False
        }
        
    position_pattern = r"Relative position: x=(\d+), y=(\d+)"
    double_click_pattern = r"Double Click"
    single_click_pattern = r"Single Click: ([\w\s]+) Click"
    keys_pattern = r"with keys: ([\w\s,]+)"
    
    
    position_match = re.search(position_pattern, action)
    if position_match:
        parsed_result['x'] = int(position_match.group(1))
        parsed_result['y'] = int(position_match.group(2))
    
    
    if re.search(double_click_pattern, action):
        parsed_result['click_type'] = "double_click"
    else:
        single_click_match = re.search(single_click_pattern, action)
        if single_click_match:
            parsed_result['click_type'] = single_click_match.group(1).strip().lower()
    
    keys_match = re.search(keys_pattern, action)
    if keys_match:
        parsed_result['keys_pressed'] = keys_match.group(1).strip()
    return parsed_result, False


def find_mouse_increments(curr_x, curr_y, target_x, target_y, values=Values(), grid=True):
        if grid:
            action_space = values.Q_action_space_indexes_grid
        else:
            action_space = values.Q_action_space_indexes

        increments = []
        x_diff = target_x - curr_x
        y_diff = target_y - curr_y
        if grid:
            # Technically the decreasing will never happen in grid-style but it is not an issue
            x_actions = [
            (5, "incr_mouse_x_5", "decr_mouse_x_5"),
            ]

            y_actions = [
            (5, "incr_mouse_y_5", "decr_mouse_y_5"),
            ]
        else:
            x_actions = [
                (500, "incr_mouse_x_500", "decr_mouse_x_500"),
                (100, "incr_mouse_x_100", "decr_mouse_x_100"),
                (50, "incr_mouse_x_50", "decr_mouse_x_50"),
                (10, "incr_mouse_x_10", "decr_mouse_x_10"),
                (5, "incr_mouse_x_5", "decr_mouse_x_5"),
                (1, "incr_mouse_x_1", "decr_mouse_x_1"),
            ]

            y_actions = [
                (500, "incr_mouse_y_500", "decr_mouse_y_500"),
                (100, "incr_mouse_y_100", "decr_mouse_y_100"),
                (50, "incr_mouse_y_50", "decr_mouse_y_50"),
                (10, "incr_mouse_y_10", "decr_mouse_y_10"),
                (5, "incr_mouse_y_5", "decr_mouse_y_5"),
                (1, "incr_mouse_y_1", "decr_mouse_y_1"),
            ]

        mouse_positions = [[curr_x, curr_y]]
        for step, incr_action, decr_action in x_actions:
            while abs(x_diff) >= step:
                if x_diff > 0:
                    increments.append(action_space[incr_action])
                    x_diff -= step
                else:
                    increments.append(action_space[decr_action])
                    x_diff += step
                mouse_positions.append([mouse_positions[-1][0] + step if x_diff > 0 else mouse_positions[-1][0] - step, mouse_positions[-1][1]])

        for step, incr_action, decr_action in y_actions:
            while abs(y_diff) >= step:
                if y_diff > 0:
                    increments.append(action_space[incr_action])
                    y_diff -= step
                else:
                    increments.append(action_space[decr_action])
                    y_diff += step
                mouse_positions.append([mouse_positions[-1][0], mouse_positions[-1][1] + step if y_diff > 0 else mouse_positions[-1][1] - step])

        return increments, mouse_positions

def find_pos_increments(current_pos, next_pos, values=Values(), grid=True):
    if grid:
        action_space = values.Q_action_space_indexes_grid
    else:
        action_space = values.Q_action_space_indexes
        
    increments = []
    positions = [current_pos[:]]  # Start with the initial position
    pos_diff = [next_pos[i] - current_pos[i] for i in range(3)]

    pos_actions = [
        (200, "incr_position_x_1000", "decr_position_x_1000"),
        (50, "incr_position_x_100", "decr_position_x_100"),
        (5, "incr_position_x_5", "decr_position_x_5"),
        (1, "incr_position_x_1", "decr_position_x_1"),
    ]

    for axis in range(3): 
        for step, incr_action, decr_action in pos_actions:
            while abs(pos_diff[axis]) >= step:
                if pos_diff[axis] > 0:
                    increments.append(action_space[incr_action.replace("x", ["x", "y", "z"][axis])])
                    pos_diff[axis] -= step
                else:
                    increments.append(action_space[decr_action.replace("x", ["x", "y", "z"][axis])])
                    pos_diff[axis] += step
                current_pos[axis] += step if pos_diff[axis] > 0 else -step
                positions.append(current_pos[:])

    return increments, positions

def find_crossSectionScale_increments(curr_scale, target_scale, values=Values(), grid=True):
    if grid:
        action_space = values.Q_action_space_indexes_grid
    else:
        action_space = values.Q_action_space_indexes
        
    increments = []
    scales = [curr_scale]
    diff = target_scale - curr_scale

    scale_actions = [
        (1, "incr_crossSectionScale_1", "decr_crossSectionScale_1"),
        (0.1, "incr_crossSectionScale_0.1", "decr_crossSectionScale_0.1"),
    ]

    for step, incr_action, decr_action in scale_actions:
        while abs(diff) >= step:
            if diff > 0:
                increments.append(action_space[incr_action])
                diff -= step
            else:
                increments.append(action_space[decr_action])
                diff += step
            curr_scale += step if diff > 0 else -step
            scales.append(curr_scale)

    return increments, scales


def find_projectionScale_increments(curr_scale, target_scale, values=Values(), grid=True):
    if grid:
        action_space = values.Q_action_space_indexes_grid
    else:
        action_space = values.Q_action_space_indexes
        
    increments = []
    scales = [curr_scale]
    diff = target_scale - curr_scale

    scale_actions = [
        (5000, "incr_projectionScale_5000", "decr_projectionScale_5000"),
        (1000, "incr_projectionScale_1000", "decr_projectionScale_1000"),
        (100, "incr_projectionScale_100", "decr_projectionScale_100"),
    ]

    for step, incr_action, decr_action in scale_actions:
        while abs(diff) >= step:
            if diff > 0:
                increments.append(action_space[incr_action])
                diff -= step
            else:
                increments.append(action_space[decr_action])
                diff += step
            curr_scale += step if diff > 0 else -step
            scales.append(curr_scale)

    return increments, scales


# Receives angles from [0,180] and [-180,0]
def find_projectionOrientation_increments(current_angle, objective_angle, values=Values(), grid=True):
    if grid:
        action_space = values.Q_action_space_indexes_grid
    else:
        action_space = values.Q_action_space_indexes
        
    increments = [] # Command indices
    angles = [] # Interpolation angles

    roll_actions = [
        (1.0, "incr_projectionOrientation_q1_1.0", "decr_projectionOrientation_q1_1.0"),
        (0.2, "incr_projectionOrientation_q1_0.2", "decr_projectionOrientation_q1_0.2")
    ]
    yaw_actions = [
        (1.0, "incr_projectionOrientation_q2_1.0", "decr_projectionOrientation_q2_1.0"),
        (0.2, "incr_projectionOrientation_q2_0.2", "decr_projectionOrientation_q2_0.2")
    ]
    pitch_actions = [
        (1.0, "incr_projectionOrientation_q3_1.0", "decr_projectionOrientation_q3_1.0"),
        (0.2, "incr_projectionOrientation_q3_0.2", "decr_projectionOrientation_q3_0.2")
    ]

    curr_yaw, curr_pitch, curr_roll = current_angle
    obj_yaw, obj_pitch, obj_roll = objective_angle

    for step, incr_action, decr_action in yaw_actions:
        diff = atan2(sin(obj_yaw - curr_yaw), cos(obj_yaw-curr_yaw)) # Get initial difference
        while abs(diff) >= step:
            if diff > 0 and (diff - step) >= 0:
                increments.append(action_space[incr_action])
                diff -= step
                curr_yaw += step
                if curr_yaw < -pi:
                    curr_yaw += 2*pi
                elif curr_yaw > pi:
                    curr_yaw -= 2*pi
                angles.append([curr_yaw, curr_pitch, curr_roll])
            elif diff < 0 and (diff + step) <= 0:
                increments.append(action_space[decr_action])
                diff += step
                curr_yaw -= step
                if curr_yaw < -pi:
                    curr_yaw += 2*pi
                elif curr_yaw > pi:
                    curr_yaw -= 2*pi
                angles.append([curr_yaw, curr_pitch, curr_roll])
    for step, incr_action, decr_action in pitch_actions:
        diff = obj_pitch - curr_pitch # Get initial difference
        while abs(diff) >= step:
            if diff > 0 and (diff - step) >= 0:
                increments.append(action_space[incr_action])
                diff -= step
                curr_pitch += step
                angles.append([curr_yaw, curr_pitch, curr_roll])
            elif diff < 0 and (diff + step) <= 0:
                increments.append(action_space[decr_action])
                diff += step
                curr_pitch -= step
                angles.append([curr_yaw, curr_pitch, curr_roll])
    for step, incr_action, decr_action in roll_actions:
        diff = atan2(sin(obj_roll - curr_roll), cos(obj_roll-curr_roll)) # Get initial difference
        while abs(diff) >= step:
            if diff > 0 and (diff - step) >= 0:
                increments.append(action_space[incr_action])
                diff -= step
                curr_roll += step
                if curr_roll  < -pi:
                    curr_roll += 2*pi
                elif curr_roll > pi:
                    curr_roll -= 2*pi
                angles.append([curr_yaw, curr_pitch, curr_roll])
            elif diff < 0 and (diff + step) <= 0:
                increments.append(action_space[decr_action])
                diff += step
                curr_roll -= step
                if curr_roll  < -pi:
                    curr_roll += 2*pi
                elif curr_roll > pi:
                    curr_roll -= 2*pi
                angles.append([curr_yaw, curr_pitch, curr_roll])

    return increments, angles

# def get_action_tensor(action_history):
#     return torch.tensor(list(action_history), dtype=torch.float32)

# def add_action(action_history, action):
#     action_history.append(action)  # Automatically removes oldest when full

# # Function to pop the oldest action
# def pop_action(action_history):
#     if action_history:
#         return action_history.popleft()  # Removes and returns oldest action
#     return None  # Handle empty case

if __name__ == "__main__":
    from Values import Values
    import numpy as np
    curr_angle = [0, 30, 0]
    obj_angle = [0, -30, 0]

    curr_angle = [pi/180 * value for value in curr_angle]
    obj_angle = [pi/180 * value for value in obj_angle]

    incr, angles = find_projectionOrientation_increments(curr_angle,obj_angle, Values().Q_action_space_indexes)
    angles_np = np.array(angles)
    angles_np = angles_np * 180 / pi
    print(angles_np)