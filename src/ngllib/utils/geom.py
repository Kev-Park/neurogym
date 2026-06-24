from math import atan2, asin
from math import sin, cos, pi
from scipy.spatial.transform import Rotation  # For proper quaternion handling
import numpy as np

def quaternion_to_euler(q):
    """
    Converts a quaternion to Euler angles (yaw, pitch, roll) in radians.
    Assumes Neuroglancer's default quaternion convention: [x, y, z, w].

    Parameters:
        q (list or np.ndarray): Quaternion [x, y, z, w].

    Returns:
        tuple: (yaw, pitch, roll) in radians.
    """
    x, y, z, w = q

    # Compute the angles using standard equations for quaternion to Euler.
    # Neuroglancer uses a ZYX (yaw, pitch, roll) intrinsic rotation order.
    yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))  # Z axis rotation
    pitch = asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))       # Y axis rotation
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2)) # X axis rotation

    return [yaw, pitch, roll]


def euler_to_quaternion(euler_angles):
    """
    Converts Euler angles [yaw, pitch, roll] in radians to a quaternion.
    Assumes Neuroglancer's default quaternion convention: [x, y, z, w].

    Parameters:
        yaw (float): Rotation around the Z-axis in radians.
        pitch (float): Rotation around the Y-axis in radians.
        roll (float): Rotation around the X-axis in radians.

    Returns:
        list: Quaternion [x, y, z, w].
    """

    # Precompute half angles for efficiency
    yaw, pitch, roll = euler_angles
    half_yaw = yaw / 2.0
    half_pitch = pitch / 2.0
    half_roll = roll / 2.0

    # Sine and cosine of half angles
    cy = cos(half_yaw)
    sy = sin(half_yaw)
    cp = cos(half_pitch)
    sp = sin(half_pitch)
    cr = cos(half_roll)
    sr = sin(half_roll)

    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]




def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles (in radians) to a 3x3 rotation matrix.
    Correct order for ZYX (yaw, pitch, roll): Rx @ Ry @ Rz
    """
    yaw, pitch, roll = euler_angles
    
    Rz = np.array([  # Z rotation (yaw)
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([  # Y rotation (pitch)
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([  # X rotation (roll)
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    return Rz @ Ry @ Rx  

def project_point_to_2d(euler_angles:np.ndarray, termination_position:np.ndarray, curr_position:np.ndarray, normalized:bool=True):
    """
    Returns 2D projection of delta vector between a point and user position in Neuroglancer view.
    """
    assert len(euler_angles) == 3
    assert len(termination_position) == 3
    assert len(curr_position) == 3
    # initial_transform = np.array([
    #     [0, 1, 0],  
    #     [0, 0, -1],  
    #     [-1, 0, 0]  
    # ])
    
    termination_pos = termination_position[0]
    rotation_matrix = euler_to_rotation_matrix(euler_angles)
    delta_vector = termination_position - curr_position
    if normalized:
        normalized_delta_vector = delta_vector / np.linalg.norm(delta_vector)
        projection = rotation_matrix @ normalized_delta_vector
    else:
        projection = rotation_matrix @ delta_vector
    #flipped_projection = np.array([-projection[0], -projection[1]])

    return np.array([projection[0], projection[1]])

def project_z_axis_to_2d(euler_angles):
    """
    Returns 2D projection of z-axis in Neuroglancer view.
    Handles edge case when z-axis points directly away from camera.
    """
    rotation_matrix = euler_to_rotation_matrix(euler_angles)
    z_transformed = rotation_matrix @ np.array([0, 0, 1])
    
    # Correct projection: keep X, invert Y for Neuroglancer's display
    projection = np.array([-z_transformed[0], -z_transformed[1]])  # Fixed coordinate flip

    norm = np.linalg.norm(projection)
    return np.array([0.0, 0.0]) if norm < 1e-6 else projection / norm


if __name__ == "__main__":
    # Test case 1: Identity quaternion (z-axis pointing away)
    quat_identity = [0, 0, 0, 1]
    euler = quaternion_to_euler(quat_identity)
    print("Identity projection:", project_z_axis_to_2d(euler))  # [0.0, 0.0]

    # Test case 2: 90° yaw rotation (z-axis should point left in 2D view)
    quat_yaw90 = Rotation.from_euler('z', 90, degrees=True).as_quat()
    print("Quaternion:", quat_yaw90)
    euler = quaternion_to_euler(quat_yaw90)
    print("Yaw 90° projection:", project_z_axis_to_2d(euler))  # [0.0, 0.0] (still along Z)

    # Test case 3: Your example quaternion
    quat_example = [
    0.7608212828636169,
    0.27864933013916016,
    -0.49907028675079346,
    -0.30730170011520386
  ]
    euler = quaternion_to_euler(quat_example)
    print("Example projection:", project_z_axis_to_2d(euler))  # Should show directio