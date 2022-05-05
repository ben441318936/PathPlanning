import numpy as np

from MotionModels import MotionModel

def simple_control(motion_model: MotionModel, curr_state: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    curr_pos = motion_model.state_2_position(curr_state)
    curr_heading = motion_model.state_2_heading(curr_state)
    curr_yaw_rate = motion_model.state_2_yaw_rate(curr_state)
    curr_velocity = motion_model.state_2_velocity(curr_state)
    goal_heading = np.arctan2(goal_pos[1], goal_pos[0])

    # outer loop, proportional for v and w
    K_V = 1
    v_goal = K_V * np.linalg.norm(goal_pos - curr_pos)
    K_W = 1
    w_goal = K_W * (1-np.cos(goal_heading-curr_heading))

    # inner loop, propertional for torque
    K_V_T = 1
    K_W_T = 10
    return np.array([K_V_T * (v_goal - curr_velocity) + K_W_T * (w_goal - curr_yaw_rate), K_V_T * (v_goal - curr_velocity) - K_W_T * (w_goal - curr_yaw_rate)])


