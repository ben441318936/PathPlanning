from abc import ABC, abstractmethod
from turtle import heading
import numpy as np

from MotionModels import MotionModel, DifferentialDrive, DifferentialDriveVelocityInput

def simple_control(motion_model: MotionModel, curr_state: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    curr_pos = motion_model.state_2_position(curr_state)
    curr_heading = motion_model.state_2_heading(curr_state)
    curr_heading = np.arctan2(np.cos(curr_heading), np.sin(curr_heading))
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

class Controller(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def control(self) -> np.ndarray:
        pass


class PVelocityControl(Controller):
    def __init__(self, motion_model: MotionModel, v_max=None, w_max=None) -> None:
        self._motion_model = motion_model
        self.v_max = v_max
        self.w_max = w_max

    def control(self, curr_state: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        curr_pos = self._motion_model.state_2_position(curr_state)
        curr_heading = self._motion_model.state_2_heading(curr_state)
        curr_heading = np.arctan2(np.sin(curr_heading), np.cos(curr_heading))
        goal_heading = np.arctan2(goal_pos[1] - curr_pos[1], goal_pos[0] - curr_pos[0])

        # outer loop, set reference v and w based on position and heading error
        pos_error = np.linalg.norm(goal_pos - curr_pos)
        # if close enough, don't move
        if pos_error < 1:
            return np.array([0,0])
        heading_error = np.arctan2(np.sin(goal_heading-curr_heading), np.cos(goal_heading-curr_heading))

        KP_V = 4
        # KP_W max is 100, because the max heading error is 180 deg
        # this corresponds to turning 180 degs in one timestep at max error
        KP_W = 100

        # ensure that the robot is moving as straight as possible
        # this will be needed if planner outputs straight line paths
        # make this value large to let the robot move smoothly
        if np.abs(heading_error) > 0.1:
            v = 0 
        else:
            v = KP_V * pos_error

        w = KP_W * heading_error

        if self.v_max is not None and np.abs(v) > self.v_max:
            v = np.sign(v) * self.v_max
        if self.w_max is not None and np.abs(w) > self.w_max:
            w = np.sign(w) * self.w_max

        return np.array([v, w])

# WIP
class DoublePDControl(Controller):
    def __init__(self, motion_model: MotionModel) -> None:
        self._motion_model = motion_model
        # initialize memory
        self.reset_memory()

    def reset_memory(self):
        self._pos_error = None
        self._heading_error = None
        self._v_error = None
        self._w_error = None

    def control(self, curr_state: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        curr_pos = self._motion_model.state_2_position(curr_state)
        curr_heading = self._motion_model.state_2_heading(curr_state)
        curr_heading = np.arctan2(np.sin(curr_heading), np.cos(curr_heading))
        # curr_yaw_rate = self._motion_model.state_2_yaw_rate(curr_state)
        # curr_velocity = self._motion_model.state_2_velocity(curr_state)
        goal_heading = np.arctan2(goal_pos[1] - curr_pos[1], goal_pos[0] - curr_pos[0])

        # outer loop, set reference v and w based on position and heading error
        pos_error = np.linalg.norm(goal_pos - curr_pos)
        # if close enough, don't move
        if pos_error < 1:
            return np.array([0,0])
        # heading_error = (1-np.cos(goal_heading-curr_heading))
        heading_error = np.arctan2(np.sin(goal_heading-curr_heading), np.cos(goal_heading-curr_heading))

        KP_V, KD_V = 4, 0
        # without derivative, KP_W max is 100, because the max heading error is 180 deg
        # this corresponds to turning 180 degs in one timestep
        KP_W, KD_W = 100, 0

        d_pos_error = 0 if self._pos_error is None else (pos_error - self._pos_error) / self._motion_model.sampling_period
        d_heading_error = 0 if self._heading_error is None else (heading_error - self._heading_error) / self._motion_model.sampling_period

        # only move forward if the heading is close to correct
        if heading_error > 0.001:
            v_goal = 0 
        else:
            v_goal = KP_V * pos_error + KD_V * d_pos_error
        w_goal = KP_W * heading_error + KD_W * d_heading_error

        # inner loop, set torque based on v and w error
        # v_error = v_goal - curr_velocity
        # w_error = w_goal - curr_yaw_rate

        # d_v_error = (v_error - self._v_error) / self._motion_model.sampling_period
        # d_w_error = (w_error - self._w_error) / self._motion_model.sampling_period

        # KP_VT, KD_VT = 1, 0
        # KP_WT, KD_WT = 1, 0

        # T_R = (KP_VT * v_error + KD_VT * d_v_error) + (KP_WT * w_error + KD_WT * d_w_error)
        # T_L = (KP_VT * v_error + KD_VT * d_v_error) - (KP_WT * w_error + KD_WT * d_w_error)

        # update memory
        # only update pos_error if we actually moving forward
        if v_goal != 0:
            self._pos_error = pos_error
        self._heading_error = heading_error
        # self._v_error = v_error
        # self._w_error = w_error

        max_v = 10 # m/s
        max_w = 180 / 180*np.pi # rad/s

        if np.abs(v_goal) > max_v:
            v_goal = np.sign(v_goal) * max_v
        if np.abs(w_goal) > max_w:
            w_goal = np.sign(w_goal) * max_w

        return np.array([v_goal, w_goal])


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    M = DifferentialDriveVelocityInput(sampling_period=0.01)
    C = PVelocityControl(M)
    curr_state = np.array([50,50,0])
    goal_pos = np.array([60,60])

    states = [curr_state]
    for i in range(1000):
        input_torque = C.control(curr_state, goal_pos)
        curr_state = M.step(curr_state, input_torque)
        states.append(curr_state)

    states = np.array(states)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(states[:,0])
    plt.ylabel("x")
    plt.subplot(3,1,2)
    plt.plot(states[:,1])
    plt.ylabel("y")
    plt.subplot(3,1,3)
    plt.plot(states[:,2])
    plt.ylabel("theta")
    plt.show()