'''
Implements different control schemes.

Uses the python controls toolbox.

Implements an abstract Controller class that defines the basic controller interface.
Controller objects should take in a MotionModel object, and use the MotionModel utilities
for state and parameter extraction.
'''

from abc import ABC, abstractmethod
import numpy as np
import control

from MotionModels import MotionModel, DifferentialDrive, DifferentialDriveVelocityInput

class Controller(ABC):
    '''
    Defines basic interface for controller objects.
    '''

    __slots__ = ("_motion_model")

    def __init__(self, motion_model: MotionModel) -> None:
        self._motion_model = motion_model

    @property
    def motion_model(self) -> MotionModel:
        return self._motion_model

    @abstractmethod
    def control(self) -> np.ndarray:
        pass

class PVelocityControl(Controller):
    '''
    Implements proportional control for linear rate and yaw rate
    using position and heading feedback.

    Velocity is proportional to L2 distance between curr and target position.
    Yaw rate is proportional to heading error.
    '''
    
    __slots__ = ("_max_rpm", "_max_phi")

    def __init__(self, motion_model: DifferentialDriveVelocityInput, max_rpm: float = 60) -> None:
        super().__init__(motion_model)
        self._motion_model = motion_model
        self._max_rpm = max_rpm
        self._max_phi = self._max_rpm / 60 * 2 * np.pi

    def control(self, curr_state: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        curr_pos = self._motion_model.state_2_position(curr_state)
        curr_heading = self._motion_model.state_2_heading(curr_state)
        curr_heading = np.arctan2(np.sin(curr_heading), np.cos(curr_heading))
        goal_heading = np.arctan2(goal_pos[1] - curr_pos[1], goal_pos[0] - curr_pos[0])

        # outer loop, set reference v and w based on position and heading error
        pos_error = np.linalg.norm(goal_pos - curr_pos)
        # if close enough, don't move
        if pos_error < 0.5:
            return self._motion_model.create_velocities_dict(v=0, w=0)
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

        if not (v == 0 and w == 0):
            # adjust reference according to max rpm constraints
            phi_r = 1/2 * (v / (self._motion_model.parameters["wheel radius"]/2) + 
                        w / (self._motion_model.parameters["wheel radius"] / self._motion_model.parameters["axel length"]))
            phi_l = 1/2 * (v / (self._motion_model.parameters["wheel radius"]/2) -
                        w / (self._motion_model.parameters["wheel radius"] / self._motion_model.parameters["axel length"]))
            phi = np.array([phi_r, phi_l])
            phi_clip = np.clip(phi, -self._max_phi, self._max_phi)
            ratio = phi_clip / phi
            min_ratio = np.nanmin(ratio)

            v = v * min_ratio
            w = w * min_ratio

        return self._motion_model.create_velocities_dict(v=v, w=w)

class PVelocitySSTorqueControl(PVelocityControl):
    '''
    Implements full state feedback torque control based on velocities reference.

    Control law is based on the state space model from torque to velocity.

    Gain is computed using LQR. Adjust Q and R to change gain.
    '''

    __slots__ = ("_Q", "_R", "_K", "_max_torque")

    def __init__(self, motion_model: DifferentialDrive, Q=np.eye(2), R=np.eye(2), max_torque: float = 100) -> None:
        super().__init__(motion_model)
        self._motion_model = motion_model
        self._Q = Q
        self._R = R
        self._max_torque = max_torque
        self.compute_gain() # this sets self._K, the feedback control gain

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def K(self) -> np.ndarray:
        return self._K

    def compute_gain(self):
        # continuous time model params
        A = np.array([[-self._motion_model.parameters["wheel friction"], 0], 
                      [0, -self._motion_model.parameters["wheel friction"]]])
        B_top = self._motion_model.parameters["wheel radius"] / 2 / self._motion_model.parameters["inertia"]
        B_bot = self._motion_model.parameters["wheel radius"] / self._motion_model.parameters["inertia"] / self._motion_model.parameters["axel length"]
        B = np.array([[B_top, B_top], 
                      [B_bot, -B_bot]])
        # convert to discrete time
        sys_c = control.ss(A, B, np.eye(2), np.zeros((2,2)))
        sys_d = control.sample_system(sys_c, self._motion_model.sampling_period)
        # this lqr uses A-BK
        self._K, S, E = control.lqr(sys_d, self._Q, self._R)
        self._K = np.array(self._K)
        
    def control(self, curr_state: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        v_w_ref = super().control(curr_state, goal_pos)
        v_w_ref = np.array([v_w_ref["v"],v_w_ref["w"]])
        curr_v = self._motion_model.state_2_velocity(curr_state)
        curr_w = self._motion_model.state_2_yaw_rate(curr_state)
        # K was computed using A-BK
        T = -self._K @ (np.array([curr_v, curr_w] - v_w_ref))
        T = np.clip(T, -self._max_torque, self._max_torque)
        return self._motion_model.create_torque_dict(T_R=T[0], T_L=T[1])


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