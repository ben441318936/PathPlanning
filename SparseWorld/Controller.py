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

from MotionModel import MotionModel, DifferentialDriveTorqueToVelocity,  DifferentialDriveTorqueInput, DifferentialDriveVelocityInput

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
    def control(self) -> dict:
        pass

class PVelocityController(Controller):
    '''
    Implements proportional control for linear velocity and yaw rate
    using position and heading (pose) feedback.

    Velocity is proportional to L2 distance between curr and target position.
    Yaw rate is proportional to heading error.
    '''
    
    __slots__ = ("_max_rpm", "_max_phi", "_KP_V", "_KP_W")

    def __init__(self, motion_model: DifferentialDriveVelocityInput, KP_V: float = 4, KP_W: float = 100, max_rpm: float = 60) -> None:
        super().__init__(motion_model)
        self._motion_model = motion_model
        self._max_rpm = max_rpm
        self._max_phi = self._max_rpm / 60 * 2 * np.pi
        self._KP_V = KP_V
        self._KP_W = KP_W

    def control(self, curr_pose: np.ndarray, goal_pos: np.ndarray) -> dict:
        curr_pos = self._motion_model.state_2_position(curr_pose)
        curr_heading = self._motion_model.state_2_heading(curr_pose)
        curr_heading = np.arctan2(np.sin(curr_heading), np.cos(curr_heading))
        goal_heading = np.arctan2(goal_pos[1] - curr_pos[1], goal_pos[0] - curr_pos[0])

        # outer loop, set reference v and w based on position and heading error
        pos_error = np.linalg.norm(goal_pos - curr_pos)
        # if close enough, don't move
        if pos_error < 0.5:
            v, w = 0.0, 0.0
        else:
            heading_error = np.arctan2(np.sin(goal_heading-curr_heading), np.cos(goal_heading-curr_heading))

            # ensure that the robot is moving as straight as possible
            # this will be needed if planner outputs straight line paths
            # make this value large to let the robot move smoothly
            if np.abs(heading_error) > 0.1:
                v = 0 
            else:
                v = self._KP_V * pos_error

            w = self._KP_W * heading_error

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

        return self._motion_model.create_input_dict(v, w)


class SSTorqueController(Controller):
    '''
    Implements full state feedback torque control based on [v,w] reference.

    Full state feedback is based on the state space model from torque to velocity.

    Gain is computed using LQR. Q is weight on state, R is weight on input.
    '''

    __slots__ = ("_Q", "_R", "_K", "_max_torque")

    def __init__(self, motion_model: DifferentialDriveTorqueToVelocity, Q=np.diag(np.array([1000,2000])), R=np.eye(2), max_torque: float = 100) -> None:
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
        
    def control(self, curr_velocity: np.ndarray, goal_velocity: np.ndarray) -> dict:
        '''
        Goal velocity is formatted as [v,w]
        '''
        # K was computed using A-BK
        T = -self._K @ (curr_velocity - goal_velocity)
        T = np.clip(T, -self._max_torque, self._max_torque)
        return self._motion_model.create_input_dict(T[0],T[1])


class PVelocitySSTorqueController(Controller):
    '''
    Implements full state feedback torque control based on velocities reference.

    Velocity references are computed using proportional control.
    Implemented as a PVelocityController object.

    Full state feedback is based on the state space model from torque to velocity.

    Gain is computed using LQR. Q is weight on state, R is weight on input.
    '''

    __slots__ = ("_velocity_to_pose_controller", "_torque_to_velocity_controller")

    def __init__(self, motion_model: DifferentialDriveTorqueInput, KP_V: float = 4, KP_W: float = 100, max_rpm: float = 60, Q=np.diag(np.array([1000,2000])), R=np.eye(2), max_torque: float = 100) -> None:
        super().__init__(motion_model)
        self._motion_model = motion_model
        self._velocity_to_pose_controller = PVelocityController(motion_model.velocity_input_submodel, KP_V=KP_V, KP_W=KP_W, max_rpm=max_rpm)
        self._torque_to_velocity_controller = SSTorqueController(motion_model.torque_to_velocity_submodel, Q=Q, R=R, max_torque=max_torque)

    @property
    def Q(self) -> np.ndarray:
        return self._torque_to_velocity_controller._Q

    @property
    def R(self) -> np.ndarray:
        return self._torque_to_velocity_controller._R

    @property
    def K(self) -> np.ndarray:
        return self._torque_to_velocity_controller._K
        
    def control(self, curr_state: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        v_w_ref = self._velocity_to_pose_controller.control(self._motion_model.state_2_pose(curr_state), goal_pos)
        v_w_names = self._motion_model.velocity_input_submodel.input_names
        for key, value in v_w_ref.items():
            if key == v_w_names[0]:
                v = value
            if key == v_w_names[1]:
                w = value
        v_w_ref = np.array([v, w])
        curr_v_w = np.array([self._motion_model.state_2_velocity(curr_state), self._motion_model.state_2_yaw_rate(curr_state)])
        return self._torque_to_velocity_controller.control(curr_v_w, v_w_ref)


def test_torque_to_velocity(init_velocity=np.array([0,0]), goal_velocity=np.array([1,0])):
    import matplotlib.pyplot as plt

    M = DifferentialDriveTorqueToVelocity(sampling_period=0.01)
    C = SSTorqueController(M, Q=np.diag(np.array([1000,2000])), max_torque=100)

    curr_state = init_velocity
    
    states = [curr_state]
    for i in range(1000):
        inputs = C.control(curr_state, goal_velocity)
        curr_state = M.step(curr_state, inputs)
        states.append(curr_state)

    states = np.array(states)

    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(states[:,0])
    plt.ylabel("v")

    plt.subplot(2,1,2)
    plt.plot(states[:,1])
    plt.ylabel("w")

    plt.tight_layout(pad=2)
    plt.show()

def test_velocity_to_pose(init_pose=np.array([50,50,0]), goal_pos=np.array([55,55])):
    import matplotlib.pyplot as plt

    M = DifferentialDriveVelocityInput(sampling_period=0.01)
    C = PVelocityController(M, KP_V=4, KP_W=100, max_rpm=60)

    curr_state = init_pose
    
    states = [curr_state]
    for i in range(1000):
        inputs = C.control(curr_state, goal_pos)
        curr_state = M.step(curr_state, inputs)
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

    plt.tight_layout(pad=2)
    plt.show()
    
def test_full_state(init_state=np.array([50,50,0,0,0]), goal_pos=np.array([55,55])):
    import matplotlib.pyplot as plt

    M = DifferentialDriveTorqueInput(sampling_period=0.01)
    C = PVelocitySSTorqueController(M, KP_V=4, KP_W=100, max_rpm=60, Q=np.diag(np.array([1000,2000])), max_torque=100)

    curr_state = init_state

    states = [curr_state]
    for i in range(1000):
        inputs = C.control(curr_state, goal_pos)
        curr_state = M.step(curr_state, inputs)
        states.append(curr_state)

    states = np.array(states)

    plt.figure()

    plt.subplot(5,1,1)
    plt.plot(states[:,0])
    plt.ylabel("x")

    plt.subplot(5,1,2)
    plt.plot(states[:,1])
    plt.ylabel("y")

    plt.subplot(5,1,3)
    plt.plot(states[:,2])
    plt.ylabel("theta")

    plt.subplot(5,1,4)
    plt.plot(states[:,3])
    plt.ylabel("v")

    plt.subplot(5,1,5)
    plt.plot(states[:,4])
    plt.ylabel("w")

    plt.tight_layout(pad=2)
    plt.show()


if __name__ == "__main__":
    test_torque_to_velocity()
    # test_velocity_to_pose()
    # test_full_state()