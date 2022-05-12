'''
Implements different state estimation schemes.

Uses the python controls toolbox.

Implements an abstract Estimator class that defines the basic estimator interface.
Estimator objects should take in a MotionModel object, and use the MotionModel utilities
for state and parameter extraction.
'''

from abc import ABC, abstractmethod
import numpy as np
import control

from MotionModel import DifferentialDriveTorqueToWheelVelocity, MotionModel, DifferentialDriveTorqueInput, DifferentialDriveVelocityInput

class Estimator(ABC):
    '''
    Defines basic interface for estimator objects.
    '''

    __slots__ = ("_motion_model", "_estimate_state")

    def __init__(self, motion_model: MotionModel) -> None:
        self._motion_model = motion_model

    # initialize the estimator
    def init_estimator(self, init_state: np.ndarray = None) -> None:
        if init_state is None:
            self._estimate_state = np.zeros((self._motion_model.state_dim))
        else:
            self._estimate_state = init_state

    # predict the next state using known input and motion model
    @abstractmethod
    def predict(self, control_input) -> None:
        pass

    # update the current state using latest observation
    @abstractmethod
    def update(self, observation) -> None:
        pass

    # extract the most probable state for control use
    @property
    def estimate(self) -> np.ndarray:
        return self._estimate_state

class WheelVelocityEstimator(Estimator):
    '''
    Stationary Kalman Filter for the wheel velocity using torque and encoder reading.

    QN is covaraince of input noise.
    RN is covariance of output noise.
    '''

    __slots__ = ("_phi", "_L", "_QN", "_RN")

    def __init__(self, motion_model: DifferentialDriveTorqueToWheelVelocity, QN=np.eye(2), RN=np.eye(2)) -> None:
        super().__init__(motion_model)
        self._motion_model = motion_model
        self._QN = QN
        self._RN = RN
        self.compute_gain() # this sets self._L, the estimator gain
        self.init_estimator(None)

    @property
    def QN(self) -> np.ndarray:
        return self._QN

    @property
    def RN(self) -> np.ndarray:
        return self._RN

    @property
    def L(self) -> np.ndarray:
        return self._L

    def compute_gain(self) -> None:
        # continuous time model params
        A = np.array([[-self._motion_model.parameters["wheel friction"], 0], 
                      [0, -self._motion_model.parameters["wheel friction"]]])
        B = np.array([[1/self._motion_model.parameters["inertia"], 0], 
                      [0, 1/self._motion_model.parameters["inertia"]]])
        C = np.eye(2)
        # convert to discrete time
        sys_c = control.ss(A, B, C, np.zeros((2,2)))
        sys_d = control.sample_system(sys_c, self._motion_model.sampling_period)
        # this lqe uses x_(t+1|t+1) = x_(t+1|t) + L @ (z - C x_(t+1|t))
        self._L, P, E = control.dlqe(sys_d, self._QN, self._RN)
        self._L = np.array(self._L)

    def predict(self, control_input) -> None:
        self._estimate_state = self._motion_model.step(self._estimate_state, control_input)

    def update(self, observation) -> None:
        self._estimate_state = self._estimate_state + self._L @ (observation - self._estimate_state)

class ParticleKalmanEstimator(Estimator):
    '''
    Particle Filter with LIDAR measurements for position and heading estimation.

    Includes a stionary Kalman Filter for wheel speed estimation.
    Implemented as a WheelSpeedEstimator object.

    QN is covaraince of input torque noise.
    RN is covariance of wheel encoder noise.
    '''

    __slots__ = ("_wheel_velocity_estimator")

    def __init__(self, motion_model: DifferentialDriveTorqueInput, QN=np.eye(2), RN=np.eye(2)) -> None:
        super().__init__(motion_model)
        self._motion_model = motion_model
        self._wheel_velocity_estimator = WheelVelocityEstimator(motion_model.torque_to_velocity_submodel, QN=QN, RN=RN)
        self._wheel_velocity_estimator.compute_gain() # this sets observer gain
        self.init_estimator(None)
        self._wheel_velocity_estimator.init_estimator(None)

    def init_estimator(self, init_state: np.ndarray = None) -> None:
        super().init_estimator(init_state)
        if init_state is None:
            self._wheel_velocity_estimator.init_estimator(None)
        else:
            self._wheel_velocity_estimator.init_estimator(self._motion_model.state_2_wheel_velocity(init_state))

    def predict(self, control_input) -> None:
        # predict wheel speed
        self._wheel_velocity_estimator.predict(control_input)
        self._estimate_state = self._motion_model.step(self._estimate_state, control_input)
        self._estimate_state[self._motion_model.wheel_velocity_state_idx] = self._wheel_velocity_estimator.estimate

    def update(self, observation) -> None:
        # update wheel speed
        self._wheel_velocity_estimator.update(observation)
        self._estimate_state[self._motion_model.wheel_velocity_state_idx] = self._wheel_velocity_estimator.estimate

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from Controller import PVelocitySSTorqueController

    input_noise_var = 1*np.eye(2)
    output_noise_var = 0.001*np.eye(2)

    # create motion model
    M = DifferentialDriveTorqueInput(sampling_period=0.01)

    # create controller
    C = PVelocitySSTorqueController(M)

    # create estimator
    E = ParticleKalmanEstimator(M, QN=input_noise_var, RN=output_noise_var)
    curr_state = np.array([50,50,0,0,0])
    E.init_estimator(curr_state)

    goal_pos = np.array([55,55])

    real_states = [curr_state]
    estimated_states = [curr_state]
    errors = [np.zeros((5))]

    for i in range(10000):
        input_torque = C.control(E.estimate, goal_pos)

        input_torque_noise = input_torque.copy()
        input_noise = np.random.multivariate_normal(np.zeros((2,)), input_noise_var, size=None)
        input_torque_noise["T_R"] += input_noise[0]
        input_torque_noise["T_L"] += input_noise[1]
        curr_state = M.step(curr_state, input_torque_noise)
        real_states.append(curr_state)

        E.predict(input_torque)
        E.update(M.state_2_wheel_velocity(curr_state) + np.random.multivariate_normal(np.zeros((2,)), output_noise_var, size=None))
        estimated_states.append(E.estimate)

        errors.append(real_states[-1] - estimated_states[-1])

    real_states = np.array(real_states)
    estimated_states = np.array(estimated_states)
    errors = np.array(errors)

    print("var err phi R:", np.var(errors[:,3]))
    print("var err phi L:", np.var(errors[:,4]))

    plt.figure()
    
    plt.subplot(5,3,1)
    plt.plot(real_states[:,0])
    plt.ylabel("real x")
    plt.ylim(49,56)
    plt.subplot(5,3,2)
    plt.plot(estimated_states[:,0])
    plt.ylabel("esti x")
    plt.ylim(49,56)
    plt.subplot(5,3,3)
    plt.plot(errors[:,0])
    plt.ylabel("erro x")

    plt.subplot(5,3,4)
    plt.plot(real_states[:,1])
    plt.ylabel("real y")
    plt.ylim(49,56)
    plt.subplot(5,3,5)
    plt.plot(estimated_states[:,1])
    plt.ylabel("esti y")
    plt.ylim(49,56)
    plt.subplot(5,3,6)
    plt.plot(errors[:,1])
    plt.ylabel("erro y")

    plt.subplot(5,3,7)
    plt.plot(real_states[:,2])
    plt.ylabel("real theta")
    plt.subplot(5,3,8)
    plt.plot(estimated_states[:,2])
    plt.ylabel("esti theta")
    plt.subplot(5,3,9)
    plt.plot(errors[:,2])
    plt.ylabel("erro theta")

    plt.subplot(5,3,10)
    plt.plot(real_states[:,3])
    plt.ylabel("real phi R")
    plt.subplot(5,3,11)
    plt.plot(estimated_states[:,3])
    plt.ylabel("esti phi R")
    plt.subplot(5,3,12)
    plt.plot(errors[:,3])
    plt.ylabel("erro phi R")

    plt.subplot(5,3,13)
    plt.plot(real_states[:,4])
    plt.ylabel("real phi L")
    plt.subplot(5,3,14)
    plt.plot(estimated_states[:,4])
    plt.ylabel("esti phi L")
    plt.subplot(5,3,15)
    plt.plot(errors[:,4])
    plt.ylabel("erro phi L")

    plt.tight_layout(pad=2)
    plt.show()


