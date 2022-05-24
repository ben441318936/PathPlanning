'''
Implements different state estimation schemes.

Uses the python controls toolbox.

Implements an abstract Estimator class that defines the basic estimator interface.
Estimator objects should take in a MotionModel object, and use the MotionModel utilities
for state and parameter extraction.
'''

from abc import ABC, abstractmethod
from typing import List

import numpy as np
np.set_printoptions(precision=5, suppress=True)

from Environment import ScanResult

from Map import OccupancyGrid, raytrace

import control

from MotionModel import MotionModel, DifferentialDriveTorqueToVelocity, DifferentialDriveTorqueInput, DifferentialDriveVelocityInput


class Estimator(ABC):
    '''
    Defines basic interface for estimator objects.
    '''

    __slots__ = ("_motion_model", "_estimate_state")

    def __init__(self, motion_model: MotionModel) -> None:
        self._motion_model = motion_model
        self.init_estimator()

    # initialize the estimator
    def init_estimator(self, init_state: np.ndarray = None) -> None:
        if init_state is None:
            self._estimate_state = np.zeros((self._motion_model.state_dim))
        else:
            self._estimate_state = init_state

    # predict the next state using known input and motion model
    @abstractmethod
    def predict(self, control_input: dict) -> None:
        pass

    # update the current state using latest observation
    @abstractmethod
    def update(self, observation: dict) -> None:
        '''
        Observation is a dictionary containing data from different sensors.
        '''
        pass

    # extract the most probable state for control use
    @property
    def estimate(self) -> np.ndarray:
        return self._estimate_state


class WheelVelocityEstimator(Estimator):
    '''
    Stationary Kalman Filter (linear quadratic estimator) for 
    linear and angular velocity using input torque and encoder reading.

    QN is covaraince of input noise.
    RN is covariance of output noise.
    '''

    __slots__ = ("_L", "_QN", "_RN", "_P")

    def __init__(self, motion_model: DifferentialDriveTorqueToVelocity, QN=np.eye(2), RN=np.eye(2)) -> None:
        super().__init__(motion_model) # calls init_estimator
        self._motion_model = motion_model # for linting typing
        self._QN = QN
        self._RN = RN
        self.compute_gain() # this sets self._L, the estimator gain

    @property
    def QN(self) -> np.ndarray:
        return self._QN

    @property
    def RN(self) -> np.ndarray:
        return self._RN

    @property
    def L(self) -> np.ndarray:
        return self._L

    @property
    def P(self) -> np.ndarray:
        return self._P

    def compute_gain(self) -> None:
        # continuous time model params
        A = np.array([[-self._motion_model.parameters["wheel friction"], 0], 
                      [0, -self._motion_model.parameters["wheel friction"]]])
        B_top = self._motion_model.parameters["wheel radius"] / 2 / self._motion_model.parameters["inertia"]
        B_bot = self._motion_model.parameters["wheel radius"] / self._motion_model.parameters["inertia"] / self._motion_model.parameters["axel length"]
        B = np.array([[B_top, B_top], 
                      [B_bot, -B_bot]])
        C_left = 1 / (self._motion_model.parameters["wheel radius"] / 2)
        C_right = 1 / (self._motion_model.parameters["wheel radius"] / self._motion_model.parameters["axel length"])
        C = 1/2 * np.array([[C_left, C_right],
                            [C_left, -C_right]])
        # convert to discrete time
        sys_c = control.ss(A, B, C, np.zeros((2,2)))
        sys_d = control.sample_system(sys_c, self._motion_model.sampling_period)
        # this lqe uses x_(t+1|t+1) = x_(t+1|t) + L @ (z - C x_(t+1|t))
        self._L, self._P, E = control.dlqe(sys_d, self._QN, self._RN)
        self._L = np.array(self._L)
        self._P = np.array(self._P)

    def predict(self, control_input: dict) -> None:
        self._estimate_state = self._motion_model.step(self._estimate_state, control_input)

    def update(self, observation: dict) -> None:
        if "ENCODER" in observation:
            self._estimate_state = self._estimate_state + self._L @ (observation["ENCODER"] - self._motion_model.state_2_wheel_velocity(self._estimate_state))


class PoseEstimator(Estimator):
    '''
    Uses Particle Filter with LIDAR measurements for pose estimation.

    Samples noisy velocity inputs to predict the pose.

    Update the pose using LIDAR scan correlation with some internal map.
    '''

    __slots__ = ("_map", "_velocity_cov", "_particles", "_num_particles")

    def __init__(self, motion_model: DifferentialDriveVelocityInput, map: OccupancyGrid, velocity_cov: np.ndarray = None, num_particles: int = 10) -> None:
        self._map: OccupancyGrid = map
        self._velocity_cov: np.ndarray = velocity_cov
        self._num_particles: int = num_particles
        
        super().__init__(motion_model) # calls init_estimator
        self._motion_model = motion_model # for linting typing

    def init_estimator(self, init_state: np.ndarray = None) -> None:
        super().init_estimator(init_state)

        # spread_mean = np.zeros((self._motion_model.state_dim))
        # spread_cov = np.diag(np.array([10,10,0.02]))
        # spread = np.random.multivariate_normal(spread_mean, spread_cov, self._num_particles)

        # creates a set of particles at the init_state
        self._particles = np.tile(init_state, (self._num_particles,1))
        
    def predict(self, control_input: dict) -> None:
        self._estimate_state = self._motion_model.step(self._estimate_state, control_input)
        # predict particles
        v_w_noise = np.random.multivariate_normal(np.zeros((self._motion_model.input_dim)), self._velocity_cov, self._num_particles)
        control_inputs = control_input.copy()
        control_inputs["v"] = control_inputs["v"] + v_w_noise[:,0]
        control_inputs["w"] = control_inputs["w"] + v_w_noise[:,1]
        self._particles = self._motion_model.step(self._particles, control_inputs)

    def update(self, observation: dict) -> None:
        if "LIDAR" in observation:
            corrs = self._compute_scan_correlations(observation["LIDAR"])
            self._estimate_state = self._particles[np.argmax(corrs),:]

    def _compute_scan_correlations(self, LIDAR_data: dict) -> np.ndarray:
        scans: List[ScanResult] = LIDAR_data["SCANS"]
        scan_max_range: float = LIDAR_data["MAX_RANGE"]

        # scans is N x (ang,rng)
        angs = np.array([scan.angle for scan in scans])
        rngs = np.array([scan.range for scan in scans])

        # process those that got inf range, i.e. did not hit an obstacle at all
        angs_inf = angs[rngs == np.inf].reshape((-1,1)) # (N,1)

         # process those that got finite range, i.e. hit an obstacle
        angs_hit = angs[rngs<np.inf].reshape((-1,1)) # (N,1)
        rngs_hit = rngs[rngs<np.inf].reshape((-1,1)) # (N,1)

        corrs = np.zeros((self._num_particles))

        for i in range(self._num_particles):
            scan_pose = self._motion_model.state_2_pose(self._particles[i,:])
            ray_start = self._map.convert_to_grid_coord(scan_pose[0:2])
            center_angle = scan_pose[2]

            endpoints = scan_pose[0:2] + scan_max_range * np.hstack((np.cos(angs_inf+center_angle), np.sin(angs_inf+center_angle))) # (N,2)
            for j in range(endpoints.shape[0]):
                endpoint = endpoints[j,:]
                endpoint = self._map.convert_to_grid_coord(endpoint)
                ray_xx, ray_yy = raytrace(ray_start[0], ray_start[1], endpoint[0], endpoint[1])
                # for empty space, negative map status is correct
                corrs[i] += np.sum(-1 * self._map.get_status((ray_xx, ray_yy)))

            endpoints = scan_pose[0:2] + rngs_hit * np.hstack((np.cos(angs_hit+center_angle), np.sin(angs_hit+center_angle))) # (N,2)
            for j in range(endpoints.shape[0]):
                endpoint = endpoints[j,:]
                endpoint = self._map.convert_to_grid_coord(endpoint)
                ray_xx, ray_yy = raytrace(ray_start[0], ray_start[1], endpoint[0], endpoint[1])
                # for obstacle, positive map status is correct
                corrs[i] += self._map.get_status((ray_xx[-1], ray_yy[-1]))
                # for empty space, negative map status is correct
                corrs[i] += np.sum(-1 * self._map.get_status((ray_xx[0:-1], ray_yy[0:-1])))


class FullStateEstimator(Estimator):
    '''
    Estimates the full state of a torque controlled differential drive model.

    Uses Particle Filter with LIDAR measurements for pose estimation.

    Uses stionary Kalman Filter (linear quadratic estimator) with encoder for wheel velocity estimation.
    Implemented as a WheelSpeedEstimator object.
    QN is covaraince of input torque noise.
    RN is covariance of wheel encoder noise.

    Because the torque-to-wheel-velocity and velocity-to-pose processes can be decoupled 
    (it's really cascaded systems), we do their estimation separately.

    The LQE will use the torque control action and encoder measurements to estimate wheel velocity.
    We know that the startionary distribution of wheel velocity satisfies phi ~ N(phi_hat, P),
    where phi_hat is the best estimated phi and P is the covariance of the estimate (conditional mean
    and stationary conditional covariance in Kalman Filter terms).
    P is given as the solution to the DARE used in LQE gain computation.

    We will sample from the phi distribution and use that as input to predict the pose.
    Then we will update the pose using LIDAR scan correlation with some internal map.
    '''

    __slots__ = ("_wheel_velocity_estimator", "_pose_estimator")

    def __init__(self, motion_model: DifferentialDriveTorqueInput, map: OccupancyGrid, QN=np.eye(2), RN=np.eye(2), ) -> None:
        self._wheel_velocity_estimator = WheelVelocityEstimator(motion_model.torque_to_velocity_submodel, QN=QN, RN=RN)
        self._pose_estimator = PoseEstimator(map, motion_model.velocity_input_submodel, velocity_cov=self._wheel_velocity_estimator.P)

        super().__init__(motion_model) # calls init_estimator
        self._motion_model = motion_model # for linting typing

    def init_estimator(self, init_state: np.ndarray = None) -> None:
        super().init_estimator(init_state)
        if init_state is None:
            self._wheel_velocity_estimator.init_estimator(None)
        else:
            self._wheel_velocity_estimator.init_estimator(self._motion_model.state_2_wheel_velocity(init_state))

    @property
    def L(self):
        return self._wheel_velocity_estimator.L

    @property
    def P(self):
        return self._wheel_velocity_estimator.P

    def predict(self, control_input: dict) -> None:
        # predict wheel speed
        self._wheel_velocity_estimator.predict(control_input)
        # predict full state
        self._estimate_state = self._motion_model.step(self._estimate_state, control_input)
        self._estimate_state[3:5] = self._wheel_velocity_estimator.estimate

    def update(self, observation: dict) -> None:
        self._wheel_velocity_estimator.update(observation)
        # update pose
        self._estimate_state[3:5] = self._wheel_velocity_estimator.estimate


def test_wheel_velocity_estimator(init_velocity=np.array([0,0]), goal_velocity=np.array([1,1])):
    import matplotlib.pyplot as plt

    from Controller import SSTorqueController

    input_noise_var = 1*np.eye(2)
    output_noise_var = 0.001*np.eye(2)

    # create motion model
    M = DifferentialDriveTorqueToVelocity(sampling_period=0.01)

    # create controller
    C = SSTorqueController(M)

    # create estimator
    E = WheelVelocityEstimator(M, QN=input_noise_var, RN=output_noise_var)
    curr_state = init_velocity
    E.init_estimator(curr_state)

    real_states = [curr_state]
    estimated_states = [curr_state]
    errors = [np.zeros((2))]

    for i in range(1000):
        control_action = C.control(E.estimate, goal_velocity)

        input_noise = np.random.multivariate_normal(np.zeros((2,)), input_noise_var, size=None)
        noisy_input = control_action.copy()
        for (name, i) in zip(M.input_names, range(input_noise.shape[0])):
            noisy_input[name] += input_noise[i]
        curr_state = M.step(curr_state, noisy_input)
        real_states.append(curr_state)

        E.predict(control_action)
        E.update(M.state_2_wheel_velocity(curr_state) + np.random.multivariate_normal(np.zeros((2,)), output_noise_var, size=None))
        estimated_states.append(E.estimate)

        errors.append(real_states[-1] - estimated_states[-1])

    real_states = np.array(real_states)
    estimated_states = np.array(estimated_states)
    errors = np.array(errors)

    print("var err v:", np.var(errors[:,0]))
    print("var err w:", np.var(errors[:,1]))
    print("computed stationary err var:\n", E.P)

    plt.figure()

    plt.subplot(2,3,1)
    plt.plot(real_states[:,0])
    plt.ylabel("real v")
    plt.subplot(2,3,2)
    plt.plot(estimated_states[:,0])
    plt.ylabel("esti v")
    plt.subplot(2,3,3)
    plt.plot(errors[:,0])
    plt.ylabel("erro v")

    plt.subplot(2,3,4)
    plt.plot(real_states[:,1])
    plt.ylabel("real w")
    plt.subplot(2,3,5)
    plt.plot(estimated_states[:,1])
    plt.ylabel("esti w")
    plt.subplot(2,3,6)
    plt.plot(errors[:,1])
    plt.ylabel("erro w")

    plt.tight_layout(pad=2)
    plt.show()

def test_pose_estimator(init_pose=np.array([50,50,0]), goal_pos=np.array([55,55])):
    import matplotlib.pyplot as plt

    from Controller import PVelocityController

    input_noise_var = np.diag(np.array([0.05, 0.005]))

    M = DifferentialDriveVelocityInput(sampling_period=0.01)
    C = PVelocityController(M)
    E = PoseEstimator(M, velocity_cov=input_noise_var, num_particles=2)
    
    curr_state = init_pose
    E.init_estimator(curr_state)

    real_states = [curr_state]
    estimated_states = [curr_state]
    errors = [np.zeros((3))]

    for i in range(1000):
        control_action = C.control(E.estimate, goal_pos)

        input_noise = np.random.multivariate_normal(np.zeros((2,)), input_noise_var, size=None)
        noisy_input = control_action.copy()
        for (name, i) in zip(M.input_names, range(input_noise.shape[0])):
            noisy_input[name] += input_noise[i]
        curr_state = M.step(curr_state, noisy_input)
        real_states.append(curr_state)

        E.predict(control_action)
        estimated_states.append(E.estimate)
        E.update(None)

        errors.append(real_states[-1] - estimated_states[-1])

    real_states = np.array(real_states)
    estimated_states = np.array(estimated_states)
    errors = np.array(errors)

    plt.figure()
    
    plt.subplot(3,3,1)
    plt.plot(real_states[:,0])
    plt.ylabel("real x")
    plt.ylim(49,56)
    plt.subplot(3,3,2)
    plt.plot(estimated_states[:,0])
    plt.ylabel("esti x")
    plt.ylim(49,56)
    plt.subplot(3,3,3)
    plt.plot(errors[:,0])
    plt.ylabel("erro x")

    plt.subplot(3,3,4)
    plt.plot(real_states[:,1])
    plt.ylabel("real y")
    plt.ylim(49,56)
    plt.subplot(3,3,5)
    plt.plot(estimated_states[:,1])
    plt.ylabel("esti y")
    plt.ylim(49,56)
    plt.subplot(3,3,6)
    plt.plot(errors[:,1])
    plt.ylabel("erro y")

    plt.subplot(3,3,7)
    plt.plot(real_states[:,2])
    plt.ylabel("real theta")
    plt.subplot(3,3,8)
    plt.plot(estimated_states[:,2])
    plt.ylabel("esti theta")
    plt.subplot(3,3,9)
    plt.plot(errors[:,2])
    plt.ylabel("erro theta")

    plt.tight_layout(pad=2)
    plt.show()

def test_full_estimator(init_state=np.array([50,50,0,0,0]), goal_pos=np.array([55,55])):
    import matplotlib.pyplot as plt

    from Controller import PVelocitySSTorqueController

    input_noise_var = 1*np.eye(2)
    output_noise_var = 0.001*np.eye(2)

    # create motion model
    M = DifferentialDriveTorqueInput(sampling_period=0.01)

    # create controller
    C = PVelocitySSTorqueController(M)

    # create estimator
    E = FullStateEstimator(M, QN=input_noise_var, RN=output_noise_var)
    curr_state = init_state
    E.init_estimator(curr_state)

    real_states = [curr_state]
    estimated_states = [curr_state]
    errors = [np.zeros((5))]

    for i in range(1000):
        control_action = C.control(E.estimate, goal_pos)

        input_noise = np.random.multivariate_normal(np.zeros((2,)), input_noise_var, size=None)
        noisy_input = control_action.copy()
        for (name, i) in zip(M.input_names, range(input_noise.shape[0])):
            noisy_input[name] += input_noise[i]
        curr_state = M.step(curr_state, noisy_input)
        real_states.append(curr_state)

        E.predict(control_action)
        E.update(M.state_2_wheel_velocity(curr_state) + np.random.multivariate_normal(np.zeros((2,)), output_noise_var, size=None))
        estimated_states.append(E.estimate)

        errors.append(real_states[-1] - estimated_states[-1])

    real_states = np.array(real_states)
    estimated_states = np.array(estimated_states)
    errors = np.array(errors)

    print("var err v:", np.var(errors[:,3]))
    print("var err w:", np.var(errors[:,4]))
    print("computed stationary err var:\n", E.P)

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
    plt.ylabel("real v")
    plt.subplot(5,3,11)
    plt.plot(estimated_states[:,3])
    plt.ylabel("esti v")
    plt.subplot(5,3,12)
    plt.plot(errors[:,3])
    plt.ylabel("erro v")

    plt.subplot(5,3,13)
    plt.plot(real_states[:,4])
    plt.ylabel("real w")
    plt.subplot(5,3,14)
    plt.plot(estimated_states[:,4])
    plt.ylabel("esti w")
    plt.subplot(5,3,15)
    plt.plot(errors[:,4])
    plt.ylabel("erro w")

    plt.tight_layout(pad=2)
    plt.show()


if __name__ == "__main__":
    # test_wheel_velocity_estimator()
    test_pose_estimator()
    # test_full_estimator()
    


