'''
Implements motion models for the robot.

Motion models define how a robot's state transitions with some known input.
They do not store state information for the robot.
Outside classes should implement state storage. 
For example, an Environment class should store the state of a robot moving in it.
An Agent class should store the (estimated) state of the robot as it moves, but 
might also stored a vectorized set of possible states, depending on the estimation method.

An abstract MotionModel class defines the basic interface and attributes of a motion model.
'''

from abc import ABC, abstractmethod
import numpy as np

class MotionModel(ABC):
    '''
    Defines MotionModel interface
    '''

    __slots__ = ("_tau", "_parameters", "_state_dim", "_input_dim")

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        self._parameters = {}
        self._tau = sampling_period
        self._state_dim = 0
        self._input_dim = 0
        if parameters_dict is not None:
            self._parameters.update(parameters_dict)

    @property
    def sampling_period(self):
        return self._tau

    @property
    def parameters(self):
        return self._parameters

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def input_dim(self):
        return self._input_dim

    @abstractmethod
    def step(self, state: np.ndarray, input_dict: dict) -> None:
        pass

    @property
    def position_state_idx(self) -> tuple:
        pass

    @abstractmethod
    def state_2_position(self, state: np.ndarray) -> np.ndarray:
        pass

    @property
    def heading_state_idx(self) -> int:
        pass

    @abstractmethod
    def state_2_heading(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def state_2_wheel_speed(self, state: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        pass

class DifferentialDriveVelocityInput(MotionModel):
    '''
    State is defined as [x,y,theta]:
        x: x coordinate
        y: y coordinate
        theta: heading

    Input is defined as [v,w]:
        v: speed in the direction of heading
        w: yaw rate (rate of change of heading)

    Use create_velocities_dict(v,w) for input formatting.
    '''

    __slots__ = ()

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        params = {
            "wheel radius": 0.5,
            "axel length": 1,
        }
        if parameters_dict is not None:
            params.update(parameters_dict)
        super().__init__(sampling_period, params)
        self._state_dim = 3
        self._input_dim = 2

    def create_velocities_dict(self, v=0.0, w=0.0) -> dict:
        return {"v": v, "w": w}

    def step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        '''
        State is defined as [x,y,theta]

        Input is defined as [v,w]

        Supports vectorized operations for N states and inputs
        States [N,3]
        v [N,1]
        w [N,1]
        '''

        input_velocities = np.vstack((input_dict["v"], input_dict["w"]))

        state = state.reshape((-1,3))
        N = state.shape[0]
        input_velocities = input_velocities.reshape((-1,2))

        v = input_velocities[:,0]
        w = input_velocities[:,1]

        new_state = state + self._tau * np.vstack((v * np.cos(state[:,2]),
                                                   v * np.sin(state[:,2]),
                                                   w)).T
        if N > 1:
            return new_state
        else:
            return np.squeeze(new_state, axis=0)

    @property
    def position_state_idx(self) -> tuple:
        return slice(0,2,None)

    def state_2_position(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:2]
        else:
            return state[0,0:2]

    @property
    def heading_state_idx(self) -> tuple:
        return 2

    def state_2_heading(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,2]
        else:
            return state[0,2]

    def state_2_wheel_speed(self, state: np.ndarray) -> np.ndarray:
        return None
    
    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        return None

    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        return None


class DifferentialDrive(DifferentialDriveVelocityInput):
    '''
    State is defined as [x,y,theta,phi_R,phi_L]:
        x: x coordinate
        y: y coordinate
        theta: heading
        phi_R: angular velocity of right wheel
        phi_L: angular velocity of left wheel

    Input is defined as [T_R,T_L]:
        T_R: torque on right wheel
        T_L: torque on left wheel

    Use create_torque_dict(T_R,T_L) for input formatting.

    Parameters are:
        wheel radius
        robot mass: includes body and both wheels
        axel length
        wheel friction: [0,1): controls how much the wheel velocity decays due to friction
    '''

    __slots__ = ()

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        params = {
            "robot mass": 20,
            "wheel friction": 0.5,
        }
        if parameters_dict is not None:
            params.update(parameters_dict)
        super().__init__(sampling_period, params)
        if "inertia" not in self._parameters:
            self._parameters["inertia"] = self._parameters["robot mass"] / 2 * self._parameters["wheel radius"]**2
        self._state_dim = 5
        self._input_dim = 2
        
    def create_torque_dict(self, T_R=0.0, T_L=0.0) -> dict:
        return {"T_R": T_R, "T_L": T_L}

    def step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        '''
        State is defined as [x,y,theta,phi_R,phi_L]
        Input is defined as [T_R,T_L]

        Supports vectorized operations for N states and inputs
        States [N,5]
        Inputs [N,2]
        '''

        state = state.reshape((-1,5))
        N = state.shape[0]

        v = (state[:,3] + state[:,4]) * self._parameters["wheel radius"] / 2
        w = (state[:,3] - state[:,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]

        new_pos_head = super().step(state[:,0:3], self.create_velocities_dict(v=v, w=w))
        new_pos_head = new_pos_head.reshape((-1,3))

        new_phis = self.torque_to_phi_step(state[:,3:5], input_dict)
        new_phis = new_phis.reshape((-1, 2))

        new_state = np.hstack((new_pos_head, new_phis))

        if N > 1:
            return new_state
        else:
            return np.squeeze(new_state, axis=0)

    def torque_to_phi_step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        state = state.reshape((-1,2))
        N = state.shape[0]

        input_torque = np.vstack((input_dict["T_R"], input_dict["T_L"]))
        
        input_torque = input_torque.reshape((-1,2))
        # input_torque = np.clip(input_torque, -self._parameters["max motor torque"], self._parameters["max motor torque"])
        
        a = input_torque / self._parameters["inertia"]
        
        phi_decay = np.exp(-self._parameters["wheel friction"] * self.sampling_period)
        phis_decay = np.array([phi_decay,phi_decay]).reshape((-1,2))

        new_phis = phis_decay * state + self._tau * np.vstack((a[:,0],
                                                               a[:,1])).T

        # new_phis = np.clip(new_phis, -self._parameters["phi max"], self._parameters["phi max"])

        if N > 1:
            return new_phis
        else:
            return np.squeeze(new_phis, axis=0)

    def state_2_wheel_speed(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,3:5]
        else:
            return state[0,3:5]
    
    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return (state[:,3] - state[:,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]
        else:
            return (state[0,3] - state[0,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]

    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return (state[:,3] + state[:,4]) * self._parameters["wheel radius"] / 2
        else:
            return (state[0,3] + state[0,4]) * self._parameters["wheel radius"] / 2
    
                                                        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = DifferentialDrive(sampling_period=0.1)
    curr_state = np.array([[0,0,0,1,0.5],
                           [0,0,0,1,1]])
    # curr_state = np.array([0,0,0,1,0.5])

    pos = []
    vel = []

    for i in range(50):
        pos.append(M.state_2_position(curr_state))
        vel.append(M.state_2_velocity(curr_state))
        curr_state = M.step(curr_state, input_dict={"T_R": np.array([1,1]), "T_L": np.array([1,0.5])})
        # curr_state = M.step(curr_state, input_torque=np.array([0.5,0.5]))

    pos = np.array(pos)
    vel = np.array(vel)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(pos[:,0], pos[:,1])
    plt.axis("equal")
    plt.subplot(1,2,2)
    plt.plot(vel)
    plt.show()
