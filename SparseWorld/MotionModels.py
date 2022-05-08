'''
Implements motion models for the robot.
'''

from abc import ABC, abstractmethod
from cmath import pi
import numpy as np

class MotionModel(ABC):
    def __init__(self, sampling_period=1) -> None:
        self._tau = sampling_period
        self._parameters = {}
        self._state_dim = 0
        self._input_dim = 0

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
    def step(self, state: np.ndarray, input: np.ndarray) -> None:
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
    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        pass

class DifferentialDriveVelocityInput(MotionModel):
    def __init__(self, sampling_period=0.1, paremeters_dict=None) -> None:
        super().__init__(sampling_period)
        self._state_dim = 3
        self._input_dim = 2
        self._parameters = {
            "wheel radius": 1,
            "robot mass": 1,
            "axel length": 1,
            "wheel friction": 0.1,
            "braking friction": 0.5
        }
        if paremeters_dict is not None:
            self._parameters.update(paremeters_dict)

    def step(self, state: np.ndarray, input_velocities: np.ndarray) -> np.ndarray:
        '''
        State is defined as [x,y,theta]
        Input is defined as [v,w]

        Supports vectorized operations for N states and inputs
        States [N,3]
        Inputs [N,2]
        '''

        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        input_velocities = input_velocities.reshape((-1,self._input_dim))

        v = input_velocities[:,0]
        w = input_velocities[:,1]
        
        # if braking:
        #     decay = self._parameters["braking friction"]
        # else:
        #     decay = self._parameters["wheel friction"]

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


class DifferentialDrive(MotionModel):
    '''
    Parameters are:
        wheel radius
        robot mass: includes body and both wheels
        axel length
        wheel friction: [0,1), controls how much the wheel velocity decays nominally
        braking friction: [0,1), controls how much the wheel velocity decays when braking

    State is defined as [x,y,theta,phi_R,phi_L]
    Input is defined as [T_R,T_L]
    '''

    def __init__(self, sampling_period=1, paremeters_dict=None) -> None:
        super().__init__(sampling_period)
        self._state_dim = 5
        self._input_dim = 2
        self._parameters = {
            "wheel radius": 0.5,
            "robot mass": 20,
            "axel length": 1,
            "wheel friction": 0.1,
            "max wheel rpm": 60,
            "max motor torque": 100
        }
        if paremeters_dict is not None:
            self._parameters.update(paremeters_dict)
        self._parameters["inertia"] = self._parameters["robot mass"] / 2 * self._parameters["wheel radius"]**2
        self._parameters["phi max"] = self._parameters["max wheel rpm"] / 60 * 2 * np.pi

    def step(self, state: np.ndarray, input_torque: np.ndarray) -> np.ndarray:
        '''
        State is defined as [x,y,theta,phi_R,phi_L]
        Input is defined as [T_R,T_L]

        Supports vectorized operations for N states and inputs
        States [N,5]
        Inputs [N,2]
        '''

        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        input_torque = input_torque.reshape((-1,self._input_dim))
        input_torque = np.clip(input_torque, -self._parameters["max motor torque"], self._parameters["max motor torque"])

        v = (state[:,3] + state[:,4]) * self._parameters["wheel radius"] / 2
        w = (state[:,3] - state[:,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]
        a = input_torque / self._parameters["inertia"]
        
        phi_decay = np.exp(-self._parameters["wheel friction"]*self.sampling_period)
        state_decay = np.array([1,1,1,phi_decay,phi_decay]).reshape((-1,self._state_dim))

        new_state = state_decay * state + self._tau * np.vstack((v * np.cos(state[:,2]),
                                                                 v * np.sin(state[:,2]),
                                                                 w,
                                                                 a[:,0],
                                                                 a[:,1])).T
        new_state[:,3:5] = np.clip(new_state[:,3:5], -self._parameters["phi max"], self._parameters["phi max"])
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
        curr_state = M.step(curr_state, input_torque=np.array([[0.5,0.5],
                                                               [0.5,0.5]]))
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

    
