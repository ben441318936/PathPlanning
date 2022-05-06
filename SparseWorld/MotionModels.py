'''
Implements motion models for the robot.
'''

from abc import ABC, abstractmethod
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
    def step(self, state: np.ndarray, input: np.ndarray, braking=False) -> None:
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

    def step(self, state: np.ndarray, input_velocities: np.ndarray, braking=False) -> np.ndarray:
        '''
        State is defined as [x,y,theta]
        Input is defined as [v,w]

        Supports vectorized operations for N states and inputs
        States [N,3]
        Inputs [N,2]
        '''
        if braking:
            input_velocities = np.array([0,0])

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
            "wheel radius": 1,
            "robot mass": 1,
            "axel length": 1,
            "wheel friction": 0.1,
            "braking friction": 0.5
        }
        if paremeters_dict is not None:
            self._parameters.update(paremeters_dict)

    def step(self, state: np.ndarray, input_torque: np.ndarray, braking=False) -> np.ndarray:
        '''
        State is defined as [x,y,theta,phi_R,phi_L]
        Input is defined as [T_R,T_L]

        Supports vectorized operations for N states and inputs
        States [N,5]
        Inputs [N,2]
        '''

        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        input_torque = input_torque.reshape((-1,2))

        v = (state[:,3] + state[:,4]) * self._parameters["wheel radius"] / 2
        w = (state[:,3] - state[:,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]
        a = input_torque / (self._parameters["robot mass"] * self._parameters["wheel radius"]**2)
        
        if braking:
            decay = self._parameters["braking friction"]
        else:
            decay = self._parameters["wheel friction"]

        new_state = state + self._tau * np.vstack((v * np.cos(state[:,2]),
                                                   v * np.sin(state[:,2]),
                                                   w,
                                                   a[:,0] - decay * state[:,3],
                                                   a[:,1] - decay * state[:,4])).T
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

    
