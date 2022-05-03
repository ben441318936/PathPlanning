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
    def step(self, input) -> None:
        pass

class DifferentialDrive(MotionModel):
    '''
    Parameters are:
        wheel radius
        wheel moment of inertial
        axel length
        wheel friction: [0,1), controls how much the wheel velocity decays

    State is defined as [x,y,theta,phi_R,phi_L]
    Input is defined as [T_R,T_L]
    '''

    def __init__(self, sampling_period=1, paremeters_dict=None) -> None:
        super().__init__(sampling_period)
        self.state_dim = 5
        self.input_dim = 2
        self.parameters = {
            "wheel radius": 1,
            "wheel moment of inertia": 1,
            "axel length": 1,
            "wheel friction": 0.1
        }
        if paremeters_dict is not None:
            self._parameters.update(paremeters_dict)

    def step(self, state: np.ndarray, input_torque: np.ndarray) -> np.ndarray:
        '''
        State is defined as [x,y,theta,phi_R,phi_L]
        Input is defined as [T_R,T_L]

        Supports vectorized operations for N states and inputs
        States [N,5]
        Inputs [N,2]
        '''

        state = state.reshape((-1,5))
        input_torque = input_torque.reshape((-1,2))

        v = (self._parameters["wheel radius"] / 2 * (state[:,3] + state[:,4]))
        w = ((state[:,3] - state[:,4]) / self._parameters["axel length"])
        a = input_torque / self._parameters["wheel moment of inertia"]
        decay = self._parameters["wheel friction"]

        new_state = state + self._tau * np.vstack((v * np.cos(state[:,2]),
                                                   v * np.sin(state[:,2]),
                                                   w,
                                                   a[:,0] - decay * state[:,3],
                                                   a[:,1] - decay * state[:,4])).T
        return new_state

                                                        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = DifferentialDrive(sampling_period=0.1)
    curr_state = np.array([[0,0,0,1,0.5],
                           [0,0,0,1,1]])
    curr_state = curr_state.reshape((-1,5))

    pos = []
    vel = []

    for i in range(50):
        pos.append(curr_state[1,0:2])
        vel.append(np.sum(curr_state[1,3:5])/2)
        curr_state = M.step(curr_state, input_torque=np.array([[0.5,0.5],
                                                               [0.5,0.5]]))

    pos = np.array(pos)
    vel = np.array(vel)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(pos[:,0], pos[:,1])
    plt.axis("equal")
    plt.subplot(1,2,2)
    plt.plot(vel)
    plt.show()

    
