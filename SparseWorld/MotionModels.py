'''
Implements motion models for the robot.
Assume inputs to the actual robot will be torque.
'''

from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np

class MotionModel(ABC):
    def __init__(self, sampling_period=1) -> None:
        self._tau = sampling_period
        self._state = None
        self._parameters = {}

    @property
    def sampling_period(self):
        return self._tau

    @property
    def state(self):
        return self._state

    @property
    def parameters(self):
        return self._parameters

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
        self._state = np.zeros((5,))
        self._parameters = {
            "wheel radius": 1,
            "wheel moment of inertia": 1,
            "axel length": 1,
            "wheel friction": 0.1
        }
        if paremeters_dict is not None:
            self._parameters.update(paremeters_dict)

    def step(self, input_torque) -> None:
        '''
        State is defined as [x,y,theta,phi_R,phi_L]
        Input is defined as [T_R,T_L]
        '''
        v = self._parameters["wheel radius"] / 2 * (self._state[3] + self._state[4])
        w = (self._state[3] - self._state[4]) / self._parameters["axel length"]
        a = input_torque / self._parameters["wheel moment of inertia"]
        decay = self._parameters["wheel friction"]
        self._state = self._state + self._tau * np.array([v * np.cos(self._state[2]),
                                                          v * np.sin(self._state[2]),
                                                          w,
                                                          a[0] - decay * self._state[3],
                                                          a[1] - decay * self._state[4]])

    @property
    def position(self):
        return self._state[0:2]

    @property
    def velocity(self):
        return self._parameters["wheel radius"] / 2 * (self._state[3] + self._state[4])

    @property
    def heading(self):
        return self._state[2]

    @property
    def yaw_rate(self):
        return (self._state[3] - self._state[4]) / self._parameters["axel length"]

                                                        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = DifferentialDrive(sampling_period=0.1)
    M._state = np.array([0,0,0,1,0.5])

    pos = []
    vel = []

    for i in range(50):
        pos.append(M.state[0:2])
        vel.append(np.sum(M.state[3:5])/2)
        M.step(np.array([0.5,0.5]))

    pos = np.array(pos)
    vel = np.array(vel)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(pos[:,0], pos[:,1])
    plt.axis("equal")
    plt.subplot(1,2,2)
    plt.plot(vel)
    plt.show()

    
