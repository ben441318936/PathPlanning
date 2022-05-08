'''
Implements agent classes.

Agent is supposed to incoporate planning and control.
    Planning: SLAM based on internal sensors and LIDAR scans.
    Control: Execute the planned path.
'''

from abc import ABC, abstractmethod
import numpy as np

from MotionModels import MotionModel
from Controller import Controller

class Agent(ABC):
    def __init__(self, motion_model: MotionModel, controller: Controller) -> None:
        self._motion_model = motion_model
        self._controller = controller

    # determine a trajectory to follow
    # or at least the next target position
    @abstractmethod
    def plan(self):
        pass

    # determine the next control action
    @abstractmethod
    def control(self):
        pass

class ParticleOccupancyAgent(Agent):
    '''
    Agent that performs SLAM using particle filter with an occupancy grid.
    '''
    def __init__(self, motion_model: MotionModel, num_particles=1) -> None:
        super().__init__(motion_model)
        self._N = num_particles
        self._particles = np.zeros((self._N, self._motion_model.state_dim))





