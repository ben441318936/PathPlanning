from abc import ABC, abstractmethod
import numpy as np

from MotionModels import MotionModel

class Agent(ABC):
    def __init__(self, motion_model: MotionModel) -> None:
        self._motion_model = motion_model

class ParticleOccupancyAgent(Agent):
    '''
    Agent that performs SLAM using particle filter with an occupancy grid.
    '''
    def __init__(self, motion_model: MotionModel, num_particles=1) -> None:
        super().__init__(motion_model)
        self._N = num_particles
        self._particles = np.zeros((self._N, self._motion_model.state_dim))





