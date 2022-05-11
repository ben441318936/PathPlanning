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

from MotionModels import MotionModel, DifferentialDrive, DifferentialDriveVelocityInput

class Estimator(ABC):
    '''
    Defines basic interface for estimator objects.
    '''

    __slots__ = ("_motion_model")

    def __init__(self, motion_model: MotionModel) -> None:
        self._motion_model = motion_model

    # predict the next state using known input and motion model
    @abstractmethod
    def predict(self, control_input) -> None:
        pass

    # update the current state using latest observation
    @abstractmethod
    def update(self, observation) -> None:
        pass

    # extract the most probable state for control use
    @abstractmethod
    def get_estimate(self) -> np.ndarray:
        pass

class WheelSpeedEstimator(Estimator):
    '''
    Stationary Kalman Filter for the wheel speeds using torque and encoder reading.
    '''

    __slots__ = ("_phi_R", "_phi_L")
    