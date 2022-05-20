'''
Implements agent classes.

Agent is supposed to incoporate planning, control and estimation.
    Planning: Goal direction navigation based on current knowledge of the environment.
    Control: Execute the planned path.
    Estimation: Track robot state and environment information.
'''

from abc import ABC, abstractmethod
import numpy as np

from MotionModel import MotionModel
from Controller import Controller
from Planner import Planner
from Estimator import Estimator

class Agent(ABC):

    __slots__ = ("_motion_model", "_planner", "_controller", "_estimator", "_target_position", "_current_stop")

    def __init__(self, motion_model: MotionModel, planner: Planner, controller: Controller, estimator: Estimator) -> None:
        self._motion_model: MotionModel = motion_model
        self._planner: Planner = planner
        self._controller: Controller = controller
        self._estimator: Estimator = estimator
        self._target_position: np.ndarray = None
        self._current_stop: np.ndarray = None

    @property
    def motion_model(self) -> MotionModel:
        return self._motion_model

    @property
    def planner(self) -> Planner:
        return self._planner

    @property
    def controller(self) -> Controller:
        return self._controller

    @property
    def estimator(self) -> Estimator:
        return self._estimator

    @property
    def path(self) -> np.ndarray:
        return self._planner.path

    # the state of the robot from the agent POV is the best estimate we have from the estimator
    @property
    def state(self) -> np.ndarray:
        return self._estimator.estimate

    @property
    def target(self) -> np.ndarray:
        return self._target_position

    def set_agent_state(self, state: np.ndarray = None) -> None:
        self._estimator.init_estimator(state)

    def set_target_position(self, target_pos: np.ndarray = None) -> None:
        self._target_position = target_pos

    def control(self) -> dict:
        if self._current_stop is None:
            self._planner.plan(self.state, self.target)
            self._current_stop = self._planner.next_stop()
        c = self._controller.control(self.state, self._current_stop)
        self._estimator.predict(c)
        return c

    def reached_target(self, tol: float = 0.5) -> bool:
        return np.linalg.norm(self._motion_model.state_2_position(self.state) - self.target)

    @abstractmethod
    def process_observation(self, observation) -> None:
        # this depends on the representation of the environment
        # as well as the type of estimation and observation
        pass



    


    







