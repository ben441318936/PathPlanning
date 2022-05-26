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
from Map import OccupancyGrid

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
    def pose(self) -> np.ndarray:
        return self._motion_model.state_2_pose(self.state)

    @property
    def position(self) -> np.ndarray:
        return self._motion_model.state_2_position(self.state)

    @property
    def target(self) -> np.ndarray:
        return self._target_position

    def set_agent_state(self, state: np.ndarray = None) -> None:
        self._estimator.init_estimator(state)

    def set_target_position(self, target_pos: np.ndarray = None) -> None:
        if self.target is not None and np.sum(np.abs(target_pos - self.target)) == 0:
            self._current_stop = None
        self._target_position = target_pos

    def control(self, tol: float = 0.5) -> dict:
        if self._current_stop is None or not self._planner.path_valid(self.position):
            # print("Replanning at", self.position)
            if self._planner.plan(self.position, self.target):
                # print("New path", self.path)
                self._current_stop = self._planner.next_stop()
            else:
                print("Couldn't plan a path from", self.position, "to", self.target)
                exit()
        elif np.linalg.norm(self.position - self._current_stop) < tol:
            self._current_stop = self._planner.next_stop()
        c = self._controller.control(self.state, self._current_stop, tol=tol)
        self._estimator.predict(c)
        return c

    def reached_target(self, tol: float = 0.5) -> bool:
        return np.linalg.norm(self.position - self.target) < tol

    @abstractmethod
    def process_observation(self, observation: dict) -> None:
        # this depends on the representation of the environment
        # as well as the type of estimation and observation
        pass

class OccupancyGridAgent(Agent):
    '''
    An agent that uses an occupancy grid to record the environment.
    '''

    __slots__ = ("_map", "_scan_angular_range", "_scan_angles", "_scan_resolution", "_scan_max_range")

    def __init__(self, motion_model: MotionModel, planner: Planner, controller: Controller, estimator: Estimator, map: OccupancyGrid, 
                 scan_angular_range: tuple = (-np.pi, np.pi), scan_resolution: float = 5/180*np.pi, scan_max_range: float = 5) -> None:
        super().__init__(motion_model, planner, controller, estimator)
        self._map: OccupancyGrid = map
        self._scan_angular_range: tuple = scan_angular_range
        self._scan_resolution: float = scan_resolution
        self._scan_max_range: float = scan_max_range
        self._init_scan_angles() # this sets self._scan_angles

    def _init_scan_angles(self) -> None:
        num_points = int(np.ceil((self._scan_angular_range[1] - self._scan_angular_range[0]) / self._scan_resolution))
        self._scan_angles = np.linspace(self._scan_angular_range[0], self._scan_angular_range[1], num_points, endpoint=True)

    @property
    def binary_map(self) -> np.ndarray:
        return self._map.get_binary_map()

    @property
    def scan_angular_range(self) -> tuple:
        return self._scan_angular_range

    @property
    def scan_angles(self) -> np.ndarray:
        return self._scan_angles

    @property
    def scan_resolution(self) -> float:
        return self._scan_resolution

    @property
    def scan_max_range(self) -> float:
        return self._scan_max_range

    def control(self, tol: float = 0.5) -> dict:
        return super().control(min([tol, self._map.resolution]))

    def reached_target(self, tol: float = 0.5) -> bool:
        return super().reached_target(min([tol, self._map.resolution]))

    def process_observation(self, observation: dict) -> None:
        # update our state estimates using observations before making updates to our map and planner
        if "LIDAR" in observation:
            observation["LIDAR"] = {"SCANS": observation["LIDAR"], "MAX_RANGE": self._scan_max_range}
        self._estimator.update(observation)
        if "LIDAR" in observation:
            self._map.update_map(self.pose, observation["LIDAR"])
            self._planner.update_environment(self.pose, observation["LIDAR"])


if __name__ == "__name__":
    pass
        




    


    







